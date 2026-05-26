import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    # wandb 是训练日志工具；WANDB_MODE=offline 时仍会记录 loss 等指标，但保存到本地，不上传服务器。
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    # TrainState 是训练时的“完整状态包”：模型参数、optimizer 状态、当前 step、EMA 参数等都放在里面。
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    # 1.创建 optimizer
    # 2.eval_shape 得到 train_state_shape
    # 3.从 gs://openpi-assets/checkpoints/pi05_base/params 加载权重
    # 4.把权重合进模型
    # 5.创建 TrainState
    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # 用 model_rng 初始化模型结构和参数。对 pi05_libero 来说，这里会创建 Pi0Config(pi05=True) 对应的 Pi0。
        model = config.model.create(model_rng)      # return Pi0(self, rngs=nnx.Rngs(rng))

        # Merge the partial params into the model.
        if partial_params is not None:
            # partial_params 通常来自 base checkpoint；这里只覆盖 checkpoint 里存在的那部分参数。
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    # eval_shape 只推导数组形状和 dtype，不真正分配完整模型权重；后面用这些形状决定参数如何分片。
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    # 从配置里的 weight_loader 加载预训练权重；pi05_libero 默认从 pi05_base checkpoint 读取。
    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    # train_step 是“一次参数更新”的完整逻辑。外层会先把它 JIT 成 ptrain_step，
    # 然后在 for step in pbar 里反复调用。

    # 1. 用当前 TrainState 里的 model_def 和 params 还原出可执行模型。
    # model_def 是模型结构，params 是当前权重；二者合起来才是一个完整模型。
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        # 2. 定义 loss 计算方式。
        # 对 pi05_libero 来说，这里实际调用 Pi0.compute_loss：
        # 给真实 action 加噪声，随机取一个 time，让模型预测从 noisy action 走回真实 action 的速度场。
        # compute_loss 返回每个 action horizon 位置的 loss，外面再求平均得到一个标量 loss。
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    # 3. 为当前 step 生成专属随机 key。
    # fold_in 用当前 step 混入随机种子，保证每一步的数据增强、噪声、time 采样都不同且可复现。
    train_rng = jax.random.fold_in(rng, state.step)

    # 4. 拆出当前 batch。
    # observation 是图像、state、tokenized prompt 等模型输入；actions 是监督学习的真实 action chunk。
    observation, actions = batch

    # 5. 指定哪些参数参与反向传播。
    # DiffState 指定哪些参数参与求梯度；被 freeze_filter 冻住的参数不会更新。
    diff_state = nnx.DiffState(0, config.trainable_filter)

    # 6. 前向计算 loss，并对可训练参数求梯度。
    # value_and_grad 会同时返回 loss 和 grads；grads 的树结构与可训练参数一致。
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    # 7. 取出当前可训练参数，用 optimizer 根据梯度计算更新量。
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)

    # 8. 把 optimizer 给出的 updates 应用到参数上，得到新的可训练参数。
    new_params = optax.apply_updates(params, updates)

    # 9. 把新参数写回模型，再重新导出完整参数树。
    # 这里的完整参数树包括可训练参数和被冻结的参数。
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    # 10. 创建新的 TrainState：step 加 1，参数和 optimizer state 都换成更新后的版本。
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        # 11. 如果启用 EMA，就维护一份参数滑动平均。
        # EMA 参数通常用于更稳定的推理 checkpoint，不直接参与当前 step 的反向传播。
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # 12. 统计训练日志需要的数值。
    # 这里只统计 kernel 类参数的 norm，跳过 bias、scale、position embedding 等辅助参数。
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        # loss：当前 batch 的训练目标值。
        "loss": loss,
        # grad_norm：梯度整体大小，用来观察训练是否爆炸或过小。
        "grad_norm": optax.global_norm(grads),
        # param_norm：主要参数整体大小，用来辅助判断参数是否异常漂移。
        "param_norm": optax.global_norm(kernel_params),
    }
    # 13. 返回更新后的训练状态和日志指标；外层 for 循环会用 new_state 进入下一 step。
    return new_state, info

# 1. 训练脚本入口，初始化config，设置 pi05_libero 等配置实际内容，如下：
    # model=Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)
    # data=LeRobotLiberoDataConfig(
    #     repo_id="physical-intelligence/libero",
    #     base_config=DataConfig(prompt_from_task=True),
    #     extra_delta_transform=False,
    # )
    # batch_size=256
    # num_workers=8
    # weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")
    # ema_decay=0.999
def main(config: _config.TrainConfig):
    # config 是 tyro 根据命令行组装出的 TrainConfig。
    # 例如 `pi05_libero --exp-name=test_openpi --num-train-steps=10 --overwrite`
    # 会先选中 pi05_libero 预设，再覆盖实验名、训练步数和 checkpoint 覆盖策略。
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        # device_count 是 JAX 当前能看到的加速设备数；CUDA_VISIBLE_DEVICES=0 时通常是 1。
        # batch 会按设备均分，所以 batch_size 必须能整除设备数，否则每张卡拿到的数据量不一致。
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    # JAX compilation cache 是“编译结果缓存”：JAX 第一次运行某个 shape 的 jit 函数会编译 XLA 程序，
    # 这一步可能较慢；缓存目录可以让后续相同程序复用编译产物，减少重复启动训练时的等待。
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    # rng 是 random number generator key，即 JAX 的随机数种子句柄。
    # JAX 随机数是显式传递的：每次需要随机数时都 split 出新 key，避免隐藏的全局随机状态。
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    # mesh/sharding 描述数组如何放到 GPU 上。单卡时基本等价于“都放在这一张卡上”，
    # 多卡时才会体现数据并行或 FSDP 参数分片。
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 2. 初始化 checkpoint 目录；--overwrite 会删除已有的同名实验目录，--resume 则尝试继续训练。
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # 3.创建训练数据流。对 pi05_libero 来说，它会读取 LeRobot 的 physical-intelligence/libero，
    # 经过 repack、LiberoInputs、Normalize、TokenizePrompt、PadStatesAndActions 后变成模型 batch。
    # 从 dataset 原字段格式：
    #     -> repack:
    #     image -> observation/image
    #     wrist_image -> observation/wrist_image
    #     state -> observation/state
    #     actions -> actions
    #     prompt -> prompt

    #     -> LiberoInputs:
    #     observation/image -> image["base_0_rgb"]
    #     observation/wrist_image -> image["left_wrist_0_rgb"]
    #     zero image -> image["right_wrist_0_rgb"]
    #     state -> state
    #     actions -> actions

    #     -> Normalize:
    #     state/actions 按 assets 中的 norm stats 归一化

    #     -> ModelTransformFactory 的 PI05 分支:
    #     InjectDefaultPrompt
    #     ResizeImages(224, 224)
    #     TokenizePrompt(PaligemmaTokenizer, discrete_state_input=False)
    #     PadStatesAndActions(action_dim=32, action_horizon=10)
    # 变成 Observation.from_dict(batch), batch["actions"]返回的
    #     observation: images / image_masks / state / tokenized_prompt
    #     actions: [batch, 10, 32]
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    # 4. 初始化模型参数、加载 pi05_base 预训练权重，并创建 optimizer/EMA 状态。
    # 4.1.创建 optimizer
    # 4.2.eval_shape 得到 train_state_shape
    # 4.3.从 gs://openpi-assets/checkpoints/pi05_base/params 加载权重
    # 4.4.把权重合进模型
    # 4.5.创建 TrainState
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)

    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # ptrain_step 是 JIT 编译后的 train_step函数（在这个python文件中定义）。第一次执行会触发编译，后续 step 会复用编译结果。
        # train_step        = Python 版的一步训练逻辑
        # ptrain_step       = JAX JIT 编译后的 train_step，更适合在 GPU 上反复跑
        # for step in pbar  = 跑很多次训练 step 的循环
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    # 后续要执行的一连串的 train_step ，预编译
    #     -> nnx.merge(state.model_def, state.params)
    #     -> model.train()
    #     -> loss_fn
    #     -> model.compute_loss(rng, observation, actions, train=True)
    #     -> mean loss
    #     -> value_and_grad
    #     -> optimizer update
    #     -> EMA update
    #     -> 返回 loss / grad_norm / param_norm
    

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        # -> 当前 batch
        #     -> ptrain_step(...)
        #     -> 内部调用 train_step(...)
        #         -> model.compute_loss(...)
        #         -> 计算梯度
        #         -> optimizer 更新参数
        #         -> 返回新的 train_state 和 loss/grad_norm 等 info
        #     -> 计算是否log、是否保存checkpoint
        # -> 取下一个 batch
        # -> 继续下一轮
        with sharding.set_mesh(mesh):
            # 真正的一次参数更新发生在这里：当前 batch -> loss/grads -> optimizer update。
            # for step in pbar 是“反复调用ptrain_step这个单步函数的训练循环”。
            train_state, info = ptrain_step(train_rng, train_state, batch)

        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            # 你的命令设置 num_train_steps=10，所以最后会在 step 9 保存一次 checkpoint。
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
