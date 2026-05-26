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
    # 2.7 初始化 wandb：WANDB_MODE=offline 时仍记录 loss 等指标，但保存到本地，不上传服务器。
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
    # 4.1 创建 optimizer：optimizer 根据梯度更新模型参数，lr_schedule 决定每一步学习率。
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # 4.2 创建模型：对 pi05_libero 来说，这里会调用 Pi0Config.create，得到 Pi0(pi05=True)。
        model = config.model.create(model_rng)

        if partial_params is not None:
            # 4.7 把预训练权重合进模型：checkpoint 中存在的参数覆盖初始化参数。
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        # 4.8 创建 TrainState：保存 step、模型参数、optimizer 状态和 EMA 参数。
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    # 4.3 eval_shape 得到 train_state_shape：只推导形状和 dtype，不真正分配完整模型权重。
    train_state_shape = jax.eval_shape(init, init_rng)
    # 4.4 根据 shape 创建 sharding：决定 TrainState 的数组如何放到设备上。
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    # 4.5/4.6 加载并验证预训练权重；pi05_libero 默认从 pi05_base checkpoint 读取。
    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 4.7/4.8 在 JIT 中完成权重合并和 TrainState 创建。
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
    # 7. 单个 train_step 内部逻辑：这是“一次参数更新”的完整流程。
    # 外层会先把它 JIT 成 ptrain_step，再在 for step in pbar 中反复调用。

    # 7.1 还原模型：用当前 TrainState 里的 model_def 和 params 还原出可执行模型。
    # model_def 是模型结构，params 是当前权重；二者合起来才是一个完整模型。
    model = nnx.merge(state.model_def, state.params)
    # 7.2 设置训练模式：启用训练态行为，例如 train=True 的预处理/增强逻辑。
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        # 7.3 定义 loss_fn：对 pi05_libero 来说，这里实际调用 Pi0.compute_loss。
        # 给真实 action 加噪声，随机取一个 time，让模型学习 flow velocity；推理时用负 dt 从 noise 走回 action。
        # compute_loss 返回每个 action horizon 位置的 loss，外面再求平均得到一个标量 loss。
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    # 7.4 为当前 step 生成随机 key：fold_in 把 step 混入 rng，保证每一步随机性不同且可复现。
    train_rng = jax.random.fold_in(rng, state.step)

    # 7.5 拆出 batch。
    # observation 是图像、state、tokenized prompt 等模型输入；actions 是监督学习的真实 action chunk。
    observation, actions = batch

    # 7.6 指定可训练参数：被冻结的参数不参与梯度更新。
    # DiffState 指定哪些参数参与求梯度；被 freeze_filter 冻住的参数不会更新。
    diff_state = nnx.DiffState(0, config.trainable_filter)

    # 7.7 前向计算 loss 并求梯度。
    # value_and_grad 会同时返回 loss 和 grads；grads 的树结构与可训练参数一致。
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    # 7.8 optimizer 根据梯度计算 updates。
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)

    # 7.9 应用 updates，得到新的可训练参数。
    new_params = optax.apply_updates(params, updates)

    # 7.10 写回模型并导出完整参数树。
    # 这里的完整参数树包括可训练参数和被冻结的参数。
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    # 7.11 更新 TrainState：step 加 1，参数和 optimizer state 都换成更新后的版本。
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        # 7.12 更新 EMA 参数：维护一份更平滑的参数滑动平均。
        # EMA 参数通常用于更稳定的推理 checkpoint，不直接参与当前 step 的反向传播。
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # 7.13 统计训练日志指标。
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
    # 7.13 返回更新后的训练状态和日志指标；外层 for 循环会用 new_state 进入下一 step。
    return new_state, info

def main(config: _config.TrainConfig):
    # 2. 进入 main 并准备训练环境。
    # config 已由 tyro 解析完成：先选中 pi05_libero，再覆盖 exp_name、num_train_steps、overwrite。
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        # 2.2 检查 batch size 和 GPU 数量：batch 会按设备均分，因此必须整除 device_count。
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    # 2.3 设置 JAX compilation cache：缓存 JIT 编译产物，减少相同 shape 训练的编译等待。
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    # 2.4 创建随机数 key：JAX 显式传递 rng，split 后分别用于训练循环和模型初始化。
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    # 2.5 创建 mesh / sharding：描述 batch 和 TrainState 的数组如何放到 JAX 设备上。
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 2.6 初始化 checkpoint 目录；--overwrite 会删除已有实验目录，--resume 则尝试继续训练。
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # 3. 创建 Data Loader：pi05_libero 走 LeRobot / Torch 分支，并输出 Observation + actions。
    # 详细格式转换在 data_loader.create_data_loader 和 LeRobotLiberoDataConfig.create 中配置。
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

    # 4. 初始化模型和 TrainState：创建 Pi0(pi05=True)，加载 pi05_base 权重，并准备 optimizer/EMA 状态。
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)

    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        # 4.9 如果是 resume，则从 checkpoint 恢复旧状态；本命令使用 --overwrite，通常不会进入这里。
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # 5. 编译单步训练函数：ptrain_step 是 JIT 编译后的 train_step，供训练循环反复调用。
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    # 6.1/6.2 设置训练起点并创建进度条；num_train_steps=10 时循环 step 0..9。
    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        # 6.3 每轮调用 ptrain_step：当前 batch 完成一次 loss、梯度和参数更新。
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)

        # 6.4/6.5 收集并定期记录 loss、grad_norm、param_norm。
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        # 6.6 取下一个 batch，供下一轮 ptrain_step 使用。
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            # 6.7 保存 checkpoint；num_train_steps=10 时最后会在 step 9 保存一次。
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
