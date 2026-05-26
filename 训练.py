# 命令：CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run scripts/train.py pi05_libero --exp-name=test_openpi --num-train-steps=10 --overwrite
# --overwrite`，如果这个目录已存在，会先删掉重建

# 1.先是调用训练函数来训练模型，然后将训练好的模型保存下来，最后创建一个WebSocket服务器来提供服务。
from scripts.train import main  # 279行
# 调用 main() 来训练模型，此时导入训练配置  pi05_libero
# 入口在 [scripts/train.py](C:/QClaw/FASTER_hy/scripts/train.py:279)

# 2.创建训练配置实例 pi05_libero 594/755行
from src.openpi.training.config import TrainConfig 
# 配置内容
#     --exp-name=test_openpi      -> config.exp_name = "test_openpi"
#     --num-train-steps=10        -> config.num_train_steps = 10
#     --overwrite                 -> config.overwrite = True
# main(_config.cli())从 _CONFIGS_DICT里按名字取配置，TrainConfig(name="pi05_libero", ...)
# 调用了：
#     model=Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)
#     data=LeRobotLiberoDataConfig(
#         repo_id="physical-intelligence/libero",
#         base_config=DataConfig(prompt_from_task=True),
#         extra_delta_transform=False,
#     )
#     batch_size=256
#     num_workers=8
#     weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")
#     ema_decay=0.999
# 所以：
#     - 模型是 `Pi0`，但配置为 `pi05=True`
#     - 每次预测/训练的 action chunk 长度是 `10`
#     - action dim 默认是 `32`，LIBERO 真实动作会 pad 到 32
#     - checkpoint 初始权重来自 `pi05_base`
#     - 训练 10 step 后保存最终 checkpoint
# checkpoint 路径会是：
#     ./checkpoints/pi05_libero/test_openpi

# 3.启动训练
from scripts.train import main  # 194行main(config)正式开始初始化
# 3.1 init_logging()
# 3.2 检查 batch_size 是否能被 device_count 整除
# 设置 JAX compilation cache
# 创建 rng
# 创建 mesh / sharding
# 初始化 checkpoint_dir
# 初始化 wandb
# 创建 data_loader
# 取第一个 batch
# 初始化模型和 optimizer
# JIT train_step
# 进入训练循环
# 3.13 保存 checkpoint




from src.openpi.serving.websocket_policy_server import create_trained_policy



# 4. data loader 构造
from src.openpi.training.data_loader import create_data_loader 
from src.openpi.training.config import TrainConfig
# data_config = config.data.create(config.assets_dirs, config.model) # 调用LeRobotLiberoDataConfig的create
# LIBERO 数据会经历这些 transforms：
# LeRobot dataset sample
#     -> PromptFromLeRobotTask
#     -> RepackTransform
#     -> LiberoInputs
#     -> Normalize
#     -> PI05 model transforms
#     -> Observation.from_dict + actions

# dataset 原字段
# -> repack:
#    image -> observation/image
#    wrist_image -> observation/wrist_image
#    state -> observation/state
#    actions -> actions
#    prompt -> prompt

# -> LiberoInputs:
#    observation/image -> image["base_0_rgb"]
#    observation/wrist_image -> image["left_wrist_0_rgb"]
#    zero image -> image["right_wrist_0_rgb"]
#    state -> state
#    actions -> actions

# -> Normalize:
#    state/actions 按 assets 中的 norm stats 归一化

# -> ModelTransformFactory 的 PI05 分支:
#    InjectDefaultPrompt
#    ResizeImages(224, 224)
#    TokenizePrompt(PaligemmaTokenizer, discrete_state_input=False)
#    PadStatesAndActions(action_dim=32, action_horizon=10)

# 最后 [DataLoaderImpl](C:/QClaw/FASTER_hy/src/openpi/training/data_loader.py:530) 会 yield：
# Observation.from_dict(batch), batch["actions"]

# 也就是训练 step 收到：
# observation: images / image_masks / state / tokenized_prompt
# actions: [batch, 10, 32]

# 5. 模型初始化
from scripts.train import init_train_state
# model = config.model.create(model_rng)
from src.openpi.models.pi0_config import Pi0Config # config.model` 是 Pi0Config(pi05=True, ...)，
# 创建一个 Pi0 模型实例，且 `pi05=True` 会让模型在 forward 时走 pi0.5 的逻辑。
# return Pi0(self, rngs=nnx.Rngs(rng))

# 创建 optimizer
# eval_shape 得到 train_state_shape
# 从 gs://openpi-assets/checkpoints/pi05_base/params 加载权重
# 把权重合进模型
# 创建 TrainState

6. 每个训练 step 做什么**

训练循环在 [train.py](C:/QClaw/FASTER_hy/scripts/train.py:260)。

train_state, info = ptrain_step(train_rng, train_state, batch)

`ptrain_step` 是 JIT 后的 [train_step](C:/QClaw/FASTER_hy/scripts/train.py:138)。

# 内部调用链：
# train_step
#     -> nnx.merge(state.model_def, state.params)
#     -> model.train()
#     -> loss_fn
#     -> model.compute_loss(rng, observation, actions, train=True)
#     -> mean loss
#     -> value_and_grad
#     -> optimizer update
#     -> EMA update
#     -> 返回 loss / grad_norm / param_norm

对 `pi05_libero` 来说，`model.compute_loss` 实际调用 [Pi0.compute_loss](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:189)。

PI0 loss 逻辑是：

```text
actions 真值
+ noise
+ random time

-> x_t = time * noise + (1 - time) * actions
-> u_t = noise - actions

observation -> embed_prefix
x_t + time -> embed_suffix
PaliGemma / action expert forward
-> v_t

loss = mean((v_t - u_t)^2)
```

也就是说，训练目标是让模型学会从 noisy action chunk 预测 flow velocity，把 noise 逐步推向真实 action。

**8. 10 step 时会发生什么**

因为你设置了：

```text
--num-train-steps=10
```

循环是：

```python
range(start_step, config.num_train_steps)
```

如果不是 resume，`start_step=0`，所以跑：

```text
step 0, 1, 2, ..., 9
```

日志：

- `log_interval=100`
- 所以只会在 `step 0` 打一次训练指标

保存：

```python
if step == config.num_train_steps - 1:
    save_state(...)
```

所以会在 `step 9` 保存最终 checkpoint。

**一句话总链路**

```text
命令行
-> tyro 选择 pi05_libero 并覆盖 exp_name/steps/overwrite
-> main()
-> 创建 LIBERO data loader
-> 加载 pi05_base 权重并初始化 Pi0(pi05=True)
-> 每个 batch 做 Pi0.compute_loss
-> optimizer 更新参数
-> 跑 10 step
-> step 9 保存 checkpoint 到 checkpoints/pi05_libero/test_openpi
```

重点记住：这条命令训练的是 **PI0.5 LIBERO 连续 action chunk 模型**；它不会调用 `Policy.infer()`，也不会调用 `sample_actions_streaming_init/step()`。训练时走的是 `Pi0.compute_loss()`，推理时才会走 `sample_actions()`。