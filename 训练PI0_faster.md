# `pi05_faster_libero` 训练命令调用链路

运行命令：

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run scripts/train.py pi05_faster_libero --exp-name=test_faster --num-train-steps=10 --overwrite
```

这是一条训练命令，不是在线推理命令。它的实际主线是：从 `scripts/train.py` 入口读取 `pi05_faster_libero` 配置，创建 LIBERO 数据加载器，初始化 `Pi0Faster` 模型，然后在训练循环里反复调用 JIT 后的 `train_step`，而 `train_step` 内部调用 `Pi0Faster.compute_loss`。

## 1. 初始化 Config

### 1.1 Shell 环境变量先生效

代码：[train.py](scripts/train.py)（第 280 行）、[pyproject.toml](pyproject.toml)（第 18 行）、[pyproject.toml](pyproject.toml)（第 33 行）

`CUDA_VISIBLE_DEVICES=0` 是在 Python 进程启动前设置的 CUDA 环境变量，含义是只让 CUDA/JAX 看到第 0 张 GPU。后面 `jax.device_count()` 会基于这个可见设备集合返回设备数。

`WANDB_MODE=offline` 是 Weights & Biases 的环境变量，表示 wandb 仍然记录训练指标，但以离线模式写到本地，不主动上传云端。代码里仍会正常执行 `wandb.init(...)`。

### 1.2 `uv run` 进入 Python 脚本

代码：[pyproject.toml](pyproject.toml)（第 1 行）、[pyproject.toml](pyproject.toml)（第 8 行）、[train.py](scripts/train.py)（第 280 行）

`uv run scripts/train.py ...` 会在当前 Python 项目的依赖环境里启动 `scripts/train.py`。`pyproject.toml` 定义了项目名、Python 版本和依赖，例如 `jax[cuda12]`、`tyro`、`wandb`、`torch`、`lerobot` 等。

进入脚本后，Python 从文件底部入口开始：

```python
if __name__ == "__main__":
    main(_config.cli())
```

也就是说，真正传入 `main()` 的不是原始字符串参数，而是 `_config.cli()` 解析出来的 `TrainConfig` 对象。

### 1.3 `tyro` 解析 CLI 并选择配置

代码：[config.py](src/openpi/training/config.py)（第 1187 行）、[config.py](src/openpi/training/config.py)（第 1184 行）

`_config.cli()` 调用 `tyro.extras.overridable_config_cli(...)`。`tyro` 是命令行参数到 dataclass 的解析库；这里它把 `_CONFIGS_DICT` 里的每个配置名都注册为可选配置。

命令里的第一个位置参数是：

```text
pi05_faster_libero
```

所以最终选中 `_CONFIGS_DICT["pi05_faster_libero"]` 对应的 `TrainConfig`。

### 1.4 命令行参数覆盖默认配置

代码：[TrainConfig](src/openpi/training/config.py)（第 579 行）、[TrainConfig.__post_init__](src/openpi/training/config.py)（第 663 行）、[train.py](scripts/train.py)（第 194 行）

选中基础配置后，命令行参数会覆盖 dataclass 字段：

```text
--exp-name=test_faster      -> config.exp_name = "test_faster"
--num-train-steps=10        -> config.num_train_steps = 10
--overwrite                 -> config.overwrite = True
```

`--resume` 没有设置，所以 `config.resume = False`。`TrainConfig.__post_init__` 会禁止 `resume` 和 `overwrite` 同时为 True。

### 1.5 当前命令的最终关键配置

代码：[pi05_faster_libero 配置](src/openpi/training/config.py)（第 861 行）、[TrainConfig.checkpoint_dir](src/openpi/training/config.py)（第 651 行）、[Pi0FasterConfig](src/openpi/models/pi0_config.py)（第 106 行）

最终关键字段如下：

```text
name = "pi05_faster_libero"
exp_name = "test_faster"
model = Pi0FasterConfig(
    pi05=True,
    action_dim=32,
    action_horizon=10,
    max_token_len=200,
    discrete_state_input=False,
    max_delay=0,
    mix_prob=0.5,
    alpha=0.6,
    u0=0.9,
)
data = LeRobotLiberoDataConfig(
    repo_id="physical-intelligence/libero",
    base_config=DataConfig(prompt_from_task=True),
    extra_delta_transform=False,
)
batch_size = 256
num_workers = 8
weight_loader = CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")
num_train_steps = 10
optimizer = AdamW(clip_gradient_norm=1.0)
lr_schedule = CosineDecaySchedule(warmup_steps=10000, peak_lr=5e-5, decay_steps=1000000, decay_lr=5e-5)
ema_decay = 0.999
seed = 42
fsdp_devices = 1
checkpoint_dir = checkpoints/pi05_faster_libero/test_faster
```

`checkpoint_dir` 在代码中会被 `.resolve()` 解析成绝对路径，但文档里按仓库相对路径理解就是 `checkpoints/pi05_faster_libero/test_faster`。

### 1.6 `Pi0FasterConfig` 的模型类型仍是 PI05

代码：[Pi0FasterConfig.model_type](src/openpi/models/pi0_config.py)（第 128 行）、[ModelTransformFactory](src/openpi/training/config.py)（第 116 行）

`Pi0FasterConfig` 没有新增 `ModelType.PI0_FASTER`。因为当前配置 `pi05=True`，所以：

```text
config.model.model_type == ModelType.PI05
```

这会影响后面的 data/model transform 分支：`Pi0Faster` 复用 PI0.5 的输入格式，不走 `PI0_FAST` token-action 分支。

## 2. 进入 `main()` 并准备训练环境

### 2.1 初始化 logging

代码：[main](scripts/train.py)（第 194 行）、[init_logging](scripts/train.py)（第 31 行）

`main(config)` 第一件事是调用 `init_logging()`，设置日志等级和格式。后续会打印当前机器名、data loader shape、train state shape、loss、grad norm、checkpoint 保存等信息。

### 2.2 检查 batch size 和设备数

代码：[main batch 检查](scripts/train.py)（第 198 行）

代码要求：

```python
config.batch_size % jax.device_count() == 0
```

当前 `batch_size=256`。如果 `CUDA_VISIBLE_DEVICES=0` 生效且 JAX 只看到 1 张 GPU，则 `256 % 1 == 0`，检查通过。这个检查保证 batch 可以均匀切到可见设备上。

### 2.3 设置 JAX compilation cache

代码：[main JAX cache](scripts/train.py)（第 203 行）

JAX compilation cache 是 JAX/XLA 的编译缓存。JIT（just-in-time compilation，即即时编译）函数第一次遇到某组 shape/dtype 时会编译成设备程序，后续相同 shape 可以复用缓存，减少等待。

这里缓存目录是：

```text
~/.cache/jax
```

### 2.4 创建 RNG key

代码：[main rng](scripts/train.py)（第 205 行）

RNG 是 random number generator，即随机数生成器。JAX 不使用隐式全局随机状态，而是显式传递 key：

```text
rng = jax.random.key(config.seed)  # seed=42
train_rng, init_rng = jax.random.split(rng)
```

`train_rng` 用于训练 step 内部随机数，`init_rng` 用于模型初始化。

### 2.5 创建 mesh 和 sharding

代码：[main mesh/sharding](scripts/train.py)（第 208 行）、[make_mesh](src/openpi/training/sharding.py)（第 17 行）、[DATA_AXIS](src/openpi/training/sharding.py)（第 7 行）

mesh 描述设备网格，sharding 描述数组如何分布到设备上。当前 `fsdp_devices=1`，如果只有 1 张可见 GPU，mesh 近似为单设备：

```text
mesh axes = ("batch", "fsdp")
data_sharding = PartitionSpec(("batch", "fsdp"))
replicated_sharding = PartitionSpec()
```

FSDP 是 fully sharded data parallel，完整参数分片数据并行。当前 `fsdp_devices=1` 时不会真正把大参数分到多张卡上，基本等价于单卡复制。

### 2.6 初始化 checkpoint 目录

代码：[main checkpoint](scripts/train.py)（第 212 行）、[initialize_checkpoint_dir](src/openpi/training/checkpoints.py)（第 20 行）

checkpoint 是训练中保存下来的模型参数、optimizer 状态、EMA 参数和 assets。当前目录是：

```text
checkpoints/pi05_faster_libero/test_faster
```

因为命令带了 `--overwrite`，如果这个目录已经存在，`initialize_checkpoint_dir` 会删除旧目录并重建。`resume=False`，所以不会从旧 checkpoint 恢复。

### 2.7 初始化 wandb

代码：[main init_wandb](scripts/train.py)（第 218 行）、[init_wandb](scripts/train.py)（第 50 行）

`init_wandb(config, resuming=False, enabled=True)` 会创建 wandb run，并把 run id 写入：

```text
checkpoints/pi05_faster_libero/test_faster/wandb_id.txt
```

由于外部设置了 `WANDB_MODE=offline`，这些指标会离线记录。代码层面仍然会执行 `wandb.log(...)`。

### 2.8 创建 data loader 并取第一个 batch

代码：[main data_loader](scripts/train.py)（第 220 行）、[main first batch](scripts/train.py)（第 225 行）

`main()` 调用：

```python
data_loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
data_iter = iter(data_loader)
batch = next(data_iter)
```

随后打印 batch shape，并把第一批图像拼接后写到 wandb，用于 sanity check。

## 3. 创建 Data Loader

### 3.1 `create_data_loader` 是数据入口

代码：[create_data_loader](src/openpi/training/data_loader.py)（第 223 行）、[create_data_loader 调 data.create](src/openpi/training/data_loader.py)（第 242 行）

`train.py` 只传入整体 `TrainConfig`，数据入口里先把 `config.data` 展开成真正的 `DataConfig`：

```python
data_config = config.data.create(config.assets_dirs, config.model)
```

当前 `config.data` 是 `LeRobotLiberoDataConfig(...)`，所以会进入 `LeRobotLiberoDataConfig.create(...)`。

### 3.2 `LeRobotLiberoDataConfig.create(...)` 生成 `DataConfig`

代码：[LeRobotLiberoDataConfig.create](src/openpi/training/config.py)（第 328 行）、[create_base_config](src/openpi/training/config.py)（第 217 行）

这个函数不直接读样本，而是定义样本进入模型前的转换流水线。基础字段来自 `create_base_config`：

```text
repo_id = "physical-intelligence/libero"
asset_id = "physical-intelligence/libero"
prompt_from_task = True
use_quantile_norm = True  # 因为 model_type 是 PI05，不是 PI0
norm_stats = 从 assets/pi05_faster_libero/physical-intelligence/libero 加载
```

如果 norm stats 不存在，后面的 `transform_dataset` 会报错，提示先运行 `scripts/compute_norm_stats.py`。

### 3.3 配置 `repack_transforms`

代码：[LIBERO repack_transform](src/openpi/training/config.py)（第 331 行）、[RepackTransform](src/openpi/transforms.py)（第 78 行）

`repack_transforms` 把 LeRobot 样本原始 key 改成 LIBERO adapter 期望的 key：

```text
image       -> observation/image
wrist_image -> observation/wrist_image
state       -> observation/state
actions     -> actions
prompt      -> prompt
```

这一步只处理字段名和嵌套结构。

### 3.4 配置 `data_transforms`

代码：[LIBERO data_transforms](src/openpi/training/config.py)（第 345 行）、[LiberoInputs](src/openpi/policies/libero_policy.py)（第 30 行）、[LiberoOutputs](src/openpi/policies/libero_policy.py)（第 68 行）

`data_transforms.inputs` 当前只有：

```python
LiberoInputs(model_type=ModelType.PI05)
```

它把 LIBERO 观测转成 openpi 通用格式：

```text
state
image["base_0_rgb"]
image["left_wrist_0_rgb"]
image["right_wrist_0_rgb"]
image_mask
actions
prompt
```

LIBERO 没有右腕图，所以 `right_wrist_0_rgb` 用零图占位；因为模型类型不是 `PI0_FAST`，右腕 mask 是 False。`extra_delta_transform=False`，所以当前命令不会额外执行 `DeltaActions`。

### 3.5 配置 `model_transforms`

代码：[ModelTransformFactory 调用](src/openpi/training/config.py)（第 357 行）、[ModelTransformFactory PI05 分支](src/openpi/training/config.py)（第 140 行）

因为 `Pi0FasterConfig(pi05=True).model_type == PI05`，所以进入 `ModelTransformFactory` 的 PI05 分支。当前模型侧 transforms 是：

```text
InjectDefaultPrompt(default_prompt=None)
ResizeImages(224, 224)
TokenizePrompt(PaligemmaTokenizer(max_token_len=200), discrete_state_input=False)
PadStatesAndActions(action_dim=32, action_horizon=10)
```

这说明 `Pi0Faster` 不走 `TokenizeFASTInputs`，也不会把 action 变成 FAST token；它仍然使用连续 action chunk。

### 3.6 `transform_dataset` 串联完整转换顺序

代码：[transform_dataset](src/openpi/training/data_loader.py)（第 172 行）、[Normalize](src/openpi/transforms.py)（第 112 行）

训练样本实际转换顺序是：

```text
repack_transforms.inputs
-> data_transforms.inputs
-> Normalize(norm_stats, use_quantiles=True)
-> model_transforms.inputs
```

也就是：

```text
RepackTransform
-> LiberoInputs
-> Normalize
-> InjectDefaultPrompt
-> ResizeImages
-> TokenizePrompt
-> PadStatesAndActions
```

Normalize 会把 state/actions 归一化到模型训练空间；`PadStatesAndActions` 会把真实 LIBERO 的 7 维 action pad 到模型需要的 32 维。

### 3.7 `create_data_loader` 判断 RLDS / Torch 分支

代码：[create_data_loader 分支](src/openpi/training/data_loader.py)（第 245 行）、[create_rlds_data_loader](src/openpi/training/data_loader.py)（第 338 行）、[create_torch_data_loader](src/openpi/training/data_loader.py)（第 272 行）

分支判断是：

```python
if data_config.rlds_data_dir is not None:
    return create_rlds_data_loader(...)
return create_torch_data_loader(...)
```

当前 `LeRobotLiberoDataConfig` 没有设置 `rlds_data_dir`，所以实际走 `create_torch_data_loader`。`create_rlds_data_loader` 只是在 DROID/RLDS 这类数据配置里才会走。

### 3.8 `create_torch_dataset` 创建 LeRobotDataset

代码：[create_torch_dataset](src/openpi/training/data_loader.py)（第 130 行）、[LeRobotDataset](src/openpi/training/data_loader.py)（第 140 行）

当前数据集来自：

```text
repo_id = "physical-intelligence/libero"
```

`LeRobotDataset` 使用 `delta_timestamps` 为每个样本取一段 action sequence。当前 `action_horizon=10`，所以会为 `"actions"` 取 10 个连续动作时间点。

### 3.9 从 LeRobot task 生成 prompt

代码：[PromptFromLeRobotTask](src/openpi/transforms.py)（第 310 行）、[create_torch_dataset prompt_from_task](src/openpi/training/data_loader.py)（第 148 行）

因为 `prompt_from_task=True`，`create_torch_dataset` 会包一层：

```python
TransformedDataset(dataset, [PromptFromLeRobotTask(dataset_meta.tasks)])
```

它从 LeRobot 样本里的 `task_index` 查表得到自然语言 prompt，并写入样本的 `"prompt"` 字段。

### 3.10 `TorchDataLoader` 做 batch、shuffle 和 worker

代码：[create_torch_data_loader](src/openpi/training/data_loader.py)（第 303 行）、[TorchDataLoader](src/openpi/training/data_loader.py)（第 379 行）、[_collate_fn](src/openpi/training/data_loader.py)（第 469 行）

当前 `framework="jax"`，所以：

```text
local_batch_size = batch_size // jax.process_count()
```

单进程时 `local_batch_size=256`。`TorchDataLoader` 内部用 `torch.utils.data.DataLoader`：

```text
shuffle=True
num_workers=8
drop_last=True
seed=42
collate_fn=_collate_fn
```

`_collate_fn` 会把单样本 stack 成 batch 维。

### 3.11 `TorchDataLoader.__iter__` 转成 sharded JAX array

代码：[TorchDataLoader.__iter__](src/openpi/training/data_loader.py)（第 450 行）、[JAX sharding 转换](src/openpi/training/data_loader.py)（第 463 行）

因为传入了 `data_sharding`，每个 batch 会通过：

```python
jax.make_array_from_process_local_data(self._sharding, x)
```

变成带 sharding 信息的 JAX array，供 JIT 后的 `ptrain_step` 直接消费。

### 3.12 `DataLoaderImpl` 最终 yield `Observation` 和 `actions`

代码：[DataLoaderImpl](src/openpi/training/data_loader.py)（第 527 行）、[Observation.from_dict](src/openpi/models/model.py)（第 109 行）、[Actions 类型](src/openpi/models/model.py)（第 141 行）

最终产出：

```python
yield _model.Observation.from_dict(batch), batch["actions"]
```

当前 batch 的核心 shape 是：

```text
Observation.images["base_0_rgb"]          [256, 224, 224, 3]
Observation.images["left_wrist_0_rgb"]    [256, 224, 224, 3]
Observation.images["right_wrist_0_rgb"]   [256, 224, 224, 3]  # 零图，占位
Observation.image_masks[...]              [256]
Observation.state                         [256, 32]
Observation.tokenized_prompt              [256, 200]
Observation.tokenized_prompt_mask         [256, 200]
actions                                   [256, 10, 32]
```

真实 LIBERO action 主要是 7 维，训练前被 pad 到 `action_dim=32`。

## 4. 初始化模型和 TrainState

### 4.1 `init_train_state` 入口

代码：[init_train_state](scripts/train.py)（第 84 行）、[main 调 init_train_state](scripts/train.py)（第 236 行）

`main()` 在拿到第一个 batch 后调用：

```python
train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
```

当前 `resuming=False`，所以会新建模型、加载预训练权重，并创建新的 `TrainState`。

### 4.2 创建 optimizer

代码：[init_train_state optimizer](scripts/train.py)（第 88 行）、[create_optimizer](src/openpi/training/optimizer.py)（第 105 行）、[AdamW](src/openpi/training/optimizer.py)（第 65 行）

`tx = create_optimizer(...)` 生成 Optax optimizer。当前是：

```text
CosineDecaySchedule(warmup_steps=10000, peak_lr=5e-5, decay_steps=1000000, decay_lr=5e-5)
AdamW(b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10, clip_gradient_norm=1.0)
```

`tx` 后面会根据梯度计算参数 updates。

### 4.3 创建 `Pi0Faster` 模型

代码：[init_train_state init](scripts/train.py)（第 90 行）、[config.model.create](scripts/train.py)（第 92 行）、[Pi0FasterConfig.create](src/openpi/models/pi0_config.py)（第 136 行）

`config.model.create(model_rng)` 调到：

```python
return Pi0Faster(self, rngs=nnx.Rngs(rng))
```

所以当前创建的具体模型类是 `Pi0Faster`，不是 `Pi0`，也不是 `Pi0FAST`。

### 4.4 `Pi0Faster.__init__` 组装模型模块

代码：[Pi0Faster.__init__](src/openpi/models/pi0_faster.py)（第 67 行）、[Pi0Faster 参数](src/openpi/models/pi0_faster.py)（第 103 行）

`Pi0Faster` 初始化内容包括：

```text
PaliGemma.llm      # 语言/动作专家 backbone
PaliGemma.img      # SigLIP 图像编码器
action_in_proj     # action -> action expert width
time_mlp_in/out    # PI05 的 timestep 条件
action_out_proj    # action expert hidden -> action_dim
max_delay = 0
mix_prob = 0.5
alpha = 0.6
u0 = 0.9
```

FASTER 相关的 `max_delay/mix_prob/alpha/u0` 是调度逻辑参数，不是可训练权重。

### 4.5 `eval_shape` 先推导 TrainState shape

代码：[jax.eval_shape](scripts/train.py)（第 114 行）、[TrainState](src/openpi/training/utils.py)（第 13 行）

`jax.eval_shape(init, init_rng)` 只推导结构、shape 和 dtype，不真正分配完整参数。得到的 `train_state_shape` 包含：

```text
step
params
model_def
opt_state
tx
ema_decay
ema_params
```

EMA 是 exponential moving average，即参数指数滑动平均。当前 `ema_decay=0.999`，所以会维护一份更平滑的 EMA 参数用于保存/推理。

### 4.6 根据 shape 创建 sharding

代码：[state_sharding](scripts/train.py)（第 115 行）、[fsdp_sharding](src/openpi/training/sharding.py)（第 48 行）

`fsdp_sharding(train_state_shape, mesh, log=True)` 根据 mesh 和参数 shape 决定每个数组是复制还是沿 FSDP 轴切分。当前 `fsdp_devices=1` 时，参数基本会复制到单设备。

### 4.7 加载 pi05_base checkpoint 权重

代码：[load weights](scripts/train.py)（第 120 行）、[_load_weights_and_validate](scripts/train.py)（第 73 行）、[CheckpointWeightLoader](src/openpi/training/weight_loaders.py)（第 37 行）

当前权重加载器是：

```text
CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")
```

它会 `maybe_download` checkpoint，然后 `restore_params`。`_load_weights_and_validate` 会检查加载出来的参数树和当前 `Pi0Faster` 参数树在 shape/dtype 上能对齐。

### 4.8 合并 checkpoint 权重并创建 TrainState

代码：[partial_params 合并](scripts/train.py)（第 94 行）、[TrainState 创建](scripts/train.py)（第 104 行）、[init_train_state JIT](scripts/train.py)（第 123 行）

如果 `partial_params` 不为空，代码会：

```text
nnx.split(model)
-> state.replace_by_pure_dict(partial_params)
-> nnx.merge(graphdef, state)
```

然后创建 `TrainState`：

```text
step = 0
params = 当前模型参数
model_def = 模型结构图
tx = optimizer
opt_state = tx.init(trainable_params)
ema_decay = 0.999
ema_params = params
```

这里的 init 本身也被 `jax.jit` 编译，输出按 `state_sharding` 放置。

### 4.9 非 resume 跳过 restore

代码：[main restore 分支](scripts/train.py)（第 241 行）、[restore_state](src/openpi/training/checkpoints.py)（第 89 行）

当前命令带 `--overwrite`，不带 `--resume`，所以 `resuming=False`，不会进入 `_checkpoints.restore_state(...)`。训练从 step 0 开始。

## 5. 编译单步训练函数

### 5.1 原始 `train_step` 是一次参数更新逻辑

代码：[train_step](scripts/train.py)（第 133 行）

`train_step(config, rng, state, batch)` 描述一次训练更新：恢复模型、计算 loss、反向传播、optimizer 更新参数、更新 EMA、返回指标。

### 5.2 用 `functools.partial` 固定 config

代码：[ptrain_step 创建](scripts/train.py)（第 244 行）

`main()` 里执行：

```python
functools.partial(train_step, config)
```

这会把 `config` 固定住。之后训练循环每步只传：

```text
train_rng
train_state
batch
```

### 5.3 用 `jax.jit` 编译成 `ptrain_step`

代码：[jax.jit ptrain_step](scripts/train.py)（第 244 行）

`ptrain_step` 是 JIT 后的训练一步函数。第一次调用时，JAX/XLA 会根据输入 shape 和 sharding 编译；之后外层 Python 循环反复调用同一个编译结果。

JIT 和外层循环的关系是：

```text
Python for step in pbar:
    调用一次已编译的 ptrain_step
```

外层循环负责取 batch、记录日志、保存 checkpoint；单步数值计算尽量在 JIT 内完成。

### 5.4 指定 `in_shardings` 和 `out_shardings`

代码：[ptrain_step sharding](scripts/train.py)（第 246 行）

`jax.jit` 明确声明输入输出如何分布：

```text
输入:
  rng         -> replicated_sharding
  train_state -> train_state_sharding
  batch       -> data_sharding

输出:
  train_state -> train_state_sharding
  info        -> replicated_sharding
```

`donate_argnums=(1,)` 表示允许 JAX 复用旧 `train_state` 的内存缓冲区，减少显存/内存压力。

## 6. 进入训练循环

### 6.1 设置起始 step

代码：[start_step](scripts/train.py)（第 250 行）

非 resume 时，`train_state.step == 0`，所以：

```text
start_step = 0
```

### 6.2 创建进度条并确定循环范围

代码：[pbar](scripts/train.py)（第 251 行）

当前 `config.num_train_steps=10`，所以循环范围是：

```text
step = 0, 1, 2, ..., 9
```

也就是最多调用 10 次 `ptrain_step`。

### 6.3 每轮调用 `ptrain_step`

代码：[训练循环](scripts/train.py)（第 259 行）、[ptrain_step 调用](scripts/train.py)（第 261 行）、[set_mesh](src/openpi/training/sharding.py)（第 26 行）

每个 step 的核心语句是：

```python
with sharding.set_mesh(mesh):
    train_state, info = ptrain_step(train_rng, train_state, batch)
```

`set_mesh` 是上下文管理器，让模型内部需要 sharding constraint 的地方能拿到当前 mesh。

### 6.4 记录 loss / grad_norm / param_norm

代码：[infos 收集](scripts/train.py)（第 263 行）、[wandb.log](scripts/train.py)（第 269 行）

`info` 包含：

```text
loss
grad_norm
param_norm
```

默认 `log_interval=100`，所以 10 step smoke test 通常只会在 step 0 打一次日志；最后 step 保存 checkpoint，但不一定额外日志。

### 6.5 取下一个 batch

代码：[next batch](scripts/train.py)（第 271 行）

每轮 `ptrain_step` 后执行：

```python
batch = next(data_iter)
```

也就是说，当前 step 用的是上一轮准备好的 batch，更新完参数后再为下一轮预取 batch。

### 6.6 保存 checkpoint

代码：[save condition](scripts/train.py)（第 273 行）、[save_state](src/openpi/training/checkpoints.py)（第 65 行）、[_split_params](src/openpi/training/checkpoints.py)（第 145 行）

保存条件是：

```python
(step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1
```

当前 `save_interval=1000`、`num_train_steps=10`，所以最后会在 `step == 9` 保存一次。

`save_state` 会把 EMA 参数单独拆成可推理用的 `"params"` item；如果存在 `ema_params`，保存的推理参数优先用 EMA。

## 7. 单个 `train_step` 内部逻辑

### 7.1 从 TrainState 还原模型

代码：[train_step merge](scripts/train.py)（第 141 行）

`TrainState` 里保存的是 `model_def` 和 `params`。单步训练先执行：

```python
model = nnx.merge(state.model_def, state.params)
```

得到可执行的 `Pi0Faster` 模型实例。

### 7.2 切到训练模式

代码：[model.train](scripts/train.py)（第 142 行）

`model.train()` 会启用训练态行为。后面 `compute_loss(..., train=True)` 时，`preprocess_observation` 会做训练图像增强。

### 7.3 定义 `loss_fn`

代码：[loss_fn](scripts/train.py)（第 144 行）、[compute_loss 调用](scripts/train.py)（第 148 行）

`loss_fn` 调用：

```python
chunked_loss = model.compute_loss(rng, observation, actions, train=True)
return jnp.mean(chunked_loss)
```

当前模型是 `Pi0Faster`，所以这里实际进入 `Pi0Faster.compute_loss`。

注意：`Pi0Faster.compute_loss` 当前实现内部已经把 batch/horizon 的 loss reduce 成一个标量；外层 `jnp.mean(...)` 对标量再求均值，数值不变。

### 7.4 用 step fold 进 RNG

代码：[fold_in](scripts/train.py)（第 151 行）

```python
train_rng = jax.random.fold_in(rng, state.step)
```

`fold_in` 把当前 step 混入随机 key，保证每个训练 step 的随机数不同，同时又能由 seed 复现。

### 7.5 拆出 Observation 和 actions

代码：[batch 拆分](scripts/train.py)（第 153 行）

batch 是 `DataLoaderImpl` yield 出来的二元组：

```text
observation: Observation
actions:     [batch, action_horizon, action_dim]
```

当前 shape 近似是：

```text
observation.state = [256, 32]
actions = [256, 10, 32]
```

### 7.6 指定可训练参数集合

代码：[DiffState](scripts/train.py)（第 155 行）、[TrainConfig.trainable_filter](src/openpi/training/config.py)（第 658 行）

`DiffState(0, config.trainable_filter)` 告诉 NNX 对哪些参数求梯度。当前 `freeze_filter` 默认是 `nnx.Nothing`，所以基本是 full fine-tuning：所有 `nnx.Param` 都参与训练。

### 7.7 前向计算 loss 并求梯度

代码：[value_and_grad](scripts/train.py)（第 157 行）

```python
loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)
```

`value_and_grad` 同时返回 loss 和 gradients。gradients 是梯度，即 loss 对可训练参数的导数。

### 7.8 optimizer 计算并应用更新

代码：[optimizer update](scripts/train.py)（第 159 行）、[apply_updates](scripts/train.py)（第 162 行）、[nnx.update](scripts/train.py)（第 164 行）

流程是：

```text
params = state.params.filter(trainable_filter)
updates, new_opt_state = tx.update(grads, state.opt_state, params)
new_params = optax.apply_updates(params, updates)
nnx.update(model, new_params)
```

这一步真正改变模型可训练权重。

### 7.9 更新 TrainState 和 EMA

代码：[new_state](scripts/train.py)（第 167 行）、[EMA 更新](scripts/train.py)（第 168 行）

先把 step 加 1，并写入新参数和 optimizer state。因为 `ema_decay=0.999`，还会更新：

```text
ema_params = 0.999 * old_ema_params + 0.001 * new_params
```

EMA 不直接参与当前 step 的反向传播，但保存 checkpoint 时会优先作为推理参数。

### 7.10 返回训练指标

代码：[info](scripts/train.py)（第 176 行）、[return](scripts/train.py)（第 192 行）

`train_step` 返回：

```text
new_state
info = {
  "loss": loss,
  "grad_norm": optax.global_norm(grads),
  "param_norm": optax.global_norm(kernel_params),
}
```

外层循环会收集这些指标，并按 `log_interval` 写日志和 wandb。

## 8. `Pi0Faster.compute_loss` 核心逻辑

### 8.1 输入 shape 和预处理

代码：[Pi0Faster.compute_loss](src/openpi/models/pi0_faster.py)（第 190 行）、[preprocess_observation](src/openpi/models/model.py)（第 144 行）

输入是：

```text
observation:
  images / image_masks / state / tokenized_prompt / tokenized_prompt_mask
actions:
  [B, AH, AD] = [256, 10, 32]
```

`compute_loss` 先 split RNG，然后调用：

```python
observation = preprocess_observation(preprocess_rng, observation, train=True)
```

训练态下，图像会做随机 crop、rotate、color jitter 等增强。

### 8.2 采样 noise

代码：[noise](src/openpi/models/pi0_faster.py)（第 197 行）

```python
b, ah, ad = actions.shape
noise = jax.random.normal(noise_rng, actions.shape)
```

shape：

```text
actions = [B, AH, AD]
noise   = [B, AH, AD]
```

`noise` 是高斯噪声，代表 flow/diffusion 起点附近的随机 action chunk。

### 8.3 采样 delay 和 prefix mask

代码：[delay](src/openpi/models/pi0_faster.py)（第 200 行）、[prefix_action_mask](src/openpi/models/pi0_faster.py)（第 201 行）

FASTER 训练希望模拟“前几个 action 已经是已知干净历史”的场景，所以采样：

```python
delay = jax.random.randint(delay_rng, (b,), 0, self.max_delay)
prefix_action_mask = jnp.arange(ah)[None, :] < delay[:, None]
```

shape：

```text
delay              = [B]
prefix_action_mask = [B, AH]
```

当前配置里 `max_delay=0`。这意味着代码实际会执行 `randint(..., 0, 0)`；如果当前 JAX 版本要求 `maxval > minval`，训练会在第一次编译/执行 `ptrain_step` 时失败。这个文档只追踪现有代码，不修改它。

### 8.4 采样 constant time 并计算 HAS time

代码：[time_const](src/openpi/models/pi0_faster.py)（第 203 行）、[compute_HAS 调用](src/openpi/models/pi0_faster.py)（第 205 行）、[compute_HAS](src/openpi/models/pi0_faster.py)（第 231 行）

先采样 constant schedule 的时间：

```text
time_const ~ Beta(1.5, 1) * 0.999 + 0.001
time_const: [B, AH]
```

然后计算 HAS（Horizon-Aware Schedule，按 action horizon 位置分配不同 denoising 时间表）：

```text
i_valid = max(i - delay, 0)
denom = max(action_horizon - 1 - delay, 1)
j = i_valid / denom
u = (1 - j^alpha) * u0
time_HAS = clip((time_const - u) / (1 - u), 0, 1)
```

直观含义：越靠近当前时刻的 action，越早被推进到 `t=0`，便于 streaming 推理时更早输出近端动作。

### 8.5 混合 constant schedule 和 HAS schedule

代码：[use_HAS](src/openpi/models/pi0_faster.py)（第 207 行）、[time](src/openpi/models/pi0_faster.py)（第 208 行）

```python
use_HAS = jax.random.bernoulli(type_rng, self.mix_prob, (b, 1))
time = jnp.where(use_HAS, time_HAS, time_const)
```

当前 `mix_prob=0.5`，表示每个样本有 50% 概率使用 HAS，50% 概率使用 constant schedule。

### 8.6 prefix 位置固定为干净 action

代码：[prefix time mask](src/openpi/models/pi0_faster.py)（第 210 行）

```python
time = jnp.where(prefix_action_mask, 0.0, time)
```

被 `prefix_action_mask=True` 的 horizon 位置，时间强制为 0，表示这些位置保持 ground-truth action，不参与噪声混合。

当前 `max_delay=0` 时，理论上 `prefix_action_mask` 全 False；但前提是 delay 采样能正常执行。

### 8.7 构造 noisy action `x_t` 和监督速度 `u_t`

代码：[x_t / u_t](src/openpi/models/pi0_faster.py)（第 212 行）

核心公式：

```text
x_t = time * noise + (1 - time) * actions
u_t = noise - actions
```

shape：

```text
time = [B, AH]
x_t  = [B, AH, AD]
u_t  = [B, AH, AD]
```

`x_t` 是当前时间的 noisy action。`u_t` 是训练目标速度场：模型要学会从真实 action 指向 noise 的速度；推理时用负 `dt` 沿相反方向从 noise 走回 action。

### 8.8 编码 observation prefix

代码：[embed_prefix](src/openpi/models/pi0_faster.py)（第 113 行）、[compute_loss embed_prefix](src/openpi/models/pi0_faster.py)（第 215 行）

`embed_prefix(observation)` 编码图像和 prompt：

```text
images -> SigLIP image tokens
tokenized_prompt -> PaliGemma token embeddings
```

输出：

```text
prefix_tokens   [B, prefix_len, emb]
prefix_mask     [B, prefix_len]
prefix_ar_mask  [prefix_len]
```

prefix 是条件信息，表示“当前看到了什么、任务是什么”。

### 8.9 编码 action suffix

代码：[embed_suffix](src/openpi/models/pi0_faster.py)（第 143 行）、[compute_loss embed_suffix](src/openpi/models/pi0_faster.py)（第 216 行）

`embed_suffix(observation, x_t, time)` 编码 noisy actions 和每个 horizon 位置自己的 timestep：

```text
action_tokens = action_in_proj(x_t)
time_emb = posemb_sincos(time)
PI05 路径: time_emb -> time_mlp_in/out -> adarms_cond
```

输出：

```text
suffix_tokens  [B, suffix_len, emb]
suffix_mask    [B, suffix_len]
suffix_ar_mask [suffix_len]
adarms_cond    [B, AH, emb]
```

这里和普通 `Pi0` 的关键差异是：`Pi0Faster` 的 `timestep` shape 是 `[B, AH]`，每个 action horizon 位置可以有不同时间。

### 8.10 前向计算得到预测速度 `v_t`

代码：[attn_mask](src/openpi/models/pi0_faster.py)（第 217 行）、[PaliGemma forward](src/openpi/models/pi0_faster.py)（第 221 行）、[v_t](src/openpi/models/pi0_faster.py)（第 224 行）

模型把 prefix 和 suffix 拼在同一次 forward 里：

```python
(prefix_out, suffix_out), _ = self.PaliGemma.llm(...)
v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
```

shape：

```text
v_t = [B, AH, AD]
```

`v_t` 是模型预测的速度场。

### 8.11 计算 masked MSE loss

代码：[loss](src/openpi/models/pi0_faster.py)（第 226 行）、[return loss](src/openpi/models/pi0_faster.py)（第 229 行）

先对 action_dim 求均方误差：

```text
per_pos_loss = mean((v_t - u_t)^2, axis=-1)  # [B, AH]
```

再忽略 prefix 位置：

```text
postfix_action_mask = not prefix_action_mask
loss = sum(per_pos_loss * postfix_action_mask) / (sum(postfix_action_mask) + 1e-8)
```

当前实现最终返回的是一个标量 loss。外层 `train_step` 又执行 `jnp.mean(chunked_loss)`，因此标量保持不变。

### 8.12 本训练命令不会调用 `sample_actions`

代码：[Pi0Faster.sample_actions](src/openpi/models/pi0_faster.py)（第 255 行）、[Policy.infer](src/openpi/policies/policy.py)（第 77 行）、[Policy.infer_streaming](src/openpi/policies/policy.py)（第 128 行）

`sample_actions`、`sample_actions_streaming_init`、`sample_actions_streaming_step` 是推理路径使用的函数。当前命令是 `scripts/train.py`，不会进入 `Policy.infer` 或 `Policy.infer_streaming`。

如果后续用同一个 `Pi0Faster` checkpoint 做推理，普通推理链路会是：

```text
Policy.infer
-> input transforms
-> Observation.from_dict
-> Pi0Faster.sample_actions
-> output transforms
```

streaming 推理链路会是：

```text
Policy.infer_streaming
-> Pi0Faster.sample_actions_streaming_init
-> Python host loop
-> Pi0Faster.sample_actions_streaming_step
-> on_actions_ready callback
-> output transforms
```

## 9. 最终整体链路

### 9.1 从命令到配置

代码：[train.py 入口](scripts/train.py)（第 280 行）、[cli](src/openpi/training/config.py)（第 1187 行）、[pi05_faster_libero](src/openpi/training/config.py)（第 861 行）

```text
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run scripts/train.py ...
-> scripts/train.py
-> _config.cli()
-> 选择 pi05_faster_libero
-> 覆盖 exp_name / num_train_steps / overwrite
```

### 9.2 从配置到数据

代码：[create_data_loader](src/openpi/training/data_loader.py)（第 223 行）、[LeRobotLiberoDataConfig.create](src/openpi/training/config.py)（第 328 行）、[DataLoaderImpl](src/openpi/training/data_loader.py)（第 527 行）

```text
TrainConfig.data
-> LeRobotLiberoDataConfig.create
-> repack_transforms / data_transforms / model_transforms
-> create_torch_data_loader
-> LeRobotDataset
-> TorchDataLoader
-> DataLoaderImpl
-> Observation, actions
```

### 9.3 从配置到模型

代码：[init_train_state](scripts/train.py)（第 84 行）、[Pi0FasterConfig.create](src/openpi/models/pi0_config.py)（第 136 行）、[Pi0Faster](src/openpi/models/pi0_faster.py)（第 67 行）

```text
TrainConfig.model
-> Pi0FasterConfig(pi05=True)
-> Pi0Faster
-> load pi05_base checkpoint
-> merge params
-> TrainState(step=0, params, opt_state, ema_params)
```

### 9.4 从训练循环到 loss

代码：[ptrain_step](scripts/train.py)（第 244 行）、[训练循环](scripts/train.py)（第 259 行）、[train_step](scripts/train.py)（第 133 行）、[Pi0Faster.compute_loss](src/openpi/models/pi0_faster.py)（第 190 行）

```text
jax.jit(functools.partial(train_step, config))
-> ptrain_step
-> for step in 0..9
-> ptrain_step(train_rng, train_state, batch)
-> train_step
-> model.compute_loss
-> loss / grads
-> optimizer update
-> EMA update
```

### 9.5 从训练状态到 checkpoint

代码：[save condition](scripts/train.py)（第 273 行）、[save_state](src/openpi/training/checkpoints.py)（第 65 行）、[_split_params](src/openpi/training/checkpoints.py)（第 145 行）

```text
step 9
-> save_state
-> 保存 train_state
-> 保存 EMA params 作为 params
-> 保存 norm stats assets
-> checkpoints/pi05_faster_libero/test_faster/9
```

## 一句话总链路

这条训练命令的主线是：`config` 选择 `Pi0FasterConfig(pi05=True)` + LIBERO 数据，data loader 产出 `Observation/actions`，`ptrain_step` 调 `Pi0Faster.compute_loss` 得到 flow matching loss，optimizer 更新 `Pi0Faster` 参数，并在最后一步保存带 EMA 参数的 checkpoint。
