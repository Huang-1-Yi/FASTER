下面按“两级结构”梳理这条命令的训练整体逻辑。这里以你的命令 `pi05_libero --num-train-steps=10` 为准，所以实际走的是 **LeRobot / Torch data loader 分支**，不是 RLDS 分支。

**1. 初始化 Config**

1.1. Shell 环境生效  
代码：[train.py 入口](C:/QClaw/FASTER_hy/scripts/train.py:313)  
`CUDA_VISIBLE_DEVICES=0`：CUDA 是 NVIDIA GPU 的计算接口，这里表示只让 JAX/CUDA 看到第 0 张 GPU。  
`WANDB_MODE=offline`：wandb 是训练日志工具，这里表示离线记录日志，不上传云端。

1.2. `uv run scripts/train.py ...` 启动训练脚本  
代码：[train.py 入口](C:/QClaw/FASTER_hy/scripts/train.py:313)  
`uv` 是 Python 项目环境和依赖管理工具；这条命令表示在 `uv` 管理的环境里运行训练脚本。入口是：

```python
main(_config.cli())
```

1.3. `_config.cli()` 解析命令行  
代码：[config.py cli](C:/QClaw/FASTER_hy/src/openpi/training/config.py:1218)  
CLI 是 command-line interface，即命令行接口。`tyro` 是把命令行参数解析成 Python dataclass 的库。它根据第一个位置参数 `pi05_libero`，从 `_CONFIGS_DICT` 里选中对应的 `TrainConfig`。

1.4. 命令行参数覆盖默认配置  
代码：[TrainConfig](C:/QClaw/FASTER_hy/src/openpi/training/config.py:610)  
`--exp-name=test_openpi` 覆盖实验名。  
`--num-train-steps=10` 覆盖训练步数。  
`--overwrite` 覆盖 checkpoint 目录处理策略。checkpoint 是训练中保存下来的模型参数和训练状态。

1.5. 得到最终 `TrainConfig`  
代码：[pi05_libero 配置](C:/QClaw/FASTER_hy/src/openpi/training/config.py:868)  
Config 是 configuration，即配置对象。核心内容是：

```text
model = Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)
data = LeRobotLiberoDataConfig(...)
batch_size = 256
num_workers = 8
weight_loader = pi05_base checkpoint
ema_decay = 0.999
num_train_steps = 10
overwrite = True
```

**2. 进入 main() 并准备训练环境**

2.1. 初始化 logging  
代码：[main](C:/QClaw/FASTER_hy/scripts/train.py:243)  
logging 是日志系统，用来把训练过程中的信息打印出来，例如当前机器、loss、checkpoint 保存等。

2.2. 检查 batch size 和 GPU 数量  
代码：[main batch 检查](C:/QClaw/FASTER_hy/scripts/train.py:252)  
`batch_size` 是一次训练喂给模型的样本数。它必须能被 `jax.device_count()` 整除，因为 batch 会按 GPU 数量均分。你的 `CUDA_VISIBLE_DEVICES=0` 通常意味着 `device_count=1`，所以 `256 % 1 == 0` 没问题。

2.3. 设置 JAX compilation cache  
代码：[main JAX cache](C:/QClaw/FASTER_hy/scripts/train.py:260)  
JAX 是本项目使用的数值计算和自动微分框架。JIT 是 just-in-time compilation，即即时编译。JAX 第一次运行 JIT 函数会把 Python 数值逻辑编译成 GPU/XLA 程序。compilation cache 是编译结果缓存，后续相同 shape 的训练可以复用编译结果，少等一些编译时间。

2.4. 创建随机数 key  
代码：[main rng](C:/QClaw/FASTER_hy/scripts/train.py:265)  
`rng` 是 random number generator，即随机数生成器。JAX 不使用隐藏的全局随机状态，而是显式传递 `rng key`。这里先创建总 key，再 split 成：

```text
train_rng：训练 step 用
init_rng：模型初始化用
```

2.5. 创建 mesh / sharding  
代码：[main mesh/sharding](C:/QClaw/FASTER_hy/scripts/train.py:271)  
`mesh` 描述当前有哪些设备参与训练。`sharding` 描述数组如何分布到设备上。单卡时基本可以理解为“都放到这一张 GPU 上”；多卡时才涉及数据并行或参数分片。

2.6. 初始化 checkpoint 目录  
代码：[initialize_checkpoint_dir](C:/QClaw/FASTER_hy/src/openpi/training/checkpoints.py:20)  
路径是：

```text
checkpoints/pi05_libero/test_openpi
```

因为有 `--overwrite`，如果目录已存在，会先删除再重建。

2.7. 初始化 wandb  
代码：[init_wandb](C:/QClaw/FASTER_hy/scripts/train.py:43)  
wandb 负责记录训练指标。由于 `WANDB_MODE=offline`，日志会本地保存，不上传。

**3. 创建 Data Loader**

3.1. 调用 `create_data_loader(config, ...)`  
代码：[create_data_loader](C:/QClaw/FASTER_hy/src/openpi/training/data_loader.py:223)  
Data Loader 是训练数据加载器，负责读取样本、做 batch、转换格式。它先执行：

```python
data_config = config.data.create(config.assets_dirs, config.model)
```

对 `pi05_libero` 来说，`config.data = LeRobotLiberoDataConfig(...)`，所以会进入 `LeRobotLiberoDataConfig.create(...)`。

3.2. `LeRobotLiberoDataConfig.create(...)` 生成 `DataConfig`  
代码：[LeRobotLiberoDataConfig.create](C:/QClaw/FASTER_hy/src/openpi/training/config.py:334)  
它不直接读数据，而是配置后续样本怎么转换。生成三类 transform。transform 是“数据转换步骤”，比如改字段名、归一化、tokenize、padding。

```text
repack_transforms
data_transforms
model_transforms
```

3.3. 配置 `repack_transforms`  
代码：[RepackTransform](C:/QClaw/FASTER_hy/src/openpi/transforms.py:80)  
作用：把 LeRobot dataset 原始 key 改成 LIBERO adapter 期望的 key。

```text
image       -> observation/image
wrist_image -> observation/wrist_image
state       -> observation/state
actions     -> actions
prompt      -> prompt
```

3.4. 配置 `data_transforms`  
代码：[LiberoInputs](C:/QClaw/FASTER_hy/src/openpi/policies/libero_policy.py:30)  
作用：把 LIBERO 格式转成 openpi 统一输入格式。核心是：

```python
LiberoInputs(model_type=model_config.model_type)
```

它会生成：

```text
state
image["base_0_rgb"]
image["left_wrist_0_rgb"]
image["right_wrist_0_rgb"]
image_mask
actions
prompt
```

同时配置 [LiberoOutputs](C:/QClaw/FASTER_hy/src/openpi/policies/libero_policy.py:78)，但训练时主要用 inputs，outputs 更偏推理阶段。

3.5. 配置 `model_transforms`  
代码：[ModelTransformFactory](C:/QClaw/FASTER_hy/src/openpi/training/config.py:104)  
调用：

```python
ModelTransformFactory()(model_config)
```

因为 `pi05_libero` 的 `model_config.model_type == PI05`，所以进入 PI05 分支。PI05 model transforms 是：

```text
InjectDefaultPrompt
ResizeImages(224, 224)
TokenizePrompt(PaligemmaTokenizer, discrete_state_input=False)
PadStatesAndActions(action_dim=32, action_horizon=10)
```

对应代码：[InjectDefaultPrompt](C:/QClaw/FASTER_hy/src/openpi/transforms.py:105)、[ResizeImages](C:/QClaw/FASTER_hy/src/openpi/transforms.py:195)、[TokenizePrompt](C:/QClaw/FASTER_hy/src/openpi/transforms.py:266)、[PadStatesAndActions](C:/QClaw/FASTER_hy/src/openpi/transforms.py:347)。

3.6. `create_data_loader` 判断走哪个 data loader 分支  
代码：[create_data_loader 分支](C:/QClaw/FASTER_hy/src/openpi/training/data_loader.py:245)  
判断条件是：

```python
if data_config.rlds_data_dir is not None:
    return create_rlds_data_loader(...)
else:
    return create_torch_data_loader(...)
```

RLDS 是 robotics datasets 常见的数据格式；本仓库里主要用于 DROID。DROID 是一个机器人操作数据集。

3.7. 对 `pi05_libero`，走 `create_torch_data_loader`  
代码：[create_torch_data_loader](C:/QClaw/FASTER_hy/src/openpi/training/data_loader.py:303)  
因为 `pi05_libero` 的 `rlds_data_dir is None`，所以它不走 RLDS。实际链路是：

```text
create_torch_data_loader
-> create_torch_dataset
-> LeRobotDataset("physical-intelligence/libero")
-> transform_dataset
-> TorchDataLoader
-> DataLoaderImpl
```

3.8. `create_torch_dataset` 创建 LeRobotDataset  
代码：[create_torch_dataset](C:/QClaw/FASTER_hy/src/openpi/training/data_loader.py:130)  
LeRobot 是 Hugging Face 生态里的机器人数据集格式。这里读取：

```text
physical-intelligence/libero
```

并按 `action_horizon=10` 取 action sequence。

3.9. `transform_dataset` 串联 transforms  
代码：[transform_dataset](C:/QClaw/FASTER_hy/src/openpi/training/data_loader.py:172)  
每个样本会按顺序走：

```text
PromptFromLeRobotTask
RepackTransform
LiberoInputs
Normalize
InjectDefaultPrompt
ResizeImages
TokenizePrompt
PadStatesAndActions
```

对应代码：[PromptFromLeRobotTask](C:/QClaw/FASTER_hy/src/openpi/transforms.py:329)、[Normalize](C:/QClaw/FASTER_hy/src/openpi/transforms.py:115)。

3.10. `TorchDataLoader` 负责 batch 和 shuffle  
代码：[TorchDataLoader](C:/QClaw/FASTER_hy/src/openpi/training/data_loader.py:413)  
底层用 `torch.utils.data.DataLoader`。虽然名字里有 Torch，但当前 `framework="jax"`，最后 batch 会转成 JAX array。

3.11. `DataLoaderImpl` 输出训练需要的 batch  
代码：[DataLoaderImpl](C:/QClaw/FASTER_hy/src/openpi/training/data_loader.py:562)  
最终 yield：

```python
Observation.from_dict(batch), batch["actions"]
```

`Observation` 是模型输入的数据结构。代码：[Observation.from_dict](C:/QClaw/FASTER_hy/src/openpi/models/model.py:110)。训练 step 收到：

```text
observation: images / image_masks / state / tokenized_prompt
actions: [batch, 10, 32]
```

3.12. RLDS 分支的嵌套关系，但本命令不走  
代码：[create_rlds_data_loader](C:/QClaw/FASTER_hy/src/openpi/training/data_loader.py:372)  
如果某个配置有 `rlds_data_dir`，才会走：

```text
create_rlds_data_loader
-> create_rlds_dataset
-> DroidRldsDataset
-> transform_iterable_dataset
-> RLDSDataLoader
-> DataLoaderImpl
```

RLDSDataLoader 主要用于已经按 batch 组织好的 RLDS / DROID 数据。代码：[RLDSDataLoader](C:/QClaw/FASTER_hy/src/openpi/training/data_loader.py:518)。

**4. 初始化模型和 TrainState**

4.1. 创建 optimizer  
代码：[init_train_state](C:/QClaw/FASTER_hy/scripts/train.py:86)、[create_optimizer](C:/QClaw/FASTER_hy/src/openpi/training/optimizer.py:105)  
optimizer 是优化器，负责根据梯度更新模型参数。根据 config 里的：

```text
optimizer = AdamW(clip_gradient_norm=1.0)
lr_schedule = CosineDecaySchedule(...)
```

创建优化器 `tx`。`tx` 是 Optax 中常用的 optimizer transformation 简写。

4.2. 创建模型  
代码：[Pi0Config.create](C:/QClaw/FASTER_hy/src/openpi/models/pi0_config.py:49)  
调用：

```python
model = config.model.create(model_rng)
```

对 `pi05_libero` 来说，`config.model = Pi0Config(pi05=True, ...)`，所以创建的是 [Pi0](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:66)，不是 `Pi0Faster`。

4.3. `eval_shape` 得到 `train_state_shape`  
代码：[init_train_state eval_shape](C:/QClaw/FASTER_hy/scripts/train.py:125)  
`eval_shape` 只推导模型参数和训练状态的形状，不真正做完整计算。它用来知道参数树长什么样，方便后续加载 checkpoint 和设置 sharding。

4.4. 根据 shape 创建 sharding  
代码：[init_train_state sharding](C:/QClaw/FASTER_hy/scripts/train.py:128)  
`state_sharding = fsdp_sharding(...)`。FSDP 是 fully sharded data parallel，即参数分片式数据并行。单卡时基本就是放在这张卡上；多卡时用于参数分片或数据并行。

4.5. 从预训练 checkpoint 加载权重  
代码：[_load_weights_and_validate](C:/QClaw/FASTER_hy/scripts/train.py:74)  
`pi05_libero` 使用：

```text
gs://openpi-assets/checkpoints/pi05_base/params
```

4.6. 验证加载的参数形状  
代码：[_load_weights_and_validate](C:/QClaw/FASTER_hy/scripts/train.py:74)  
检查 checkpoint 中的参数 shape / dtype 是否能对上当前模型。dtype 是 data type，即数组数据类型，例如 float32、bfloat16。

4.7. 把预训练权重合进模型  
代码：[init_train_state 合并权重](C:/QClaw/FASTER_hy/scripts/train.py:99)  
已有 checkpoint 参数会覆盖初始化参数。没有被 checkpoint 覆盖的参数保留初始化值。

4.8. 创建 `TrainState`  
代码：[TrainState 创建](C:/QClaw/FASTER_hy/scripts/train.py:108)  
`TrainState` 是训练状态包，包含：

```text
step
params
model_def
optimizer tx
optimizer state
ema params
```

EMA 是 exponential moving average，即参数指数滑动平均，常用于保存更平滑的推理参数。

4.9. 如果是 resume，则恢复旧状态  
代码：[restore_state](C:/QClaw/FASTER_hy/src/openpi/training/checkpoints.py:87)  
resume 是继续训练。你的命令用了 `--overwrite`，所以不是 resume。

**5. 编译单步训练函数**

5.1. 定义原始 `train_step`  
代码：[train_step](C:/QClaw/FASTER_hy/scripts/train.py:146)  
`train_step` 是一次参数更新的 Python 逻辑。

5.2. 用 `functools.partial(train_step, config)` 固定 config  
代码：[ptrain_step 创建](C:/QClaw/FASTER_hy/scripts/train.py:284)  
`partial` 是把函数的一部分参数提前固定住。这样后面调用时只需要传：

```text
rng
train_state
batch
```

5.3. 用 `jax.jit(...)` 编译成 `ptrain_step`  
代码：[ptrain_step 创建](C:/QClaw/FASTER_hy/scripts/train.py:284)  
`ptrain_step` 是 GPU/XLA 编译后的训练一步函数。XLA 是 JAX 用来把计算图编译成高效设备程序的编译器。第一次调用会编译，后续循环复用编译结果。

5.4. 设置输入输出 sharding  
代码：[ptrain_step sharding](C:/QClaw/FASTER_hy/scripts/train.py:286)  
告诉 JAX：

```text
rng 怎么放
train_state 怎么放
batch 怎么放
返回的新 train_state 和日志怎么放
```

**6. 进入训练循环**

6.1. 设置起始 step  
代码：[start_step](C:/QClaw/FASTER_hy/scripts/train.py:293)  
非 resume 时：

```text
start_step = 0
```

6.2. 创建进度条 `pbar`  
代码：[pbar](C:/QClaw/FASTER_hy/scripts/train.py:294)  
`pbar` 是 progress bar，即进度条。你的命令 `num_train_steps=10`，所以循环范围是：

```text
0, 1, 2, ..., 9
```

6.3. 每轮调用 `ptrain_step`  
代码：[训练循环调用 ptrain_step](C:/QClaw/FASTER_hy/scripts/train.py:304)  
核心语句：

```python
train_state, info = ptrain_step(train_rng, train_state, batch)
```

每调用一次，就完成一次参数更新。

6.4. 收集训练指标  
代码：[infos 收集](C:/QClaw/FASTER_hy/scripts/train.py:305)  
`info` 包含：

```text
loss
grad_norm
param_norm
```

6.5. 按 `log_interval` 打日志  
代码：[训练日志](C:/QClaw/FASTER_hy/scripts/train.py:306)  
默认 `log_interval=100`。因为你只跑 10 step，所以通常只在 step 0 打一次日志。

6.6. 取下一个 batch  
代码：[next batch](C:/QClaw/FASTER_hy/scripts/train.py:313)  
每轮训练结束后：

```python
batch = next(data_iter)
```

准备下一轮训练。

6.7. 保存 checkpoint  
代码：[save_state](C:/QClaw/FASTER_hy/src/openpi/training/checkpoints.py:65)  
默认 `save_interval=1000`，但最后一步一定保存。你跑 10 step，所以会在：

```text
step 9
```

保存最终 checkpoint。

**7. 单个 train_step 内部逻辑**

7.1. 还原模型  
代码：[train_step](C:/QClaw/FASTER_hy/scripts/train.py:146)  
用当前 `train_state` 里的：

```text
model_def
params
```

合成可执行模型。

7.2. 设置训练模式  
代码：[model.train](C:/QClaw/FASTER_hy/scripts/train.py:153)  
调用：

```python
model.train()
```

7.3. 定义 `loss_fn`  
代码：[loss_fn](C:/QClaw/FASTER_hy/scripts/train.py:156)  
`loss_fn` 内部调用：

```python
model.compute_loss(rng, observation, actions, train=True)
```

7.4. 为当前 step 生成随机 key  
代码：[fold_in](C:/QClaw/FASTER_hy/scripts/train.py:169)  
用：

```python
jax.random.fold_in(rng, state.step)
```

保证每一步的噪声、time、数据增强随机性不同。

7.5. 拆出 batch  
代码：[batch unpack](C:/QClaw/FASTER_hy/scripts/train.py:174)  
得到：

```text
observation
actions
```

7.6. 指定可训练参数  
代码：[DiffState](C:/QClaw/FASTER_hy/scripts/train.py:179)  
通过 `config.trainable_filter` 排除 frozen 参数。frozen 是冻结参数，即不更新。

7.7. 前向计算 loss 并求梯度  
代码：[value_and_grad](C:/QClaw/FASTER_hy/scripts/train.py:184)  
调用：

```python
nnx.value_and_grad(...)
```

得到：

```text
loss
grads
```

`grads` 是 gradients，即梯度。

7.8. optimizer 计算 updates  
代码：[optimizer update](C:/QClaw/FASTER_hy/scripts/train.py:187)  
根据梯度和 optimizer state 得到参数更新量。

7.9. 应用 updates  
代码：[apply_updates](C:/QClaw/FASTER_hy/scripts/train.py:191)  
更新当前可训练参数。

7.10. 写回模型并导出完整参数树  
代码：[nnx.update](C:/QClaw/FASTER_hy/scripts/train.py:196)  
包含训练参数和 frozen 参数。

7.11. 更新 `TrainState`  
代码：[new_state](C:/QClaw/FASTER_hy/scripts/train.py:201)  
step 加 1，并保存新的 params / optimizer state。

7.12. 更新 EMA 参数  
代码：[EMA 更新](C:/QClaw/FASTER_hy/scripts/train.py:203)  
`ema_decay=0.999`，所以维护一份平滑后的参数。

7.13. 统计日志指标  
代码：[info](C:/QClaw/FASTER_hy/scripts/train.py:224)  
返回：

```text
loss
grad_norm
param_norm
```

**8. Pi0.compute_loss 的训练目标**

8.1. 输入真实 actions  
代码：[Pi0.compute_loss](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:189)  
形状大致是：

```text
[batch, action_horizon=10, action_dim=32]
```

8.2. 随机采样 noise  
代码：[noise](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:196)  
noise 是噪声。这里生成和 actions 一样形状的高斯噪声。

8.3. 随机采样 time  
代码：[time](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:197)  
每个样本采一个 flow matching 时间点。flow matching 是让模型学习从噪声流向真实数据的速度场。

8.4. 构造 noisy action  
代码：[x_t](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:199)  
公式：

```text
x_t = time * noise + (1 - time) * actions
```

8.5. 构造监督目标速度  
代码：[u_t](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:200)  
公式：

```text
u_t = noise - actions
```

8.6. 编码 observation prefix  
代码：[embed_prefix](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:106)  
图像、文本、state 进入 `embed_prefix`。prefix 是条件信息，即“当前看到了什么、任务是什么”。

8.7. 编码 action suffix  
代码：[embed_suffix](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:140)  
`x_t` 和 `time` 进入 `embed_suffix`。suffix 是动作生成相关的 token。

8.8. 模型 forward  
代码：[PaliGemma forward](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:207)  
forward 是前向计算，即模型根据输入算输出。PaliGemma / action expert 输出 action token 表征。

8.9. 投影得到预测速度  
代码：[action_out_proj](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:210)  
通过：

```text
action_out_proj -> v_t
```

得到模型预测的速度场。

8.10. 计算 MSE loss  
代码：[loss return](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:212)  
MSE 是 mean squared error，即均方误差。公式：

```text
loss = mean((v_t - u_t)^2)
```

训练目标就是让模型学会从 noisy action chunk 预测速度场，最终推回真实 action chunk。

**9. 最终整体链路**

```text
1. 初始化 config
   -> 选择 pi05_libero
   -> 覆盖 exp_name / num_train_steps / overwrite

2. 准备训练环境
   -> GPU / wandb / rng / sharding / checkpoint

3. 创建 data loader
   -> LeRobotLiberoDataConfig.create
   -> create_torch_data_loader
   -> LeRobotDataset
   -> transforms
   -> Observation + actions

4. 初始化模型
   -> Pi0Config(pi05=True).create
   -> Pi0
   -> 加载 pi05_base 权重
   -> TrainState

5. 编译 train_step
   -> train_step
   -> jax.jit
   -> ptrain_step

6. 训练循环
   -> for step in 0..9
   -> ptrain_step
   -> next batch
   -> log
   -> save

7. 单步训练
   -> compute_loss
   -> grad
   -> optimizer update
   -> EMA update

8. 保存 checkpoint
   -> step 9 保存到 checkpoints/pi05_libero/test_openpi
```

最关键的一句话：  
这条训练命令的主线是 **config 选择 PI0.5 + LIBERO 数据，data loader 产出 Observation/actions，train_step 调 Pi0.compute_loss，optimizer 更新 Pi0 参数**。
