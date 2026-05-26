下面按“两级结构”梳理这条命令的训练整体逻辑。这里以你的命令 `pi05_libero --num-train-steps=10` 为准，所以实际走的是 **LeRobot / Torch data loader 分支**，不是 RLDS 分支。

**1. 初始化 Config**

1.1. Shell 环境生效  
`CUDA_VISIBLE_DEVICES=0` 让 JAX/CUDA 只看到第 0 张 GPU。  
`WANDB_MODE=offline` 让 wandb 离线记录日志。

1.2. `uv run scripts/train.py ...` 启动训练脚本  
入口是：

```python
main(_config.cli())
```

1.3. `_config.cli()` 解析命令行  
tyro 根据第一个位置参数 `pi05_libero`，从 `_CONFIGS_DICT` 里选中对应的 `TrainConfig`。

1.4. 命令行参数覆盖默认配置  
`--exp-name=test_openpi` 覆盖实验名。  
`--num-train-steps=10` 覆盖训练步数。  
`--overwrite` 覆盖 checkpoint 目录处理策略。

1.5. 得到最终 `TrainConfig`  
核心内容是：

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
设置日志格式，让训练过程中输出更清楚。

2.2. 检查 batch size 和 GPU 数量  
`batch_size` 必须能被 `jax.device_count()` 整除。  
你的 `CUDA_VISIBLE_DEVICES=0` 通常意味着 `device_count=1`，所以 `256 % 1 == 0` 没问题。

2.3. 设置 JAX compilation cache  
JAX 第一次运行 JIT 函数会编译 GPU 程序。  
cache 用来保存编译结果，后续相同 shape 的训练可以少等一些编译时间。

2.4. 创建随机数 key  
JAX 用显式 `rng key` 管理随机数。  
这里先创建总 key，再 split 成：

```text
train_rng：训练 step 用
init_rng：模型初始化用
```

2.5. 创建 mesh / sharding  
`mesh` 描述当前有哪些设备参与训练。  
`sharding` 描述 batch 或参数如何放到设备上。  
单卡时基本可以理解为“都放到这一张 GPU 上”。

2.6. 初始化 checkpoint 目录  
路径是：

```text
checkpoints/pi05_libero/test_openpi
```

因为有 `--overwrite`，如果目录已存在，会先删除再重建。

2.7. 初始化 wandb  
由于 `WANDB_MODE=offline`，日志会本地保存，不上传。

**3. 创建 Data Loader**

3.1. 调用 `create_data_loader(config, ...)`  
这是训练数据入口。  
它先执行：

```python
data_config = config.data.create(config.assets_dirs, config.model)
```

对 `pi05_libero` 来说：

```text
config.data = LeRobotLiberoDataConfig(...)
```

所以会进入 `LeRobotLiberoDataConfig.create(...)`。

3.2. `LeRobotLiberoDataConfig.create(...)` 生成 `DataConfig`  
它不直接读数据，而是配置后续样本怎么转换。

生成三类 transform：

```text
repack_transforms
data_transforms
model_transforms
```

3.3. 配置 `repack_transforms`  
作用：把 LeRobot dataset 原始 key 改成 LIBERO adapter 期望的 key。

```text
image       -> observation/image
wrist_image -> observation/wrist_image
state       -> observation/state
actions     -> actions
prompt      -> prompt
```

3.4. 配置 `data_transforms`  
作用：把 LIBERO 格式转成 openpi 统一输入格式。

核心是：

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

同时配置 `LiberoOutputs()`，但训练时主要用 inputs，outputs 更偏推理阶段。

3.5. 配置 `model_transforms`  
调用：

```python
ModelTransformFactory()(model_config)
```

因为 `pi05_libero` 的 `model_config.model_type == PI05`，所以进入 PI05 分支。

PI05 model transforms 是：

```text
InjectDefaultPrompt
ResizeImages(224, 224)
TokenizePrompt(PaligemmaTokenizer, discrete_state_input=False)
PadStatesAndActions(action_dim=32, action_horizon=10)
```

3.6. `create_data_loader` 判断走哪个 data loader 分支  
判断条件是：

```python
if data_config.rlds_data_dir is not None:
    return create_rlds_data_loader(...)
else:
    return create_torch_data_loader(...)
```

3.7. 对 `pi05_libero`，走 `create_torch_data_loader`  
因为 `pi05_libero` 的 `rlds_data_dir is None`。  
所以它不走 RLDS。

实际链路是：

```text
create_torch_data_loader
-> create_torch_dataset
-> LeRobotDataset("physical-intelligence/libero")
-> transform_dataset
-> TorchDataLoader
-> DataLoaderImpl
```

3.8. `create_torch_dataset` 创建 LeRobotDataset  
它会读取：

```text
physical-intelligence/libero
```

并按 `action_horizon=10` 取 action sequence。

3.9. `transform_dataset` 串联 transforms  
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

3.10. `TorchDataLoader` 负责 batch 和 shuffle  
底层用 `torch.utils.data.DataLoader`。  
虽然名字里有 Torch，但当前 `framework="jax"`，最后 batch 会转成 JAX array。

3.11. `DataLoaderImpl` 输出训练需要的 batch  
最终 yield：

```python
Observation.from_dict(batch), batch["actions"]
```

训练 step 收到：

```text
observation: images / image_masks / state / tokenized_prompt
actions: [batch, 10, 32]
```

3.12. RLDS 分支的嵌套关系，但本命令不走  
如果某个配置有 `rlds_data_dir`，才会走：

```text
create_rlds_data_loader
-> create_rlds_dataset
-> DroidRldsDataset
-> transform_iterable_dataset
-> RLDSDataLoader
-> DataLoaderImpl
```

RLDSDataLoader 主要用于已经按 batch 组织好的 RLDS / DROID 数据。

**4. 初始化模型和 TrainState**

4.1. 创建 optimizer  
根据 config 里的：

```text
optimizer = AdamW(clip_gradient_norm=1.0)
lr_schedule = CosineDecaySchedule(...)
```

创建优化器 `tx`。

4.2. 创建模型  
调用：

```python
model = config.model.create(model_rng)
```

对 `pi05_libero` 来说：

```text
config.model = Pi0Config(pi05=True, ...)
```

所以创建的是：

```text
Pi0
```

不是 `Pi0Faster`。

4.3. `eval_shape` 得到 `train_state_shape`  
`eval_shape` 只推导模型参数和训练状态的形状，不真正做完整计算。  
它用来知道参数树长什么样，方便后续加载 checkpoint 和设置 sharding。

4.4. 根据 shape 创建 sharding  
`state_sharding = fsdp_sharding(...)`。  
单卡时基本就是放在这张卡上。  
多卡时用于参数分片或数据并行。

4.5. 从预训练 checkpoint 加载权重  
`pi05_libero` 使用：

```text
gs://openpi-assets/checkpoints/pi05_base/params
```

4.6. 验证加载的参数形状  
检查 checkpoint 中的参数 shape / dtype 是否能对上当前模型。

4.7. 把预训练权重合进模型  
已有 checkpoint 参数会覆盖初始化参数。  
没有被 checkpoint 覆盖的参数保留初始化值。

4.8. 创建 `TrainState`  
`TrainState` 包含：

```text
step
params
model_def
optimizer tx
optimizer state
ema params
```

4.9. 如果是 resume，则恢复旧状态  
你的命令用了 `--overwrite`，所以不是 resume。

**5. 编译单步训练函数**

5.1. 定义原始 `train_step`  
`train_step` 是一次参数更新的 Python 逻辑。

5.2. 用 `functools.partial(train_step, config)` 固定 config  
这样后面调用时只需要传：

```text
rng
train_state
batch
```

5.3. 用 `jax.jit(...)` 编译成 `ptrain_step`  
`ptrain_step` 是 GPU/XLA 编译后的训练一步函数。  
第一次调用会编译，后续循环复用编译结果。

5.4. 设置输入输出 sharding  
告诉 JAX：

```text
rng 怎么放
train_state 怎么放
batch 怎么放
返回的新 train_state 和日志怎么放
```

**6. 进入训练循环**

6.1. 设置起始 step  
非 resume 时：

```text
start_step = 0
```

6.2. 创建进度条 `pbar`  
你的命令 `num_train_steps=10`，所以循环范围是：

```text
0, 1, 2, ..., 9
```

6.3. 每轮调用 `ptrain_step`  
核心语句：

```python
train_state, info = ptrain_step(train_rng, train_state, batch)
```

每调用一次，就完成一次参数更新。

6.4. 收集训练指标  
`info` 包含：

```text
loss
grad_norm
param_norm
```

6.5. 按 `log_interval` 打日志  
默认 `log_interval=100`。  
因为你只跑 10 step，所以通常只在 step 0 打一次日志。

6.6. 取下一个 batch  
每轮训练结束后：

```python
batch = next(data_iter)
```

准备下一轮训练。

6.7. 保存 checkpoint  
默认 `save_interval=1000`，但最后一步一定保存。  
你跑 10 step，所以会在：

```text
step 9
```

保存最终 checkpoint。

**7. 单个 train_step 内部逻辑**

7.1. 还原模型  
用当前 `train_state` 里的：

```text
model_def
params
```

合成可执行模型。

7.2. 设置训练模式  
调用：

```python
model.train()
```

7.3. 定义 `loss_fn`  
`loss_fn` 内部调用：

```python
model.compute_loss(rng, observation, actions, train=True)
```

7.4. 为当前 step 生成随机 key  
用：

```python
jax.random.fold_in(rng, state.step)
```

保证每一步的噪声、time、数据增强随机性不同。

7.5. 拆出 batch  
得到：

```text
observation
actions
```

7.6. 指定可训练参数  
通过 `config.trainable_filter` 排除 frozen 参数。

7.7. 前向计算 loss 并求梯度  
调用：

```python
nnx.value_and_grad(...)
```

得到：

```text
loss
grads
```

7.8. optimizer 计算 updates  
根据梯度和 optimizer state 得到参数更新量。

7.9. 应用 updates  
更新当前可训练参数。

7.10. 写回模型并导出完整参数树  
包含训练参数和 frozen 参数。

7.11. 更新 `TrainState`  
step 加 1，并保存新的 params / optimizer state。

7.12. 更新 EMA 参数  
`ema_decay=0.999`，所以维护一份平滑后的参数。

7.13. 统计日志指标  
返回：

```text
loss
grad_norm
param_norm
```

**8. Pi0.compute_loss 的训练目标**

8.1. 输入真实 actions  
形状大致是：

```text
[batch, action_horizon=10, action_dim=32]
```

8.2. 随机采样 noise  
生成和 actions 一样形状的高斯噪声。

8.3. 随机采样 time  
每个样本采一个 flow matching 时间点。

8.4. 构造 noisy action  
公式：

```text
x_t = time * noise + (1 - time) * actions
```

8.5. 构造监督目标速度  
公式：

```text
u_t = noise - actions
```

8.6. 编码 observation prefix  
图像、文本、state 进入 `embed_prefix`。

8.7. 编码 action suffix  
`x_t` 和 `time` 进入 `embed_suffix`。

8.8. 模型 forward  
PaliGemma / action expert 输出 action token 表征。

8.9. 投影得到预测速度  
通过：

```text
action_out_proj -> v_t
```

8.10. 计算 MSE loss  
公式：

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