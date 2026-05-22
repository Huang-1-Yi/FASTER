# FASTER vs openpi 目录结构差异分析

> 基于 FASTER (`C:\QClaw\FASTER`) 与上游 openpi (`C:\QClaw\openpi`) 的完整对比。

---

## 一、仅存在于 FASTER 的文件/目录

| 路径 | 类型 | 说明 |
|------|------|------|
| `compare.md` | 文件 | 本对比文档 |
| `examples/calvin/` | 目录 | **新增示例**：CALVIN 任务环境，包含 `calvin_env_wrapper.py`、`main.py`、`README.md` |
| `src/openpi/models/pi0_faster.py` | 文件 | **核心新文件**：`Pi0Faster` 模型类，实现了 HAS（Horizon-Aware Schedule）调度策略、`compute_HAS`、流式推理初始化/单步方法 |
| `src/openpi/policies/agilex_policy.py` | 文件 | **新增策略**：`AgilexInputs`/`AgilexOutputs`，适配 AgileX Piper 机械臂的数据格式 |
| `src/openpi/policies/calvin_policy.py` | 文件 | **新增策略**：`CalvinInputs`/`CalvinOutputs`，适配 CALVIN 任务的数据格式 |
| `start_server.sh` | 文件 | **新增启动脚本**：内含流式推理服务示例命令 |
| `third_party/libero` | 子模块 | **替换**：openpi 中是 `third_party/aloha`，FASTER 中替换为 `third_party/libero` |

---

## 二、仅存在于 openpi 的文件/目录

| 路径 | 类型 | 说明 |
|------|------|------|
| `.github/` | 目录 | openpi 的 CI/CD 工作流（FASTER 未携带） |
| `third_party/aloha` | 子模块 | **替换**：FASTER 中被移除，替换为 `third_party/libero` |

---

## 三、两边都存在但内容有差异的文件

> 以下文件中，部分 README 因翻译而内容不同（中文 vs 英文），部分代码文件因功能增删而有实质差异。

### 3.1 根目录配置

| 文件 | 差异说明 |
|------|---------|
| `.gitmodules` | FASTER 将子模块从 `third_party/aloha`（Physical-Intelligence/aloha）替换为 `third_party/libero`（Lifelong-Robot-Learning/LIBERO） |
| `README.md` | FASTER 完全重写，增加了 FASTER 方法介绍（TL;DR、Abstract）、HAS 超参数说明、三种推理模式（Sync/Async/Streaming）部署指南、FASTER 专用训练 config 列表 |
| `pyproject.toml` | FASTER 新增 `nvitop>=1.5.3`、`datasets==3.6.0`、`av>=15.0.0,<16.0.0` 三个依赖；移除 `chex==0.1.90` |

### 3.2 examples/ 示例目录

| 文件 | 差异说明 |
|------|---------|
| `examples/aloha_real/README.md` | FASTER 将全部英文内容翻译为简体中文（结构完全相同，仅语言变更） |
| `examples/aloha_sim/README.md` | FASTER 将全部英文内容翻译为简体中文（结构完全相同，仅语言变更） |
| `examples/droid/README.md` | FASTER 将全部英文内容翻译为简体中文（结构完全相同，仅语言变更） |
| `examples/droid/README_train.md` | FASTER 将全部英文内容翻译为简体中文；安装步骤简化为 `uv sync --group rlds`，openpi 为详细的 Python 3.11 venv 说明 |
| `examples/libero/README.md` | FASTER 将英文翻译为中文；结果表格新增 `pi05_faster_libero` 对比数据（π0.5 96.9% vs π0.5+FASTER 96.5%）；评估命令改用 `pi05_faster_libero` 配置和对应 checkpoint 路径；新增清华镜像源配置 |
| `examples/libero/main.py` | FASTER 新增 `save_videos`（bool）、`out_path`（默认 `data/libero_eval`）、`save_name`（必填参数）三个 CLI 参数；视频保存路径重构为 `video_out_path / videos / save_name / task_suite_name`；文件名增加时间戳后缀；新增结果文本文件输出到 `{out_path}/results/{save_name}_results.txt` |
| `examples/simple_client/README.md` | FASTER 将英文内容翻译为简体中文（仅语言变更） |
| `examples/ur5/README.md` | FASTER 将英文内容翻译为简体中文（仅语言变更） |
| `examples/inference.ipynb` | 在 openpi 中不存在，仅存在于 FASTER |
| `examples/policy_records.ipynb` | FASTER 修改了内容（具体差异未展开） |
| `examples/convert_jax_model_to_pytorch.py` | FASTER 修改了内容（具体差异未展开） |
| `examples/aloha_real/requirements.in` | FASTER 将 8 个依赖扩展为 19 个，新增 dm_control、einops、h5py、modern_robotics、pexpect、pyquaternion、pyrealsense2、rospkg、requests、Pillow、opencv-python 等，用于 ROS/Realsense 支持 |
| `examples/aloha_sim/requirements.in` | FASTER 将 8 个依赖扩展为 19 个（与 aloha_real 相同，增加了 ROS/Realsense 支持相关依赖） |
| `examples/libero/requirements.in` | FASTER 将 5 个依赖扩展为 12 个，新增 imageio[ffmpeg]、torch、torchvision、robosuite、matplotlib、opencv-python、PyYaml 等重型依赖 |
| `examples/simple_client/requirements.in` | FASTER 将 5 个依赖扩展为 12 个（与 libero 相同） |

### 3.3 核心模型文件 — `src/openpi/models/`

| 文件 | 差异说明 |
|------|---------|
| `models/pi0.py` | FASTER 修改了 `embed_suffix` 返回类型注解：`adarms_cond` 从 `Float[Array, "b emb"]` 改为 `Float[Array, "b a emb"]`（增加了 action horizon 维度 `a`）；openpi 中的 TODO 注释 `# TODO: rewrite gemma in NNX` 在 FASTER 中被删除 |
| `models/pi0_config.py` | FASTER 新增 `Pi0FasterConfig` dataclass，包含 `max_delay=10`、`mix_prob=0.5`、`alpha=0.6`、`u0=0.9` 四个 HAS 相关字段，以及对应的 `create()` 方法返回 `Pi0Faster` 实例 |
| `models/gemma.py` | FASTER 修改了 RMSNorm 调制向量的分割方式：openpi 在 split 前使用 `modulation[:, None, :]` 增加中间维度，FASTER 直接使用 `modulation` 进行 `jnp.split(..., 3, axis=-1)` |

### 3.4 PyTorch 模型 — `src/openpi/models_pytorch/`

| 文件 | 差异说明 |
|------|---------|
| `models_pytorch/gemma_pytorch.py` | FASTER 为 `past_key_values` 参数新增 `pytest.Cache` 类型注解，用于支持 PyTorch 模型中的缓存机制 |
| `models_pytorch/pi0_pytorch.py` | FASTER 将硬编码的 action_dim 值 `32` 替换为 `config.action_dim` 和 `config.dtype`；新增 `pytorch_compile_mode` 配置支持和对应的 `torch.compile` 调用 |

### 3.5 策略文件 — `src/openpi/policies/`

| 文件 | 差异说明 |
|------|---------|
| `policies/policy.py` | FASTER 的 `Policy` 类新增 `infer_streaming()` 方法，实现高速异步流式推理：打破 JAX Host-Device 同步壁垒，通过 Python 展开循环调用 `_sample_actions_streaming_init` 和 `_sample_actions_streaming_step`；`sample_actions` 的 JIT 增加 `static_argnames`：`infer_time_schedule`、`alpha`、`u0`；新增 `action_prefix` / `delay` 参数支持（prefix mode）；新增 `concurrent.futures.ThreadPoolExecutor` 用于异步回调 |

### 3.6 服务端 — `src/openpi/serving/`

| 文件 | 差异说明 |
|------|---------|
| `serving/websocket_policy_server.py` | FASTER 新增 `StreamingWebsocketPolicyServer` 类，支持流式 action 输出：在自适应降噪过程中，提前完成的 actions 通过 `on_actions_ready` 回调立即发送。协议格式：连接打开时发 metadata → 接收 observation → 发送零个或多个 `{"type": "partial", "actions": ndarray}` → 发送一个 `{"type": "final", ...}` → 循环 |

### 3.7 共享工具 — `src/openpi/shared/`

| 文件 | 差异说明 |
|------|---------|
| `shared/download.py` | FASTER 新增 `_download_gsutil()` 函数，使用 `gsutil -m cp -r` 下载 `gs://openpi-assets/` bucket 内容，避免 gcsfs 认证问题；其他 gs:// URL 继续使用 fsspec |

### 3.8 训练配置 — `src/openpi/training/`

| 文件 | 差异说明 |
|------|---------|
| `training/config.py` | FASTER 新增 `agilex_policy` 和 `calvin_policy` 的 import；新增 `LeRobotAgilexDataConfig` dataclass（含 AgileX 机械臂数据转换配置）；新增 `LeRobotCalvinDataConfig` dataclass（含 CALVIN 任务数据转换配置）；`ModelTransformFactory.__call__` 中 `PI05` 分支增加了对 `Pi0FasterConfig` 的类型兼容判断；新增预定义训练配置：`pi05_agilex`、`pi05_rtc_agilex`（RTC 推理）、`pi05_faster_agilex`、`pi05_calvin`、`pi05_faster_calvin`、`pi05_faster_libero` |

### 3.9 数据变换 — `src/openpi/transforms.py`

| 文件 | 差异说明 |
|------|---------|
| `transforms.py` | FASTER 修改了四处：`Normalize.__call__` 新增对 `action_prefix` 的归一化处理（当 action_prefix 不在 norm_stats 中时复用 actions 的统计量）；`Unnormalize.__call__` 新增从 norm_stats 中删除 action_prefix 的逻辑；`DeltaActions.__call__` 新增对 `action_prefix` 的 delta 转换支持；`PadStatesAndActions.__call__` 新增 `model_action_horizon` 参数并增加对 `action_prefix` 的 padding 处理 |

### 3.10 脚本 — `scripts/`

| 文件 | 差异说明 |
|------|---------|
| `scripts/serve_policy.py` | FASTER 新增 `--streaming`（启用流式服务器）、`--early-stop-actions`（提前停止的 action 数）、`--use-custom-sample-kwargs`（传递自定义采样参数）、`--infer-time-schedule`（`const` 或 `HAS`）、`--alpha`、`--u0`、`--num_steps` 七个 CLI 参数；根据 `--streaming` 标志选择创建 `WebsocketPolicyServer` 或 `StreamingWebsocketPolicyServer` |
| `scripts/compute_norm_stats.py` | FASTER 将 norm_stats 的输出路径从 `{assets_dirs}/{repo_id}` 改为 `{assets_dirs}/{asset_id}` |

### 3.11 openpi-client 客户端包 — `packages/openpi-client/`

| 文件 | 差异说明 |
|------|---------|
| `pyproject.toml` | FASTER 将 dev 依赖配置从 `[dependency-groups]` 改为 `[tool.uv] dev-dependencies`；新增 `tree>=0.2.4` 依赖 |
| `src/openpi_client/base_policy.py` | FASTER 在 `BasePolicy` 中新增 `infer_streaming()` 方法，默认实现直接回退到 `infer()`（非流式） |
| `src/openpi_client/websocket_client_policy.py` | FASTER 新增 `infer_streaming()` 方法：与 `StreamingWebsocketPolicyServer` 配合使用，读取 `type=partial` 消息时触发 `on_actions_ready` 回调，读取 `type=final` 消息时返回最终结果 |

---

## 四、两端完全相同的文件（无差异）

以下文件在 FASTER 和 openpi 中内容完全一致，列出以避免重复检查：

| 文件 |
|------|
| `examples/droid/main.py` |
| `examples/droid/compute_droid_nonidle_ranges.py` |
| `examples/aloha_real/requirements.txt` |
| `examples/aloha_sim/requirements.txt` |
| `examples/libero/requirements.txt` |
| `examples/simple_client/requirements.txt` |
| `examples/aloha_real/convert_aloha_data_to_lerobot.py` |
| `examples/droid/convert_droid_data_to_lerobot.py` |
| `examples/libero/convert_libero_data_to_lerobot.py` |
| `src/openpi/policies/aloha_policy.py` |
| `src/openpi/policies/droid_policy.py` |
| `src/openpi/policies/libero_policy.py` |
| `src/openpi/policies/policy_config.py` |
| `src/openpi/policies/policy_test.py` |
| `src/openpi/training/checkpoints.py` |
| `src/openpi/training/data_loader.py` |
| `src/openpi/training/data_loader_test.py` |
| `src/openpi/training/weight_loaders.py` |
| `src/openpi/training/sharding.py` |
| `src/openpi/training/optimizer.py` |
| `src/openpi/training/utils.py` |
| `scripts/train.py` |
| `scripts/train_pytorch.py` |
| `scripts/train_test.py` |

---

## 五、Pi0Faster 新模型类 — 核心差异详解

[`pi0_faster.py`](src/openpi/models/pi0_faster.py) 是 FASTER 最核心的新增文件，与 openpi 的 `Pi0` 相比：

### 5.1 构造函数新增属性
- `self.max_delay`：最大延迟前缀长度（训练时模拟推理延迟）
- `self.mix_prob`：HAS 混合概率（每个 batch 以此概率使用 HAS）
- `self.alpha`：HAS 超参数，控制动作索引间命中时间的变化方式
- `self.u0`：首个 action 被确定时的全局时间步（默认为 (N-1)/N）

### 5.2 新增方法

| 方法 | 功能 |
|------|------|
| `compute_HAS(time, delay, alpha, u0)` | 实现 Horizon-Aware Schedule：近时域 action 先完成降噪，远时域 action 后完成 |
| `sample_actions_streaming_init(...)` | 流式推理初始化：预计算 KV cache 和完整时间调度表 |
| `sample_actions_streaming_step(...)` | 单步流式推理：设计为在 host 循环中异步调用，无 JAX 同步开销 |

### 5.3 `embed_suffix` 签名变化
- openpi：`timestep: Float[Array, " b"]`（所有 action 位置共享同一 timestep）
- FASTER：`timestep: Float[Array, " b ah"]`（每个 action 位置可有不同 timestep，实现 HAS）

### 5.4 `sample_actions` 新增参数
- `delay`：推理延迟（对应 action prefix 长度）
- `action_prefix`：已知的干净 action 前缀
- `infer_time_schedule`：`"const"`（常数调度）或 `"HAS"`（自适应调度）
- `alpha`、`u0`：HAS 超参数

---

## 六、子模块差异

| | openpi | FASTER |
|---|---|---|
| `.gitmodules` | `third_party/aloha` → Physical-Intelligence/aloha | `third_party/libero` → Lifelong-Robot-Learning/LIBERO |

openpi 的 `third_party/aloha` 子模块在 FASTER 中被完全移除，替换为 `third_party/libero`。

---

## 七、Git 历史差异

openpi 额外拥有 `.github/` 目录（CI/CD workflow 配置），FASTER 完全不包含此目录。
