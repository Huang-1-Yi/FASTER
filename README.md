<p align="center">
  <h1 align="center">FASTER: Rethinking Real-Time Flow VLAs</h1>
  <p align="center">
    <strong><a href="https://innovator-zero.github.io/">Yuxiang Lu</a><sup>1,2</sup></strong>
    &nbsp;&nbsp;
    <strong><a href="https://happinesslz.github.io/">Zhe Liu</a><sup>1</sup></strong>
    &nbsp;&nbsp;
    <strong><a href="https://www.linkedin.com/in/xianzhefan">Xianzhe Fan</a><sup>1</sup></strong>
    &nbsp;&nbsp;
    <strong><a href="https://huster-yzy.github.io/">Zhenya Yang</a><sup>1</sup></strong>
    &nbsp;&nbsp;
    <strong><a href="https://scholar.google.com/citations?user=aoqtBAsAAAAJ&hl=en">Jinghua Hou</a><sup>1</sup></strong>
    &nbsp;&nbsp;
    <br>
    <strong><a href="https://provencestar.github.io/">Junyi Li</a><sup>1</sup></strong>
    &nbsp;&nbsp;
    <strong><a href="https://kxding.github.io">Kaixin Ding</a><sup>1</sup></strong>
    &nbsp;&nbsp;
    <strong><a href="https://i.cs.hku.hk/~hszhao/">Hengshuang Zhao</a><sup>1</sup></strong>
  </p>

  <p align="center">
    <sup>1</sup> The University of Hong Kong
    <sup>2</sup> ACE Robotics
  </p>

  <p align="center">
  <a href="https://arxiv.org/abs/2603.19199"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2603.19199-b31b1b.svg"></a>
  <a href="https://innovator-zero.github.io/FASTER"><img alt='proj' src="https://img.shields.io/badge/Project Page-82B366.svg"></a>
  </p>
</p>

本仓库提供了 FASTER 的官方实现，同时基于 [openpi](https://github.com/Physical-Intelligence/openpi) 构建了 [Training-time RTC](https://arxiv.org/abs/2512.05964) 的非官方实现。


## TL;DR

VLA 模型的实时反应不仅受推理延迟约束，还受动作分块的生成和执行方式影响。**FASTER** 引入了一种在异步执行下快速采样动作的新范式。通过将即时反应所需的采样过程压缩为单步，FASTER 相比 $\pi_{0.5}$ 和 X-VLA 实现了 **10 倍加速**，使高度动态任务（如打乒乓球）中的实时响应成为可能。

<img width="2960" height="836" alt="teaser" src="https://github.com/user-attachments/assets/121bf40a-20dc-41ff-bac0-c6d96edfb1c0" />

[Demo](https://github.com/user-attachments/assets/c16bd3fa-48ac-4d1b-aef9-4b0f4d839011)

## 📰 News

- **Apr 30 2026**: Code released.
- **Mar 19 2026**: Paper released.


## 📰 Abstract
实时执行对于将视觉-语言-动作（Vision-Language-Action, VLA）模型部署到物理世界至关重要。现有异步推理方法主要优化轨迹平滑性，但忽略了应对环境变化时的关键延迟。通过重新审视动作分块策略中“反应”的概念，本文对影响反应时间的因素进行了系统分析。我们表明，反应时间由首动时间（Time to First Action, TTFA）和执行 horizon 共同决定，并服从均匀分布。此外，我们发现基于流的 VLA 中应用常数调度策略的标准做法可能效率低下，会迫使系统在开始任何动作之前完成所有采样步骤，从而成为反应延迟的瓶颈。为解决这一问题，我们提出了 **F**ast **A**ction **S**ampling for Immedia**TE** **R**eaction（**FASTER**）。通过引入 Horizon-Aware Schedule，FASTER 在流采样过程中自适应地优先处理近时域动作，将即时反应的降噪过程压缩十倍（*例如* 在 $\pi_{0.5}$ 和 X-VLA 中压缩为单步），同时保持长时域轨迹的质量。结合流式客户端-服务器管道，FASTER 显著降低了真实机器人上的有效反应延迟，尤其是在消费级 GPU 上部署时。包括高度动态的乒乓球任务在内的真实世界实验表明，FASTER 为通用策略解锁了前所未有的实时响应能力，能够快速生成精确且平滑的轨迹。

## ⚙️ Setup
> 本仓库基于 [openpi](https://github.com/Physical-Intelligence/openpi)（JAX 版本）构建。在继续之前，我们强烈建议先熟悉原始 openpi 的工作流程。

### Requirements

FASTER 的硬件要求与 openpi 相同。运行本仓库中的模型需要 NVIDIA GPU，且至少满足以下规格。这些估算基于单 GPU，但你也可以通过在训练配置中设置 `fsdp_devices`，使用多 GPU 模型并行来降低每张 GPU 的显存需求。请注意，当前训练脚本尚不支持多节点训练。

| Mode               | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Inference          | > 8 GB          | RTX 4090           |
| Fine-Tuning (LoRA) | > 22.5 GB       | RTX 4090           |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H100 |

本仓库已在 Ubuntu 22.04 上使用 $\pi_{0.5}$ 完整微调进行测试。

### Installation

克隆本仓库时，请确保更新子模块：

```bash
git clone --recurse-submodules https://github.com/innovator-zero/FASTER.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

我们使用 [uv](https://docs.astral.sh/uv/) 管理 Python 依赖。请参考 [uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/)完成安装。安装 uv 后，运行以下命令配置环境：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

注意：拉取作为依赖项的 LeRobot（数据集 `v2.1`，commit `0cf8648`）时需要设置 `GIT_LFS_SKIP_SMUDGE=1`。

## 🚀 Usage

我们提供了在 AgileX Piper 平台上训练和部署 FASTER 的详细指南。如果你已经将 openpi 适配到自己的机器人平台（包括数据处理和映射），那么运行 FASTER 应该非常直接。

对于仿真基准测试，可以参考 [LIBERO](examples/libero/README.md) 和 [CALVIN](examples/calvin/README.md) 指南。

### Policy Training

FASTER 遵循 openpi 的标准微调流程。主要区别在于使用 **HAS (Horizon-Aware Schedule)** 替代传统的常数调度策略。

#### 1. Prepare Data

我们使用 LeRobot 数据集 v2.1 作为数据加载器。如果你使用 [AgileX Piper](https://global.agilex.ai/products/piper) 机械臂或 [Cobot Magic](https://global.agilex.ai/products/cobot-magic) 系统，我们强烈推荐使用 [piper-aio](https://github.com/innovator-zero/piper-aio) 工具包。它覆盖了完整的机器人学习流程：硬件设置、遥操作数据采集、数据回放、LeRobot 数据集转换和策略推理。

#### 2. Define Config

AgileX Piper 数据的数据处理配置位于 [`AgilexInputs`](src/openpi/policies/agilex_policy.py)、[`AgilexOutputs`](src/openpi/policies/agilex_policy.py) 和 [`LeRobotAgilexDataConfig`](src/openpi/training/config.py)。
我们在 [`config.py`](src/openpi/training/config.py) 中提供了以下示例训练配置：

- `pi05_agilex`：使用常数调度策略微调 $\pi_{0.5}$ 模型。
- `pi05_rtc_agilex`：使用 [Training-time RTC](https://arxiv.org/abs/2512.05964) 提出的动作条件策略微调 $\pi_{0.5}$ 模型。
- `pi05_faster_agilex`：使用 FASTER 微调 $\pi_{0.5}$ 模型，并采用以下超参数：
  - `max_delay`：最大前缀长度 $d_\text{max}$，通过动作条件策略在训练期间模拟推理延迟。默认值为 `10`，支持在 30 Hz 机器人上实现最长 333.3 ms 的 TTFA (Time to First Action)。
  - `mix_prob`：**混合调度策略** 中的混合概率 $p$。每个动作样本以概率 $p$ 使用 HAS，以概率 $1 - p$ 保留原始常数调度。默认值为 `0.5`。
  - `alpha`：HAS 超参数，控制动作索引之间命中时间的变化方式。默认值为 `0.6`。
  - `u0`：首个动作被确定时的全局时间步，设为 $N$ 次推理采样步中的 $(N-1)/N$。由于 $\pi_{0.5}$ 使用 10 步，默认设置 `u0=0.9`。

#### 3. Launch Training

启动训练前，记得先计算训练数据的归一化统计量。这些统计量可以通过符号链接（`ln -s`）在多个配置之间共享：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_faster_agilex
```

然后即可启动训练：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_faster_agilex --exp-name=my_experiment
```



### Policy Deployment

由于机器人控制器通常需要不同的环境或独立的机器，我们使用客户端-服务器接口进行策略部署。服务器处理策略推理，而客户端运行机器人控制器，并通过 WebSocket 建立通信。你可以在同一台机器上启动策略服务器，也可以在局域网连接的工作站上启动。对于远程推理，建议使用有线局域网以降低延迟和丢包。

**在客户端侧，我们使用 [piper-aio](https://github.com/innovator-zero/piper-aio/tree/main/inference) 工具包中提供的推理脚本。如果你使用自己的机器人平台，可以根据实际设置适配这些脚本。**

#### Sync Inference

这是动作分块策略的标准做法，也是 $\pi_{0.5}$ 等 VLA 的基准方式。机器人执行一个动作分块，只有在当前分块被完全消费后才请求下一个分块。在策略推理期间，机器人控制器暂停，只有在新动作到达时才恢复。

##### 使用常数调度策略启动策略服务器：

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_agilex --policy.dir=checkpoints/pi05_agilex/my_experiment/49999
```
###### 强制当前环境启动
```bash
uv run --active scripts/serve_policy.py --use-custom-sample-kwargs --infer-time-schedule=HAS --alpha=0.6 --u0=0.9 --streaming --early-stop-actions=4 policy:checkpoint --policy.config=pi05_faster_agilex --policy.dir=checkpoints/pi05_faster_agilex/my_experiment/49999
```
###### python启动
```bash
python scripts/serve_policy.py --use-custom-sample-kwargs --infer-time-schedule=HAS --alpha=0.6 --u0=0.9 --streaming --early-stop-actions=4 policy:checkpoint --policy.config=pi05_faster_agilex --policy.dir=checkpoints/pi05_faster_agilex/my_experiment/49999
```
以 sync 模式启动机器人客户端：

```bash
python inference/infer_sync.py # ... other arguments
```
###### 数据集自动下载到/home/hy/.cache/huggingface/lerobot/
```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero                    # 给 pi05 配置也算一次
uv run --active scripts/compute_norm_stats.py --config-name pi05_faster_libero    # 再给 FASTER 配置也算一次

WANDB_MODE=offline uv run scripts/train.py pi05_libero --exp-name=test_openpi --num-train-steps=10 --overwrite
WANDB_MODE=offline uv run scripts/train.py pi05_faster_libero --exp-name=test_faster --num-train-steps=10 --overwrite

```

#### Async Inference

机器人在当前分块完全执行之前就开始为下一个分块发起推理请求。一旦下一个推理请求被触发，机器人继续执行当前分块中的持续动作。在最后一个动作完成之前，新预测的分块应该已经就位，从而实现无缝执行而不会停止。

使用 HAS 启动策略服务器（需与训练时使用的 `alpha` 和 `u0` 保持一致）：

```bash
uv run scripts/serve_policy.py --use-custom-sample-kwargs --infer-time-schedule=HAS --alpha=0.6 --u0=0.9 policy:checkpoint --policy.config=pi05_faster_agilex --policy.dir=checkpoints/pi05_faster_agilex/my_experiment/49999
```

以 async (rtc) 模式启动机器人客户端：

```bash
python inference/infer_async.py --mode=rtc --delay=4 --exec_horizon=25 # ... other arguments
```

参数：

- `--delay`：推理延迟 $d:= \lfloor \Delta t_\text{infer}/\Delta t_{\text{ctrl}}\rfloor$，由推理延迟 $\Delta t_\text{infer}$ 和控制周期 $\Delta t_{\text{ctrl}}$ 决定。我们建议将其设置得大一个控制步，以覆盖推理时间和传输延迟的波动。
- `--exec_horizon`：动作分块的执行 horizon $s$。客户端只执行前 $s$ 个有效动作（不包含延迟动作），然后触发新的推理请求。该值应大于或等于 $d$。

#### Streaming Inference

该模式基于我们的 **Streaming Client-Server Interface** 构建，实现了最低的 TTFA (Time to First Action)。早期动作一旦完成即可立即分发给机器人客户端。当机器人执行这些初始动作时，策略服务器继续并行优化后续动作，并逐步补充客户端的动作缓冲区。该模式专为需要快速响应的任务而设计，例如打乒乓球。

以 streaming 模式启动策略服务器：

```bash
uv run scripts/serve_policy.py --use-custom-sample-kwargs --infer-time-schedule=HAS --alpha=0.6 --u0=0.9 --streaming --early-stop-actions=4 policy:checkpoint --policy.config=pi05_faster_agilex --policy.dir=checkpoints/pi05_faster_agilex/my_experiment/49999
```

参数：

- `--early-stop-actions`：如果策略已生成并向客户端分发了指定数量的有效动作，则剩余的动作采样迭代将提前停止。该值应大于或等于 `exec_horizon`，因为剩余的带噪声动作不会被机器人执行，无需生成。这样策略服务器就可以为下一次推理请求做好准备。

以 streaming 模式启动机器人客户端：

```bash
python inference/infer_async.py --mode=rtc --delay=3 --exec_horizon=4 --streaming # ... other arguments
```

参数：

- `--delay`：由于以 TTFA 衡量的推理延迟缩短，因此可以比 async 模式下设置得更小。
- `--exec_horizon`：较小的 $s$ 可以通过提高推理频率并收紧推理-执行循环来帮助高度动态任务。对于日常任务（如拾取放置或叠毛巾），较小的 $s$ 通常不必要，反而可能增加运动抖动。

## 📖 Citation

```
 @article{lu2026faster,
  title={FASTER: Rethinking Real-Time Flow VLAs}, 
  author={Yuxiang Lu and Zhe Liu and Xianzhe Fan and Zhenya Yang and Jinghua Hou and Junyi Li and Kaixin Ding and Hengshuang Zhao},
  year={2026},
  journal={arXiv preprint arXiv:2603.19199}
}
```

## 🙏 Acknowledgements

我们感谢以下仓库提供的参考和前期工作：

- [openpi](https://github.com/Physical-Intelligence/openpi)
- [real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)
- [AgiBot-World](https://github.com/OpenDriveLab/AgiBot-World)
