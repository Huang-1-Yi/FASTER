# FASTER 仓库阅读与实验路线指南

> **项目背景**：基于 openpi 实现 FASTER: Rethinking Real-Time Flow VLAs，核心是为 π0.5 / flow VLA 加入 Horizon-Aware Schedule、action_prefix 和 streaming inference，以降低真实机器人实时反应延迟。

---

## 目录

- [项目定位](#项目定位)
- [三阶段路线](#三阶段路线)
- [核心文件地图](#核心文件地图)
- [推荐阅读顺序](#推荐阅读顺序)
- [仿真阶段指南](#仿真阶段指南)
- [真实机器人部署指南](#真实机器人部署指南)
- [动手修改建议](#动手修改建议)
- [最小可行实验路线](#最小可行实验路线)

---

## 项目定位

FASTER = **openpi 主体 + FASTER 对实时推理链路的改造**

核心创新点：
| 模块 | 创新内容 |
|------|---------|
| Horizon-Aware Schedule (HAS) | 根据推理延迟自适应 action horizon |
| action_prefix | 流式推理时前置动作更快响应 |
| streaming inference | time-to-first-action 优化 |

**最重要的事**：不要按文件从头读，而是按**数据流**读：

```
数据集/机器人观测 → policy adapter → transforms → model → Policy wrapper → WebSocket server/client → 机器人执行动作
```

---

## 三阶段路线

> 当前无真实硬件，优先仿真，真实部署在硬件到位后进行。

| 阶段 | 目标 | 入口 |
|------|------|------|
| **仿真阶段** | 跑通算法链路 | `examples/libero`、`examples/calvin` |
| **Franka 阶段** | 7DoF + DROID 体系 | `examples/droid` |
| **Aubo 阶段** | 6DoF + UR5 体系 | `examples/ur5` |

### 1. 仿真阶段：先跑通算法链路

目标是确认环境、checkpoint、数据格式、server/client、FASTER streaming 都能工作。
优先看 `examples/libero` 和 `examples/calvin`，因为它们已经是仓库里现成的仿真入口。

### 2. Franka 阶段：优先借 DROID / Franka 体系

Franka 是 7DoF 机械臂，仓库文档里已有 DROID/Franka 相关动作空间说明。
先把 Franka 看成"7 关节 + 夹爪"的 8 维 action/state 问题，再决定是否沿用 DROID 风格的 joint position / joint velocity。

### 3. Aubo 阶段：优先借 UR5 体系

Aubo 通常更接近 UR5 这类 6DoF 工业臂。仓库里有 `examples/ur5/README.md`，它不是完整可运行示例，但非常适合当作 Aubo 适配模板：6 个关节 + 夹爪通常是 7 维 action；如果末端没有夹爪，则可以是 6 维。

---

## 核心文件地图

### 1. 总览和命令

- [README.md](README.md) — 项目总览
- [compare.md](compare.md) — FASTER 相比 openpi 的新增点汇总

### 2. 配置中心

**重点**：[src/openpi/training/config.py](src/openpi/training/config.py)

这个文件决定用哪个模型、哪个数据集、哪些 transform、加载哪个预训练权重、训练多少步。

| Config Name | 用途 |
|-------------|------|
| `pi05_faster_libero` | LIBERO 仿真 |
| `pi05_faster_calvin` | CALVIN 仿真 |
| `pi05_faster_agilex` | AgileX 实机 |
| `pi05_droid_finetune` | DROID/Franka 微调 |

### 3. 数据适配

**路径**：[src/openpi/policies](src/openpi/policies)

负责把不同环境的数据转成模型统一格式。

| 文件 | 职责 |
|------|------|
| [agilex_policy.py](src/openpi/policies/agilex_policy.py) | AgileX 相机、state、action → `image/state/actions/prompt` |
| [droid_policy.py](src/openpi/policies/droid_policy.py) | Franka/DROID → `state/image/image_mask/prompt/actions` |
| [libero_policy.py](src/openpi/policies/libero_policy.py) | LIBERO → 统一格式 |

### 4. 通用变换

**路径**：[src/openpi/transforms.py](src/openpi/transforms.py)

做归一化、图像 resize、prompt tokenize、action padding、delta action。FASTER 额外支持 `action_prefix`。

### 5. 核心创新

**路径**：[src/openpi/models/pi0_faster.py](src/openpi/models/pi0_faster.py)

| 函数 | 作用 |
|------|------|
| `compute_loss()` | 训练时模拟 delay 和 HAS |
| `compute_HAS()` | Horizon-Aware Schedule 核心算法 |
| `sample_actions()` | 普通推理 |
| `sample_actions_streaming_init()` | 流式推理初始化 |
| `sample_actions_streaming_step()` | 流式推理单步 |

### 6. 部署链路

```
scripts/serve_policy.py
    ↓
src/openpi/policies/policy.py (Policy.infer() / infer_streaming())
    ↓
src/openpi/serving/websocket_policy_server.py
    ↓
packages/openpi-client/src/openpi_client/websocket_client_policy.py
```

负责 server/client 推理，尤其是 streaming 下的 partial actions。

---

## 推荐阅读顺序

> 不要泛读全仓库。建议按"仿真 → Franka → Aubo"的顺序读。

### 第一轮：仿真路径

先建立可运行闭环。

1. [examples/libero/README.md](examples/libero/README.md) 和 [examples/calvin/README.md](examples/calvin/README.md)
2. 从 `config.py` 找 `pi05_faster_libero`
3. 看它用的 `Pi0FasterConfig`，再跳到 `pi0_config.py`
4. 看 `pi0_faster.py` 的 `compute_loss()`
5. 回到 `scripts/train.py`，理解训练循环只是反复调用 `model.compute_loss()`
6. 看推理：`serve_policy.py` → `Policy.infer()` / `infer_streaming()` → `Pi0Faster.sample_actions()`
7. 看 `examples/libero/main.py` 或 `examples/calvin/main.py` 的评估脚本

### 第二轮：Franka / DROID 迁移路径

1. `docs/norm_stats.md` — 重点看 `franka`、`droid`、动作维度和控制频率
2. `examples/droid/README.md` — 理解远程 policy server + 机器人客户端配合
3. `src/openpi/policies/droid_policy.py` — 数据格式转换
4. `config.py` 的 `pi05_droid_finetune` — 有自己 Franka 数据后怎么微调

### 第三轮：Aubo / UR5 迁移路径

1. `examples/ur5/README.md` — 如何写一个新机械臂 adapter 模板
2. 对照 `libero_policy.py` 和 `agilex_policy.py` — 理解不同机器人只是输入键、维度不同
3. 想象未来会新增 `AuboInputs`、`AuboOutputs`、`LeRobotAuboDataConfig` 和 `pi05_faster_aubo`

---

## 仿真阶段指南

仿真阶段最重要的不是马上追求高分，而是确认**四件事**：

### 1. 环境能跑

先用 LIBERO 或 CALVIN 跑 baseline，不要先动模型。
- **LIBERO**：更贴近 manipulation benchmark
- **CALVIN**：更适合看长任务链

### 2. server/client 能通

先跑普通 `infer()`，再跑 `--streaming`。
- 普通推理通了 → checkpoint、transform、归一化、observation 格式没问题
- streaming 通了 → FASTER 的实时链路可用

### 3. shape 能对

对每个环境都确认：

| 检查项 | 说明 |
|--------|------|
| `state` 维度 | 是几维 |
| `actions` 维度 | 是几维 |
| `action_horizon` | 是多少 |
| 图像 key | 能否映射到 `base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb` |

### 4. 指标要分开看

- **仿真成功率**：证明性能没有明显掉
- **TTFA / streaming latency**：FASTER 的核心价值

---

## 真实机器人部署指南

### Franka 部署前要看什么

Franka 的阅读重点是"动作空间、归一化、相机视角、控制频率"。

#### 1. 动作空间

7DoF robot 通常使用前 7 维表示 joint actions，第 8 维表示 gripper。
要明确到底是 joint position、joint velocity，还是末端位姿增量。

#### 2. 归一化统计

先看 `docs/norm_stats.md`。Franka 有两个可能入口：

| asset_id | 风格 |
|----------|------|
| `droid` | 更贴近 DROID 数据/控制方式 |
| `franka` | 更贴近非 DROID 的 Franka/FR3 统计 |

> 真正微调时，建议两条都试：复用已有 stats 和重新计算自己数据 stats。

#### 3. 数据格式

如果采集 Franka 数据，最好从一开始就整理成 **LeRobot 格式**。这样可以复用 `create_torch_dataset()`、`compute_norm_stats.py` 和训练脚本。

#### 4. 客户端

真机侧写一个轻量客户端：
```
读取相机和关节状态 → 组 observation → 调 openpi-client → 执行动作块
```

### Aubo 部署前要看什么

Aubo 的阅读重点是"自己定义 adapter"，不要强行套 DROID。

#### 1. 先以 UR5 示例为模板

Aubo 如果是 6 轴 + 夹爪，基本可以按 UR5 的结构走：
```
state = [6 joints, gripper]
actions = [6 joints, gripper]
```

#### 2. 确认 action 语义

最早就要确定模型输出到底控制什么：
- 关节位置目标
- 关节速度
- 末端位姿增量
- 夹爪开合

这个决定 `DeltaActions` 是否使用、mask 怎么写、机器人客户端怎么执行。

#### 3. 相机设计要尽早固定

openpi/pi0 默认吃一个 base view 和两个 wrist view。Aubo 实机未必有腕部相机；没有就用零图像和 mask 关掉，对应写法可以参考 `libero_policy.py` 和 `droid_policy.py`。

#### 4. 仿真先对齐 Aubo 数据结构

在真机未到位前，可以先用假数据或仿真数据做一个 `AuboInputs` 的 smoke test。
目标不是训练出好策略，而是提前发现 key、shape、action_dim、padding、normalization 的问题。

---

## 动手修改建议

如果你想基于它做创新，建议分三层动手：

### 第一层：低风险 — 改 config

比如修改 `max_delay`、`mix_prob`、`alpha`、`u0`、`action_horizon`、`num_steps`。
这类创新主要验证 FASTER 的 schedule 和实时性权衡。

### 第二层：中风险 — 改数据/机器人适配

如果接自己的机器人，复制一个 `*_policy.py`，实现新的 `Inputs/Outputs`，再在 `config.py` 里加一个新的 `LeRobotXXXDataConfig` 和 `TrainConfig`。

### 第三层：高风险 — 改算法

核心入口是 `compute_HAS()`。

可尝试方向：
- 动态 `alpha/u0`，根据实时推理延迟自适应
- 不同 action 维度使用不同 schedule
- 让近端动作更快收敛、远端动作保留更多探索
- 训练时加入 latency-aware loss
- 改 streaming early-stop 逻辑，让 server 根据动作置信度而不是固定数量提前返回

---

## 最小可行实验路线

> 先别直接上真实机器人。建议按下面顺序做。

### 第一步：最小训练 smoke test

```bash
# 计算归一化统计
uv run scripts/compute_norm_stats.py --config-name pi05_faster_libero

# 最小训练（10 步）
WANDB_MODE=offline uv run scripts/train.py pi05_faster_libero \
  --exp-name=test_faster \
  --num-train-steps=10 \
  --overwrite
```

### 第二步：普通推理 server/client smoke test

```bash
# 启动 server
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_faster_libero \
  --policy.dir=checkpoints/pi05_faster_libero/test_faster/9

# 另开终端，用 simple client 或 LIBERO/CALVIN client 测试连接
```

### 第三步：streaming 推理 smoke test

```bash
uv run scripts/serve_policy.py \
  --use-custom-sample-kwargs \
  --infer-time-schedule=HAS \
  --alpha=0.6 \
  --u0=0.9 \
  --streaming \
  --early-stop-actions=4 \
  policy:checkpoint \
  --policy.config=pi05_faster_libero \
  --policy.dir=checkpoints/pi05_faster_libero/test_faster/9
```

### 第四步：仿真 benchmark

用 LIBERO/CALVIN 跑完整评估，分别记录：

| 指标 | 说明 |
|------|------|
| 成功率 | 任务完成率 |
| 普通推理耗时 | end-to-end 延迟 |
| streaming TTFA | time-to-first-action |
| early_stop_actions | 不同取值对成功率和延迟的影响 |

### 第五步：准备 Franka/Aubo 的 adapter 草图

暂时没有硬件也可以先写"数据契约"：

- [ ] observation 里有哪些图像
- [ ] state/action 是几维
- [ ] gripper 如何归一化
- [ ] 控制频率是多少
- [ ] action 是 absolute 还是 delta

---

## 一句话抓重点

> **你现在要先把 FASTER 当成"仿真里可测的实时策略系统"看懂；等 Franka/Aubo 硬件和数据到位后，再把重点从 `Pi0Faster` 转到 `policies/*_policy.py`、`transforms.py`、`training/config.py` 和真机 client。**
