# PI0 与 FASTER / Pi0Faster 的理念和代码链路对比指南

> 本文档融合 `guidence_v1.md` ~ `guidence_v4.md` 与 `compare.md` 的分析，用“理念 + 函数调用流 + 每个模块输出”的方式说明：原始 PI0/PI0.5 和 FASTER 论文、本仓库 `Pi0Faster` 实现到底差在哪里。

## 0. 先澄清三个名字

| 名字 | 代码入口 | 本质 | 输出是否是连续 action | 和 FASTER 论文关系 |
|---|---|---|---|---|
| `PI0` / `PI0.5` | [Pi0](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:66), [Pi0Config](C:/QClaw/FASTER_hy/src/openpi/models/pi0_config.py:20) | flow matching 连续动作生成器 | 是，`sample_actions()` 返回 `x_0` | FASTER 的基础模型范式 |
| `PI0_FAST` | [Pi0FAST](C:/QClaw/FASTER_hy/src/openpi/models/pi0_fast.py:134) | FAST action token 自回归生成器 | 否，先返回 `output_tokens` | 不是本文 FASTER；名字容易混淆 |
| `Pi0Faster` | [Pi0Faster](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:67), [Pi0FasterConfig](C:/QClaw/FASTER_hy/src/openpi/models/pi0_config.py:106) | 连续 action flow + prefix/HAS/streaming | 是，仍返回连续 `x_0` | 本仓库对应 FASTER 论文的核心实现 |

最重要的一点：`Pi0Faster` 不是 `PI0_FAST`。`PI0_FAST` 把动作变成 token 来生成，再由 [ExtractFASTActions](C:/QClaw/FASTER_hy/src/openpi/transforms.py:294) 解码回连续动作；`Pi0Faster` 不走 token action 路线，它保留 PI0/PI0.5 的连续 flow action 生成，只改变时间调度、前缀条件和流式输出。

## 1. 理念对比

| 维度 | PI0 / PI0.5 | FASTER / Pi0Faster | 关键代码 |
|---|---|---|---|
| 核心目标 | 生成高质量完整 action chunk | 降低机器人对环境变化的首次反应延迟，即 TTFA | [Pi0Faster.sample_actions_streaming_init](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:365) |
| 动作生成单位 | 一次输出完整 `[action_horizon, action_dim]` | 内部仍有完整 chunk，但近端 action 可先 ready | [sample_actions_streaming_step](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:419) |
| 时间调度 | 所有 horizon 位置共享同一个 `time` | 每个 horizon 位置有独立 timestep，近端动作更早到 `t=0` | [compute_HAS](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:231) |
| 推理方式 | 等全部 denoising steps 完成后返回 | streaming 模式中每步检查 `newly_ready` 并提前发出 | [Policy.infer_streaming](C:/QClaw/FASTER_hy/src/openpi/policies/policy.py:127) |
| 已执行动作处理 | 普通 PI0 不建模 `delay/action_prefix` | 支持把 horizon 前缀视为已知干净 action | [Pi0Faster.sample_actions 参数](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:256) |
| 训练目标 | flow matching，预测 `v_t` | 仍是 flow matching，但混合 const/HAS，并 mask 掉 prefix loss | [Pi0.compute_loss](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:189), [Pi0Faster.compute_loss](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:190) |
| 代码侵入点 | `pi0.py` 主模型 | 新增 `pi0_faster.py`，并改 `policy.py`、`transforms.py`、server/client | [compare.md](C:/QClaw/FASTER_hy/compare.md:7) |

一句话心智模型：

```text
PI0：看到 observation 后，从 noise 迭代 10 步，最后一次性给完整 action chunk。
FASTER：仍从 noise 迭代，但让靠前的 action 更早完成，并在完成时立刻流式发给执行端。
```

## 2. 总函数调用流

普通 PI0 / Pi0Faster 非 streaming 推理共享同一个外壳：

```text
env / dataset obs
-> environment adapter: LiberoInputs / CalvinInputs / DroidInputs / AgilexInputs
-> create_trained_policy 组装 input transforms
-> Normalize / ResizeImages / TokenizePrompt / PadStatesAndActions
-> Policy.infer
-> Observation.from_dict
-> model.sample_actions
-> {"state": inputs["state"], "actions": model_return}
-> output transforms: model outputs -> Unnormalize -> environment Outputs adapter
-> executable actions
```

对应代码主入口：

| 环节 | 函数 | 输入 | 输出 |
|---|---|---|---|
| policy 创建 | [create_trained_policy](C:/QClaw/FASTER_hy/src/openpi/policies/policy_config.py:16) | `TrainConfig`, checkpoint, `sample_kwargs` | `Policy`，含 input/output transforms |
| 输入 transform | [Policy.__init__](C:/QClaw/FASTER_hy/src/openpi/policies/policy.py:25) | transform 列表 | `_input_transform`, `_output_transform`, `_sample_actions` |
| 推理入口 | [Policy.infer](C:/QClaw/FASTER_hy/src/openpi/policies/policy.py:77) | 原始 obs dict | 最终 action dict |
| 结构化输入 | [Observation.from_dict](C:/QClaw/FASTER_hy/src/openpi/models/model.py:109) | `image/image_mask/state/tokenized_prompt` | `Observation` dataclass |
| 模型生成 | [Pi0.sample_actions](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:217) 或 [Pi0Faster.sample_actions](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:256) | `Observation`, rng, optional kwargs | 归一化空间中的连续 actions |
| 输出 transform | [policy_config.py output_transforms](C:/QClaw/FASTER_hy/src/openpi/policies/policy_config.py:84) | `{"state", "actions"}` | 环境动作空间中的 actions |

## 3. 配置层：Pi0Faster 如何复用 PI0/PI0.5 的 transform

`Pi0FasterConfig` 没有新建 `ModelType.PI0_FASTER`。它的 [model_type](C:/QClaw/FASTER_hy/src/openpi/models/pi0_config.py:128) 仍然根据 `pi05` 返回 `PI05` 或 `PI0`：

```text
Pi0FasterConfig(pi05=True)  -> ModelType.PI05
Pi0FasterConfig(pi05=False) -> ModelType.PI0
```

这意味着在 [ModelTransformFactory](C:/QClaw/FASTER_hy/src/openpi/training/config.py:104) 中，`Pi0Faster` 不会进入 `PI0_FAST` 分支，而是复用普通连续 action 模型的输入输出约定：

| 配置 | 分支 | 输入 transform | 输出 transform |
|---|---|---|---|
| `Pi0Config(pi05=False)` | [PI0 分支](C:/QClaw/FASTER_hy/src/openpi/training/config.py:121) | default prompt, resize, `TokenizePrompt`, pad | 空 |
| `Pi0Config(pi05=True)` | [PI05 分支](C:/QClaw/FASTER_hy/src/openpi/training/config.py:140) | default prompt, resize, `TokenizePrompt(discrete_state_input=...)`, pad | 空 |
| `Pi0FasterConfig(pi05=True)` | [PI05 分支](C:/QClaw/FASTER_hy/src/openpi/training/config.py:140) | 和 PI05 一样，但 `PadStatesAndActions` 也服务于 `action_prefix` | 空 |
| `Pi0FASTConfig` | [PI0_FAST 分支](C:/QClaw/FASTER_hy/src/openpi/training/config.py:166) | `TokenizeFASTInputs` | [ExtractFASTActions](C:/QClaw/FASTER_hy/src/openpi/transforms.py:294) |

本仓库中直接给出的 FASTER 配置包括：

| 配置名 | 代码位置 | 模型 | 说明 |
|---|---|---|---|
| `pi05_faster_libero` | [config.py](C:/QClaw/FASTER_hy/src/openpi/training/config.py:870) | `Pi0FasterConfig(pi05=True, action_horizon=10, max_delay=0, mix_prob=0.5, alpha=0.6, u0=0.9)` | LIBERO FASTER 配置 |
| `pi05_faster_calvin` | [config.py](C:/QClaw/FASTER_hy/src/openpi/training/config.py:915) | `Pi0FasterConfig(pi05=True, action_horizon=10, max_delay=0, ...)` | CALVIN FASTER 配置 |
| `pi05_rtc_agilex` | [config.py](C:/QClaw/FASTER_hy/src/openpi/training/config.py:1120) | `Pi0FasterConfig(pi05=True, max_delay=10, mix_prob=0.0)` | prefix/RTC 风格，不混 HAS |
| `pi05_faster_agilex` | [config.py](C:/QClaw/FASTER_hy/src/openpi/training/config.py:1136) | `Pi0FasterConfig(pi05=True, max_delay=10, mix_prob=0.5, alpha=0.6, u0=0.9)` | Agilex 上启用 prefix + HAS |

## 4. 输入链路对比：图像、文本、本体感知

### 4.1 环境 adapter 输出什么

环境 adapter 的职责是把每个数据集/机器人自己的字段，改成 openpi 统一字段：

```text
state
image: dict[camera_name, HWC uint8 image]
image_mask: dict[camera_name, bool]
prompt
actions        # 仅训练样本有
delay/action_prefix  # FASTER prefix mode 可选
```

| 环境 | 输入函数 | 主要输入 | 输出 |
|---|---|---|---|
| LIBERO | [LiberoInputs.__call__](C:/QClaw/FASTER_hy/src/openpi/policies/libero_policy.py:40) | `observation/state`, `observation/image`, `observation/wrist_image`, `prompt` | `state`, 三个 camera slot，mask，optional `actions/prompt` |
| CALVIN | [CalvinInputs.__call__](C:/QClaw/FASTER_hy/src/openpi/policies/calvin_policy.py:43) | `state_ee_pos`, `state_ee_rot`, `state_gripper`, `image`, `wrist_image` | 拼接后的 7-D state，camera slots，optional actions/prompt |
| DROID | [DroidInputs.__call__](C:/QClaw/FASTER_hy/src/openpi/policies/droid_policy.py:35) | joint position, gripper, exterior image, wrist image | 8-D state，camera slots，optional actions/prompt |
| Agilex | [AgilexInputs.__call__](C:/QClaw/FASTER_hy/src/openpi/policies/agilex_policy.py:38) | `state`, `images`, `prompt`, optional `delay/action_prefix` | 14-D state，camera slots，optional `actions/prompt/delay/action_prefix` |

和 PI0 相比，FASTER 在 adapter 层最大的新增点是：`AgilexInputs` 会透传 [delay](C:/QClaw/FASTER_hy/src/openpi/policies/agilex_policy.py:79) 和 [action_prefix](C:/QClaw/FASTER_hy/src/openpi/policies/agilex_policy.py:82)。普通 PI0 没有这个条件输入。

### 4.2 图像输入链路

图像从环境到模型大致走：

```text
raw image
-> *_Inputs: 转成 HWC uint8，并映射到 base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb
-> ResizeImages: resize_with_pad 到 224x224
-> Observation.from_dict: uint8 转成 [-1, 1] float32
-> preprocess_observation: 检查 camera key，必要时 resize/augment/mask
-> embed_prefix: SigLIP/PaliGemma image tower 变成 image tokens
```

| 模块 | 函数 | 输入 | 输出 |
|---|---|---|---|
| adapter 图像解析 | [Libero `_parse_image`](C:/QClaw/FASTER_hy/src/openpi/policies/libero_policy.py:20), [Droid `_parse_image`](C:/QClaw/FASTER_hy/src/openpi/policies/droid_policy.py:21), [Agilex `_decode_agilex`](C:/QClaw/FASTER_hy/src/openpi/policies/agilex_policy.py:97) | float/uint8, CHW/HWC | HWC uint8 |
| resize | [ResizeImages.__call__](C:/QClaw/FASTER_hy/src/openpi/transforms.py:184) | `data["image"]` | 224x224 images |
| 结构化 | [Observation.from_dict](C:/QClaw/FASTER_hy/src/openpi/models/model.py:109) | `image`, `image_mask` | `Observation.images`, `Observation.image_masks` |
| 预处理 | [preprocess_observation](C:/QClaw/FASTER_hy/src/openpi/models/model.py:144) | `Observation` | 检查/补齐后的 `Observation` |
| prefix embedding | [Pi0.embed_prefix](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:105), [Pi0Faster.embed_prefix](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:113) | images, masks | `prefix_tokens`, `prefix_mask`, `prefix_ar_mask` |

PI0 和 Pi0Faster 在图像输入上基本一致。FASTER 没有发明新的视觉编码器；它复用相同的 prefix KV cache，只是在 streaming init 阶段把 prefix 的 KV cache 预先算出来，见 [sample_actions_streaming_init](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:390)。

### 4.3 文本 prompt 输入链路

文本从环境到模型大致走：

```text
prompt string
-> InjectDefaultPrompt: 缺 prompt 时补默认任务文本
-> TokenizePrompt: PaligemmaTokenizer
-> tokenized_prompt / tokenized_prompt_mask
-> Observation.from_dict
-> embed_prefix: LLM embedding 变成 language tokens
```

| 模块 | 函数 | 输入 | 输出 |
|---|---|---|---|
| 默认 prompt | [InjectDefaultPrompt.__call__](C:/QClaw/FASTER_hy/src/openpi/transforms.py:102) | data dict | 若缺失则增加 `prompt` |
| prompt tokenize | [TokenizePrompt.__call__](C:/QClaw/FASTER_hy/src/openpi/transforms.py:249) | `prompt`, optional `state` | `tokenized_prompt`, `tokenized_prompt_mask` |
| prompt embedding | [Pi0.embed_prefix](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:127), [Pi0Faster.embed_prefix](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:133) | token ids + mask | language token embeddings |

PI0.5 / Pi0Faster 常走 `discrete_state_input`，即 `TokenizePrompt(..., discrete_state_input=model_config.discrete_state_input)`，代码在 [ModelTransformFactory PI05 分支](C:/QClaw/FASTER_hy/src/openpi/training/config.py:156)。这意味着本体 state 可以作为离散 token 融入 prompt/token 序列。Pi0Faster 复用这条路径。

### 4.4 本体感知 state 输入链路

本体感知链路：

```text
robot/env state
-> *_Inputs: 拼接成统一 state 向量
-> Normalize: 使用 norm stats 归一化
-> PadStatesAndActions: pad 到 model.action_dim
-> Observation.from_dict: 写入 Observation.state
-> PI0/FASTER: 根据 pi05 与否进入 prefix/token 或 suffix state token 逻辑
```

| 模块 | 函数 | 输入 | 输出 |
|---|---|---|---|
| state 拼接 | [CalvinInputs.__call__](C:/QClaw/FASTER_hy/src/openpi/policies/calvin_policy.py:47), [DroidInputs.__call__](C:/QClaw/FASTER_hy/src/openpi/policies/droid_policy.py:36), [AgilexInputs.__call__](C:/QClaw/FASTER_hy/src/openpi/policies/agilex_policy.py:66) | 环境原始 state 字段 | `state` |
| 归一化 | [Normalize.__call__](C:/QClaw/FASTER_hy/src/openpi/transforms.py:122) | `state`, `actions`, optional `action_prefix` | normalized fields |
| padding | [PadStatesAndActions.__call__](C:/QClaw/FASTER_hy/src/openpi/transforms.py:327) | `state/actions/action_prefix` | pad 到模型 `action_dim/action_horizon` |
| Observation | [Observation.from_dict](C:/QClaw/FASTER_hy/src/openpi/models/model.py:121) | dict fields | `Observation.state` |

Pi0Faster 对 state 本身没有改变；它新增的是 `action_prefix` 也必须和 `actions` 使用同一套数值约定。相关处理在：

| FASTER prefix 处理 | 代码 | 作用 |
|---|---|---|
| prefix 复用 action norm stats | [Normalize.__call__](C:/QClaw/FASTER_hy/src/openpi/transforms.py:126) | `action_prefix` 使用 `actions` 的归一化统计 |
| prefix 参与 delta 转换 | [DeltaActions.__call__](C:/QClaw/FASTER_hy/src/openpi/transforms.py:222) | 保证 prefix 与待预测 actions 在同一坐标系 |
| prefix pad 到固定 shape | [PadStatesAndActions.__call__](C:/QClaw/FASTER_hy/src/openpi/transforms.py:338) | 从短 prefix 补成 `[action_horizon, action_dim]` |
| 输出时移除 prefix stats | [Unnormalize.__call__](C:/QClaw/FASTER_hy/src/openpi/transforms.py:156) | prefix 只用于输入，不应要求输出也包含它 |

## 5. 引导信息生成链路：prefix、suffix、time、action_prefix

这里的“引导信息”不是额外生成一段语言提示，而是模型生成动作时用到的条件信息：

```text
视觉 tokens + 文本 tokens + state 条件 + noisy action tokens + timestep 条件 + optional action_prefix
```

### 5.1 PI0 的引导信息

PI0 在推理时先算 observation prefix：

```text
Observation.images + tokenized_prompt
-> Pi0.embed_prefix
-> prefix_tokens / prefix_mask / prefix_ar_mask
-> PaliGemma.llm prefill
-> kv_cache
```

然后每个 denoising step 构造 suffix：

```text
x_t + scalar time
-> Pi0.embed_suffix
-> suffix action tokens + adarms_cond
-> PaliGemma action expert
-> action_out_proj
-> v_t
```

| 函数 | 输入 | 输出 | 说明 |
|---|---|---|---|
| [Pi0.embed_prefix](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:105) | `Observation` | `prefix_tokens`, `prefix_mask`, `prefix_ar_mask` | 图像/文本条件 |
| [Pi0.embed_suffix](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:139) | `obs`, `noisy_actions`, `timestep: [b]` | `suffix_tokens`, masks, `adarms_cond` | 所有 action 位置共享同一个 timestep |
| [Pi0.sample_actions step](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:239) | `x_t`, `time` | `x_t + dt * v_t` | 每步统一更新整个 chunk |

PI0 的关键点：`timestep` 是 `[b]`，也就是同一个 batch 样本中的所有 horizon action 共享一个 denoising 时间。

### 5.2 Pi0Faster 的引导信息

Pi0Faster 的 prefix 仍然是 observation prefix，但 suffix 的 timestep 变成 `[b, action_horizon]`：

| 函数 | 输入 | 输出 | 与 PI0 的差异 |
|---|---|---|---|
| [Pi0Faster.embed_prefix](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:113) | `Observation` | `prefix_tokens`, `prefix_mask`, `prefix_ar_mask` | 与 PI0 类似，可缓存为 KV cache |
| [Pi0Faster.embed_suffix](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:143) | `obs`, `noisy_actions`, `timestep: [b, ah]` | `suffix_tokens`, masks, `adarms_cond: [b, ah, emb]` | 每个 action horizon 位置有自己的 timestep |
| [Pi0Faster.compute_HAS](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:231) | `time`, `delay`, `alpha`, `u0` | `time_schedule: [b, ah]` 或 `[steps, b, ah]` | 近端 action 更早接近 `0` |
| [prefix_action_mask](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:287) | `delay` | bool mask `[b, ah]` | 标记 horizon 前面哪些 action 是已知 prefix |

Pi0Faster 的引导逻辑可以记成：

```text
observation prefix: 说明“现在看到什么、任务是什么”
action_prefix: 说明“前几个动作已经确定/已经执行”
HAS timestep: 说明“每个未来动作当前降噪到哪里了”
noisy action x_t: 说明“当前动作草稿是什么”
```

## 6. 动作生成链路对比

### 6.1 PI0：连续 action flow，一次性返回完整 chunk

训练：

```text
actions + noise + time
-> x_t = time * noise + (1 - time) * actions
-> u_t = noise - actions
-> embed_prefix + embed_suffix
-> action_out_proj 得到 v_t
-> loss = mean((v_t - u_t)^2)
```

代码入口：[Pi0.compute_loss](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:189)。

推理：

```text
noise x_t
-> for each step:
   embed_suffix(observation, x_t, scalar time)
   PaliGemma + action expert
   v_t = action_out_proj(...)
   x_t = x_t + dt * v_t
-> return x_0
```

关键返回：[Pi0.sample_actions](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:278)。

| 阶段 | 输入 | 输出 |
|---|---|---|
| 初始 | rng, `Observation`, optional `noise` | `noise: [b, ah, ad]` |
| prefix prefill | `prefix_tokens` | `kv_cache` |
| denoising step | `x_t`, scalar `time` | `v_t`, next `x_t` |
| final | final `x_t` | `x_0: [b, action_horizon, action_dim]` |

### 6.2 Pi0Faster：连续 action flow + delay/action_prefix + HAS

Pi0Faster 训练仍是 flow matching，但多了三件事：

1. 采样 `delay`，构造 [prefix_action_mask](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:200)。
2. 用 [compute_HAS](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:205) 得到 horizon-aware time。
3. 用 [postfix_action_mask](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:227) 只对非 prefix 的动作算 loss。

推理时 `sample_actions()` 支持两种 schedule：

| `infer_time_schedule` | 代码 | 行为 |
|---|---|---|
| `"const"` | [const 分支](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:348) | 类似 PI0，所有非 prefix action 共享统一 step |
| `"HAS"` | [HAS 分支](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:350) | `t_schedule: [num_steps+1, b, ah]`，每个 horizon 单独更新 |

Pi0Faster 非 streaming 仍然最后返回完整 chunk：[return x_0](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:363)。

| 阶段 | 输入 | 输出 | 与 PI0 的差异 |
|---|---|---|---|
| 初始 | rng, `Observation`, optional `noise/delay/action_prefix` | `noise`, `delay`, padded `action_prefix` | 支持 prefix mode |
| prefix prefill | `prefix_tokens` | `kv_cache` | 同 PI0，但后续 HAS/streaming 复用 |
| mask 构造 | `delay` | `prefix_action_mask: [b, ah]` | 前 `delay` 个 action 被视为已知 |
| const step | `x_t`, scalar `time` | `x_next` | 可用 `action_prefix` 覆盖前缀 |
| HAS step | `x_t`, `t_curr: [b, ah]`, `dt_curr: [b, ah]` | `x_next` | 每个 horizon 的时间不同 |
| final | final `x_t` | continuous actions | 不需要 FAST 解码 |

### 6.3 Streaming：init / step / partial ready

streaming 的核心是把一次完整 `sample_actions()` 拆成：

```text
sample_actions_streaming_init
-> Python host loop:
   sample_actions_streaming_step
   check newly_ready
   output_transform newly_ready actions
   callback / websocket partial send
-> final output
```

| 函数 | 输入 | 输出 |
|---|---|---|
| [sample_actions_streaming_init](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:365) | rng, `Observation`, `num_steps`, optional `delay/action_prefix`, `alpha/u0` | `noise`, `already_output`, `t_starts`, `dt_schedule`, `is_ready_after_step`, `kv_cache`, `prefix_mask`, `prefix_action_mask`, `action_prefix`, `observation` |
| [sample_actions_streaming_step](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:419) | current `x_t`, readiness state, current HAS step, cache, masks | `x_next`, `already_output_next`, `newly_ready` |
| [Policy.infer_streaming loop](C:/QClaw/FASTER_hy/src/openpi/policies/policy.py:192) | `num_steps` 次 step 输出 | 对 newly ready 的 actions 做 output transform 后 callback |

`newly_ready` 的语义在 [sample_actions_streaming_step](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:452)：

```text
newly_ready = step_ready & ~already_output
```

它表示“本 step 第一次 ready、应该发出去的 action horizon 位置”。这正是 FASTER 相比普通 PI0 的实时性核心。

## 7. 输出链路：模型输出如何变成可执行 action

`Policy.infer()` 对所有模型都用同一个字段名 `"actions"`，但这个字段的语义取决于模型：

| 模型 | `sample_actions()` 返回 | 是否需要 model output transform |
|---|---|---|
| PI0 | normalized continuous actions | 不需要 |
| Pi0Faster | normalized continuous actions | 不需要 |
| PI0_FAST | action tokens | 需要 [ExtractFASTActions](C:/QClaw/FASTER_hy/src/openpi/transforms.py:294) |

统一输出链路在 [create_trained_policy](C:/QClaw/FASTER_hy/src/openpi/policies/policy_config.py:84)：

```text
model_transforms.outputs
-> Unnormalize
-> data_transforms.outputs
-> repack_transforms.outputs
```

每个阶段输出：

| 阶段 | 输入 | 输出 |
|---|---|---|
| `Policy.infer` 暂存 | model return | `{"state": inputs["state"], "actions": actions}`，见 [policy.py](C:/QClaw/FASTER_hy/src/openpi/policies/policy.py:112) |
| `model_transforms.outputs` | 模型原始输出 | 对 PI0/Pi0Faster 通常为空；对 PI0_FAST 把 token 解码成连续 action |
| [Unnormalize](C:/QClaw/FASTER_hy/src/openpi/transforms.py:147) | normalized continuous actions | 真实数据尺度上的 continuous actions |
| 环境 Outputs adapter | continuous actions | 裁剪/映射后的可执行动作 |

环境输出 adapter：

| 环境 | 函数 | 输出动作维度 |
|---|---|---|
| LIBERO | [LiberoOutputs.__call__](C:/QClaw/FASTER_hy/src/openpi/policies/libero_policy.py:76) | 前 7 维 |
| CALVIN | [CalvinOutputs.__call__](C:/QClaw/FASTER_hy/src/openpi/policies/calvin_policy.py:88) | 前 7 维 |
| DROID | [DroidOutputs.__call__](C:/QClaw/FASTER_hy/src/openpi/policies/droid_policy.py:81) | 前 8 维 |
| Agilex | [AgilexOutputs.__call__](C:/QClaw/FASTER_hy/src/openpi/policies/agilex_policy.py:88) | 前 14 维 |

streaming partial actions 也会经过同一个 `_output_transform`。在 [Policy.infer_streaming](C:/QClaw/FASTER_hy/src/openpi/policies/policy.py:218) 中，`ready_actions_np` 被包装成 `{"state", "actions"}` 后调用 `_output_transform`，所以 callback 收到的已经是环境尺度和环境维度的 actions。

## 8. Server / Client 部署链路对比

普通 server：

```text
client sends obs
-> WebsocketPolicyServer._handler
-> policy.infer(obs)
-> one complete response
```

代码：[WebsocketPolicyServer._handler](C:/QClaw/FASTER_hy/src/openpi/serving/websocket_policy_server.py:48)。

streaming server：

```text
client sends obs
-> StreamingWebsocketPolicyServer._handler
-> background thread runs policy.infer_streaming
-> on_actions_ready puts partial messages into asyncio queue
-> server sends zero or more {"type": "partial", "actions": ...}
-> server sends {"type": "final", ...}
```

代码：[StreamingWebsocketPolicyServer](C:/QClaw/FASTER_hy/src/openpi/serving/websocket_policy_server.py:85)。

| 环节 | 普通 PI0 server | FASTER streaming server |
|---|---|---|
| server 创建 | [serve_policy.py 普通分支](C:/QClaw/FASTER_hy/scripts/serve_policy.py:137) | [serve_policy.py streaming 分支](C:/QClaw/FASTER_hy/scripts/serve_policy.py:129) |
| policy 方法 | `policy.infer` | `policy.infer_streaming` |
| WebSocket 响应 | 一次完整 action dict | 多个 partial + 一个 final |
| client 方法 | [WebsocketClientPolicy.infer](C:/QClaw/FASTER_hy/packages/openpi-client/src/openpi_client/websocket_client_policy.py:46) | [WebsocketClientPolicy.infer_streaming](C:/QClaw/FASTER_hy/packages/openpi-client/src/openpi_client/websocket_client_policy.py:55) |
| 默认 fallback | 无 | [BasePolicy.infer_streaming](C:/QClaw/FASTER_hy/packages/openpi-client/src/openpi_client/base_policy.py:10) 默认退化到 `infer` |

注意：只有 `Pi0Faster` 实现了 `sample_actions_streaming_init/step`。`Policy.__init__` 通过 [hasattr 检查](C:/QClaw/FASTER_hy/src/openpi/policies/policy.py:68) 判断模型是否支持 streaming。普通 PI0/PI0.5 不应该直接开 `--streaming`。

## 9. PI0 和 FASTER 的一页总表

| 模块 | PI0 / PI0.5 | FASTER / Pi0Faster |
|---|---|---|
| 配置类 | [Pi0Config](C:/QClaw/FASTER_hy/src/openpi/models/pi0_config.py:20) | [Pi0FasterConfig](C:/QClaw/FASTER_hy/src/openpi/models/pi0_config.py:106)，新增 `max_delay/mix_prob/alpha/u0` |
| 模型类 | [Pi0](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:66) | [Pi0Faster](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:67) |
| 输入 transform | `TokenizePrompt`, `PadStatesAndActions` | 复用 PI0/PI05 transform，但额外照顾 `action_prefix` |
| 图像输入 | image tower -> prefix tokens | 相同 |
| 文本输入 | prompt tokenizer -> language tokens | 相同 |
| state 输入 | continuous state 或 PI05 discrete state token | 相同 |
| 时间条件 | scalar `time: [b]` | horizon-aware `time: [b, ah]` |
| action 前缀 | 无 | `delay/action_prefix/prefix_action_mask` |
| 训练 loss | 所有 horizon 位置参与 flow loss | prefix 位置不算 loss，postfix 位置参与 |
| 推理 | 完整 denoising 后返回 chunk | 可 const/HAS；streaming 可提前 partial 输出 |
| 模型输出 | continuous `x_0` | continuous `x_0` |
| 部署 | one request -> one full response | partial actions + final response |

## 10. 最短阅读路线

如果只想快速抓住 FASTER 相比 PI0 改在哪里，按这个顺序读：

1. [Pi0Config](C:/QClaw/FASTER_hy/src/openpi/models/pi0_config.py:20) 和 [Pi0FasterConfig](C:/QClaw/FASTER_hy/src/openpi/models/pi0_config.py:106)：先看配置差异。
2. [ModelTransformFactory](C:/QClaw/FASTER_hy/src/openpi/training/config.py:104)：确认 Pi0Faster 复用 PI05/PI0 transform，不走 PI0_FAST。
3. [Normalize](C:/QClaw/FASTER_hy/src/openpi/transforms.py:112)、[DeltaActions](C:/QClaw/FASTER_hy/src/openpi/transforms.py:203)、[PadStatesAndActions](C:/QClaw/FASTER_hy/src/openpi/transforms.py:327)：看 `action_prefix` 如何被归一化、delta 化、padding。
4. [Pi0.sample_actions](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:217)：先理解原始连续 action flow。
5. [Pi0Faster.compute_HAS](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:231)、[Pi0Faster.sample_actions](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:256)：看 HAS 和 prefix 如何改变 denoising。
6. [Pi0Faster.sample_actions_streaming_init](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:365)、[sample_actions_streaming_step](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:419)：看 streaming 如何产生 `newly_ready`。
7. [Policy.infer_streaming](C:/QClaw/FASTER_hy/src/openpi/policies/policy.py:127)、[StreamingWebsocketPolicyServer](C:/QClaw/FASTER_hy/src/openpi/serving/websocket_policy_server.py:85)、[WebsocketClientPolicy.infer_streaming](C:/QClaw/FASTER_hy/packages/openpi-client/src/openpi_client/websocket_client_policy.py:55)：看 partial actions 如何真正发到客户端。

## 11. 最终结论

FASTER 论文和本仓库的 `Pi0Faster` 可以理解为：在 PI0/PI0.5 的连续 action flow 生成器上，增加实时机器人更关心的三件能力。

```text
1. delay/action_prefix：把已经执行或已知的近端动作作为条件。
2. HAS：给不同 horizon 位置分配不同 denoising 时间，近端动作更早 ready。
3. streaming：ready 一个发一个，避免等完整 action chunk。
```

所以它和 PI0 的关系不是“替代”，而是“继承 PI0 的 VLA + flow action chunk 生成范式，并把执行时延和首次动作输出时间作为一等问题来优化”。
