**第一优先级：代码主线**

1. [src/openpi/training/config.py](C:/QClaw/FASTER_hy/src/openpi/training/config.py:104)  
   先看 `ModelTransformFactory`，再看 `LeRobotLiberoDataConfig`、`LeRobotCalvinDataConfig`、`LeRobotDROIDDataConfig`、`RLDSDroidDataConfig`，最后看 `pi05_faster_libero`、`pi05_faster_calvin`、`pi05_faster_agilex`、`pi05_droid_finetune`。

2. 环境/机器人 adapter：  
   [libero_policy.py](C:/QClaw/FASTER_hy/src/openpi/policies/libero_policy.py:30)、[calvin_policy.py](C:/QClaw/FASTER_hy/src/openpi/policies/calvin_policy.py:33)、[droid_policy.py](C:/QClaw/FASTER_hy/src/openpi/policies/droid_policy.py:31)、[agilex_policy.py](C:/QClaw/FASTER_hy/src/openpi/policies/agilex_policy.py:27)  
   重点看 `*Inputs` 和 `*Outputs`，理解每个环境怎么把自己的 `state/images/actions/prompt` 变成 openpi 统一格式。

3. [src/openpi/transforms.py](C:/QClaw/FASTER_hy/src/openpi/transforms.py:115)  
   重点看 `Normalize`、`DeltaActions`、`PadStatesAndActions`、`Unnormalize`，尤其是 `action_prefix` 怎么被归一化、delta 化和 padding。

4. FASTER 核心：  
   [pi0_config.py](C:/QClaw/FASTER_hy/src/openpi/models/pi0_config.py:112) 的 `Pi0FasterConfig`  
   [pi0_faster.py](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:206) 的 `compute_loss()`、`compute_HAS()`、`sample_actions()`、`sample_actions_streaming_init()`、`sample_actions_streaming_step()`。

5. 推理/部署链路：  
   [policy_config.py](C:/QClaw/FASTER_hy/src/openpi/policies/policy_config.py) → [policy.py](C:/QClaw/FASTER_hy/src/openpi/policies/policy.py:25) → [serve_policy.py](C:/QClaw/FASTER_hy/scripts/serve_policy.py) → [websocket_policy_server.py](C:/QClaw/FASTER_hy/src/openpi/serving/websocket_policy_server.py:87) → [websocket_client_policy.py](C:/QClaw/FASTER_hy/packages/openpi-client/src/openpi_client/websocket_client_policy.py:57)

最小阅读顺序可以记成：

`config.py → *_policy.py → transforms.py → pi0_config.py/pi0_faster.py → policy/server/client`

先别急着读 `gemma.py`、`vit.py`、`siglip.py`、`models_pytorch/` 这些底层模型实现。当前阶段最重要的是先画出数据流：  
`dataset/env obs → adapter → normalize/pad → model → unnormalize → env action`。


是的，可以把这三个文件看成 **三类 action 生成路线**：

- [pi0.py](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:66)：`PI0`，连续 action diffusion / flow 生成器
- [pi0_fast.py](C:/QClaw/FASTER_hy/src/openpi/models/pi0_fast.py:134)：`PI0_FAST`，FAST token 自回归 action 生成器
- [pi0_faster.py](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:67)：`Pi0Faster`，连续 action 生成器的 prefix/HAS/streaming 增强版

**共同输入输出骨架**
三者都接收 [Observation](C:/QClaw/FASTER_hy/src/openpi/models/model.py:83)，主要内容是：

```text
images / image_masks
state
tokenized_prompt / tokenized_prompt_mask
可选 token_ar_mask / token_loss_mask
```

三者都被 [Policy.infer](C:/QClaw/FASTER_hy/src/openpi/policies/policy.py:80) 调用。公共流程是：

```text
env obs
-> data transforms
-> Normalize
-> model_transforms.inputs
-> Observation.from_dict
-> model.sample_actions(...)
-> {"state": inputs["state"], "actions": model_return}
-> output_transforms
-> executable actions
```

输出链路在 [policy_config.py](C:/QClaw/FASTER_hy/src/openpi/policies/policy_config.py:84)：

```text
model_transforms.outputs
-> Unnormalize
-> data_transforms.outputs
-> repack_transforms.outputs
```

**1. PI0：连续 action 直接生成**
入口：[Pi0.sample_actions](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:217)

数据流：

```text
images/state/prompt tokens
-> embed_prefix
-> 从 noise 初始化 x_t
-> embed_suffix(observation, x_t, time)
-> PaliGemma + action expert
-> action_out_proj 得到 v_t
-> x_t + dt * v_t
-> return x_0
```

交换内容：

```text
输入：Observation + noise
内部交换：x_t, time, suffix tokens, v_t
输出：连续 actions，shape [batch, action_horizon, action_dim]
```

所以 `PI0` 的 `sample_actions()` 返回值已经是连续 action，只需要：

```text
Unnormalize -> 环境 Outputs adapter
```

**2. PI0_FAST：先生成 action tokens，再解码**
入口：[Pi0FAST.sample_actions](C:/QClaw/FASTER_hy/src/openpi/models/pi0_fast.py:236)

数据流：

```text
images/state/prompt
-> TokenizeFASTInputs
-> embed_inputs
-> prefill KV cache
-> 自回归生成 token
-> return output_tokens
-> ExtractFASTActions
-> FASTTokenizer.extract_actions
-> 连续 actions
```

交换内容：

```text
输入：Observation，含 token_ar_mask / token_loss_mask
内部交换：prefix embeddings, logits, token, output_tokens, kv_cache
模型输出：output_tokens，不是连续 action
transform 输出：连续 actions，shape [action_horizon, action_dim]
```

关键转换点：

- [ExtractFASTActions](C:/QClaw/FASTER_hy/src/openpi/transforms.py:310)
- [FASTTokenizer.extract_actions](C:/QClaw/FASTER_hy/src/openpi/models/tokenizer.py:119)

所以 `PI0_FAST` 的 `"actions"` 字段在 output transform 前其实是 token，之后才变成连续 action。

**3. Pi0Faster：连续 action，但加入 prefix/HAS/streaming**
入口：[Pi0Faster.sample_actions](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:284)

它不是 `PI0_FAST`。它仍然像 `PI0` 一样生成连续 action：

```text
images/state/prompt tokens
-> embed_prefix
-> 从 noise 初始化 x_t
-> 可选 action_prefix 覆盖前 delay 个 action
-> const schedule 或 HAS schedule
-> embed_suffix(observation, x_t, t_curr)
-> PaliGemma + action expert
-> action_out_proj 得到 v_t
-> x_t + dt * v_t
-> return x_0
```

交换内容：

```text
输入：Observation + noise + 可选 delay/action_prefix
内部交换：prefix_action_mask, time schedule, x_t, v_t
输出：连续 actions x_0
```

额外 streaming 交换内容：

```text
sample_actions_streaming_init
-> 返回 noise, already_output, t_starts, dt_schedule, is_ready_after_step, kv_cache, prefix_mask, prefix_action_mask, action_prefix, observation

sample_actions_streaming_step
-> 输入上一步 x_t 和 schedule
-> 输出 x_next, already_output_next, newly_ready
```

对应代码：

- [compute_HAS](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:257)
- [sample_actions_streaming_init](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:404)
- [sample_actions_streaming_step](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:463)

**一张总表**

| 文件 | 类别 | 模型返回值 | 是否 token action | 主要交换内容 |
|---|---|---|---|---|
| `pi0.py` | 连续 action flow 生成 | `x_0` 连续 actions | 否 | `noise/x_t/time/v_t` |
| `pi0_fast.py` | FAST token action 生成 | `output_tokens` | 是 | `logits/token/output_tokens/kv_cache` |
| `pi0_faster.py` | 连续 action + FASTER 调度 | `x_0` 连续 actions | 否 | `delay/action_prefix/HAS/newly_ready` |

最短心智模型：

```text
PI0：直接生成连续动作。
PI0_FAST：生成动作 token，再解码成连续动作。
Pi0Faster：仍生成连续动作，但支持前缀条件、HAS 调度和流式提前输出。
```