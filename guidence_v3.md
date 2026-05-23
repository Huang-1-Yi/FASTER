# PI0_FAST 如何区别于 PI0，以及 output 如何变成可执行 action

## 主结论

`PI0` 和 `PI0_FAST` 最大的区别不在 `Policy.infer()` 外壳，而在 **模型内部 action 生成方式** 和 **模型侧 output transform**。

- `PI0`：模型直接生成连续 action chunk。也就是说 [Pi0.sample_actions](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:217) 的返回值已经是归一化空间里的连续动作。
- `PI0_FAST`：模型先自回归生成 FAST action tokens。也就是说 [Pi0FAST.sample_actions](C:/QClaw/FASTER_hy/src/openpi/models/pi0_fast.py:236) 的返回值虽然被 `Policy` 暂存在 `"actions"` 字段里，但它此时实际是 token 序列，不是机器人能执行的连续动作。

因此，`PI0_FAST` 必须多走一步 [ExtractFASTActions](C:/QClaw/FASTER_hy/src/openpi/transforms.py:310)，把 token 解码成连续 action chunk，然后才能进入 `Unnormalize` 和环境输出 adapter。

## 名字提醒：PI0_FAST 不是 Pi0Faster

你刚才的理解更接近 `PI0_FAST`：

```text
prompt/images/state
-> action chunk 真值被编码成 FAST action tokens
-> 训练模型预测这些 action tokens
-> 推理时生成 output_tokens
-> 解码回连续 actions
```

但要注意，`PI0_FAST` 不是“用 PI0 生成的语言提示作为输入”。它用的是数据集/环境原本给出的 task prompt、images、state；真值来自数据集里的 action chunk。训练时 [FASTTokenizer.tokenize](C:/QClaw/FASTER_hy/src/openpi/models/tokenizer.py:64) 把真值 actions 编码成 FAST tokens，[Pi0FAST.compute_loss](C:/QClaw/FASTER_hy/src/openpi/models/pi0_fast.py:198) 学的是 next-token prediction。

`Pi0Faster` 是另一件事。代码里确实有 [Pi0FasterConfig](C:/QClaw/FASTER_hy/src/openpi/models/pi0_config.py:112) 和 [Pi0Faster](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:67)，但它没有使用 FAST action tokens，也没有走 [ExtractFASTActions](C:/QClaw/FASTER_hy/src/openpi/transforms.py:310)。它保留 `PI0/PI05` 的连续 action denoising 路线，只是在训练和推理调度上加入 `delay/action_prefix`、HAS schedule 和 streaming early action 输出。

## 入口：ModelTransformFactory 如何分流

先看 [ModelTransformFactory](C:/QClaw/FASTER_hy/src/openpi/training/config.py:104)。它根据 `model_config.model_type` 决定模型侧 transform。

### PI0 分支

入口是 [ModelTransformFactory 的 PI0 分支](C:/QClaw/FASTER_hy/src/openpi/training/config.py:121)。

`PI0` 只配置 `inputs`：

1. 注入默认 prompt。
2. resize 图像。
3. 用 [TokenizePrompt](C:/QClaw/FASTER_hy/src/openpi/transforms.py:266) 把 prompt 变成 PaliGemma tokens。
4. pad `state/actions` 到模型维度。

它没有单独配置 `outputs`，因为模型本身已经直接返回连续 actions。

### Pi0Faster 如何分流

`Pi0Faster` 的配置入口是 [Pi0FasterConfig](C:/QClaw/FASTER_hy/src/openpi/models/pi0_config.py:112)，但它没有单独的 `ModelType.PI0_FASTER`。关键代码在 [Pi0FasterConfig.model_type](C:/QClaw/FASTER_hy/src/openpi/models/pi0_config.py:141)：

```text
pi05=True  -> model_type 返回 PI05
pi05=False -> model_type 返回 PI0
```

所以 `Pi0Faster` 在 [ModelTransformFactory](C:/QClaw/FASTER_hy/src/openpi/training/config.py:104) 里不会进入 `PI0_FAST` 分支，而是复用普通连续 action 模型的 transform：

```text
Pi0FasterConfig(pi05=True)
-> ModelType.PI05
-> 进入 PI05 分支
-> TokenizePrompt
-> PadStatesAndActions(action_dim, action_horizon)
-> outputs 为空
```

这点非常重要：`Pi0Faster` 虽然名字里有 `Faster`，但它不是 `PI0_FAST`，也不需要 `ExtractFASTActions`。它的输入还是 prompt/image/state，输出仍是连续 action chunk；只是模型内部的 denoising schedule、prefix 条件和 streaming 机制变了。

代码里的 FASTER 配置基本都走 `pi05=True`，例如 [pi05_faster_libero](C:/QClaw/FASTER_hy/src/openpi/training/config.py:876)、[pi05_faster_calvin](C:/QClaw/FASTER_hy/src/openpi/training/config.py:921)、[pi05_faster_agilex](C:/QClaw/FASTER_hy/src/openpi/training/config.py:1143)，因此它们通常会复用 [ModelTransformFactory 的 PI05 分支](C:/QClaw/FASTER_hy/src/openpi/training/config.py:140)。

### PI0_FAST 分支

入口是 [ModelTransformFactory 的 PI0_FAST 分支](C:/QClaw/FASTER_hy/src/openpi/training/config.py:168)。

`PI0_FAST` 同时配置 `inputs` 和 `outputs`：

1. 输入侧用 [TokenizeFASTInputs](C:/QClaw/FASTER_hy/src/openpi/transforms.py:288)，它调用 [FASTTokenizer.tokenize](C:/QClaw/FASTER_hy/src/openpi/models/tokenizer.py:64)，把 `prompt/state` 组织成 FAST 模型需要的 token 序列。
2. 输出侧用 [ExtractFASTActions](C:/QClaw/FASTER_hy/src/openpi/transforms.py:310)，它调用 [FASTTokenizer.extract_actions](C:/QClaw/FASTER_hy/src/openpi/models/tokenizer.py:119)，把模型生成的 token 序列解码成连续 action chunk。

所以可以把 `PI0_FAST` 理解成：**把 action 生成器从连续扩散/flow 输出，替换成了 FAST token 自回归生成，再通过 tokenizer 解码回连续动作。**

## PI0 的 action 生成链路

模型类入口是 [Pi0](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:66)。

训练时，[Pi0.compute_loss](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:189) 学的是 flow matching / denoising 目标：

```text
真实 actions + noise + time
-> 构造 x_t
-> 模型预测速度 v_t
-> loss 比较 v_t 和目标 u_t
```

推理时，[Pi0.sample_actions](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:217) 从噪声开始迭代：

```text
noise x_t
-> embed_suffix(observation, x_t, time)
-> PaliGemma + action expert
-> action_out_proj 得到 v_t
-> x_t + dt * v_t
-> 循环到 t=0
-> return x_0
```

关键返回在 [Pi0.sample_actions](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:278)：

```python
x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
return x_0
```

这里的 `x_0` 已经是连续 action tensor，形状语义就是：

```text
[batch, action_horizon, action_dim]
```

所以 `PI0` 的返回值可以直接进入公共输出链路：

```text
连续 actions
-> Unnormalize
-> 环境 Outputs adapter
-> 可执行 action
```

## Pi0Faster 的 action 生成链路

模型类入口是 [Pi0Faster](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:67)。

它的生成链路最像 `PI0`，不是 `PI0_FAST`：

```text
noise x_t
-> embed_suffix(observation, x_t, time 或 HAS time schedule)
-> PaliGemma + action expert
-> action_out_proj 得到 v_t
-> x_t + dt * v_t
-> 循环或 scan 到 t=0
-> return x_0
```

也就是说，[Pi0Faster.sample_actions](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:284) 返回的仍然是连续 action chunk，而不是 token。关键返回在 [Pi0Faster.sample_actions](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:385) 的 `const/HAS` 分支之后：

```python
return x_0
```

它和 `PI0` 的差别主要在三处：

1. `delay/action_prefix`：推理时可以传入已经知道或已经执行的前缀动作。代码入口在 [Pi0Faster.sample_actions 参数](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:291)，训练时对应 [prefix_action_mask](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:215)。
2. `HAS`：用 [compute_HAS](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:257) 给不同 horizon 位置分配不同的 denoising 时间，让近端动作更早 ready。
3. streaming：用 [sample_actions_streaming_init](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:404) 预计算 KV cache 和 ready schedule，再用 [sample_actions_streaming_step](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:463) 一步步更新动作并返回 `newly_ready`。

所以 `Pi0Faster` 的完整非 token 输出路径是：

```text
Pi0Faster.sample_actions()
-> x_0 连续 action chunk
-> Policy.infer 暂存为 {"actions": x_0}
-> model_transforms.outputs 为空
-> Unnormalize
-> environment Outputs adapter
-> executable actions
```

streaming 路径则是：

```text
Pi0Faster.sample_actions_streaming_init()
-> 预计算 prefix KV cache / HAS schedule / ready mask
-> Pi0Faster.sample_actions_streaming_step()
-> 每步得到 x_next 和 newly_ready
-> Policy.infer_streaming 对 newly_ready 做 output_transform
-> callback 提前发出 partial actions
-> 最后返回 final action chunk
```

## PI0_FAST 的 action 生成链路

模型类入口是 [Pi0FAST](C:/QClaw/FASTER_hy/src/openpi/models/pi0_fast.py:134)。

训练时，[Pi0FAST.compute_loss](C:/QClaw/FASTER_hy/src/openpi/models/pi0_fast.py:198) 不再学习连续动作上的 denoising velocity，而是学习 token 预测：

```text
TokenizeFASTInputs 把 prompt/state/actions 变成 token 序列
-> 模型预测下一个 token
-> cross entropy loss 只打在 action/postfix tokens 上
```

推理时，[Pi0FAST.sample_actions](C:/QClaw/FASTER_hy/src/openpi/models/pi0_fast.py:236) 做的是语言模型式自回归解码：

```text
prompt/state/image prefix
-> prefill KV cache
-> 从最后一个 logit 开始
-> 每一步 sample/argmax 得到一个 token
-> token 写入 output_tokens
-> token embedding 送回 LLM 继续解码
-> 遇到 EOS 或 max_decoding_steps 停止
-> return output_tokens
```

关键返回在 [Pi0FAST.sample_actions](C:/QClaw/FASTER_hy/src/openpi/models/pi0_fast.py:309)：

```python
_, _, output_tokens, _, _, _ = jax.lax.while_loop(
    cond, step, (rng, last_logit, output_tokens, kv_cache, False, 0)
)
return output_tokens
```

这里的 `output_tokens` 不是连续 action。它只是模型生成的 token 序列，后面还要解码。

## 为什么 sample_actions 可以直接 return output_tokens

关键点：`Policy` 把 `sample_actions()` 的返回值当作 **模型原始 action 表示**，不是马上假定它一定是最终可执行 action。

看 [Policy.infer](C:/QClaw/FASTER_hy/src/openpi/policies/policy.py:80)：

```python
actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
outputs = {"state": inputs["state"], "actions": actions}
...
outputs = self._output_transform(outputs)
```

这里的字段名叫 `"actions"`，是为了统一 `Policy` 接口。它的真实含义是：

```text
模型 sample_actions 返回的原始输出，暂存在 actions 槽位里。
```

对于 `PI0`，这个槽位里已经是连续动作。

对于 `PI0_FAST`，这个槽位里还是 token，所以必须靠 output transform 修正语义。

这个设计让 [Policy.infer](C:/QClaw/FASTER_hy/src/openpi/policies/policy.py:125) 不需要针对 `PI0`、`PI0_FAST` 写两套分支；差异被封装在 `model_transforms.outputs` 里。

## output_transforms 的执行顺序

输出链路在 [create_trained_policy](C:/QClaw/FASTER_hy/src/openpi/policies/policy_config.py:16) 里组装，重点看 [output_transforms](C:/QClaw/FASTER_hy/src/openpi/policies/policy_config.py:84)：

```python
output_transforms=[
    *data_config.model_transforms.outputs,
    transforms.Unnormalize(...),
    *data_config.data_transforms.outputs,
    *repack_transforms.outputs,
]
```

这个顺序非常重要：

1. `model_transforms.outputs`：先把模型原始输出变成标准连续 actions。对 `PI0` 来说这里通常为空；对 `PI0_FAST` 来说这里就是 [ExtractFASTActions](C:/QClaw/FASTER_hy/src/openpi/transforms.py:310)。
2. `Unnormalize`：把归一化空间里的连续 action 还原到数据集/机器人动作尺度。
3. `data_transforms.outputs`：比如 [LiberoOutputs](C:/QClaw/FASTER_hy/src/openpi/policies/libero_policy.py:78)、[CalvinOutputs](C:/QClaw/FASTER_hy/src/openpi/policies/calvin_policy.py:91)、[DroidOutputs](C:/QClaw/FASTER_hy/src/openpi/policies/droid_policy.py:81)、[AgilexOutputs](C:/QClaw/FASTER_hy/src/openpi/policies/agilex_policy.py:98)，负责裁剪或重映射到环境真正消费的 action 维度。

如果 `PI0_FAST` 跳过第一步，`Unnormalize` 会把离散 token 当成连续动作处理，数值和 shape 语义都会错。

## ExtractFASTActions 具体怎么把 output 转成 action

入口是 [ExtractFASTActions](C:/QClaw/FASTER_hy/src/openpi/transforms.py:310)：

```python
tokens = data.pop("actions")
actions = self.tokenizer.extract_actions(
    tokens.astype(np.int32),
    self.action_horizon,
    self.action_dim,
)
return {**data, "actions": actions}
```

它做两件事：

1. 从 `"actions"` 字段取出模型生成的 token。
2. 调用 [FASTTokenizer.extract_actions](C:/QClaw/FASTER_hy/src/openpi/models/tokenizer.py:119)，得到连续 action chunk 后再写回 `"actions"`。

[FASTTokenizer.extract_actions](C:/QClaw/FASTER_hy/src/openpi/models/tokenizer.py:119) 的内部步骤是：

```text
output token ids
-> PaliGemma tokenizer decode 成文本
-> 找到 "Action: " 后、"|" 前的 action token 文本
-> 重新 encode 成 PaliGemma token ids
-> 映射回 FAST action token ids
-> FAST tokenizer decode
-> [action_horizon, action_dim] 连续动作
```

对应关键代码：

```python
decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

if "Action: " not in decoded_tokens:
    return np.zeros((action_horizon, action_dim), dtype=np.float32)

raw_action_tokens = np.array(
    self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
)
action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
return self._fast_tokenizer.decode(
    [action_tokens.tolist()],
    time_horizon=action_horizon,
    action_dim=action_dim,
)[0]
```

所以 `PI0_FAST` 的完整输出路径是：

```text
Pi0FAST.sample_actions()
-> output_tokens
-> Policy.infer 暂存为 {"actions": output_tokens}
-> ExtractFASTActions
-> FASTTokenizer.extract_actions
-> normalized continuous actions
-> Unnormalize
-> environment Outputs adapter
-> executable actions
```

## Pi0Faster 的位置和作用

`Pi0Faster` 的配置入口是 [Pi0FasterConfig](C:/QClaw/FASTER_hy/src/openpi/models/pi0_config.py:112)。这里有一个容易误解的点：[Pi0FasterConfig.model_type](C:/QClaw/FASTER_hy/src/openpi/models/pi0_config.py:141) 仍然返回 `PI05` 或 `PI0`，没有单独的 `PI0_FASTER` enum。

这意味着它在 [ModelTransformFactory](C:/QClaw/FASTER_hy/src/openpi/training/config.py:104) 中不会走 `PI0_FAST` 分支，而是复用 [PI05 分支](C:/QClaw/FASTER_hy/src/openpi/training/config.py:140) 或 [PI0 分支](C:/QClaw/FASTER_hy/src/openpi/training/config.py:121)。所以它的输入输出仍然是连续 action 路线：

```text
prompt/images/state
-> TokenizePrompt
-> PadStatesAndActions
-> Pi0Faster.sample_actions
-> 连续 action chunk
-> Unnormalize
-> 环境 Outputs adapter
```

代码里的具体配置包括 [pi05_faster_libero](C:/QClaw/FASTER_hy/src/openpi/training/config.py:876)、[pi05_faster_calvin](C:/QClaw/FASTER_hy/src/openpi/training/config.py:921) 和 [pi05_faster_agilex](C:/QClaw/FASTER_hy/src/openpi/training/config.py:1143)。

### Pi0Faster 训练时改了什么

[Pi0Faster.compute_loss](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:206) 仍然是连续 action denoising / flow matching 思路，但加入了两类 FASTER 机制：

1. `delay/action_prefix`：训练时随机采样 `delay`，把 horizon 开头一段 action 当作已知干净 prefix。对应代码是 [delay 采样和 prefix_action_mask](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:215)。
2. HAS schedule：用 [compute_HAS](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:257) 给不同 horizon 位置分配不同 denoising 时间表，让近端动作更早接近 `t=0`。

所以它不是把动作变成 token，而是让同一个连续 action denoising 模型学会：

```text
前面若干 action 已知时，如何补后面的 suffix；
不同 horizon 位置可以用不同降噪进度；
近端 action 可以更早 ready。
```

### Pi0Faster 推理时改了什么

[Pi0Faster.sample_actions](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:284) 的返回值仍然是连续 action chunk。关键返回在 [Pi0Faster.sample_actions](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:385) 的 constant/HAS 分支之后：

```python
return x_0
```

它和 `PI0_FAST` 的差别非常大：

```text
PI0_FAST 返回 output_tokens，需要 ExtractFASTActions 解码。
Pi0Faster 返回 x_0，已经是连续 actions，不需要 ExtractFASTActions。
```

`Pi0Faster` 额外提供 streaming API：

- [sample_actions_streaming_init](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:404)：预计算 prefix KV cache、HAS 时间表、ready mask。
- [sample_actions_streaming_step](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:463)：每一步更新连续 action，并返回 `newly_ready`，告诉上层哪些 horizon 位置可以提前发送。

因此 `Pi0Faster` 的核心目标是低延迟和流式输出，不是 action token 化。

## 和 PI0 的一页对照

| 环节 | PI0 | PI0_FAST |
|---|---|---|
| 配置入口 | [PI0 分支](C:/QClaw/FASTER_hy/src/openpi/training/config.py:121) | [PI0_FAST 分支](C:/QClaw/FASTER_hy/src/openpi/training/config.py:168) |
| 输入 tokenizer | [TokenizePrompt](C:/QClaw/FASTER_hy/src/openpi/transforms.py:266) | [TokenizeFASTInputs](C:/QClaw/FASTER_hy/src/openpi/transforms.py:288) |
| 训练目标 | [Pi0.compute_loss](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:189)：连续动作 denoising / flow matching | [Pi0FAST.compute_loss](C:/QClaw/FASTER_hy/src/openpi/models/pi0_fast.py:198)：next-token cross entropy |
| 推理生成 | [Pi0.sample_actions](C:/QClaw/FASTER_hy/src/openpi/models/pi0.py:217)：从 noise 迭代到 x0 | [Pi0FAST.sample_actions](C:/QClaw/FASTER_hy/src/openpi/models/pi0_fast.py:236)：自回归生成 output tokens |
| sample_actions 返回值 | 连续 action tensor | FAST/PaliGemma token 序列 |
| model outputs | 通常为空 | [ExtractFASTActions](C:/QClaw/FASTER_hy/src/openpi/transforms.py:310) |
| 返回值为何能继续执行 | 已经是连续 actions，直接反归一化 | 先被 output transform 解码成连续 actions，再反归一化 |

## PI0_FAST 和 Pi0Faster 的一页对照

| 问题 | PI0_FAST | Pi0Faster / FASTER |
|---|---|---|
| 是否使用 FAST action tokens | 是 | 否 |
| 是否需要 [ExtractFASTActions](C:/QClaw/FASTER_hy/src/openpi/transforms.py:310) | 需要 | 不需要 |
| `sample_actions()` 返回什么 | [output_tokens](C:/QClaw/FASTER_hy/src/openpi/models/pi0_fast.py:309) | [连续 action chunk x_0](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:402) |
| 训练目标 | [next-token cross entropy](C:/QClaw/FASTER_hy/src/openpi/models/pi0_fast.py:198) | [连续 action denoising / flow loss](C:/QClaw/FASTER_hy/src/openpi/models/pi0_faster.py:206) |
| 核心变化 | 换 action 表示和生成方式 | 保留连续 action 生成，改 schedule/prefix/streaming |
| 适合怎样理解 | action chunk 被 tokenizer 编码成 token，再训练模型预测 token | action chunk 仍是连续值，但训练/推理允许前缀条件和 horizon-aware 早发 |

## 读代码时的心智模型

不要把 `Policy.infer()` 里的 `"actions"` 字段理解成“永远已经是最终动作”。更准确的理解是：

```text
"actions" 是 Policy 输出链路中的统一槽位。
在 output_transforms 之前，它可以是模型原始输出；
在 output_transforms 之后，它才是环境/机器人可消费的 action。
```

这就是为什么 `PI0_FAST` 可以 `return output_tokens`：因为它并不是直接把 tokens 发给环境，而是交给 `model_transforms.outputs` 先解码。这个职责边界让 `Policy` 保持统一，模型差异则留在 `ModelTransformFactory` 和 tokenizer/transform 里处理。
