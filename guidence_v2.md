**阅读顺序**
1. 先读总览和差异：  
   [README.md](C:/QClaw/FASTER/README.md)、[compare.md](C:/QClaw/FASTER/compare.md)、[guidence.md](C:/QClaw/FASTER/guidence.md)

2. 再读配置中心：  
   [src/openpi/training/config.py](C:/QClaw/FASTER/src/openpi/training/config.py:109)  
   重点看 `ModelTransformFactory`、`LeRobotLiberoDataConfig`、`LeRobotCalvinDataConfig`、`LeRobotDROIDDataConfig`、`RLDSDroidDataConfig`，以及 `pi05_faster_libero`、`pi05_faster_calvin`、`pi05_faster_agilex`、`pi05_droid_finetune`。

3. 然后读环境/机器人 adapter：  
   [src/openpi/policies/libero_policy.py](C:/QClaw/FASTER/src/openpi/policies/libero_policy.py)  
   [src/openpi/policies/calvin_policy.py](C:/QClaw/FASTER/src/openpi/policies/calvin_policy.py)  
   [src/openpi/policies/droid_policy.py](C:/QClaw/FASTER/src/openpi/policies/droid_policy.py)  
   [src/openpi/policies/agilex_policy.py](C:/QClaw/FASTER/src/openpi/policies/agilex_policy.py)  
   你要建立的核心心智模型是：不同机器人/仿真只是在这里把自己的 `state/images/actions/prompt` 映射到 openpi 统一格式。

4. 再读通用 transform：  
   [src/openpi/transforms.py](C:/QClaw/FASTER/src/openpi/transforms.py:115)  
   FASTER 额外关心 `action_prefix`：`Normalize` 复用 `actions` 统计量，`DeltaActions` 支持 prefix，`PadStatesAndActions` 同时 pad action dim 和 horizon。

5. 最后读 FASTER 核心：  
   [src/openpi/models/pi0_config.py](C:/QClaw/FASTER/src/openpi/models/pi0_config.py:113) 的 `Pi0FasterConfig`  
   [src/openpi/models/pi0_faster.py](C:/QClaw/FASTER/src/openpi/models/pi0_faster.py:206) 的 `compute_loss()`、`compute_HAS()`、`sample_actions()`、`sample_actions_streaming_init()`、`sample_actions_streaming_step()`

6. 部署链路按这个顺序读：  
   [src/openpi/policies/policy_config.py](C:/QClaw/FASTER/src/openpi/policies/policy_config.py) → [src/openpi/policies/policy.py](C:/QClaw/FASTER/src/openpi/policies/policy.py:134) → [scripts/serve_policy.py](C:/QClaw/FASTER/scripts/serve_policy.py) → [src/openpi/serving/websocket_policy_server.py](C:/QClaw/FASTER/src/openpi/serving/websocket_policy_server.py:87) → [packages/openpi-client/src/openpi_client/websocket_client_policy.py](C:/QClaw/FASTER/packages/openpi-client/src/openpi_client/websocket_client_policy.py:57)

**LIBERO / CALVIN 怎么选**
- 先跑 `LIBERO`。它更适合做第一轮 smoke test：manipulation benchmark、任务套件清楚、client 代码短。入口是 [examples/libero/README.md](C:/QClaw/FASTER/examples/libero/README.md) 和 [examples/libero/main.py](C:/QClaw/FASTER/examples/libero/main.py)。默认 `replan_steps=5`，当前 eval 走普通 `client.infer()`。
- 再跑 `CALVIN`。它更适合验证长任务链和语言连续指令鲁棒性。入口是 [examples/calvin/README.md](C:/QClaw/FASTER/examples/calvin/README.md) 和 [examples/calvin/main.py](C:/QClaw/FASTER/examples/calvin/main.py)。结果看 `avg_seq_len`、`chain_sr`、`task_info`。
- 简单说：`LIBERO` 验证“基本 manipulation 成功率和流程闭环”，`CALVIN` 验证“长序列、多指令、时间一致性”。

**FASTER 核心改动**
- `Pi0FasterConfig`：在 `Pi0Config` 基础上加了 `max_delay`、`mix_prob`、`alpha`、`u0`，但 `model_type` 仍走 `PI05/PI0`，没有单独的 `PI0_FASTER` enum。
- `compute_loss()`：训练时随机采样 `delay`，用 `prefix_action_mask` 把前缀 action 当作已知干净动作；然后以 `mix_prob` 混合 HAS schedule 和 const schedule。
- `compute_HAS()`：按 action horizon 位置生成不同 timestep，近端动作更早接近 0，远端动作保留更多降噪步。
- `sample_actions()`：普通推理支持 `infer_time_schedule="const"` 或 `"HAS"`，也支持 `delay/action_prefix`。
- `sample_actions_streaming_init()`：预计算 prefix KV cache、HAS 时间表、`is_ready_after_step`。
- `sample_actions_streaming_step()`：每一步返回 `newly_ready`，供上层一旦有动作 ready 就发送。
- `Policy.infer_streaming()`：Python 展开采样循环，用 callback 异步发 partial actions。
- `StreamingWebsocketPolicyServer`：协议是 `partial` 多次发送，最后发一个 `final`。

**源码里两个要标记的点**
- `pi05_faster_libero` 和 `pi05_faster_calvin` 当前配置里 `max_delay=0`；如果后续训练 smoke test 遇到 `jax.random.randint` 相关问题，优先检查这里。`pi05_faster_agilex` 是 `max_delay=10`。
- 只有 `Pi0Faster` 有 streaming init/step。用普通 `pi05_droid` 或 `pi05_libero` 开 `--streaming` 会缺少 streaming 方法；要测 FASTER streaming，配置必须走 `Pi0FasterConfig`。

**分阶段 Checklist**
**阶段 1：读懂代码**
- [ ] 按上面的顺序读 `config.py → *_policy.py → transforms.py → pi0_faster.py → policy/server/client`
- [ ] 手动画出一次数据流：dataset/env obs → adapter → normalize → pad/tokenize → model → unnormalize → env action
- [ ] 确认每个环境的 `state_dim/action_dim/action_horizon/image keys`

**阶段 2：跑通仿真**
- [ ] 先 `LIBERO`：`pi05_faster_libero`，跑 norm stats、10 step 训练、serve、eval
- [ ] 再 `CALVIN`：确认 `CALVIN_ROOT` 和 dataset 路径，再跑 `pi05_faster_calvin`
- [ ] 记录成功率，不急着追高分，先确认 checkpoint、transform、WebSocket、eval 都闭环

**阶段 3：实时性实验**
- [ ] 普通 server：`scripts/serve_policy.py policy:checkpoint ...`
- [ ] HAS server：加 `--use-custom-sample-kwargs --infer-time-schedule=HAS --alpha=0.6 --u0=0.9`
- [ ] streaming server：再加 `--streaming --early-stop-actions=4`
- [ ] 用 [examples/simple_client/main.py](C:/QClaw/FASTER/examples/simple_client/main.py:165) 先测 `client_time_to_first_action_ms`
- [ ] 分开记录：总推理延迟、TTFA、server infer、policy infer、成功率

**阶段 4：准备 Franka 数据契约**
- [ ] 重点读 [src/openpi/policies/droid_policy.py](C:/QClaw/FASTER/src/openpi/policies/droid_policy.py)：输入是外部相机、wrist 相机、7 关节、gripper；输出前 8 维。
- [ ] 重点读 [docs/norm_stats.md](C:/QClaw/FASTER/docs/norm_stats.md)：Franka 有 `droid` 和 `franka` 两套 stats；DROID 是 7 维 joint velocity + 1 gripper，15 Hz；普通 Franka/UR5e 是 20 Hz。
- [ ] 先决定你们 Franka 走哪种契约：DROID 风格 joint velocity，还是 non-DROID/FR3 风格 joint position。
- [ ] 若要 FASTER 化 Franka，后面应基于 `pi05_droid_finetune` 新增一个 `Pi0FasterConfig` 版本，而不是直接拿现有 `pi05_droid` 开 streaming。

**阶段 5：准备 Aubo 数据契约**
- [ ] 先按 [examples/ur5/README.md](C:/QClaw/FASTER/examples/ur5/README.md) 建模：`state=[6 joints, gripper]`，`actions=[6 joints, gripper]`。
- [ ] 如果没有 gripper，就明确是 6 维 action；如果有夹爪，按第 7 维 gripper。
- [ ] 若复用 `ur5e` norm stats，必须尽量匹配：关节角弧度、gripper `[0,1]`、20 Hz、动作语义一致。
- [ ] 未来新增：`src/openpi/policies/aubo_policy.py`、`LeRobotAuboDataConfig`、`pi05_faster_aubo`、Aubo WebSocket client/执行器。

**阶段 6：后续实机部署**
- [ ] 先只做 client/server 网络闭环，不接电机执行。
- [ ] 再做相机、state、action shape 录制和回放。
- [ ] 再接低速、限幅、急停保护下的 open-loop chunk 执行。
- [ ] 最后才打开 streaming：控制 `delay`、`exec_horizon/early_stop_actions`、控制频率和网络抖动。