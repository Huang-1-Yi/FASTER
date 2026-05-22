import concurrent.futures
import logging
import pathlib
import time
from collections.abc import Sequence
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX 模型初始化；sample_actions 被 JIT 后，num_steps/infer_time_schedule/alpha/u0 作为静态参数参与编译缓存。
            self._sample_actions = nnx_utils.module_jit(
                model.sample_actions, static_argnames=("num_steps", "infer_time_schedule", "alpha", "u0")
            )

            # FASTER: 只有 Pi0Faster 暴露 init/step streaming API；普通 pi0/pi05 只能走完整 infer。
            # 显式拆分并 JIT init/step，是为了避免 io_callback 带来的 host-device 同步瓶颈。
            if hasattr(model, "sample_actions_streaming_init"):
                self._sample_actions_streaming_init = nnx_utils.module_jit(
                    model.sample_actions_streaming_init, static_argnames=("num_steps", "alpha", "u0")
                )
                self._sample_actions_streaming_step = nnx_utils.module_jit(model.sample_actions_streaming_step)

            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # transform 可能原地修改输入，因此先复制一层 tree。
        inputs = jax.tree.map(lambda x: x, obs)

        if "action_prefix" in inputs or "delay" in inputs:
            # FASTER: prefix-conditioned inference 要求 delay 和 action_prefix 同时存在；
            # delay 表示前多少个 action 已知干净，action_prefix 承载这些值，原始形状是 (delay, action_dim)。
            assert "action_prefix" in inputs and "delay" in inputs, "action_prefix and delay must be present"
            # FASTER: action_prefix 的第一维长度必须等于 delay；后续 input transform 会 pad 成模型固定 horizon。
            assert (
                inputs["action_prefix"].shape[0] == inputs["delay"]
            ), f"{inputs['action_prefix'].shape[0]} != {inputs['delay']}"
            assert not self._is_pytorch_model, "FASTER is not supported for PyTorch models"
            prefix_mode = True
        else:
            prefix_mode = False

        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # JAX 路径需要增加 batch 维并转成 jax.Array。
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # PyTorch 路径需要增加 batch 维并移动到目标 device。
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # sample_kwargs 汇总推理时传给 sample_actions 的可选控制参数；
        # 这里复制一份，避免单次 infer 注入 noise/prefix 时污染 Policy 的默认配置。
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # noise 若是 (action_horizon, action_dim)，需要补 batch 维
                noise = noise[None, ...]  # 变成 (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        if prefix_mode:
            # FASTER: action_prefix 已经经过 input transform 的 normalize/pad，再传入 sample_actions；
            # delay 同样带 batch 维，和模型侧 prefix_action_mask 对齐。
            sample_kwargs["delay"] = inputs["delay"]
            sample_kwargs["action_prefix"] = inputs["action_prefix"]

        actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        outputs = {"state": inputs["state"], "actions": actions}
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @override
    def infer_streaming(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        on_actions_ready=None,
        early_stop_actions: int | None = None,
    ) -> dict:  # type: ignore[misc]
        """Streaming inference via Python-unrolled loop.

        Massively improves performance by breaking JAX Host-Device sync barriers
        and utilizing asynchronous execution without `io_callback`.
        """
        assert not self._is_pytorch_model, "Streaming inference is only supported for JAX models"
        assert (
            self._sample_actions_streaming_init is not None
        ), "Model does not support streaming (no sample_actions_streaming_init method)"

        inputs = jax.tree.map(lambda x: x, obs)
        if "action_prefix" in inputs or "delay" in inputs:
            # FASTER: streaming 与普通 infer 使用同一套 delay/action_prefix 契约；
            # 输入 transform 之前仍要求短 prefix 长度等于 delay。
            assert "action_prefix" in inputs and "delay" in inputs, "action_prefix and delay must be present"
            assert (
                inputs["action_prefix"].shape[0] == inputs["delay"]
            ), f"{inputs['action_prefix'].shape[0]} != {inputs['delay']}"
            prefix_mode = True
        else:
            prefix_mode = False

        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)

        # FASTER: streaming 固定走模型的 HAS init/step 路径；infer_time_schedule 只对非 streaming sample_actions 生效。
        # 因此这里显式丢弃 infer_time_schedule，避免把 const/HAS 字符串传给 streaming init。
        sample_kwargs = {k: v for k, v in self._sample_kwargs.items() if k != "infer_time_schedule"}
        if noise is not None:
            noise = jnp.asarray(noise)

            if noise.ndim == 2:  # noise 若是 (action_horizon, action_dim)，需要补 batch 维
                noise = noise[None, ...]  # 变成 (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        state_np = np.asarray(inputs["state"][0])

        if prefix_mode:
            sample_kwargs["delay"] = inputs["delay"]
            sample_kwargs["action_prefix"] = inputs["action_prefix"]

        num_steps = sample_kwargs.get("num_steps", 10)

        callback_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        start_time = time.monotonic()
        emitted_action_count = 0

        # FASTER: init 预计算静态 observation、KV cache、HAS schedule 和 partial action readiness mask；
        # 这些结果会被 host loop 逐 step 复用，避免重复编码 prompt/images/state。
        (
            x_t,
            already_output,
            t_starts,
            dt_schedule,
            is_ready_after_step,
            kv_cache,
            prefix_mask,
            prefix_action_mask,
            action_prefix_init,
            observation_preprocessed,
        ) = self._sample_actions_streaming_init(sample_rng, observation, **sample_kwargs)

        # FASTER: host loop 每个 denoising step 检查 newly_ready，并通过 callback 尽早发出 partial actions。
        # 这段故意留在 Python 侧展开，让 server 能在模型 step 之间发送已 ready 的动作。
        for i in range(num_steps):
            x_next, already_output, newly_ready = self._sample_actions_streaming_step(
                x_t,
                already_output,
                t_starts[i],
                dt_schedule[i],
                is_ready_after_step[i],
                kv_cache,
                prefix_mask,
                prefix_action_mask,
                action_prefix_init,
                observation_preprocessed,
            )

            newly_ready_np = np.asarray(newly_ready[0])
            newly_ready_count = int(np.count_nonzero(newly_ready_np))
            selected_count = newly_ready_count

            if early_stop_actions is not None:
                # FASTER: early_stop_actions 用于实时控制，只取前几个 ready actions，避免等待完整 chunk；
                # final 输出仍返回当前 x_t，因此调用方可同时记录完整推理状态。
                remaining = early_stop_actions - emitted_action_count
                selected_count = min(newly_ready_count, remaining)
                emitted_action_count += selected_count

            if on_actions_ready is not None:
                ready_actions_np = np.asarray(x_next[0][newly_ready_np])[:selected_count]

                def process_and_send(actions_data, state_data):
                    temp_output = {"state": state_data, "actions": actions_data}
                    temp_output = self._output_transform(temp_output)
                    # FASTER: callback 收到的是已 unnormalize 且裁剪到环境 action_dim 的 actions。
                    on_actions_ready(temp_output["actions"])

                callback_executor.submit(process_and_send, ready_actions_np, state_np)

            x_t = x_next
            if early_stop_actions is not None and emitted_action_count >= early_stop_actions:
                break

        callback_executor.shutdown(wait=True)
        model_time = time.monotonic() - start_time

        outputs = {"state": inputs["state"], "actions": x_t}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
