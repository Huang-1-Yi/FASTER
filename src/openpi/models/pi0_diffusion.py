import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """改自 big_vision。

    token 可以 attend 到有效输入 token；这些 token 的累计 mask_ar 必须小于或等于当前 token。
    因此 `mask_ar` bool[?B, N] 可以表达多种 attention 结构，例如：

      [[1 1 1 1 1 1]]: 纯 causal attention。

      [[0 0 0 1 1 1]]: prefix-lm attention。前 3 个 token 互相可见，
          后 3 个 token 使用 causal attention。第一个值也可以是 1，行为不变。

      [[1 0 1 0 1 0 0 1 0 0]]: 4 个 block 之间是 causal attention；
          同一个 block 内 token 互相可见，并且可以 attend 到所有之前的 block。

    Args:
      input_mask: bool[B, N]，true 表示有效输入，false 表示 padding。
      mask_ar: bool[?B, N]，true 表示之前的 token 不能依赖该位置；
        false 表示该位置和前一个 token 共享相同 attention mask。
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """为标量位置计算 sine-cosine positional embedding 向量。"""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0Diffusion(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0DiffusionConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.num_diffusion_train_timesteps = config.num_diffusion_train_timesteps
        self.diffusion_prediction_type = config.diffusion_prediction_type
        self.diffusion_schedule = config.diffusion_schedule
        self.diffusion_min_alpha_bar = config.diffusion_min_alpha_bar
        self.diffusion_clip_sample = config.diffusion_clip_sample

        if self.diffusion_prediction_type not in ("epsilon", "sample", "v"):
            raise ValueError(f"Unsupported diffusion_prediction_type: {self.diffusion_prediction_type}")
        if self.diffusion_schedule not in ("cosine", "linear"):
            raise ValueError(f"Unsupported diffusion_schedule: {self.diffusion_schedule}")
        if self.num_diffusion_train_timesteps <= 0:
            raise ValueError("num_diffusion_train_timesteps must be positive")
        if not 0.0 < self.diffusion_min_alpha_bar <= 1.0:
            raise ValueError("diffusion_min_alpha_bar must be in (0, 1]")

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: 后续可把 gemma 重写成 NNX；当前先通过 bridge 复用已有实现。
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # 该属性会由 model.train() 和 model.eval() 自动切换。
        self.deterministic = True

    def _alpha_bar(self, time: jax.Array) -> jax.Array:
        """连续 alpha_bar(t)；t=0 是干净 action，t=1 接近纯噪声。"""
        time = jnp.clip(time, 0.0, 1.0)
        if self.diffusion_schedule == "cosine":
            s = 0.008
            f_t = jnp.cos((time + s) / (1.0 + s) * jnp.pi / 2.0) ** 2
            f_0 = jnp.cos(s / (1.0 + s) * jnp.pi / 2.0) ** 2
            alpha_bar = f_t / f_0
        elif self.diffusion_schedule == "linear":
            alpha_bar = 1.0 - time * (1.0 - self.diffusion_min_alpha_bar)
        else:
            raise ValueError(f"Unsupported diffusion_schedule: {self.diffusion_schedule}")
        return jnp.clip(alpha_bar, self.diffusion_min_alpha_bar, 1.0)

    def _expand_to_actions(self, value: jax.Array, actions: jax.Array) -> jax.Array:
        while value.ndim < actions.ndim:
            value = value[..., None]
        return value

    def _diffusion_target(self, actions: jax.Array, noise: jax.Array, time: jax.Array) -> jax.Array:
        alpha_bar = self._expand_to_actions(self._alpha_bar(time), actions)
        sqrt_alpha_bar = jnp.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = jnp.sqrt(1.0 - alpha_bar)

        if self.diffusion_prediction_type == "epsilon":
            return noise
        if self.diffusion_prediction_type == "sample":
            return actions
        if self.diffusion_prediction_type == "v":
            return sqrt_alpha_bar * noise - sqrt_one_minus_alpha_bar * actions
        raise ValueError(f"Unsupported diffusion_prediction_type: {self.diffusion_prediction_type}")

    def _ddim_step(self, x_t: jax.Array, model_output: jax.Array, time: jax.Array, next_time: jax.Array) -> jax.Array:
        alpha_bar = self._expand_to_actions(self._alpha_bar(time), x_t)
        next_alpha_bar = self._expand_to_actions(self._alpha_bar(next_time), x_t)
        sqrt_alpha_bar = jnp.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = jnp.sqrt(1.0 - alpha_bar)

        if self.diffusion_prediction_type == "epsilon":
            eps_pred = model_output
            x0_pred = (x_t - sqrt_one_minus_alpha_bar * eps_pred) / jnp.maximum(sqrt_alpha_bar, 1e-6)
        elif self.diffusion_prediction_type == "sample":
            x0_pred = model_output
            eps_pred = (x_t - sqrt_alpha_bar * x0_pred) / jnp.maximum(sqrt_one_minus_alpha_bar, 1e-6)
        elif self.diffusion_prediction_type == "v":
            x0_pred = sqrt_alpha_bar * x_t - sqrt_one_minus_alpha_bar * model_output
            eps_pred = sqrt_one_minus_alpha_bar * x_t + sqrt_alpha_bar * model_output
        else:
            raise ValueError(f"Unsupported diffusion_prediction_type: {self.diffusion_prediction_type}")

        if self.diffusion_clip_sample:
            x0_pred = jnp.clip(x0_pred, -1.0, 1.0)

        return jnp.sqrt(next_alpha_bar) * x0_pred + jnp.sqrt(1.0 - next_alpha_bar) * eps_pred

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # 编码图像。
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # 图像 token 之间互相可见。
            ar_mask += [False] * image_tokens.shape[1]

        # 加入语言 token。
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # 图像和语言输入之间使用 full attention。
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b a emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # 非 pi05 路径加入单个连续 state token。
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # 图像/语言输入不 attend 到 state 或 action。
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # 用 sine-cosine positional encoding 编码 timestep；输入范围是 [0, 1]。
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP 用于生成 adaRMS 条件。
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb[:, None, :]
        else:
            # 非 pi05 路径用 MLP 融合 timestep 和 action 信息，不使用 adaRMS。
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # 图像/语言/state 输入不 attend 到 action token。
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        # diffusion 训练对应 Diffusion Policy：采样 timestep、加噪，并预测配置指定的 target。
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        timestep = jax.random.randint(time_rng, batch_shape, 0, self.num_diffusion_train_timesteps)
        time = (timestep.astype(jnp.float32) + 1.0) / self.num_diffusion_train_timesteps
        alpha_bar = self._expand_to_actions(self._alpha_bar(time), actions)
        x_t = jnp.sqrt(alpha_bar) * actions + jnp.sqrt(1.0 - alpha_bar) * noise
        target = self._diffusion_target(actions, noise, time)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        pred = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(pred - target), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """PI0 diffusion 完整 chunk 推理。

        从一整块 Gaussian action noise 出发，所有 horizon 位置共享一个标量 diffusion time，
        通过 deterministic DDIM 逐步更新到 t=0。
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` 形状是 (b, suffix_len, suffix_len)，表示 suffix token 之间如何互相 attend。
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` 形状是 (b, suffix_len, prefix_len)，表示 suffix token 如何 attend 到 prefix token。
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `full_attn_mask` 形状是 (b, suffix_len, prefix_len + suffix_len)，表示作为 query 的 suffix token
            # 如何 attend 到生成 key/value 的完整 prefix + suffix 序列。
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` 形状是 (b, suffix_len)，表示 suffix token 的位置。
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            model_output = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            next_time = jnp.maximum(time + dt, 0.0)
            return self._ddim_step(x_t, model_output, time, next_time), next_time

        def cond(carry):
            x_t, time = carry
            # 对浮点误差稍微宽松一些。
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
