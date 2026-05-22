import functools
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
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
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
    """Computes sine-cosine positional embedding vectors for scalar positions."""
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


class Pi0Faster(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0FasterConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
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

        # deterministic 会由 model.train()/model.eval() 自动切换。
        self.deterministic = True
        self.max_delay = config.max_delay

        self.mix_prob = config.mix_prob
        assert 0.0 <= self.mix_prob <= 1.0, "mix_prob must be in [0, 1]"
        self.alpha = config.alpha
        assert 0.0 <= self.alpha <= 1.0, "alpha must be in [0, 1]"
        self.u0 = config.u0
        assert 0.0 <= self.u0 <= 1.0, "u0 must be in [0, 1]"
        print(f"mix_prob: {self.mix_prob}, alpha: {self.alpha}, u0: {self.u0}")

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # FASTER: observation prefix 只包含图像和语言等静态条件，后续可缓存为 KV cache 供多步 denoising 复用。
        # 图像 tokens 先编码，并与语言 tokens 共享 full attention。
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
            # 图像 tokens 之间使用 full attention。
            ar_mask += [False] * image_tokens.shape[1]

        # 再加入语言 prompt tokens。
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # 图像和语言 tokens 之间使用 full attention。
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b ah"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b ah emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # 非 pi05 路径用单个连续 state token；pi05 的 state 已在 prefix 侧作为离散 token 处理。
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # state/action suffix 走 causal 约束，不能被前面的 image/language prefix 反向看到。
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # FASTER: 每个 horizon 位置都有自己的 timestep，因此 HAS 可以让近端 action 更早到 t=0。
        # timestep 用 sine-cosine positional encoding 表示，范围对应 [0, 1]。
        time_emb = jax.vmap(
            functools.partial(
                posemb_sincos, embedding_dim=self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
            )
        )(timestep)
        if self.pi05:
            # pi05 用 time MLP 生成 adaRMS 条件。
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # 非 pi05 路径用 MLP 融合 timestep 与 action 信息，不走 adaRMS。
            # 旧实现曾把 time_emb repeat 成 time_tokens，这里保持每个 horizon 位置独立。
            time_tokens = time_emb
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # FASTER: prefix tokens 不 attend 到 action tokens，避免 observation 条件被 noisy action suffix 污染；
        # action tokens 内部按顺序自回归，以保留 chunk 内的时序依赖。
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng, delay_rng, type_rng = jax.random.split(rng, 5)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        b, ah, ad = actions.shape
        noise = jax.random.normal(noise_rng, actions.shape)

        # FASTER: 为每个 batch 随机采样 delay，模拟 horizon 开头已有干净 action_prefix 的场景。
        # Data contract: delay 的形状是 (b,)，prefix_action_mask 会扩展成 (b, action_horizon)。
        # NOTE: 走随机 delay 路径时 max_delay 必须大于 0，否则 randint 的上下界无效。
        delay = jax.random.randint(delay_rng, (b,), 0, self.max_delay)
        # FASTER: prefix_action_mask 标记哪些 horizon 位置属于已知干净 prefix。
        prefix_action_mask = jnp.arange(ah)[None, :] < delay[:, None]  # 形状: (b, ah)

        time_const = jax.random.beta(time_rng, 1.5, 1, (b, 1)) * 0.999 + 0.001
        time_const = jnp.broadcast_to(time_const, (b, ah))
        time_HAS = self.compute_HAS(time_const, delay, alpha=self.alpha, u0=self.u0)

        # FASTER: 训练时混合 constant schedule 和 HAS，让同一模型同时适配整块推理与 streaming 早发动作；
        # mix_prob 控制每个样本是否使用 HAS，而不是每个 horizon 位置单独切换。
        use_HAS = jax.random.bernoulli(type_rng, self.mix_prob, (b, 1))
        time = jnp.where(use_HAS, time_HAS, time_const)

        # FASTER: prefix 位置的 time 置 0，使其保持 ground truth action 而不参与降噪。
        time = jnp.where(prefix_action_mask, 0.0, time)  # 形状: (b, ah)

        x_t = time[..., None] * noise + (1 - time[..., None]) * actions
        u_t = noise - actions

        # FASTER: 训练时一次性前向 prefix + suffix，让模型同时看到 observation tokens 与 noisy action suffix；
        # 推理阶段才把 prefix KV cache 拆出来复用。
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # FASTER: loss 只计算 postfix action；已知干净 prefix 不应反向推动模型重复学习。
        # prefix_action_mask 是 (b, ah)，会广播到 v_t/u_t 的 action_dim 之外做逐位置筛选。
        loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # 形状: (b, ah)
        postfix_action_mask = jnp.logical_not(prefix_action_mask)  # 形状: (b, ah)
        loss = jnp.sum(loss * postfix_action_mask) / (jnp.sum(postfix_action_mask) + 1e-8)
        return loss

    def compute_HAS(
        self, time: jax.Array, delay: jax.Array | None = None, alpha: float = 1.0, u0: float = 0.9
    ) -> jax.Array:
        """
        Horizon-Aware Schedule
        time: (b, 1) or (a, b, 1)
        delay: (b,)
        return: (b, ah) or (a, b, ah)
        """
        # FASTER: 近端 action 更早推向 t=0，远端 action 保留更多 denoising 步；delay 会平移可预测 suffix 起点。
        # Data contract: time 可为 (b, 1) 或 (steps, b, 1)，返回形状分别为 (b, ah) 或 (steps, b, ah)。
        i = jnp.arange(self.action_horizon)[None, :]  # 形状: (1, ah)
        i_valid = jnp.maximum(i - delay[:, None], 0)  # (b, ah)，prefix 之前的位置被截到 0
        denom = jnp.maximum(self.action_horizon - 1 - delay, 1)[:, None]  # 形状: (b, 1)

        j = i_valid / denom  # 形状: (b, ah)
        u = (1 - j**alpha) * u0  # 形状: (b, ah)

        if time.ndim == 3:
            u = u[None, :, :]

        time_schedule = (time - u) / (1 - u)  # 形状: (b, ah) 或 (a, b, ah)

        time_schedule = jnp.clip(time_schedule, 0.0, 1.0)
        return time_schedule

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        delay: at.Int[at.Array, "b"] | None = None,
        action_prefix: at.Float[at.Array, "b ah ad"] | None = None,
        infer_time_schedule: str = "const",
        alpha: float = 1.0,
        u0: float = 0.9,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # FASTER: 这里采用 diffusion 常见约定，t=1 是 noise，t=0 是目标 action；这与 pi0 论文记法相反。
        # Data contract: delay/action_prefix 缺省时表示没有已知 prefix，action_prefix 仍补成固定 (b, ah, ad)。
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        if delay is None:
            delay = jnp.zeros((batch_size,), dtype=jnp.int32)
            action_prefix = jnp.zeros((batch_size, self.action_horizon, self.action_dim))

        assert action_prefix.shape == (batch_size, self.action_horizon, self.action_dim)

        # FASTER: prefix KV cache 只依赖 prompt/images/state，可在 constant 和 HAS 的所有 denoising 步中复用。
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        prefix_action_mask = jnp.arange(self.action_horizon)[None, :] < delay[:, None]  # 形状: (b, ah)

        def step(carry, _):
            x_t, time = carry
            x_t = jnp.where(prefix_action_mask[..., None], action_prefix, x_t)
            time_ = jnp.broadcast_to(time, batch_size)  # 形状: (b,)
            time_ = jnp.where(prefix_action_mask, 0.0, time_[:, None])  # 形状: (b, ah)

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time_)
            # FASTER: suffix token 通过当前 suffix attention 与预先缓存的 prefix KV cache 一起完成 denoising。
            # full_attn_mask 的最后一维是 prefix_len + suffix_len，表示 suffix query 同时看 prefix KV 和 suffix KV。
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # FASTER: suffix positions 延续 prefix 长度，保证复用 KV cache 时 token 位置连续。
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            x_next = x_t + dt * v_t
            return (x_next, time + dt), None

        def step_adaptive(carry, step_params):
            x_t, _ = carry
            x_t = jnp.where(prefix_action_mask[..., None], action_prefix, x_t)
            t_curr, dt_curr = step_params  # 形状: (b, ah)

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, t_curr)
            # FASTER: HAS 路径中每个 horizon 位置有独立 t_curr/dt_curr，但 attention 结构仍与 constant 路径一致。
            # 这里复用同一套 prefix_attn_mask/full_attn_mask 契约，只改变每个 action token 的时间条件。
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # FASTER: suffix positions 延续 prefix 长度，保证复用 KV cache 时 token 位置连续。
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            x_next = x_t + dt_curr[..., None] * v_t
            return (x_next, None), None

        if infer_time_schedule == "const":
            # FASTER: constant schedule 需要完整 chunk 降噪完成后再返回 actions。
            (x_0, _), _ = jax.lax.scan(step, (noise, 1.0), None, length=num_steps)
        elif infer_time_schedule == "HAS":
            # FASTER: HAS 为每个 horizon 位置分配独立时间表，使近端 action 可先于远端 action ready。
            base_times = jnp.linspace(1.0, 0.0, num_steps + 1)[:, None, None]  # 形状: (num_steps + 1, b, 1)

            t_schedule = self.compute_HAS(base_times, delay, alpha=alpha, u0=u0)  # 形状: (num_steps + 1, b, ah)
            t_schedule = jnp.where(prefix_action_mask[None, :, :], 0.0, t_schedule)

            dt_schedule = t_schedule[1:] - t_schedule[:-1]
            t_starts = t_schedule[:-1]

            (x_0, _), _ = jax.lax.scan(step_adaptive, (noise, None), (t_starts, dt_schedule), length=num_steps)
        else:
            raise ValueError(f"Invalid infer_time_schedule: {infer_time_schedule}")

        return x_0

    def sample_actions_streaming_init(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        delay: at.Int[at.Array, "b"] | None = None,
        action_prefix: at.Float[at.Array, "b ah ad"] | None = None,
        alpha: float = 1.0,
        u0: float = 0.9,
    ):
        """Precomputes kv_cache and time schedules before streaming."""
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]

        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        if delay is None:
            delay = jnp.zeros((batch_size,), dtype=jnp.int32)
            action_prefix = jnp.zeros((batch_size, self.action_horizon, self.action_dim))

        assert action_prefix.shape == (batch_size, self.action_horizon, self.action_dim)

        # FASTER: streaming 将推理拆成 init 和多次 step，host loop 才能在模型调用之间发送 ready actions。
        # init 阶段只计算静态 observation 的 KV cache 和完整 HAS 时间表，不实际发送动作。
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask_init = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask_init, positions=positions)

        prefix_action_mask = jnp.arange(self.action_horizon)[None, :] < delay[:, None]

        time = jnp.linspace(1.0, 0.0, num_steps + 1)[:, None, None]
        t_schedule = self.compute_HAS(time, delay, alpha=alpha, u0=u0)  # 形状: (num_steps + 1, b, ah)
        t_schedule = jnp.where(prefix_action_mask[None, :, :], 0.0, t_schedule)

        dt_schedule = t_schedule[1:] - t_schedule[:-1]
        t_starts = t_schedule[:-1]
        # FASTER: is_ready_after_step 是 streaming readiness 信号；当 HAS time 足够接近 0 时对应 action 可发送。
        # 它的形状是 (num_steps, b, ah)，host loop 会逐 step 取出 newly_ready 的 action。
        is_ready_after_step = t_schedule[1:] < 0.01  # time 接近 0 时认为 action ready

        already_output_init = prefix_action_mask  # 已知干净 prefix 不需要重复发送

        return (
            noise,
            already_output_init,
            t_starts,
            dt_schedule,
            is_ready_after_step,
            kv_cache,
            prefix_mask,
            prefix_action_mask,
            action_prefix,
            observation,
        )

    def sample_actions_streaming_step(
        self,
        x_t: jax.Array,
        already_output: jax.Array,
        t_curr: jax.Array,
        dt_curr: jax.Array,
        step_ready: jax.Array,
        kv_cache,
        prefix_mask: jax.Array,
        prefix_action_mask: jax.Array,
        action_prefix: jax.Array,
        observation: _model.Observation,
    ):
        """Single streaming step, designed to be called asynchronously in a host loop."""
        x_t = jnp.where(prefix_action_mask[..., None], action_prefix, x_t)

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, t_curr)
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        x_next = x_t + dt_curr[..., None] * v_t

        # FASTER: newly_ready 是本 step 唯一应发送的 action 集合；already_output 防止同一 horizon 重复 partial send。
        # 最终 x_next 仍保留完整 horizon，供 final message 返回完整 action chunk。
        newly_ready = step_ready & ~already_output
        already_output_next = already_output | step_ready

        return x_next, already_output_next, newly_ready
