import dataclasses
import enum
import logging
import socket

import torch
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # 训练配置名，例如 "pi0_aloha_sim"。
    config: str
    # checkpoint 目录，例如 "checkpoints/pi0_aloha_sim/exp/10000"。
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # 要服务的环境；仅在使用默认 policy 时生效。
    env: EnvMode = EnvMode.ALOHA_SIM

    # Data contract: obs 缺少 "prompt" 且模型没有默认 prompt 时，使用这里的 default_prompt。
    default_prompt: str | None = None

    # policy server 监听端口。
    port: int = 8000
    # 调试时记录 policy 行为。
    record: bool = False
    max_cuda_mem_fraction: float = 0.9
    # FASTER: 使用 streaming server，动作在 denoising 过程中逐步发送。
    streaming: bool = False
    # FASTER: streaming 模式下，发出指定数量的 newly ready actions 后提前停止；None 表示跑完整推理。
    # 这个参数只影响 StreamingWebsocketPolicyServer，不改变普通 infer。
    early_stop_actions: int | None = None

    # 指定 policy 加载方式；未提供时使用当前环境的默认 policy。
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # FASTER: 是否把自定义 sample_kwargs 传入推理；关闭时保持 checkpoint 默认行为。
    use_custom_sample_kwargs: bool = False

    # FASTER: 以下 sample kwargs 仅在 use_custom_sample_kwargs=True 时生效；
    # infer_time_schedule 控制普通 sample_actions，streaming server 会固定走 HAS init/step。
    infer_time_schedule: str = "const"
    alpha: float = 0.6
    u0: float = 0.9
    num_steps: int = 10


# 每个环境对应的默认 checkpoint；这里只定义加载入口，具体数据契约由 config.py 的 DataConfig 决定。
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            # FASTER: 自定义 sample kwargs 是显式 opt-in，避免默认 checkpoint 的推理行为被 HAS/步数实验意外改变。
            # 关闭时不传这些参数，模型使用自身 sample_actions 的默认值。
            sample_kwargs = (
                {
                    "infer_time_schedule": args.infer_time_schedule,
                    "alpha": args.alpha,
                    "u0": args.u0,
                    "num_steps": args.num_steps,
                }
                if args.use_custom_sample_kwargs
                else {}
            )
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                sample_kwargs=sample_kwargs,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def _maybe_configure_cuda(args: Args) -> None:
    if not torch.cuda.is_available():
        return

    try:
        frac = max(0.0, min(1.0, args.max_cuda_mem_fraction))
        for dev_idx in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(frac, dev_idx)
        logging.info("Set CUDA memory fraction to %.2f for %d device(s)", frac, torch.cuda.device_count())
    except Exception as e:  # pragma: no cover - defensive
        logging.warning("Failed to set CUDA memory fraction: %s", e)


def main(args: Args) -> None:
    _maybe_configure_cuda(args)
    if args.early_stop_actions is not None and not args.streaming:
        raise ValueError("--early-stop-actions can only be used with --streaming")
    if args.early_stop_actions is not None and args.early_stop_actions <= 0:
        raise ValueError("--early-stop-actions must be positive when provided")

    policy = create_policy(args)
    policy_metadata = policy.metadata

    # 调试时包装 recorder。
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    if args.streaming:
        # FASTER: streaming 需要模型实现 sample_actions_streaming_init/step；普通 pi0/pi05 应使用非 streaming server。
        # WebSocket 协议会先发 partial actions，再发 final 完整结果。
        server = websocket_policy_server.StreamingWebsocketPolicyServer(
            policy=policy,
            host="0.0.0.0",
            port=args.port,
            metadata=policy_metadata,
            early_stop_actions=args.early_stop_actions,
        )
    else:
        server = websocket_policy_server.WebsocketPolicyServer(
            policy=policy,
            host="0.0.0.0",
            port=args.port,
            metadata=policy_metadata,
        )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
