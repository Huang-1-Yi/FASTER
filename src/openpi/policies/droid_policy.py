import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_droid_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DroidInputs(transforms.DataTransformFn):
    # Data contract: model_type 决定 DROID camera slots 与 image_mask 约定；自定义 DROID-like 数据通常只改 key 映射。
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            # Data contract: gripper 必须是一维数组，才能和 7 维 joint_position 拼接。
            gripper_pos = gripper_pos[np.newaxis]
        # Data contract: DROID/Franka-style state 是 7 个 joint position 加 1 个 gripper 值。
        state = np.concatenate([data["observation/joint_position"], gripper_pos])

        # Data contract: DROID 图像统一转成 uint8 (H,W,C)，避免 LeRobot 的 float32 CHW 与在线推理不一致。
        # 若数据使用不同相机 key，改这里的读取 key，并保持下方 openpi camera slot 稳定。
        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # Data contract: pi0-FAST 不屏蔽占位图，保持 FAST tokenization 的输入槽位稳定；缺失的 base_1 用零图占位。
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        # Data contract: 这里把 DROID camera 命名映射到 openpi 模型期望的 camera slots；
        # actions 和 prompt 透传给后续 model/data transforms，在线推理通常只传 observation/prompt。
        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Data contract: DROID 执行端只消费前 8 维动作：7 个机器人维度加 1 个 gripper；其余维度是模型 padding。
        return {"actions": np.asarray(data["actions"][:, :8])}
