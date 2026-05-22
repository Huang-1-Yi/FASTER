import dataclasses
import math
from typing import ClassVar

import einops
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing_extensions import Literal

from openpi import transforms


def make_agilex_example() -> dict:
    """Creates a random input example for the Agilex policy."""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class AgilexInputs(transforms.DataTransformFn):
    """Inputs for the Agilex policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # Data contract: 输入 camera 名称必须在该集合内；缺失 camera 用黑图占位，并将对应 image_mask 置 False。
    # 自定义 Agilex 相机名时，需要同时更新这个集合、repack 映射和下方 extra_image_names。
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_agilex(data)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Data contract: cam_high 是必选主视角，用来决定缺失 wrist 图像的占位尺寸。
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Data contract: wrist 相机可选；缺失时用零图和 False mask 保持 camera slot 稳定。
        # 模型侧仍看到固定的 left/right wrist slot，避免训练和在线推理的结构不一致。
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        # Data contract: Agilex 在这里把 camera dict 和 14-D 双臂 state 映射到 openpi 统一输入键。
        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Data contract: actions 只在训练样本中出现，在线推理只提供 observation、prompt 和可选 prefix；
        # 后续 model transform 会把 14-D action pad 到模型 action_dim。
        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "delay" in data:
            inputs["delay"] = data["delay"]

        if "action_prefix" in data:
            # FASTER: action_prefix 原样透传，用于在 denoising 剩余 horizon 时保留已执行/已知动作。
            inputs["action_prefix"] = data["action_prefix"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AgilexOutputs(transforms.DataTransformFn):
    """Outputs for the Agilex policy."""

    def __call__(self, data: dict) -> dict:
        # Data contract: Agilex 执行端消费完整 14-D 双臂 action 向量；如果机器人维度变化，这里和 state 契约要一起改。
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": actions}


def _decode_agilex(data: dict) -> dict:
    # Data contract: state 的排列是 [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]。
    # 维度为 [6, 1, 6, 1]。
    state = np.asarray(data["state"])

    def convert_image(img):
        img = np.asarray(img)
        # Data contract: float 图像统一转成 uint8。
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Data contract: 图像统一从 CHW 转成 HWC。
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data
