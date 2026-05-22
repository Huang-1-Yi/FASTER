import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
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
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Data contract: model_type 只决定占位图是否被 image_mask 屏蔽；自定义 dataset 通常不改这个字段。
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Data contract: LIBERO 图像在这里统一转成 uint8 HWC，补齐 LeRobot 常见的 float32 CHW 存储差异。
        # 若自定义 dataset 的图像 key 不同，改这里的读取 key，但保持下面 openpi camera slot 名称不变。
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Data contract: 从这里开始，LIBERO 专用字段被转换成 openpi 统一格式：
        # state、image、image_mask、actions、prompt 这些输出 key 不能随意改名。
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Data contract: 缺失的右腕视角用零图占位，保持模型侧 camera slot 固定。
                # 如果你的数据真的有右腕图像，替换这里的零图即可。
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Data contract: pi0/pi05 屏蔽占位图；pi0-FAST 保留占位图参与 tokenization，避免 FAST token 槽位变化。
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Data contract: actions 只在训练样本中出现，在线推理只提供 observation 和 prompt；后续 model transform 会 pad 到模型 action_dim。
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Data contract: 语言指令统一放在 prompt，供后续 tokenizer 处理；若源字段不叫 prompt，只改右侧读取 key。
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Data contract: LIBERO 执行端只消费前 7 维动作，其余维度是模型 action_dim padding；自定义机器人要改成真实 action_dim。
        return {"actions": np.asarray(data["actions"][:, :7])}
