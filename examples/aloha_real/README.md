# Run Aloha (Real Robot)

本示例演示如何使用 [ALOHA setup](https://github.com/tonyzhaozh/aloha) 在真实机器人上运行。关于如何加载 checkpoint 并运行推理，请参见[这里](../../docs/remote_inference.md)。下面列出了每个已提供微调模型对应的 checkpoint 路径。

## Prerequisites

本仓库使用 ALOHA 仓库的一个 fork，只做了很小的修改，以支持 Realsense 相机。

1. 按照 ALOHA 仓库中的[硬件安装说明](https://github.com/tonyzhaozh/aloha?tab=readme-ov-file#hardware-installation)进行操作。
1. 修改 `third_party/aloha/aloha_scripts/realsense_publisher.py` 文件，为你的相机设置序列号。

## With Docker

```bash
export SERVER_ARGS="--env ALOHA --default_prompt='take the toast out of the toaster'"
docker compose -f examples/aloha_real/compose.yml up --build
```

## Without Docker

终端窗口 1：

```bash
# Create virtual environment
uv venv --python 3.10 examples/aloha_real/.venv
source examples/aloha_real/.venv/bin/activate
uv pip sync examples/aloha_real/requirements.txt
uv pip install -e packages/openpi-client

# Run the robot
python -m examples.aloha_real.main
```

终端窗口 2：

```bash
roslaunch aloha ros_nodes.launch
```

终端窗口 3：

```bash
uv run scripts/serve_policy.py --env ALOHA --default_prompt='take the toast out of the toaster'
```

## **ALOHA Checkpoint Guide**


`pi0_base` 模型可以在 ALOHA 平台上以 zero shot 方式用于一个简单任务；此外，我们还提供了两个示例微调 checkpoint，分别对应 "fold the towel" 和 "open the tupperware and put the food on the plate"，可在 ALOHA 上执行更高级的任务。

虽然我们发现这些策略可以在多个 ALOHA 工作站的未见过条件下运行，但这里仍提供一些场景设置建议，以尽可能提高策略成功率。我们会说明策略应使用的 prompt、已验证效果较好的物体，以及较充分覆盖的初始状态分布。以 zero shot 方式运行这些策略仍然是非常实验性的功能，不能保证它们一定能在你的机器人上工作。推荐的 `pi0_base` 使用方式是基于目标机器人采集的数据进行微调。


---

### **Toast Task**

该任务要求机器人从烤面包机中取出两片吐司，并将它们放到盘子上。

- **Checkpoint path**: `gs://openpi-assets/checkpoints/pi0_base`
- **Prompt**: "take the toast out of the toaster"
- **所需物体**：两片吐司、一个盘子和一个标准烤面包机。
- **物体分布**：
  - 适用于真实吐司和橡胶假吐司
  - 兼容标准 2 片式烤面包机
  - 适用于多种颜色的盘子

### **Scene Setup Guidelines**
<img width="500" alt="Screenshot 2025-01-31 at 10 06 02 PM" src="https://github.com/user-attachments/assets/3d043d95-9d1c-4dda-9991-e63cae61e02e" />

- 烤面包机应放在工作空间的左上象限。
- 两片吐司的初始位置都应在烤面包机内，并且顶部至少露出 1 cm。
- 盘子应大致放在工作空间的下方中央。
- 该策略适用于自然光和人造光，但请避免场景过暗，例如不要把装置放在封闭空间内或帘子下面。


### **Towel Task**

该任务要求机器人将一块小毛巾（例如大约手巾大小）折成八等份。

- **Checkpoint path**: `gs://openpi-assets/checkpoints/pi0_aloha_towel`
- **Prompt**: "fold the towel"
- **物体分布**：
  - 适用于多种纯色毛巾
  - 在纹理很重或带条纹的毛巾上表现较差

### **Scene Setup Guidelines**
<img width="500" alt="Screenshot 2025-01-31 at 10 01 15 PM" src="https://github.com/user-attachments/assets/9410090c-467d-4a9c-ac76-96e5b4d00943" />

- 毛巾应摊平，并大致放在桌面中央。
- 请选择不会与桌面颜色混在一起的毛巾。


### **Tupperware Task**

该任务要求机器人打开装有食物的保鲜盒，并将其中内容倒到盘子上。

- **Checkpoint path**: `gs://openpi-assets/checkpoints/pi0_aloha_tupperware`
- **Prompt**: "open the tupperware and put the food on the plate"
- **所需物体**：保鲜盒、食物（或类似食物的物体）和一个盘子。
- **物体分布**：
  - 适用于多种假食物，例如假鸡块、薯条和炸鸡。
  - 兼容不同盖子颜色和形状的保鲜盒，在带角部翻片的方形保鲜盒上效果最好（见下图）。
  - 该策略见过多种纯色盘子。

### **Scene Setup Guidelines**
<img width="500" alt="Screenshot 2025-01-31 at 10 02 27 PM" src="https://github.com/user-attachments/assets/60fc1de0-2d64-4076-b903-f427e5e9d1bf" />

- 当保鲜盒和盘子都大致位于工作空间中央时，观察到的效果最好。
- 位置：
  - 保鲜盒应位于左侧。
  - 盘子应位于右侧或下方。
  - 保鲜盒翻片应朝向盘子。

## Training on your own Aloha dataset

1. 将数据集转换为 LeRobot 数据集 v2.0 格式。

    我们提供了脚本 [convert_aloha_data_to_lerobot.py](./convert_aloha_data_to_lerobot.py)，用于将数据集转换为 LeRobot 数据集 v2.0 格式。作为示例，我们已经从 [BiPlay repo](https://huggingface.co/datasets/oier-mees/BiPlay/tree/main/aloha_pen_uncap_diverse_raw) 转换了 `aloha_pen_uncap_diverse_raw` 数据集，并将其上传到 HuggingFace Hub，名称为 [physical-intelligence/aloha_pen_uncap_diverse](https://huggingface.co/datasets/physical-intelligence/aloha_pen_uncap_diverse)。


2. 定义一个使用自定义数据集的训练配置。

    我们提供了 [pi0_aloha_pen_uncap config](../../src/openpi/training/config.py) 作为示例。关于如何使用新配置运行训练，请参考根目录 [README](../../README.md)。

重要提示：我们的 base checkpoint 包含来自多种常见机器人配置的归一化统计量。当使用这些配置之一的自定义数据集对 base checkpoint 进行微调时，建议使用 base checkpoint 中提供的对应归一化统计量。在该示例中，这是通过在 AssetsConfig 中指定 trossen asset_id 以及预训练 checkpoint 的 assets 目录路径来完成的。
