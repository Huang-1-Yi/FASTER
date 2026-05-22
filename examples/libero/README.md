# LIBERO Benchmark

本目录包含 [LIBERO benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO) 的客户端评估入口，遵循 [openpi LIBERO 示例](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md)。

对于策略训练，你可以使用 LeRobot 数据集 [physical-intelligence/libero](https://huggingface.co/datasets/physical-intelligence/libero)，也可以通过 [convert_libero_data_to_lerobot.py](convert_libero_data_to_lerobot.py) 将 RLDS 数据集 [openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds) 转换为 LeRobot 格式。

## Setup

请在仓库根目录运行以下命令。

```bash
# Initialize LIBERO repo
git submodule update --init --recursive

# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate

# Install dependencies (Add Tsinghua mirror)
uv pip sync \
  examples/libero/requirements.txt \
  third_party/libero/requirements.txt \
  --extra-index-url "https://download.pytorch.org/whl/cu113 https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple" \
  --index-strategy=unsafe-best-match

uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
```

## Evaluation

评估使用两个进程：

1. 在主 openpi 环境中运行的策略服务器。
2. 在 `examples/libero/.venv` 中运行的 LIBERO 客户端。

### 启动策略服务器

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_faster_libero \
  --policy.dir=checkpoints/pi05_faster_libero/my_experiment/29999
```

### 运行评估

```bash
source examples/libero/.venv/bin/activate
export LIBERO_CONFIG_PATH=$PWD/third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

python examples/libero/main.py \
  --args.task-suite-name libero_spatial \
  --args.save-name pi05_faster \
  --args.host 0.0.0.0 \
  --args.port 8000
```

常用选项：

- `--args.task-suite-name`：要评估的任务套件名称（`libero_spatial`、`libero_object`、`libero_goal`、`libero_10`）。
- `--args.out-path`：输出目录。
- `--args.save-name`：本次评估运行的标识符。结果会追加写入 `{out_path}/results/{save_name}_results.txt`。
- `--args.save-videos`：将 rollout 视频保存到 `{out_path}/videos/{save_name}/{task_suite_name}/`。


完整 CLI 选项列表请见 [main.py](main.py)。

## Results

以下 checkpoint 在四个任务套件上联合微调了 30k 步。

| Model       | Config               | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
| ----------- | -------------------- | -------------- | ------------- | ----------- | --------- | ------- |
| π0.5        | `pi05_libero`        | 98.8           | 98.2          | 98.0        | 92.4      | 96.9    |
| π0.5+FASTER | `pi05_faster_libero` | 98.6           | 97.8          | 97.8        | 91.6      | 96.5    |
