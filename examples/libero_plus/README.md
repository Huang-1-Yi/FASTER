# LIBERO-Plus Evaluation for FASTER_hy

[LIBERO-Plus](https://arxiv.org/abs/2510.13626) is a robustness-oriented benchmark built on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO). It introduces **7 perturbation dimensions** to evaluate generalization beyond in-distribution evaluation:

| Dimension | Description |
|---|---|
| Objects Layout | Confounding objects and target object displacement |
| Camera Viewpoints | Position, orientation, and field-of-view changes |
| Robot Initial States | Manipulator initial pose variations |
| Language Instructions | LLM-based instruction rewriting |
| Light Conditions | Intensity, direction, color, and shadow variations |
| Background Textures | Scene and surface appearance changes |
| Sensor Noise | Photometric distortions and image degradation |

GuidedVLA achieves **75.4% average success rate** on LIBERO-Plus, vs. 68.2% for the π₀ baseline.

## Requirements

This example requires the LIBERO-Plus submodule. FASTER_hy keeps it separate from the original `third_party/libero`
checkout; make sure `third_party/LIBERO-plus` is initialized:

```bash
git submodule update --init --recursive
```

## Setup (without Docker)

Create a Python 3.8 environment for the LIBERO-Plus simulator:

```bash
# System dependencies
sudo apt install -y libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev

# Create virtual environment and install dependencies
uv venv --python 3.8 examples/libero_plus/.venv
source examples/libero_plus/.venv/bin/activate

uv pip sync examples/libero_plus/requirements.txt third_party/LIBERO-plus/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy=unsafe-best-match

uv pip install -e packages/openpi-client
uv pip install -e third_party/LIBERO-plus
uv pip install -r third_party/LIBERO-plus/extra_requirements.txt

export PYTHONPATH=$PYTHONPATH:$(pwd)/third_party/LIBERO-plus
```

If `third_party/LIBERO-plus` is still empty after the command above, add or re-sync the submodule explicitly:

```bash
git submodule sync --recursive
git submodule update --init --recursive third_party/LIBERO-plus
```

## Running Evaluation

### Step 1: Launch the policy server

In one terminal, launch the policy server pointing to your trained checkpoint:

```bash
uv run --no-sync scripts/serve_policy.py \
    --env LIBERO \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_faster_libero \
    --policy.dir checkpoints/pi05_faster_libero/<exp_name>/<step>
```

For a smoke test with the public LIBERO checkpoint, use:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --no-sync scripts/serve_policy.py \
    --env LIBERO \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir gs://openpi-assets/checkpoints/pi05_libero
```

### Step 2: Run a single perturbation category

In a second terminal:

```bash
source examples/libero_plus/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/third_party/LIBERO-plus

python examples/libero_plus/main.py \
    --host 127.0.0.1 \
    --port 8000 \
    --task-suite-name libero_object \
    --task-ids 0 \
    --video-out-path data/libero_plus/videos \
    --num-trials-per-task 1 \
    --results-json-path data/libero_plus/libero_object.json
```

Available `--task-suite-name` values: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `all`

Available `--category` values: `"Objects Layout"`, `"Camera Viewpoints"`, `"Robot Initial States"`, `"Language Instructions"`, `"Light Conditions"`, `"Background Textures"`, `"Sensor Noise"`

Useful `main.py` arguments:
- `--task-ids`: e.g. `0`, `0,3,7`, or `10-19`
- `--replan-steps`: action chunk size requested from the server
- `--results-json-path`: rolling JSON summary; when `--category` is set, the category suffix is appended automatically

### Step 3: Run all task suites and perturbations at once

```bash
uv run examples/libero_plus/eval_libero_plus.py \
    --checkpoint-dir checkpoints/pi05_faster_libero/<exp_name>/<step> \
    --policy-config pi05_faster_libero \
    --gpu-ids 0,1,2,3 \
    --client-python examples/libero_plus/.venv/bin/python \
    --libero-plus-path third_party/LIBERO-plus
```

Useful `eval_libero_plus.py` arguments:
- `--task-suites`: comma-separated suites, default is `libero_spatial,libero_object,libero_goal,libero_10`
- `--categories`: comma-separated perturbation categories
- `--task-ids`: restrict to a subset of tasks
- `--num-trials-per-task`: number of rollouts per task

Outputs are written under:
- `data/libero_plus/` for JSON results and rollout videos
- `logs/libero_plus/` for per-worker logs
