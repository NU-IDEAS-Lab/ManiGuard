# Installation

ManiGuard runs in the **`behavior` conda env** — OmniGibson simulation, BDDL,
teleop, task generation, RL, and eval. Policy training/serving via openpi uses
its own venv (see the [SFT](../openpi_sim_teleop_sft.md) pages).

## 1. Clone with submodules

```bash
git clone --recursive https://github.com/NU-IDEAS-Lab/SENTINEL-Lite.git
cd ManiGuard
# or, if already cloned:
git submodule update --init --recursive
```

## 2. Install BEHAVIOR-1K + dataset

This runs upstream's `setup.sh` from inside the submodule. `--dataset` downloads
the encrypted assets into `behavior-1k/datasets/`, which matches OmniGibson's
default resolver — no env var needed afterwards.

```bash
cd behavior-1k
./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval --primitives
cd ..
```

Available flags: `--omnigibson`, `--bddl`, `--joylo`, `--dataset`, `--eval`,
`--asset-pipeline`, `--primitives`, `--dev`.
Dependencies: `--omnigibson` requires `--bddl`; `--primitives` requires `--omnigibson`.

## 3. Install ManiGuard (editable)

```bash
conda activate behavior
pip install -e .                 # base
pip install -e ".[rl,serve]"     # with RL + policy-server extras
```

## Optional: override the dataset path

Only needed if you keep BEHAVIOR assets outside the repo (e.g. HPC shared storage).
Default resolves to `behavior-1k/datasets/`.

```bash
export OMNIGIBSON_DATA_PATH=/abs/path/to/datasets
```

## Other environment variables

For headless deployment:

| Variable | Purpose |
|---|---|
| `ISAAC_PATH` | Path to Isaac Sim package |
| `OMNIGIBSON_DATA_PATH` | Path to BEHAVIOR datasets (override only) |
| `BEHAVIOR_PATH` | Path to ManiGuard root |
| `OMNIGIBSON_HEADLESS=1` | Required for server / headless rendering |
| `VK_ICD_FILENAMES` | Vulkan ICD config for headless GPU rendering |
| `CUDA_VISIBLE_DEVICES` | GPU selection (esp. when transitioning between multi-GPU tasks) |

## Common issues

!!! warning "PhysX CUDA error 700"
    Set `CUDA_VISIBLE_DEVICES=0` to pin a single GPU.

!!! warning "`typing_extensions` errors with torch 2.6.0"
    Remove the outdated `typing_extensions` from Isaac Sim so conda's version is used.

!!! warning "Vulkan `ERROR_INCOMPATIBLE_DRIVER`"
    Fix `VK_ICD_FILENAMES` to point to a valid local ICD JSON.

!!! warning "CUDA OOM"
    Reduce `total_num_envs` in the YAML config; ensure `component_placement` GPUs don't overlap.
