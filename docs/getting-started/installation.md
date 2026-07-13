# Installation

ManiGuard runs in the **`behavior` conda env** — OmniGibson simulation, BDDL,
teleop, task generation, scripted data generation, and eval. Policy training/serving
runs in each model's own environment (openpi / GR00T / SmolVLA — see the
[SFT](../sft/index.md) pages).

## 1. Clone with submodules

```bash
git clone --recursive https://github.com/NU-IDEAS-Lab/ManiGuard.git
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
pip install -e ".[serve]"        # with policy-server extras
```

## 4. ManiGuard-Bench + robot asset

To **run the benchmark** you need two ManiGuard-owned artifacts (both separate from
the Stanford-licensed BEHAVIOR asset bundle, both hosted on HuggingFace).

**4a. Robot asset (required).** The benchmark uses a Franka Panda with extended
**fin-ray fingers** — not part of the stock OmniGibson robot set. Drop it into the
robot-assets tree so the runtime patch (`maniguard._omnigibson_patches`) finds it.
It must land at `<data_root>/omnigibson-robot-assets/models/franka/franka_panda_longfinger/`,
where `<data_root>` is `behavior-1k/datasets/` (or `$OMNIGIBSON_DATA_PATH`):

```bash
hf download IDEAS-Lab-Northwestern/franka-panda-longfinger --repo-type dataset \
  --local-dir behavior-1k/datasets/omnigibson-robot-assets/models/franka/franka_panda_longfinger
```

On `import maniguard`, `FrankaPanda.usd_path` is auto-redirected to this bundle when
present (it ships the OmniGibson runtime USD + cuRobo description; no URDF needed at
runtime). Set `SENTINEL_SKIP_LONGFINGER=1` to keep the stock Franka instead.

**4b. Benchmark scenes.** The frozen benchmark (per-task `scene_ep1.json` +
`diagnostics.jsonl` + review videos) lives at
[`IDEAS-Lab-Northwestern/ManiGuard-Bench`](https://huggingface.co/datasets/IDEAS-Lab-Northwestern/ManiGuard-Bench).
`maniguard.eval.benchmark` accepts the HF repo id directly (snapshot-downloaded into
the HF cache) or a local directory:

```bash
# A) let eval pull it (run `hf auth login` first)
python -m maniguard.eval.benchmark --benchmark-root IDEAS-Lab-Northwestern/ManiGuard-Bench ...
# B) or download once and pass the local dir
hf download IDEAS-Lab-Northwestern/ManiGuard-Bench --repo-type dataset --local-dir datasets/maniguard-bench
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
