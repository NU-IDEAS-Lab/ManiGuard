# Installation

ManiGuard runs in the **`behavior` conda env** — OmniGibson simulation, BDDL,
teleop, task generation, scripted data generation, and eval. Policy training/serving
runs in each model's own environment (openpi / GR00T / SmolVLA — see the
[Fine-Tuning](../fine_tuning/index.md) pages).

## System requirements

| | Requirement |
|---|---|
| OS | Linux x86_64 (tested on Ubuntu 22.04 / 24.04); fully headless servers are supported (`OMNIGIBSON_HEADLESS=1`) |
| GPU | NVIDIA **RTX**-class GPU (Isaac Sim requires ray-tracing hardware); tested on RTX 3090 / 4080 / 4090 |
| NVIDIA driver | **A 5xx-series driver up to the 580 series** (CUDA ≤ 13.0) — see the warning below |
| CUDA toolkit | Not needed system-wide — the `behavior` env ships its own runtime (torch 2.6.0 + cu124) |
| VRAM | **16 GB runs the full eval pipeline on one card** — simulation + the heaviest policy server (π0 / π0.5) peaks at ~13 GB measured (GR00T ~11 GB, SmolVLA ~6 GB). The simulation client alone needs only ~4 GB, so an 8 GB card works when the policy server sits on another GPU or machine |
| RAM | ≥ 32 GB (64 GB comfortable) |
| Disk | ≥ 80 GB free: BEHAVIOR assets ~36 GB + the `behavior` conda env ~22 GB + benchmark & headroom |

!!! warning "NVIDIA drivers newer than the 580 series crash Isaac Sim"
    We have repeatedly observed Isaac Sim 4.5 **segfault at startup** on driver
    versions above the 580 series (CUDA 13.0) — across multiple machines and GPU
    models. Check yours with `nvidia-smi`; if it is newer, downgrade to a
    580-series (or older 5xx) driver before installing. Driver 580.x + CUDA 12.x
    userland is the verified configuration.

!!! note "RTX 50-series (Blackwell) GPUs"
    `sm_120` needs a newer torch than the env default: after setup, replace torch
    with **2.7.0 + cu128** inside the `behavior` env. The rest of the stack is
    unchanged.

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
Dependencies: `--omnigibson` requires `--bddl`; `--primitives` and `--dataset` require
`--omnigibson`; `--eval` requires `--omnigibson` + `--joylo`.

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
runtime). Set `MANIGUARD_SKIP_LONGFINGER=1` to keep the stock Franka instead.

!!! warning "A missing bundle fails silently"
    If the directory is absent, the **stock** Franka hand loads with no error —
    and policies trained on ManiGuard data will approach objects but never quite
    grasp them. Verify the path above exists before your first eval or datagen run.

**4b. Benchmark scenes.** The frozen benchmark (per-task `scene_ep1.json` +
`diagnostics.jsonl` + review videos) lives at
[`IDEAS-Lab-Northwestern/ManiGuard-Bench`](https://huggingface.co/datasets/IDEAS-Lab-Northwestern/ManiGuard-Bench).
`maniguard.eval.benchmark` accepts the HF repo id directly (snapshot-downloaded into
the HF cache) or a local directory:

```bash
# A) let eval pull it (run `hf auth login` first)
python -m maniguard.eval.benchmark --benchmark-root IDEAS-Lab-Northwestern/ManiGuard-Bench ...
# B) or download once to the family runner's default location
hf download IDEAS-Lab-Northwestern/ManiGuard-Bench --repo-type dataset \
  --local-dir outputs/lerobot_datasets/maniguard-bench
```

Option B's target is where `scripts/eval_family.sh` looks by default (any other
path works via `BENCH_ROOT=...`). Next step:
**[Run the benchmark](../evaluation/run_benchmark.md)**.

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
    The simulator wants most of a GPU. Put the policy server on a second GPU
    (`CUDA_VISIBLE_DEVICES`), or lower `camera_resolution` in the eval config.
