# RL Training (SB3 PPO)

Train a PPO policy on a benchmark pick-and-lift task using OmniGibson + Stable-Baselines3.

```
OmniGibson (PhysX)  →  SB3 VecEnv  →  PPO  →  wandb / TensorBoard
```

## Prerequisites

- `behavior` conda env with `pip install -e ".[rl]"`
- A benchmark task directory containing `diagnostics.jsonl` + `scene_ep1.json`
- GPU with ≥10 GB VRAM (RTX 4090 24 GB tested)

## Quick Start

```bash
conda activate behavior

# Single-scene, 4 parallel envs, 200k steps
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
OMNIGIBSON_HEADLESS=1 \
python -m maniguard.rl.algorithms.ppo \
    --diagnostics-file datasets/<benchmark>/table/task_0022/base/diagnostics.jsonl \
    --num-envs 4 \
    --total-timesteps 200000 \
    --n-steps 128 \
    --batch-size 256 \
    --save-freq 50000 \
    --output-dir outputs/rl_ppo_run \
    --wandb --wandb-run-name "ppo-4env-200k"
```

`--diagnostics-file` auto-infers `--scene-file` from the sibling `scene_ep1.json` and uses the benchmark's goal region (sphere + AABB intersection) instead of a hardcoded offset. Grasp reset is skipped when no grasp dataset exists for the target object.

## Resume from Checkpoint

```bash
python -m maniguard.rl.algorithms.ppo \
    --diagnostics-file <same as above> \
    --num-envs 4 \
    --total-timesteps 10000000 \
    --resume-from outputs/rl_ppo_run/ppo_final.zip \
    --output-dir outputs/rl_ppo_10M \
    --save-freq 500000 \
    --wandb --wandb-run-name "ppo-4env-10M"
```

The timestep counter is preserved — TB and wandb continue from where the previous run left off.

## Key Arguments

| Arg | Default | Notes |
|-----|---------|-------|
| `--num-envs` | 1 | Parallel envs via OG scene tiling. Requires `--scene-file` (auto-inferred from diagnostics). See scaling notes below. |
| `--n-steps` | 128 | Rollout length per env before PPO update. |
| `--batch-size` | 32 | Minibatch size for PPO gradient steps. |
| `--learning-rate` | 3e-4 | Adam LR. |
| `--arm-controller` | joint | `joint` (stable) or `osc` (fragile under exploration). |
| `--reset-mode` | cached | `cached` (multi-env OK) or `ik` (single-env only, supports pose perturbation). |
| `--save-freq` | 250000 | Checkpoint every N env-steps. |
| `--eval-freq` | 0 | Deterministic eval every N steps (0=disabled). Use with `--n-eval-episodes`. |
| `--video-freq` | 0 | Record viewer-camera clip every N steps (0=disabled). |
| `--wandb` | off | Enable Weights & Biases logging. Also: `--wandb-project`, `--wandb-entity`, `--wandb-mode`. |

## Scaling (RTX 4090 24 GB)

| Envs | Throughput | VRAM | Status |
|------|-----------|------|--------|
| 1 | ~28 steps/s | 7.3 GB | stable |
| 4 | ~60 steps/s | ~8 GB | stable, recommended |
| 8 | ~48 steps/s | 8.1 GB | stable |
| 12+ | ~47 steps/s | ~8.5 GB | diminishing returns, 16+ flaky |

**Bottleneck**: CPU-side per-env post-processing (obs gathering, RGB rendering), not GPU compute or VRAM. GPU utilization drops to ~3-6% at 4+ envs while CPU saturates.

**Dual concurrent runs** work (two separate tasks on the same GPU). VRAM roughly doubles (~16 GB for two 4-env runs). Throughput per run drops ~8%.

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `OMNIGIBSON_HEADLESS=1` | Yes (server) | Headless rendering |
| `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json` | Yes | Select NVIDIA Vulkan driver (avoids segfault from multiple ICDs) |
| `CUDA_VISIBLE_DEVICES=0` | Optional | Pin to specific GPU |

## Troubleshooting

- **Segfault on startup**: Wrong Vulkan ICD. Set `VK_ICD_FILENAMES` to the nvidia-only JSON.
- **Segfault at 16+ envs**: OG scene-tiling limit. Use ≤8 envs per process.
- **`LinAlgError: Matrix is singular`**: Switch from `--arm-controller osc` to `joint` (default).
- **PhysX CUDA error 700**: Set `CUDA_VISIBLE_DEVICES=0`.
- **No grasp dataset warning**: Expected when using `--diagnostics-file` on a benchmark task without a pre-collected grasp dataset. Training runs without grasp reset.
