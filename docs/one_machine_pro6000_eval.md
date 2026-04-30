# One-Machine Pro 6000 Eval Walkthrough

This runs both the OpenPI policy server and OmniGibson eval client on the same
RTX PRO 6000 machine. The policy server uses the pi05 SFT checkpoint, and the
eval saves per-scene MP4 video.

## Assumptions

- Repo root: `/workspace/SENTINEL-Lite`
- Behavior env: `/workspace/miniconda3/bin/conda run -n behavior`
- OpenPI venv: `/workspace/SENTINEL-Lite/openpi/.venv`
- Vulkan ICD: `/etc/vulkan/icd.d/nvidia_icd.json`
- Checkpoint:
  `vla_models/pi05-sim-table-lora/checkpoints/pi05_clutter_libero_lora/sim_table_lora/5000`
- Benchmark:
  `datasets/final_unique_accepted-goal_region_sphere-full-perturbed_with_base-20260426/table`

## 1. Quick GPU/Vulkan Check

```bash
nvidia-smi

VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
vulkaninfo --summary
```

If `vulkaninfo` cannot see the NVIDIA device, fix Vulkan before starting
OmniGibson.

## 2. Start the Local Policy Server

Run this in one shell and leave it running:

```bash
cd /workspace/SENTINEL-Lite/openpi

XLA_PYTHON_CLIENT_PREALLOCATE=false \
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_clutter_libero_lora \
  --policy.dir /workspace/SENTINEL-Lite/vla_models/pi05-sim-table-lora/checkpoints/pi05_clutter_libero_lora/sim_table_lora/5000
```

Wait for:

```text
Server listening on 0.0.0.0:8000
```

`XLA_PYTHON_CLIENT_PREALLOCATE=false` matters on one GPU because it keeps JAX
from reserving nearly all VRAM before Isaac starts.

## 3. Run a One-Scene Smoke Eval with Video

Run this from the repo root in a second shell:

```bash
cd /workspace/SENTINEL-Lite

PYTHONPATH=/workspace/SENTINEL-Lite/openpi/packages/openpi-client/src \
OMNI_KIT_ACCEPT_EULA=YES \
OMNIGIBSON_HEADLESS=1 \
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
CUDA_VISIBLE_DEVICES=0 \
/workspace/miniconda3/bin/conda run -n behavior python -m sentinel.eval.benchmark \
  --benchmark-root /workspace/SENTINEL-Lite/datasets/final_unique_accepted-goal_region_sphere-full-perturbed_with_base-20260426/table \
  --host 127.0.0.1 --port 8000 \
  --use-openpi-client \
  --eval-profile pi05_sentinel_table \
  --max-steps 100 \
  --max-scenes 1 \
  --headless \
  --save-video \
  --output-dir /workspace/SENTINEL-Lite/outputs/eval_pi05_sim_table_local_one_scene_100
```

Expected output files:

```text
outputs/eval_pi05_sim_table_local_one_scene_100/results.jsonl
outputs/eval_pi05_sim_table_local_one_scene_100/summary.json
outputs/eval_pi05_sim_table_local_one_scene_100/<scene_name>.mp4
```

The verified smoke run produced a 101-frame MP4 for:

```text
task_0097/semantic/table__task_0097__semantic__instr_01.mp4
```

## 4. Run More Scenes

After the one-scene smoke passes, remove `--max-scenes 1` or set a larger cap:

```bash
  --max-steps 500 \
  --max-scenes 10 \
  --save-video \
  --output-dir /workspace/SENTINEL-Lite/outputs/eval_pi05_sim_table_local_10
```

For a full table sweep, omit `--max-scenes`.

## 5. Shutdown

Stop the policy server with `Ctrl-C`. Confirm GPU memory is released:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

No output means no active compute process is holding VRAM.

## Notes and Gotchas

- Use `127.0.0.1` for `--host` because server and eval client are on the same
  machine.
- Keep `PYTHONPATH=/workspace/SENTINEL-Lite/openpi/packages/openpi-client/src`
  unless `openpi-client` has been installed into the `behavior` env.
- The first policy call can be slow because JAX/XLA compiles and autotunes
  kernels. The local client should not use an aggressive websocket ping timeout.
- The longfinger Franka asset should exist at:
  `behavior-1k/datasets/omnigibson-robot-assets/models/franka/franka_panda_longfinger.zip`
  or already be extracted under `franka_panda_longfinger/`.
- If OmniGibson reports Vulkan errors, keep the explicit
  `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`.
- If Isaac or JAX runs out of VRAM, verify the server was launched with
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
