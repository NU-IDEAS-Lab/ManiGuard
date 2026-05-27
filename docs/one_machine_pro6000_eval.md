# One-Machine Pro 6000 Eval Walkthrough

This runs both the OpenPI policy server and OmniGibson eval client on the same
RTX PRO 6000 machine. The policy server uses the pi05 SFT checkpoint, and the
eval saves per-scene MP4 video.

## Assumptions

- Repo root: `/workspace/ManiGuard`
- Behavior env: `/workspace/miniconda3/bin/conda run -n behavior`
- OpenPI venv: `/workspace/ManiGuard/openpi/.venv`
- Vulkan ICD: `/etc/vulkan/icd.d/nvidia_icd.json`
- Eval config: `configs/eval/sim_table_25k.yaml` (edit paths as needed)

## 1. Quick GPU/Vulkan Check

```bash
nvidia-smi

VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
vulkaninfo --summary
```

If `vulkaninfo` cannot see the NVIDIA device, fix Vulkan before starting
OmniGibson.

## 2. Start the Local Policy Server

Run this in one shell and leave it running. The checkpoint path and config name
should match the `checkpoint` and `serve_config_name` fields in your eval YAML.

```bash
cd /workspace/ManiGuard/openpi

XLA_PYTHON_CLIENT_PREALLOCATE=false \
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_clutter_libero_lora \
  --policy.dir /workspace/ManiGuard/vla_models/pi05_sim_table_lora/checkpoints/pi05_clutter_libero_lora/sim_table_lora/25000
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
cd /workspace/ManiGuard

PYTHONPATH=/workspace/ManiGuard/openpi/packages/openpi-client/src \
OMNI_KIT_ACCEPT_EULA=YES \
OMNIGIBSON_HEADLESS=1 \
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
CUDA_VISIBLE_DEVICES=0 \
/workspace/miniconda3/bin/conda run -n behavior python -m maniguard.eval.benchmark \
  --config configs/eval/sim_table_25k.yaml \
  --max-steps 100 \
  --max-scenes 1 \
  --headless \
  --output-dir /workspace/ManiGuard/outputs/eval_smoke_one_scene
```

Expected output files:

```text
outputs/eval_smoke_one_scene/eval_config.json
outputs/eval_smoke_one_scene/results.jsonl
outputs/eval_smoke_one_scene/summary.json
outputs/eval_smoke_one_scene/<scene_name>.mp4
```

## 4. Run All Scenes (Per-Scene Isolation)

OmniGibson segfaults on `og.clear()` between scenes, so the batch script runs
one python process per scene:

```bash
cd /workspace/ManiGuard

PYTHONPATH=/workspace/ManiGuard/openpi/packages/openpi-client/src \
OMNI_KIT_ACCEPT_EULA=YES \
OMNIGIBSON_HEADLESS=1 \
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_benchmark_all_scenes.sh \
  --config configs/eval/sim_table_25k.yaml \
  --headless
```

Results append to `<output_dir>/results.jsonl` (output_dir is set in the YAML).
CLI flags after `--config` override any YAML field for that run.

## 5. Custom Eval Configs

Each experiment gets its own YAML. Copy and edit:

```bash
cp configs/eval/sim_table_25k.yaml configs/eval/my_experiment.yaml
```

Key fields:
- `benchmark_root` — path to scene dataset
- `scene_filter` — glob to select scenes (e.g. `"*/base"`)
- `checkpoint` / `serve_config_name` — informational, match your server
- `longfinger` — `true` for sim-table, `false` for sim2real mug-into-bowl
- `max_steps` — rollout horizon per scene
- `output_dir` — where results/videos go

## 6. Shutdown

Stop the policy server with `Ctrl-C`. Confirm GPU memory is released:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

No output means no active compute process is holding VRAM.

## Notes and Gotchas

- Use `127.0.0.1` for `host` in the YAML because server and eval client are on
  the same machine.
- Keep `PYTHONPATH=/workspace/ManiGuard/openpi/packages/openpi-client/src`
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
