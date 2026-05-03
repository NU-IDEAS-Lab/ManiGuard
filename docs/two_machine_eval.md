# Two-Machine Eval Setup

Policy inference runs on **Machine A** (GPU); OmniGibson simulation runs on **Machine B** (GPU + Isaac Sim). They communicate over a websocket (msgpack).

```
Machine A (policy server)          Machine B (simulator + eval client)
┌──────────────────────┐           ┌──────────────────────────────┐
│  serve_policy.py     │◄─ws:8000─►│  sentinel.eval.benchmark     │
│  (JAX, OpenPI)       │           │  (OmniGibson + Isaac Sim)    │
│  GPU: ~18 GB VRAM    │           │  GPU: ~8-12 GB VRAM          │
└──────────────────────┘           └──────────────────────────────┘
```

## Prerequisites

| Machine | Conda env | Key packages |
|---------|-----------|-------------|
| A | `openpi` (or RLinf `.venv`) | openpi, jax, flax |
| B | `behavior` | omnigibson, sentinel (editable) |

Download the checkpoint on Machine A:
```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'IDEAS-Lab-Northwestern/pi05-sim-table-lora',
    local_dir='vla_models/pi05_sim_table_lora',
    allow_patterns='checkpoints/pi05_clutter_libero_lora/sim_table_lora/25000/**',
)
"
```

## Step 1 — Start the policy server (Machine A)

```bash
cd /path/to/SENTINEL-Lite/openpi

CKPT_DIR=/path/to/SENTINEL-Lite/vla_models/pi05_sim_table_lora/checkpoints/pi05_clutter_libero_lora/sim_table_lora/25000

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_clutter_libero_lora \
  --policy.dir "$CKPT_DIR"
```

Wait for `Server listening on 0.0.0.0:8000` before proceeding.

## Step 2 — Run the benchmark (Machine B)

Create or copy an eval config YAML. For a two-machine setup, set `host` to
Machine A's LAN IP:

```bash
cp configs/eval/sim_table_25k.yaml configs/eval/sim_table_25k_remote.yaml
# Edit host: <MACHINE_A_IP>
```

Single-process run (multiple scenes sequentially in one python process):

```bash
OMNI_KIT_ACCEPT_EULA=yes \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
CUDA_VISIBLE_DEVICES=0 \
conda run -n behavior python -m sentinel.eval.benchmark \
  --config configs/eval/sim_table_25k_remote.yaml
```

Per-scene isolation (one python process per scene, avoids og.clear() segfaults):

```bash
OMNI_KIT_ACCEPT_EULA=yes \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_benchmark_all_scenes.sh \
  --config configs/eval/sim_table_25k_remote.yaml
```

CLI flags after `--config` override any YAML field for that run (e.g.
`--max-steps 500 --max-scenes 5`).

## Eval config fields

All settings live in one YAML file per experiment. Key fields:

| Field | Default | Notes |
|-------|---------|-------|
| `host` | `127.0.0.1` | Policy server IP. Set to Machine A's LAN IP for two-machine setup |
| `port` | `8000` | Policy server port |
| `benchmark_root` | — | Path to scene dataset |
| `scene_filter` | `""` | Glob to select scenes (e.g. `"*/base"`) |
| `state_mode` | `eef_8d_axisangle` | Must match training config |
| `action_dim` | `7` | Must match training config |
| `execute_horizon` | `5` | Action chunk execution length |
| `longfinger` | `true` | Enable longfinger Franka patch |
| `max_steps` | `1000` | Per-scene rollout limit |
| `max_scenes` | all | Cap number of scenes to eval |
| `save_video` | `true` | Save per-scene MP4 at 10 fps |
| `headless` | `false` | Run OmniGibson without display |
| `random_policy` | `false` | Smoke-test with random actions (no server needed) |
| `camera_resolution` | `256` | Policy input image size |
| `checkpoint` | — | Informational: which checkpoint the server should load |
| `serve_config_name` | — | Informational: openpi config name for the server |

Example configs: `configs/eval/sim_table_25k.yaml`, `configs/eval/mug_into_bowl_sim2real.yaml`.

## Diagnosing lag

The benchmark prints per-segment timing every 50 steps:

```
Step 50/2000 | success=False | goals={}
```

Quick network checks:
```bash
ping <MACHINE_A_IP> -c 20        # latency (each policy query does 1 round-trip)
iperf3 -c <MACHINE_A_IP>         # bandwidth (~600 KB per observation payload)
```

## Output

Results go to `output_dir` (set in the YAML or overridden via `--output-dir`):
- `eval_config.json` — resolved config snapshot for reproducibility
- `results.jsonl` — one JSON line per scene (success, steps, goal_detail)
- `summary.json` — aggregate stats
- `<scene_name>.mp4` — video if `save_video: true`

## Common issues

| Symptom | Fix |
|---------|-----|
| PhysX CUDA error 700 | Set `CUDA_VISIBLE_DEVICES=0` on Machine B |
| Connection refused on port 8000 | Check firewall; verify server is listening |
| `KeyError: 'observation/image'` | Ensure `use_openpi_client: true` in config (default) |
| Vulkan driver error | Fix `VK_ICD_FILENAMES` path on Machine B |
| Very slow sim times | Reduce scene complexity or check GPU utilization on Machine B |
| Very slow policy times | Check GPU memory pressure on Machine A; ensure model fits in VRAM |
