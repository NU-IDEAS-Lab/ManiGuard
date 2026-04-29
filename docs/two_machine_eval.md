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
    local_dir='vla_models/pi05-sim-table-lora',
    allow_patterns='checkpoints/pi05_clutter_libero_lora/sim_table_lora/5000/**',
)
"
```

## Step 1 — Start the policy server (Machine A)

```bash
cd /path/to/SENTINEL-Lite/openpi

# Pick the checkpoint step (3000 or 5000)
CKPT_STEP=5000
CKPT_DIR=/path/to/SENTINEL-Lite/vla_models/pi05-sim-table-lora/checkpoints/pi05_clutter_libero_lora/sim_table_lora/${CKPT_STEP}

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_clutter_libero_lora \
  --policy.dir "$CKPT_DIR"
```

Wait for `Server listening on 0.0.0.0:8000` before proceeding.

## Step 2 — Run the benchmark (Machine B)

```bash
OMNI_KIT_ACCEPT_EULA=yes \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
CUDA_VISIBLE_DEVICES=0 \
conda run -n behavior python -m sentinel.eval.benchmark \
  --benchmark-root datasets/final_unique_accepted-goal_region_sphere-full-perturbed_with_base-20260426/table \
  --host <MACHINE_A_IP> --port 8000 \
  --use-openpi-client \
  --eval-profile pi05_sentinel_table \
  --max-steps 500 \
  --save-video \
  --output-dir outputs/eval_pi05_sim_table
```

Replace `<MACHINE_A_IP>` with Machine A's LAN IP. Use `127.0.0.1` if both run on the same machine.

## Eval profiles

| Profile | State mode | Action dim | Horizon | Use case |
|---------|-----------|------------|---------|----------|
| `pi05_sentinel_table` | eef_8d_axisangle | 7 | 5 | Sentinel table/clutter tasks |
| `pi05_stack_cube` | eef_8d | 7 | 5 | IsaacLab stack-cube |
| `pi05_libero` | eef_7d | 7 | 5 | LIBERO tasks |
| `pi0_libero` | eef_7d | 7 | 4 | LIBERO (pi0 base) |
| `gr00t` | eef_8d | 7 | 1 | GR00T single-step |
| `gr00t_n16` | joint | 8 | 16 | GR00T 16-step joint |

The profile must match the checkpoint's training config. `pi05_sentinel_table` pairs with `pi05_clutter_libero_lora`.

## Useful flags

| Flag | Default | Notes |
|------|---------|-------|
| `--max-steps` | 500 | Per-episode step limit |
| `--scenes SCENE1 SCENE2` | all | Restrict to specific scenes |
| `--max-scenes N` | all | Cap number of scenes |
| `--save-video` | off | Save per-scene MP4 at 10 fps |
| `--headless` | off | Run OmniGibson without display |
| `--random-policy` | off | Smoke-test with random actions (no server needed) |
| `--camera-resolution` | 256 | Policy input image size |

## Diagnosing lag

The benchmark prints per-segment timing every 50 steps:

```
Step 50/500 | success=False | goals={}
  timing(last50): policy=85ms (infer=42ms + net~43ms) | sim=120ms | obs=15ms
```

| Metric | Measures |
|--------|----------|
| `policy` | Full round-trip: serialize + network + inference + network + deserialize |
| `infer` | Pure GPU inference (reported by server) |
| `net` | `policy - infer` = serialization + network latency |
| `sim` | `env.step()` — PhysX simulation |
| `obs` | `extract_obs()` — camera rendering + state extraction |

Quick network checks:
```bash
ping <MACHINE_A_IP> -c 20        # latency (each policy query does 1 round-trip)
iperf3 -c <MACHINE_A_IP>         # bandwidth (~600 KB per observation payload)
```

## Output

Results go to `--output-dir`:
- `results.jsonl` — one JSON line per scene (success, steps, goal_detail)
- `<scene_name>.mp4` — video if `--save-video`

## Common issues

| Symptom | Fix |
|---------|-----|
| PhysX CUDA error 700 | Set `CUDA_VISIBLE_DEVICES=0` on Machine B |
| Connection refused on port 8000 | Check firewall; verify server is listening; try `curl http://<IP>:8000/healthz` |
| Action dim mismatch | Ensure `--eval-profile` matches the checkpoint's training config |
| Vulkan driver error | Fix `VK_ICD_FILENAMES` path on Machine B |
| Very slow `sim` times | Reduce scene complexity or check GPU utilization on Machine B |
| Very slow `policy` times | Check GPU memory pressure on Machine A; ensure model fits in VRAM |
