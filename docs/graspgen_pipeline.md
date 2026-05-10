# GraspGen-driven grasp evaluation pipeline

Run NVlabs/GraspGen as a local ZMQ server, then drive the
`sentinel.rl.grasps.render_grasps` per-object validator against it to
produce `.pt` grasp datasets (consumed by `GraspDatasetResetter`),
diagnostic PNGs, and success-grasp `.mp4` videos.

## Why a separate server

GraspGen pins `torch==2.1.0+cu121` and a custom `pointnet2_ops` C
extension that conflict with OmniGibson's `torch==2.6.0+cu124`. Running
GraspGen in its own `uv` venv and talking to it over ZMQ keeps the OG
env clean — the only client-side dependency is `pyzmq + msgpack-numpy`.

## One-time setup

All commands assume your project root is the SENTINEL-Lite repo.

### 1. Clone GraspGen + its model checkpoints

```bash
cd $PROJECT_ROOT  # /path/to/SENTINEL-Lite
git clone https://github.com/NVlabs/GraspGen.git
git clone https://huggingface.co/adithyamurali/GraspGenModels
```

Both directories are gitignored at the project root (see `.gitignore`),
so they stay local.

### 2. Pull the actual checkpoint files

The HuggingFace clone only fetches Git LFS pointers (134-byte stubs).
Pull the real weights:

```bash
# Option A: use git-lfs if installed
cd GraspGenModels && git lfs install && git lfs pull && cd ..

# Option B: curl directly (no git-lfs needed)
cd GraspGenModels/checkpoints
for f in graspgen_franka_panda_gen.pth graspgen_franka_panda_dis.pth; do
  curl -L -o "$f" \
    "https://huggingface.co/adithyamurali/GraspGenModels/resolve/main/checkpoints/$f"
done
cd ../..
```

Sanity check sizes — `_gen.pth` should be ~866 MB, `_dis.pth` ~159 MB.
A few hundred bytes means the LFS pull didn't run.

### 3. Set up the GraspGen uv venv

```bash
cd GraspGen
uv python install 3.10
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -e .

# Install pointnet2_ops (custom C extension, requires CUDA toolkit + g++).
# The script sets TORCH_CUDA_ARCH_LIST=8.6 — adjust if your GPU is different.
bash install_uv_pointnet.sh

# Install ZMQ serving deps (the upstream pyproject doesn't pull these).
uv pip install pyzmq msgpack msgpack-numpy

cd ..
```

Tested on RTX 4080 (sm_89, falls back to `8.6` arch fine) with CUDA 12.x
runtime. If `pointnet2_ops` build errors with CUDA arch mismatches, edit
`TORCH_CUDA_ARCH_LIST` inside `install_uv_pointnet.sh`.

### 4. Client-side deps

The `behavior` conda env needs the ZMQ wire-protocol libs (the GraspGen
client is a tiny msgpack-over-ZMQ thin client; no torch / CUDA needed
client-side):

```bash
conda activate behavior
pip install pyzmq msgpack msgpack-numpy
```

## Running

### Start the server (one terminal)

```bash
cd $PROJECT_ROOT/GraspGen
.venv/bin/python client-server/graspgen_server.py \
    --gripper_config $PROJECT_ROOT/GraspGenModels/checkpoints/graspgen_franka_panda.yml \
    --port 5556
```

Wait until you see `GraspGen ZMQ server listening on tcp://0.0.0.0:5556`
(takes ~30 s to load the diffusion + discriminator models). The server
runs idle until clients send requests, so you can leave it up across
many runs.

To run in background and survive shell exit:
```bash
cd $PROJECT_ROOT/GraspGen
nohup .venv/bin/python client-server/graspgen_server.py \
    --gripper_config $PROJECT_ROOT/GraspGenModels/checkpoints/graspgen_franka_panda.yml \
    --port 5556 > /tmp/graspgen_server.log 2>&1 &
```

### Run the per-object evaluator (other terminal, project root)

```bash
conda activate behavior
cd $PROJECT_ROOT

SENTINEL_SKIP_LONGFINGER=1 \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \
  python -m sentinel.rl.grasps.render_grasps \
    --targets alarm_clock:cvknrh apple:bwteqh mug:fapsrj \
    --output-dir outputs/grasp_datasets/run01 \
    --save-grasp-dataset outputs/grasp_datasets/run01/datasets \
    --save-video \
    --num-target-grasps 10 \
    --per-object-timeout 150
```

Flags worth knowing:

| Flag | Default | Effect |
|---|---|---|
| `--targets cat:model ...` | (use CSV) | Skip CSV, run on listed objects only |
| `--csv path` | `sentinel/task_generation/utils/franka_graspability.csv` | Drive from a CSV with `category,model,status,...` columns |
| `--exclude-statuses` | `too_large,no_grasp,no_candidates,timeout` | CSV statuses to skip; default keeps only `graspable` rows |
| `--num-target-grasps N` | 1 | Phase A stops once N valid grasps collected per object |
| `--save-grasp-dataset DIR` | unset | Write `grasps_{cat}_{model}.pt` per object |
| `--save-video` | off | Phase B replays first valid grasp with frame capture and writes `{cat}_{model}.mp4` |
| `--graspgen-num-grasps` | 200 | Diffusion samples drawn server-side per inference |
| `--graspgen-topk` | 100 | Server-side top-K by discriminator confidence |
| `--graspgen-host / --port` | `localhost:5556` | Override server address |
| `--per-object-timeout` | 120 | Hard wall-clock budget per object (s) |
| `--viz-topk` | 100 | Top-K poses drawn in `{stem}_grasps_*.png`. 0 disables. |
| `--no-pcd-on-fail` | off | Skip `{stem}_pcd_*.png` on Phase A failure |

`SENTINEL_SKIP_LONGFINGER=1` matches GraspGen's training distribution
(default Franka Panda fingers). Drop it if you want OG's longfinger
patch (longer fingers help thin objects but the grasp center drifts
~3 cm vs GraspGen's predicted pose).

### Per-object outputs (in `--output-dir`)

Always written:
- `{cat}_{model}_grasps_{top,iso,front,side}.png` — top-K GraspGen
  candidates overlaid on the sampled point cloud (line = approach axis,
  colour = discriminator confidence).

On Phase A success (`held >= 1`):
- `{cat}_{model}.pt` (in `--save-grasp-dataset` dir) — keys
  `rel_position, rel_orientation_xyzw, gripper_qpos, arm_joint_pos,
  approach_traj`. Loaded directly by
  `sentinel.rl.grasps.reset.GraspDatasetResetter`.
- `{cat}_{model}.mp4` (only with `--save-video`) — Phase B replay of
  the first held grasp through the same physics kernel Phase A used.

On Phase A failure (`held == 0`):
- `{cat}_{model}_pcd_{top,iso,front,side}.png` — diagnostic point cloud
  scatter of the object that GraspGen + cuRobo couldn't grip.

### Resume

The CSV-driven main loop skips a row if any of these exist:
- `{stem}.pt` in the dataset dir (Phase A succeeded)
- `{stem}_pcd_top.png` in the output dir (Phase A produced 0 grasps)
- `{stem}.mp4` in the output dir (Phase B succeeded → A also succeeded)

Re-running the same command picks up where it left off.

## Pipeline at a glance

```
                   ┌──────────────────────────────────────┐
                   │  GraspGen ZMQ server (uv venv)       │
                   │    diffusion + discriminator         │
                   └──────────────────┬───────────────────┘
                                      │ (N,3) point cloud
                                      ▼
   ┌───────────────────── render_grasps (conda behavior) ────────────────────┐
   │                                                                         │
   │  for each (category, model) in CSV / --targets:                         │
   │      spawn target floating, disable gravity                             │
   │      mesh = trimesh from OG visual                                      │
   │      candidates = GraspGen.infer(mesh.sample_surface(8000))             │
   │      ── grasp PNG dump ──                                               │
   │                                                                         │
   │      Phase A — collect_valid_grasps:                                    │
   │        for cand in candidates by descending conf:                       │
   │          reach prefilter, cuRobo motion plan (ik_only=False)            │
   │          run_grasp_attempt(traj, frame_callback=None):                  │
   │            replay traj (hard pin) → close → gravity hold → check        │
   │          if hold: append saved-grasp dict {rel_pose, joints, traj}      │
   │          stop at num_target_grasps                                      │
   │                                                                         │
   │      if held: save .pt; if --save-video:                                │
   │        Phase B — run_grasp_attempt(saved_traj, frame_callback=_capture) │
   │                  → MP4                                                  │
   │      else: pcd PNG dump                                                 │
   └─────────────────────────────────────────────────────────────────────────┘
```

## Troubleshooting

**`zmq.error.Again: Resource temporarily unavailable`**: server not
listening on the port. Check `ss -tlnp | grep 5556` and the server log.

**`UnpicklingError: invalid load key, 'v'`** at server boot: HuggingFace
LFS pointer file instead of real `.pth`. Re-pull (step 2).

**`ImportError: cannot import name '...' from omnigibson...`**: OG
version drift. The pipeline is tested on `behavior-1k @ v3.7.2`.

**`pointnet2_ops` build errors**: edit `TORCH_CUDA_ARCH_LIST` in
`install_uv_pointnet.sh` to your GPU's compute capability (e.g. `8.9`
for RTX 4080/4090, `9.0` for H100).
