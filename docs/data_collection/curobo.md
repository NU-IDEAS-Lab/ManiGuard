# cuRobo demo collection

A scripted alternative to teleop: take a frozen task-generation scene, stage it
in an empty scene, and let **cuRobo** motion planning execute the task
autonomously while recording the rollout as SFT data. This scales demonstration
collection without a human in the loop. Everything lives in
`maniguard/data/curobo/`.

## The flow

```
frozen task dump (scene_ep1.json + diagnostics.jsonl)
        │
        ▼
replay_empty_from_dataset      load support + task objects into an EMPTY scene
        │
        ▼
<task>_from_dataset            cuRobo plan + execute the task (pnp / lid / …)
        │   └── _sft_recorder  record per-step obs + actions → SFT HDF5
        ▼
SFT HDF5  ─►  data/lerobot export  ─►  SFT
```

| Module | What it does |
|---|---|
| `replay_empty_from_dataset.py` | Load a captured task into a bare scene containing only the support surface + task objects; optionally re-render and re-save the staged snapshot. The other drivers consume its output. |
| `pick_and_place_from_dataset.py` | cuRobo-planned pick-and-place driven by a replay-empty dump; multi-candidate grasp search + transport, records SFT. |
| `lid_transport_from_dataset.py` | Lid-transport variant (close/seal then move); supports multi-seed batches and a success cap. |
| `pick_up_lid_from_dataset.py` | Pick a lid and place it onto a container; can seed the grasp from a prior dataset (`--phase-a-grasp-from-dataset`). |
| `_sft_recorder.py` | Shared SFT recorder — writes per-step observations/actions, and provides the canonical wrist-camera patch reused at eval time (`install_wrist_camera_patch`). |

## Run

```bash
conda activate behavior

# 1) stage a frozen task into an empty scene
python -m maniguard.data.curobo.replay_empty_from_dataset \
  --task-dir datasets/6fam-base-20260513/clutter_pickup/task_0000/base

# 2) cuRobo pick-and-place on that staged task, recording SFT + video
python -m maniguard.data.curobo.pick_and_place_from_dataset \
  --task-dir datasets/6fam-base-20260513/clutter_pickup/task_0000/base \
  --out-dir outputs/curobo_pnp --save-video
```

The drivers share flags like `--task-dir`, `--episode`, `--seed`,
`--max-candidates` (grasp search breadth), `--pick-timeout` /
`--transport-timeout`, `--max-reach`, and `--save-video`. Several can live-write
a LeRobot dataset directly (`--lerobot-repo-id` / `--lerobot-root`).

## Grasp collection

```bash
python -m maniguard.data.curobo.collect_grasps_in_scene \
  --scene-file <scene_ep1.json> --diagnostics-file <diagnostics.jsonl> \
  --target-name <obj> --output-dir outputs/grasps
```

`collect_grasps_in_scene.py` proposes grasps for a target object in the actual
benchmark scene (GraspGen over `--graspgen-host`, `--graspgen-num-grasps` /
`--graspgen-topk`) and filters them by reachability (`--max-reach`), writing the
`--num-target-grasps` best. These feed both scripted collection and the
[GraspGen / cuRobo grasp pipeline](../graspgen_pipeline.md).

## Interactive: ghost-gripper teleop

`gripper_target_teleop.py` is a hands-on, cuRobo-backed mode for **specifying
grasp poses by hand** — the manual alternative to the OBB grasp sampler when it
can't find a good candidate on a hard-to-grasp object.

It spawns a **translucent "ghost" gripper** (palm + fingers + assisted-grasp
zone) attached to an anchor cube. You:

1. **Drag the ghost** to the desired end-effector pose with Isaac's viewport
   gizmo (`W` translate / `E` rotate), previewing the grasp live.
2. Press **Enter** — cuRobo motion-plans from the current joint state to the
   ghost's pose and the real Franka follows the trajectory.
3. **Space** toggles the gripper (grasp / release); **Q** quits.

It can live-write a LeRobot dataset (`--lerobot-repo-id` / `--lerobot-root`) and
has a lid-transport mode (`--lid-at-edge`, `--lid-mass`).

```bash
DISPLAY=:1 python -m maniguard.data.curobo.gripper_target_teleop \
  --task-dir <task>/base --episode 1 --lerobot-repo-id maniguard/pnp_clicks
```

## Validation

`validate_joint_replay.py` open-loop replays the absolute-joint-action route of
a recorded HDF5 to confirm the saved joint trajectory reproduces the demo:

```bash
python -m maniguard.data.curobo.validate_joint_replay \
  --hdf5 <episode.hdf5> --config <env_config.yaml> --max-steps 500
```

## See also

- [Data collection overview](index.md) — where this fits in the pipeline.
- [GraspGen / cuRobo grasp pipeline](../graspgen_pipeline.md) — per-object grasp eval for RL resets.
- [SFT](../openpi_sim_teleop_sft.md) — exporting the recorded HDF5 to LeRobot v2.1.
