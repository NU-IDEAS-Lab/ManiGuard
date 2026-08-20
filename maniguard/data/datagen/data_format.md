# ManiGuard datagen — dataset format

Schema source of truth: `maniguard/data/datagen/data_format.py`.

The cuRobo-collected SFT data is **joint-native**: the env's cuRobo emits a joint
trajectory → JointController execution → record joints directly (no eef↔joint
conversion, no sim-state reverse-engineering).

## Per-timestep record

| field | shape | dtype | meaning |
|---|---|---|---|
| `state` | (8,) | f32 | `[arm_q(7), gripper(1, mean finger)]` — current joints |
| `actions` | (8,) | f32 | `[arm_q[t+1](7), gripper_cmd(1, binary)]` — **DEFAULT = (b) next-achieved** absolute joint (DROID-style) |
| `actions_commanded` | (8,) | f32 | `[curobo_target_q(7), gripper_cmd(1)]` — **(a) commanded** cuRobo target (extra, free, for compare/future) |
| `image_opposite` | 256×256×3 | video | third-person (bench `cam_opposite`) |
| `image_left` | 256×256×3 | video | third-person (bench `cam_left`) |
| `image_right` | 256×256×3 | video | third-person (bench `cam_right`) |
| `image_left_shoulder` | 256×256×3 | video | third-person (bench `cam_left_shoulder`) |
| `wrist_image` | 256×256×3 | video | wrist (injected under `panda_hand`) |

- Dataset = **LeRobot v2.1**, `robot_type=FrankaPanda`, `fps=30`.
- The four third-person cameras come from the **shared bench `camera_setup`**
  (`maniguard/utils/camera_setup.py`) → record / SFT / eval are camera-consistent.
- **Why default action = (b) achieved**: the env's vendored StanfordVL cuRobo (v0.7.0)
  emits rough commands; achieved joints are robust to them + self-consistent
  with the recorded images + match the legacy `...-clutter-joint` data. (a) commanded
  is kept as an extra column to switch/compare without re-collecting.

## Downstream SFT/eval camera selection

The dataset always ships all five streams. Exactly one third-person view is fed to
the policy, chosen by the data config's **`external_cam ∈ {opposite, left, right,
left_shoulder}`** (extended in `openpi_sft/data_configs.py`). It is routed to
`observation/image_left` → pi0.5 `base_0_rgb`; `wrist_image` → `left_wrist_0_rgb`;
`right_wrist_0_rgb` is masked off (single-arm pi0.5 2-cam layout). Per family the
best-quality view is reviewed + set in the config; record/SFT/eval stay consistent.

## Two stages: RAW collection → LeRobot conversion

Collection writes a **reviewable RAW form first** (so the curobo trajectories can be
eyeballed before any SFT conversion); LeRobot v2.1 is produced by a **separate
downstream converter** (MP4 passthrough — no re-encode).

### Stage 1 — RAW trajectory folder (per trajectory, written by `record.Recorder`)

```
<out>/<family>/task_NNNN/<variant>/
  image_opposite.mp4        256x256 h264 yuv420p 30fps  (bench rollout spec, byte-for-byte)
  image_left.mp4
  image_right.mp4
  image_left_shoulder.mp4
  wrist_image.mp4
  traj.hdf5                 state(N,8) + actions(N,8) + actions_commanded(N,8)
                            + states(N,*) [sim dumps] + datagen_info/gripper_action(N,)
                            + attrs: prompt, n_steps, fps, resolution, <task meta>
  meta.json                 prompt, success, n_steps, fps, resolution, video_keys, <attrs>
```

The five MP4s match the bench `rollout_*` videos exactly — same PyAV `h264` /
`yuv420p` encode at the camera's native render size (256² @ 30 fps). Review = open
the videos. `traj.hdf5` carries the joint trajectory + the MimicGen sim-state dump.

### Stage 2 — LeRobot v2.1 (separate converter, only when ready for SFT)

A converter reads N raw folders → one LeRobot dataset using `lerobot_features()`
below: the 5 MP4s pass straight through (no re-encode), the `state` / `actions` /
`actions_commanded` columns come from each `traj.hdf5`.

## MimicGen hook (inside `traj.hdf5`, NOT consumed by LeRobot)

Kept now so the future MimicGen amplification layer (doc §8) needs no re-collection:
- `states` — `og.sim.dump_state(serialized=True)` per step (replay).
- `datagen_info/gripper_action` — per-step binary gripper command. (Object-centric
  `eef_pose` / `object_poses` / `subtask_term_signals` are derivable later from the
  sim states + the family skeleton; not written at collection time.)
