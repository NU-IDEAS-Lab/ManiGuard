# ManiGuard datagen — dataset format

Schema source of truth: `maniguard/data/datagen/data_format.py`.

The cuRobo-collected SFT data is **joint-native**: the env's cuRobo emits a joint
trajectory → JointController execution → record joints directly (no eef↔joint
conversion, no sim-state reverse-engineering).

## Per-timestep record

| field | shape | dtype | meaning |
|---|---|---|---|
| `state` | (8,) | f32 | `[arm_q(7), gripper(1, mean finger)]` — current joints |
| `actions` | (8,) | f32 | `[arm_q[t+1](7), gripper_cmd(1, binary)]` — **DEFAULT = (b) next-achieved** absolute joint |
| `actions_commanded` | (8,) | f32 | `[curobo_target_q(7), gripper_cmd(1)]` — **(a) commanded** cuRobo target (commanded targets) |
| `image_opposite` | 256×256×3 | video | third-person (bench `cam_opposite`) |
| `image_left` | 256×256×3 | video | third-person (bench `cam_left`) |
| `image_right` | 256×256×3 | video | third-person (bench `cam_right`) |
| `image_left_shoulder` | 256×256×3 | video | third-person (bench `cam_left_shoulder`) |
| `wrist_image` | 256×256×3 | video | wrist (injected under `panda_hand`) |

- Dataset = **LeRobot v2.1**, `robot_type=FrankaPanda`, `fps=30`.
- The four third-person cameras come from the **shared bench `camera_setup`**
  (`maniguard/utils/camera_setup.py`) → record / SFT / eval are camera-consistent.
- Default `actions` pair the next recorded arm joints with the current recorded
  gripper command. The last row repeats the final arm position.
  `actions_commanded` retains the targets submitted for each recorded step;
  both columns use absolute joint positions.

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
  image_opposite.mp4        256x256 h264 yuv420p 30fps (default settings)
  image_left.mp4
  image_right.mp4
  image_left_shoulder.mp4
  wrist_image.mp4
  traj.hdf5                 state(N,8) + actions(N,8) + actions_commanded(N,8)
                            + datagen_info/gripper_action(N,)
                            + optional states(N,*), states_len(N,) for padded states
                            + attrs: prompt, n_steps, fps, resolution, <task meta>
  meta.json                 prompt, success, n_steps, fps, resolution, video_keys, <attrs>
```

The recorder uses PyAV H.264/yuv420p at each sensor's actual frame size
(default 256², 30 fps). Shared camera definitions do not imply identical
video contents or action frequencies between collection and evaluation.
`traj.hdf5` stores the joint trajectory; serialized simulator states are optional.

### Stage 2 — LeRobot v2.1 (separate converter, only when ready for SFT)

Run the converter in the separate `lerobot==0.3.3` environment installed with
`pip install -e '.[conversion]'`. It writes the v2.1 dataset format.

A converter reads N raw folders → one LeRobot dataset using `lerobot_features()`
below: the 5 MP4s pass straight through (no re-encode), the `state` / `actions` /
`actions_commanded` columns come from each `traj.hdf5`.

## Auxiliary replay fields (RAW HDF5 only)

The RAW HDF5 file stores auxiliary replay data separately from LeRobot Parquet:
- `states` — `og.sim.dump_state(serialized=True)` per recorded step, only when
  `record_sim_states=True`. When lengths vary, rows are padded to the maximum.
- `states_len` — original lengths, present only with padded `states` rows.
- `datagen_info/gripper_action` — per-step binary gripper command. (Object-centric
  `eef_pose` / `object_poses` / `subtask_term_signals` are derivable later from the
  sim states + the family skeleton; not written at collection time.)
