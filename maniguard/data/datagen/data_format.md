# ManiGuard datagen — dataset format

Schema source of truth: `maniguard/data/datagen/data_format.py`. Design rationale +
locked decisions: the Obsidian `ManiGuard 6fam_Data_Collection_TODO_Checklist.md` §5.

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
- **Why default action = (b) achieved**: the env's cuRobo is an old/rough StanfordVL
  fork (v0.7.0); achieved joints are robust to its rough commands + self-consistent
  with the recorded images + match the legacy `...-clutter-joint` data. (a) commanded
  is kept as an extra column to switch/compare without re-collecting.

## Downstream SFT/eval camera selection

The dataset always ships all five streams. Exactly one third-person view is fed to
the policy, chosen by the data config's **`external_cam ∈ {opposite, left, right,
left_shoulder}`** (extended in `openpi_sft/data_configs.py`). It is routed to
`observation/image_left` → pi0.5 `base_0_rgb`; `wrist_image` → `left_wrist_0_rgb`;
`right_wrist_0_rgb` is masked off (single-arm pi0.5 2-cam layout). Per family the
best-quality view is reviewed + set in the config; record/SFT/eval stay consistent.

## MimicGen sidecar (per episode, NOT in the LeRobot parquet)

Kept now so the future MimicGen amplification layer (doc §8) needs no re-collection:
- `states` — `og.sim.dump_state(serialized=True)` per step (replay).
- `datagen_info/{eef_pose, object_poses/<obj>, gripper_action, subtask_term_signals/<sig>}`
  — object-centric per-step info + subtask-termination flags.
