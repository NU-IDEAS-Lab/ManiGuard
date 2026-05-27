# Controller, data, action & eval — end to end

For SFT to transfer, **four choices must agree**: the data-collection dynamics,
the **state/action format** the policy learns, and the **eval controller** that
executes the policy. If they disagree, the motion realized at eval diverges from
what the policy saw in training even when the network is "correct." This page is
the single place that keeps them aligned.

## The common SFT schema

Every collection path records the same per-step observation **state** (8-D,
robot-base-relative) via `grab_state` (`maniguard/data/curobo/_sft_recorder.py`):

```
state = [ eef_pos(3) | eef_axisangle(3) | gripper_q(2) ]   # float32, base frame
```

Only the **action** differs between the two tracks below; the state is shared
(the absolute-joint track swaps in joint state at export time).

## Two action tracks

| | **Delta-EEF** (LIBERO / OpenPI pi0.5) | **Absolute-joint** |
|---|---|---|
| Policy action | 7-D: `dpos(3) + drot_axisangle(3) + gripper(1)`, **base frame** | 8-D: next-step `arm_q(7) + gripper(1)` |
| State | 8-D eef (above) | joint state (`arm_q` + gripper) |
| Export | `multitask_lerobot_export --eef-delta-actions` | `reexport_joint_actions` (uses `joint_actions`) |
| Eval controller | `osc`, **or** `joint_position_impedance` + `ik_eef_to_joint` | `joint_position` / `joint_position_impedance` |

## End to end

### 1. Pick the action space (policy-driven)

- A VLA pretrained on EEF deltas (pi0.5 / OpenPI, LIBERO schema) → **delta-EEF**.
- A joint-space policy, or when you want the eval path to reproduce the
  collected joint trajectory exactly → **absolute-joint**.

### 2. Collect demonstrations

Any method works; the recorder logs the common 8-D state plus the raw action of
whatever drove the arm:

| Method | Arm controller while collecting | Raw action logged |
|---|---|---|
| GELLO teleop | `JointController` (abs position) | absolute joint targets |
| SO-101 teleop | `InverseKinematicsController` (EE delta → IK) | EE delta |
| Scripted cuRobo | `joint_position_impedance` (Phase A), `osc` (Phase B) | joint targets / EEF delta |

`SFTRecorder` also offers `record_fk_step`, which derives an EEF-delta action
from consecutive base-relative eef poses (FK) regardless of the controller — so
joint-tracked data can still be exported as delta-EEF.

### 3. Export to LeRobot with the matching action

- **Delta-EEF:** `python -m maniguard.data.lerobot.multitask_lerobot_export --eef-delta-actions`
  → 7-D `[dpos, drot_axisangle, gripper]`, computed from consecutive eef-pose
  states (base-frame rotation delta `R_{t+1}·R_tᵀ`); state stays 8-D eef.
- **Absolute-joint:** `python -m maniguard.data.lerobot.reexport_joint_actions`
  → swaps in 8-D joint state/action (`arm_q[t+1]`, gripper).
- Then `python -m maniguard.data.lerobot.norm_stats` for openpi-format
  normalization. (See [SFT (sim)](../openpi_sim_teleop_sft.md).)

### 4. Eval with the matching controller — the part that bites

Set these on the [`EvalConfig`](../one_machine_pro6000_eval.md)
(`maniguard/eval/eval_config.py`); the eval loader overrides the scene-baked
controller with `controller_preset`:

- **Delta-EEF → OSC** (`controller_preset: osc`): the policy's 6-D EEF delta
  drives an `OperationalSpaceController` directly. Simplest, but the *realized
  joint path* can differ from how the data was generated.
- **Delta-EEF → joint (recommended when data came from joint tracking)**:
  `controller_preset: joint_position_impedance` **+ `ik_eef_to_joint: true`**.
  Eval converts the policy's 6-D EEF delta to an absolute joint target via a
  Jacobian-IK step, then a `JointController` tracks it — reproducing the
  cuRobo/GELLO joint-position dynamics the SFT data was generated under, so the
  realized path follows training instead of diverging. Raise `joint_pos_kp` so
  the controller actually reaches each per-step target.
- **Absolute-joint → joint** (`controller_preset: joint_position`): the policy
  emits joint targets, fed straight to a `JointController`.

## Consistency matrix

Read it as: *how the data moved the arm* → *what the policy learns* → *how to
drive the policy at eval*.

| Collected with | Train action | Eval controller |
|---|---|---|
| cuRobo / GELLO joint tracking | delta-EEF | `joint_position_impedance` + `ik_eef_to_joint` |
| cuRobo / GELLO joint tracking | absolute-joint | `joint_position(_impedance)` |
| OSC / SO-101 EEF-delta | delta-EEF | `osc` |

!!! warning "Common mismatches"
    - **Frame:** the exported rotation delta is in the **robot base frame**
      (poses come from `get_relative_eef_*`). The eval controller and any policy
      preprocessing must use the same frame.
    - **Soft tracking:** the default `JointController` `kp` (≈50) is too soft to
      reach a per-step joint target in one control step → realized EEF ≠
      commanded delta. Bump `joint_pos_kp`.
    - **OSC vs joint:** evaluating a joint-tracked-data policy through `osc` can
      diverge; prefer the `ik_eef_to_joint` path for those datasets.

## Where each piece lives

| Concern | Code |
|---|---|
| Controller presets (`joint_position`, `joint_position_impedance`, `osc`, `ik`) | `maniguard/envs/frozen_task_runtime.py` → `CONTROLLER_PRESETS` |
| Recorded state/action (`grab_state`, `record_step`, `record_fk_step`) | `maniguard/data/curobo/_sft_recorder.py` |
| Action export (delta-EEF vs joint) + norm stats | `maniguard/data/lerobot/{multitask_lerobot_export,reexport_joint_actions,joint_actions,norm_stats}.py` |
| Eval knobs (`controller_preset`, `ik_eef_to_joint`, `joint_pos_kp`, `state_mode`, `action_dim`) | `maniguard/eval/eval_config.py` |

## See also

- [Data collection](../data_collection/index.md) — producing the demos.
- [OpenPI sim teleop SFT](../openpi_sim_teleop_sft.md) — the concrete export → train recipe.
- [Environment layer](../foundations/env_layer.md) — the controller presets in detail.
