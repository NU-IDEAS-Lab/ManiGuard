# Controller · data · action · eval — end to end

For SFT to transfer, **four choices must agree**: how the demo moved the arm, the
**state/action** the dataset stores, how the **model** consumes it, and the **eval
controller** that executes the policy. If they disagree, the motion realized at
eval diverges from what the policy saw in training even when the network is
"correct." ManiGuard removes most of this risk by standardizing the whole loop on
one convention.

## One convention: JointController, absolute joint

Collection, dataset, SFT, and eval are all **joint-space** — no end-effector / IK
anywhere:

- **Collected** via joint tracking — scripted datagen (cuRobo drives a
  `JointController`) or GELLO teleop (1:1 joint mirroring). SO-101 teleop drives
  the arm through IK, but its recordings are normalized to the same absolute-joint
  convention at the render stage
  ([playback](../data_collection/index.md#playback-render)).
- **Stored** as absolute joint (see [Dataset & config](dataset_and_config.md)):
  `state = [joint_0..6, gripper]`, `actions = [joint_0..6_target, gripper]`.
- **Trained**: model-agnostic. A model may re-encode internally — e.g. openpi
  converts the 7 arm joints to per-step deltas and reconstructs them to absolute
  at inference (`use_delta_joint_actions=True`) — but that is a per-model config
  detail, not a change to the dataset. See [openpi SFT](openpi.md).
- **Evaluated** with a `JointController`: the policy's (reconstructed) **absolute
  joint target** is fed straight to the controller.

## Consistency matrix

| Collected with | Dataset action | Eval controller |
|---|---|---|
| scripted datagen (cuRobo joint) | absolute joint | `joint_position` / `joint_position_impedance` |
| GELLO teleop (joint mirroring) | absolute joint | `joint_position` / `joint_position_impedance` |
| SO-101 teleop (IK deltas → joint at render) | absolute joint | `joint_position` / `joint_position_impedance` |

Set the controller on the [`EvalConfig`](../evaluation/index.md)
(`maniguard/eval/eval_config.py`); the eval loader overrides the scene-baked
controller with `controller_preset`. The policy emits absolute joint targets, so
`controller_preset: joint_position` (or `joint_position_impedance` for tighter
tracking) drives the arm directly.

!!! warning "Watch these"
    - **Soft tracking:** the default `JointController` `kp` (≈50) is too soft to
      reach a per-step joint target in one control step. Raise `joint_pos_kp`
      (use `joint_position_impedance`) so the realized motion matches training.
    - **`external_cam`:** the policy uses one overview + wrist; eval must read the
      overview choice back from the checkpoint's train config, or it sees an
      out-of-distribution viewpoint.

!!! note "EEF-native policies"
    ManiGuard datasets are absolute joint, and the pipeline is joint-space
    end-to-end. An EEF-native VLA can still be evaluated: serve it with its own
    data config and run through the `osc` controller, or map its EEF deltas to
    joint targets with the Jacobian-IK `ik_eef_to_joint` shim.

## Where each piece lives

| Concern | Code |
|---|---|
| Controller presets (`joint_position`, `joint_position_impedance`, `osc`, `ik`) | `maniguard/envs/frozen_task_runtime.py` → `CONTROLLER_PRESETS` |
| Recorded state/action (scripted datagen) | `maniguard/data/datagen/primitives/record.py` |
| RAW → LeRobot export | `maniguard/data/datagen/to_lerobot.py` (+ teleop `data/lerobot/multitask_lerobot_export.py`) |
| Per-model action encoding (e.g. openpi delta) | `maniguard/openpi_sft/data_configs.py` |
| Eval knobs (`controller_preset`, `joint_pos_kp`, `state_mode`, `action_dim`, `external_cam`) | `maniguard/eval/eval_config.py` |

## See also

- [Dataset & data-source configs](dataset_and_config.md) — producing the demos.
- [openpi SFT](openpi.md) — the concrete register → train → push recipe.
- [Environment layer](../foundations/env_layer.md) — the controller presets in detail.
