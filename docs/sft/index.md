# Supervised fine-tuning (SFT)

ManiGuard fine-tunes vision-language-action (VLA) policies on manipulation
demonstrations collected in OmniGibson (and on real-robot teleop). The guiding
principle here is **one dataset, any model**:

> A single **joint-controller** LeRobot v2.1 dataset is the source of truth. It is
> **foundation-model-agnostic** — the same dataset feeds openpi (pi0.5), GR00T,
> SmolVLA, or any other VLA. Each model only differs in *its own config* (how it
> maps the shared cameras/state/action into its expected slots), not in the data.

Everything in this project standardizes on the **JointController** convention
(post-June "正本清源" refactor): collection, dataset, SFT, and eval are all
joint-space — no end-effector / IK anywhere in the loop.

## The shared dataset (model-agnostic)

| field | shape | meaning |
|---|---|---|
| `state` | (8,) f32 | `[joint_0..6, gripper]` — absolute joint config |
| `actions` | (8,) f32 | `[joint_0..6_target, gripper_cmd]` — absolute joint target + binary gripper |
| `image_*` | 256×256×3 | third-person + wrist camera streams (video, passthrough) |

Details, LeRobot v2.1 conventions, and how each data source is produced:
**[Dataset & data-source configs](dataset_and_config.md)**.

## Data-source variants

All three produce the same joint schema above; they differ only in *how* the
demos are generated (and therefore what config declares the source):

| source | how | dataset naming | status |
|---|---|---|---|
| **Scripted datagen** (primary) | the mature 6-family pipeline → `data/datagen` → `to_lerobot` | `datagen-<fam>-v1-joint-5cam` | current main source |
| **Sim teleop** | GELLO/SO-101 teleop → `data/playback` → `multitask_lerobot_export` | `sim-<fam>-30-joint-3cam` | supported |
| **Real teleop** | real Franka teleop npz → `real_teleop_to_droid` | `<task>` (DROID joint) | supported |

## Per-model SFT recipes

| model | page | notes |
|---|---|---|
| **openpi / pi0.5** | [openpi SFT](openpi.md) | LoRA via openpi's JAX trainer through `maniguard/openpi_sft`; the reference recipe |
| **GR00T (N1.6)** | [GR00T SFT](gr00t.md) | NEW_EMBODIMENT joint-space, PyTorch/HF Trainer, component-freeze |
| **SmolVLA** | [SmolVLA SFT](smolvla.md) | LeRobot-native (`lerobot-train`), freeze-VLM + train-expert |

## Keeping collection, training, and eval consistent

The one thing that most often breaks SFT transfer is a mismatch between how the
data moved the arm, the action the policy learns, and the controller that
executes the policy at eval. See
**[Controller · data · action · eval — end to end](end_to_end.md)**.
