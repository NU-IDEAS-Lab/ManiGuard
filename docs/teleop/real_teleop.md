# Real-robot teleop → LeRobot

Converts **real Franka teleop captures** (`.npz`, one per episode under
`outputs/real_teleop/`) into a LeRobot v2.1 SFT dataset. This is the only
real-robot path in the pipeline — everything else is collected in simulation.
Two converters cover the two target conventions (`maniguard/data/real_teleop/`):

## Direct → DROID (joint) — `real_teleop_to_droid`

Real npz → LeRobot v2.1 in the **DROID joint** convention (openpi's DROID
pretrained convention is joint-space, so this stays consistent with the sim
tracks):

```bash
.venv-lerobot/bin/python -m maniguard.data.real_teleop.real_teleop_to_droid \
  --input-dir outputs/real_teleop \
  --repo-id <org>/<task> --prompt "<instruction>" \
  --root outputs/lerobot_datasets/<org>/<task> \
  --push-to-hub <org>/<task> --hub-private
```

It assembles 8-D state `[joint_position(7), gripper]` + 8-D action
`[joint_velocity(7), gripper[t+1]]`, decodes / crops / resizes the cameras, and
(via `--push-to-hub`) creates the required v2.1 tag. fps 15; the DROID schema
keeps the state columns separate rather than a single `state` column.

## Via sim-compatible HDF5 — `real_teleop_to_hdf5`

For the eef-convention path, first emit an HDF5 that matches the sim teleop
Stage-2 input schema, then reuse the shared export:

```bash
python -m maniguard.data.real_teleop.real_teleop_to_hdf5 \
  --input-dir outputs/real_teleop --output-dir outputs/real_rendered --img-size 256
```

Each episode becomes `state` = `eef_pos(3) + axisangle(3) + gripper(2)` (8-D) and
`action` = `dpos(3) + drot_axisangle(3) + gripper(1)` (7-D), with `image` + `wrist_image`.
That HDF5 then goes through the same **Stage 2** export as sim teleop — see
[Sim teleop → LeRobot](teleop_to_lerobot.md).

The resulting datasets use the same LeRobot v2.1 conventions as the sim sources —
see [SFT dataset & data-source configs](../sft/dataset_and_config.md).
