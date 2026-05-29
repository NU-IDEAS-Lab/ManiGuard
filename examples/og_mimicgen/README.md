# OmniGibson MimicGen Examples

This directory contains a small, self-contained example pipeline for
generating MimicGen-style demonstrations in OmniGibson / BEHAVIOR-1K.

The four stages are:

1. `01_teleop_spacemouse.py` records one or more source demonstrations to HDF5.
2. `02_annotate_src_demo.py` replays the source HDF5 and detects task signals.
3. `03_extract_waypoints.py` extracts object-frame end-effector waypoints.
4. `04_generate_demos.py` randomizes object poses and replays the waypoints.

## Example

```bash
cd /path/to/ManiGuard
conda activate behavior

python examples/og_mimicgen/01_teleop_spacemouse.py \
  --config /path/to/og_env_config.yaml \
  --output-hdf5 outputs/mimicgen_example/source_demo.hdf5

python examples/og_mimicgen/02_annotate_src_demo.py \
  --input-hdf5 outputs/mimicgen_example/source_demo.hdf5 \
  --output-dir outputs/mimicgen_example/annotations

python examples/og_mimicgen/03_extract_waypoints.py \
  --input-hdf5 outputs/mimicgen_example/source_demo.hdf5 \
  --annotations outputs/mimicgen_example/annotations/annotations.json \
  --output-hdf5 outputs/mimicgen_example/waypoints.hdf5 \
  --expected-sequence pick,place

python examples/og_mimicgen/04_generate_demos.py \
  --env-config outputs/mimicgen_example/annotations/env_config.json \
  --waypoints outputs/mimicgen_example/waypoints.hdf5 \
  --output-hdf5 outputs/mimicgen_example/generated_demos.hdf5 \
  --randomize-object object_0:0.05,0.05,0.0,0.17 \
  --randomize-object target_0:0.05,0.05,0.0,0.17 \
  --n-demos 10 \
  --max-attempts 50
```

## Notes

- The scripts use `omnigibson.envs.HDF5CollectionWrapper`, so the source and
  generated HDF5 files store simulator states, actions, metadata, and the
  environment config.
- Signal detection is heuristic: pick is gripper closing plus gripper contact
  with an object's root link; place is a transition into `OnTop` or `Inside`;
  open / close are detected from `Open` state changes.
- Waypoints are stored relative to the reference object pose at the source
  signal frame. During generation, they are transformed through the current
  randomized object pose.
- The generation script defaults to linear freespace interpolation and
  IK-delta actions. It intentionally avoids project-specific task classes.
