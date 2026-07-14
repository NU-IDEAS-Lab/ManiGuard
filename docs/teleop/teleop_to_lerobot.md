# Sim teleop → LeRobot

Turns raw sim-teleop captures (SO-101 / GELLO → the OmniGibson Franka) into a
LeRobot v2.1 dataset for SFT. Two stages, both driven by the template
`scripts/render_teleop_to_lerobot.sh` (edit only its CONFIG block per family):

```bash
conda activate behavior
bash scripts/render_teleop_to_lerobot.sh            # both stages
bash scripts/render_teleop_to_lerobot.sh --stage1   # re-render raw teleop → joint+3cam HDF5
bash scripts/render_teleop_to_lerobot.sh --stage2   # rendered HDF5 → LeRobot v2.1 (local build)
```

- **Stage 1 — render** — `maniguard.data.playback --input <raw> --output <rendered>`
  (defaults `--controller joint --cams 3`). Replays the raw teleop HDF5 with
  physics and records 8-D joint state/action + `image_left` / `image_right` /
  `wrist_image` at 256×256. Resume-safe; the `og.clear()` teardown segfault
  *after* a complete write is expected and harmless (success = a non-empty
  `action` dataset, not the exit code). See [Playback / render](playback.md).
- **Stage 2 — export** — `maniguard.data.lerobot.multitask_lerobot_export`
  discovers `task_*_traj_*.hdf5`, looks up each task's prompt from
  `<diag-root>/<task>/base/diagnostics.jsonl`, and writes one multitask dataset
  with per-frame `task_index`. The schema is auto-detected from the playback
  fingerprint (no schema flags). The template builds locally (no push).
- **Push** (separate, explicit) — do **not** re-run the exporter with
  `--push-to-hub` on an already-built dataset (`LeRobotDataset.create()` aborts
  with `FileExistsError`). Push the local dataset directly with
  `LeRobotDataset(...).push_to_hub(tag_version=True, push_videos=True, private=True)`.

Naming: `<org>/sim-<fam>-30-joint-3cam` (e.g. `<org>/sim-dusty-transfer-30-joint-3cam`).
The resulting dataset uses the same absolute-joint LeRobot schema as the other
sources — see [SFT dataset & data-source configs](../sft/dataset_and_config.md).
