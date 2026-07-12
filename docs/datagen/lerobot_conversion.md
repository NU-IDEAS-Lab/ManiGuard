# Datagen — RAW → LeRobot v2.1 conversion

The last stage of the [datagen pipeline](pipeline.md): repackage the raw scripted demos
(`outputs/datagen/<dataset>/<family>/task_*/traj_*/`) into **one LeRobot v2.1 dataset per family**,
ready for SFT. It is a **pure offline file repackage** — no sim, no replay, no policy. Each demo
already stored its pixels as MP4s and its numbers as an HDF5, so conversion just reads those and
writes the LeRobot layout, with the **videos passed through (no re-encode)**.

`maniguard/data/datagen/{to_lerobot.py, to_lerobot_parallel.py, lerobot_merge.py, lerobot_diff.py, reader.py, data_format.py}`.
Family-agnostic — the same converter handles all 6 families.

---

## Environment

Runs in the **lerobot `uv` venv** (Python 3.11 / `lerobot` 0.3.3), **not** the `behavior` conda env.
The converter imports `lerobot`; the sim env does not have it, and the two dependency stacks
conflict. The pure helpers (`build_prompt_table`, `frame_rows`) import nothing heavy and are unit-
testable in any env; the `lerobot` / `h5py` imports inside `convert()` are lazy.

```bash
# from the repo root (so `python -m` puts `maniguard` on the shard subprocesses' path)
source .venv-lerobot/bin/activate      # or however the lerobot uv env is activated
```

---

## Input (RAW) → Output (LeRobot v2.1)

**Input** — one directory per kept demo (see [pipeline.md § Data layout](pipeline.md#data-layout--schema)):

```
outputs/datagen/<dataset>/<family>/task_<NNNN>/traj_<NNN>/
    image_opposite.mp4  image_left.mp4  image_right.mp4  image_left_shoulder.mp4  wrist_image.mp4
    traj.hdf5     state (N,8) · actions (N,8) · actions_commanded (N,8) · states (sim dump)
    meta.json     family / target_key / grasp_id / prompt / success / ...
```

`reader.py` (`iter_traj_dirs` / `load_meta` / `load_traj` / `read_frames`) is the **single** entry
point the converter uses to walk this layout — keep the two in sync.

**Output** — standard LeRobot v2.1, one dataset per family:

```
<out_root>/<family>/
    meta/{info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl}
    data/chunk-000/episode_000000.parquet ...        (chunks_size = 1000)
    videos/chunk-000/<image_key>/episode_000000.mp4 ...
```

Feature schema (`data_format.py`, 30 fps, 256²):

| feature | shape | meaning |
|---|---|---|
| `image_opposite` / `image_left` / `image_right` / `image_left_shoulder` / `wrist_image` | (256,256,3) uint8 | the 5 passthrough camera streams |
| `state` | (8,) f32 | `[arm_q(7), gripper]` (mean finger) |
| `actions` | (8,) f32 | **(b)** next-achieved absolute joint `[arm_q[t+1](7), gripper_cmd]` — DROID-style, robust to PD tracking error; the default SFT target |
| `actions_commanded` | (8,) f32 | **(a)** the cuRobo COMMANDED joint target `[target_q(7), gripper_cmd]` — recorded for free, kept as an alternative target |

**Prompt / task index.** `build_prompt_table(metas)` collects the distinct `meta["prompt"]` strings
in **first-seen order**; each episode's `task_index` points into that list (LeRobot `tasks.jsonl`).

---

## Serial converter — `to_lerobot.py`

```bash
python -m maniguard.data.datagen.to_lerobot \
    --dataset v1 --family clutter_pickup \
    --repo-id IDEAS-Lab-Northwestern/datagen-clutter-v1-joint-5cam
# --out-root default: outputs/datagen/<dataset>_lerobot_format   (writes <out_root>/<family>/)
# --limit N  : convert only the first N trajs (smoke)
```

`convert()` creates the dataset with `LeRobotDataset.create(...)`, then for each traj: reads the
numeric rows, **places the 5 MP4s directly** into `videos/.../episode_*.mp4`, and `add_frame` +
`save_episode`. The video passthrough is done by temporarily monkey-patching, inside the
`_passthrough_images()` context, `LeRobotDataset._save_image` to a no-op (so no PNGs are written)
and `sample_images` to be MP4-aware (stats sampled from the placed videos). A traj is **skipped**
(not fatal) if its videos are missing or the MP4 frame count ≠ the HDF5 row count.

`_frame_count` full-decodes an MP4 to count frames — this is the ~25 s/episode bottleneck that
motivates the parallel path below.

---

## Parallel converter — `to_lerobot_parallel.py` (recommended for a full family)

The serial converter is single-threaded and video-decode bound. On a many-core box, shard the family
**by task** and run the **UNMODIFIED serial converter** on each task in its own process, then merge:

```bash
python -m maniguard.data.datagen.to_lerobot_parallel \
    --dataset v1 --family dusty_transfer \
    --repo-id IDEAS-Lab-Northwestern/datagen-dusty-v1-joint-5cam \
    [--procs N]        # max concurrent shard procs (default: cpu-2)
    [--no-verify]      # skip the merged-dataset self-check (on by default)
```

Drop-in replacement for a serial run (same `--dataset/--family/--repo-id/--out-root`). Measured
**18.5×** (dusty: 243 min → 13 min, 2026-07-11).

**Why it is safe (byte-identical to serial).**
- **Per-episode data + videos**: each shard runs `to_lerobot.convert` UNCHANGED on one task, through
  a symlink view (`_shard_one`), so every episode's parquet + MP4 is byte-identical **by
  construction**.
- **The merge** (`lerobot_merge.merge_shards`): concatenates the per-task shards **in task order**
  and re-offsets only the **3 global columns** — `episode_index`, `index` (global frame counter),
  `task_index` (against the merged prompt table) — recomputes just those columns' stats in
  `episodes_stats.jsonl`, and rebuilds the 4 `meta/` files. This is the ONLY new logic, and it is
  proven byte-identical to a full serial run by `lerobot_diff`.
- **`--verify`** (default on) self-checks the merged dataset against the converter's OWN logic:
  prompt table match + per-episode index correctness + `LeRobotDataset` load + file counts.

`lerobot_diff.py` (`diff_datasets`) is the field-level checker used to prove the above: it compares
`info.json`, all 3 jsonl, every parquet column (exact), and every video's md5 between two datasets,
printing `VERDICT: IDENTICAL`.

---

## Publishing to Hugging Face

Publishing is a **separate** step from conversion, driven by the **`datagen-publish` skill**
(`.claude/skills/datagen-publish/`). Key points that bite:

- Push **PRIVATE by default**.
- You MUST `create_tag` the dataset's `codebase_version` (**`v2.1`**) after upload, or
  `LeRobotDataset` raises `RevisionNotFoundError` when someone loads it.
- HF ops (upload / tag / verify) run under the **`behavior` conda** python (has `huggingface_hub`);
  the conversion itself runs under the **lerobot uv** env. Don't mix them.
- Repo id convention: `IDEAS-Lab-Northwestern/datagen-<family-short>-v1-joint-5cam`.

Always keep a **local backup** of both the raw dataset and the converted LeRobot dataset
(e.g. `~/Desktop/maniguard-finalize-datagen/{v1,v1_lerobot_format}/<family>/`) before tearing down
any collection box.

---

## Files at a glance

| file | role |
|---|---|
| `reader.py` | walk the RAW layout (`iter_traj_dirs` / `load_meta` / `load_traj` / `read_frames`) |
| `data_format.py` | the single source of truth for FPS / resolution / camera keys / feature schema |
| `to_lerobot.py` | serial converter (`convert`) + pure helpers (`build_prompt_table`, `frame_rows`) + video passthrough |
| `to_lerobot_parallel.py` | shard-by-task → serial converter per task → merge → verify |
| `lerobot_merge.py` | `merge_shards`: concat shards, re-offset the 3 global columns, rebuild meta |
| `lerobot_diff.py` | `diff_datasets`: field-level byte comparison (the identity proof) |
