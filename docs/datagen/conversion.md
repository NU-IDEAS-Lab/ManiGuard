# Review & conversion (steps 5–6)

Both steps are offline — no sim: eyeball the collected demos, then repackage them into
LeRobot v2.1 for SFT.

## 5. Review montage

`review.py` tiles every demo of a task into one MP4 for efficient visual review.

```bash
python -m maniguard.data.datagen.review --dataset v1 --family clutter_pickup --all   # one MP4 per task
python -m maniguard.data.datagen.review --dataset v1 --family clutter_pickup --task task_0000 --wrist
# → outputs/datagen/v1/clutter_pickup/<task>/_review_ls.mp4  (or _review_lsw[_pN].mp4)
```

??? note "▸ Montage layout"
    Default = **`left_shoulder` only, 10 cells/row** (50 demos → 5 rows, one file); add
    wrist with `--wrist` (5 `left_shoulder|wrist` pairs/row). Native 256² pixels; size is
    controlled by frame rate (`--stride 3` = 10 fps). All demos play in lockstep; a shorter
    clip freezes on its last frame. Streaming decode → ~10 MB memory even at full res.

## 6. RAW → LeRobot conversion

Repackages the raw demos into **one LeRobot v2.1 dataset per family**, ready for SFT.
It is a **pure offline file repackage** — no sim, no replay: each demo already stored
pixels as MP4s + numbers as an HDF5, so conversion reads those and writes the LeRobot
layout with the **videos passed through (no re-encode)**.

!!! note "Runs in the `lerobot` env, not `behavior`"
    Step 6 imports `lerobot` (Python 3.11 / `lerobot` 0.3.3 `uv` venv); the two dependency
    stacks conflict, so it does **not** run in the `behavior` conda env.

```bash
# serial (single task / smoke)
python -m maniguard.data.datagen.to_lerobot \
    --dataset v1 --family clutter_pickup --repo-id <org>/datagen-clutter-v1-joint-5cam

# parallel — recommended for a full family (~18× faster; shard-by-task → merge → verify)
python -m maniguard.data.datagen.to_lerobot_parallel \
    --dataset v1 --family dusty_transfer --repo-id <org>/datagen-dusty-v1-joint-5cam [--procs N]
```

??? info "▸ Input → output layout & feature schema"
    **Input** — one directory per kept demo (see the RAW layout below).

    **Output** — standard LeRobot v2.1, one dataset per family (`meta/`, `data/chunk-000/…parquet`,
    `videos/chunk-000/<key>/…mp4`). Feature schema (`data_format.py`, 30 fps, 256²):

    | feature | shape | meaning |
    |---|---|---|
    | 5 camera streams (`image_*`, `wrist_image`) | (256,256,3) uint8 | passthrough MP4 |
    | `state` | (8,) f32 | `[arm_q(7), gripper]` |
    | `actions` | (8,) f32 | next-achieved absolute joint — the default SFT target |
    | `actions_commanded` | (8,) f32 | the cuRobo commanded joint target — alternative target |

    `reader.py` (`iter_traj_dirs` / `load_meta` / `load_traj` / `read_frames`) is the single
    entry point the converter uses to walk the RAW layout — keep the two in sync.
    `build_prompt_table(metas)` collects distinct prompts in first-seen order; each episode's
    `task_index` points into that list.

??? note "▸ Why the parallel path is byte-identical to serial"
    Each shard runs the **unchanged** serial `convert` on one task through a symlink view, so
    every episode's parquet + MP4 is byte-identical by construction. The only new logic is the
    merge (`lerobot_merge.merge_shards`): concatenate shards in task order, re-offset the **3
    global columns** (`episode_index`, `index`, `task_index`), recompute just those stats, and
    rebuild the 4 `meta/` files. `lerobot_diff.diff_datasets` proves `VERDICT: IDENTICAL`
    field-by-field (info.json, all jsonl, every parquet column, every video md5); the
    merged-dataset self-check runs by default (disable with `--no-verify`).

??? warning "▸ Publishing to Hugging Face (separate step)"
    - Push **PRIVATE by default**.
    - You MUST `create_tag` the dataset's `codebase_version` (**`v2.1`**) after upload, or
      `LeRobotDataset` raises `RevisionNotFoundError` on load.
    - HF ops (upload / tag / verify) run under **`behavior` conda** python (has
      `huggingface_hub`); the conversion runs under the **lerobot uv** env. Don't mix them.
    - Repo id: `<org>/datagen-<family-short>-v1-joint-5cam`.

??? note "▸ Files at a glance"
    | file | role |
    |---|---|
    | `reader.py` | walk the RAW layout |
    | `data_format.py` | single source of truth for FPS / resolution / camera keys / schema |
    | `to_lerobot.py` | serial converter + pure helpers + video passthrough |
    | `to_lerobot_parallel.py` | shard-by-task → serial per task → merge → verify |
    | `lerobot_merge.py` | `merge_shards`: concat, re-offset the 3 global columns, rebuild meta |
    | `lerobot_diff.py` | `diff_datasets`: field-level byte comparison (the identity proof) |

## Data layout & schema (RAW)

The on-disk layout every step reads/writes (step 6 repackages it into LeRobot):

```
outputs/datagen/<dataset>/<bench_family>/<task>/traj_<NNN>/
    image_opposite.mp4  image_left.mp4  image_right.mp4  image_left_shoulder.mp4  wrist_image.mp4
    traj.hdf5      state (N,8)=[arm_q(7),gripper] · actions (N,8)=[arm_q[t+1],gripper_cmd]
                   actions_commanded (N,8)=[cuRobo cmd] · states (N,*)=sim-state dump (MimicGen hook)
    meta.json      family/source_task/target_key/grasp_id/approach/draw/standoff_m/…/success/ltl_violated/prompt
    _summary.json  (per task) n_success / n_attempts / elapsed_s
```

`<bench_family>` matches the bench dataset dir name; `traj_NNN` is sequential over **kept**
demos only (gap-free). Each demo ≈ 1.1 MB (mostly the 5 videos). Naming and resolution
(256², 30 fps, h264/yuv420p) are byte-for-byte consistent with the bench rollout videos.

**Next →** [Families & gotchas](families.md)
