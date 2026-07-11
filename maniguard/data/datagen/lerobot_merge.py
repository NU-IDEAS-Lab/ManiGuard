"""Merge per-task LeRobot shard datasets into ONE, replicating the serial ``to_lerobot`` converter's
global indexing / prompt-table / meta EXACTLY.

Each shard is a LeRobot dataset produced by the UNMODIFIED serial converter (``to_lerobot.convert``)
run on a SINGLE task, so its per-episode parquet data columns + videos are already byte-identical to
what a full serial run would emit for that task. Merging therefore only has to:

  * concatenate episodes in task order (== ``reader.iter_traj_dirs`` order),
  * re-offset the 3 global columns ``episode_index`` / ``index`` / ``task_index`` and recompute
    THEIR per-episode stats (the image / state / action / timestamp / frame_index stats are
    deterministic and carry over unchanged),
  * rebuild the 4 meta files (``episodes`` / ``episodes_stats`` / ``tasks`` / ``info``).

Proven byte-identical to a full serial ``to_lerobot`` run by field-level diff on a 2-task subset and
on a shared-prompt task pair (dusty, 2026-07-11). See ``lerobot_diff.diff_datasets``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CHUNK = 1000


def _set_col(table, name, arr):
    i = table.schema.get_field_index(name)
    return table.set_column(i, name, pa.array(arr, table.column(name).type))


def _col_stats(a):
    """Recompute a scalar column's stats the way lerobot does: numpy over ALL frames, population
    std; min/max keep native (int) type, mean/std float, count = n frames."""
    a = np.asarray(a)
    return {
        "min": [a.min().item()],
        "max": [a.max().item()],
        "mean": [float(a.mean())],
        "std": [float(a.std())],
        "count": [int(a.shape[0])],
    }


def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def merge_shards(out, shard_dirs, *, chunk: int = CHUNK) -> dict:
    """Merge ``shard_dirs`` (LeRobot dataset roots, one per task, IN SORTED TASK ORDER == the order
    ``reader.iter_traj_dirs`` yields tasks) into a single LeRobot dataset at ``out``. Non-destructive
    to the shards. Returns ``{episodes, frames, tasks}``."""
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)

    info = json.load(open(Path(shard_dirs[0]) / "meta" / "info.json"))
    vkeys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

    global_prompts: list[str] = []
    pidx: dict[str, int] = {}
    ep_rows, stat_rows = [], []
    g_ep = g_frame = 0

    for sh in shard_dirs:
        sh = Path(sh)
        eps = {}
        for line in open(sh / "meta" / "episodes.jsonl"):
            d = json.loads(line)
            eps[d["episode_index"]] = d
        stats = {}
        for line in open(sh / "meta" / "episodes_stats.jsonl"):
            d = json.loads(line)
            stats[d["episode_index"]] = d

        for loc in sorted(eps):
            ep = eps[loc]
            prompt = ep["tasks"][0]
            length = ep["length"]
            if prompt not in pidx:
                pidx[prompt] = len(global_prompts)
                global_prompts.append(prompt)
            gti = pidx[prompt]
            new_ep = g_ep
            ch = new_ep // chunk

            t = pq.read_table(sh / "data" / f"chunk-{loc // chunk:03d}" / f"episode_{loc:06d}.parquet")
            n = t.num_rows
            fi = t.column("frame_index").to_numpy()
            epi_arr = np.full(n, new_ep)
            idx_arr = g_frame + fi
            tsk_arr = np.full(n, gti)
            t = _set_col(t, "episode_index", epi_arr)
            t = _set_col(t, "index", idx_arr)
            t = _set_col(t, "task_index", tsk_arr)
            dst = out / "data" / f"chunk-{ch:03d}" / f"episode_{new_ep:06d}.parquet"
            dst.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(t, dst)

            for k in vkeys:
                src = sh / "videos" / f"chunk-{loc // chunk:03d}" / k / f"episode_{loc:06d}.mp4"
                dv = out / "videos" / f"chunk-{ch:03d}" / k / f"episode_{new_ep:06d}.mp4"
                dv.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dv)

            r = dict(ep)
            r["episode_index"] = new_ep
            ep_rows.append(r)
            st = dict(stats[loc])
            st["episode_index"] = new_ep
            ss = dict(st["stats"])
            ss["episode_index"] = _col_stats(epi_arr)
            ss["index"] = _col_stats(idx_arr)
            ss["task_index"] = _col_stats(tsk_arr)
            st["stats"] = ss
            stat_rows.append(st)

            g_ep += 1
            g_frame += length

    _write_jsonl(out / "meta" / "episodes.jsonl", ep_rows)
    _write_jsonl(out / "meta" / "episodes_stats.jsonl", stat_rows)
    _write_jsonl(out / "meta" / "tasks.jsonl", [{"task_index": i, "task": p} for i, p in enumerate(global_prompts)])
    info["total_episodes"] = g_ep
    info["total_frames"] = g_frame
    info["total_tasks"] = len(global_prompts)
    info["total_videos"] = g_ep * len(vkeys)
    info["total_chunks"] = (g_ep - 1) // chunk + 1
    info["splits"] = {"train": f"0:{g_ep}"}
    json.dump(info, open(out / "meta" / "info.json", "w"), indent=4)
    return {"episodes": g_ep, "frames": g_frame, "tasks": len(global_prompts)}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Merge per-task LeRobot shards into one dataset.")
    ap.add_argument("out", help="output dataset root")
    ap.add_argument("shards", nargs="+", help="shard dataset roots, IN SORTED TASK ORDER")
    a = ap.parse_args()
    print("MERGED", merge_shards(a.out, a.shards))
