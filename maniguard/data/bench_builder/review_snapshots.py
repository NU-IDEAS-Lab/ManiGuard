"""Extract a per-task review frame for a ManiGuard-Bench family.

For visual QC it is far faster to eyeball one still per task than to scrub 35 short
videos. This tool walks the *currently-existing* tasks of a finalized family, grabs the
LAST frame of each task's "opposite" review video (the wide front view that shows the whole
scene layout), and writes it as ``<family>/snapshots/<task_id>.png`` — a ``snapshots/`` folder
sibling to the ``task_*`` folders. The reviewer then flips through ``snapshots/`` instead of
opening every video.

It is deliberately drop-agnostic: it simply processes the ``task_*`` folders that exist right now
(run it whenever you want to review), writing one PNG per existing task, named one-to-one by task id.

Usage:
  python -m maniguard.data.bench_builder.review_snapshots --family cabinet_pickup
  python -m maniguard.data.bench_builder.review_snapshots --family jar_transport --view opposite_side_front
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

OUT_ROOT_DEFAULT = "outputs/lerobot_datasets/maniguard-bench"
TASK_RE = re.compile(r"^task_\d+$")


def _last_frame(video_path: Path):
    """Decode the final frame of an mp4 as an HxWx3 uint8 RGB array (None on failure)."""
    import av

    last = None
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            last = frame
    if last is None:
        return None
    return last.to_ndarray(format="rgb24")


def _find_video(base_dir: Path, view: str, episode: int) -> Path | None:
    """Locate the task's review video for the requested view (exact name, then a loose glob)."""
    exact = base_dir / f"rollout_{view}_ep{episode}.mp4"
    if exact.exists():
        return exact
    hits = sorted(base_dir.glob(f"*{view}*_ep{episode}.mp4"))
    return hits[0] if hits else None


def extract_review_snapshots(
    family: str,
    out_root: str = OUT_ROOT_DEFAULT,
    *,
    episode: int = 1,
    view: str = "opposite_side_front",
    subdir: str = "base",
) -> dict:
    """Write one last-frame PNG per existing task into ``<family>/snapshots/`` (named by task id)."""
    import imageio.v2 as imageio

    fam_dir = Path(out_root) / family
    if not fam_dir.is_dir():
        raise FileNotFoundError(f"family dir not found: {fam_dir}")

    tasks = sorted(d.name for d in fam_dir.iterdir() if d.is_dir() and TASK_RE.match(d.name))
    if not tasks:
        raise FileNotFoundError(f"no task_* folders under {fam_dir}")

    snap_dir = fam_dir / "snapshots"
    snap_dir.mkdir(exist_ok=True)

    written, missing = [], []
    for task in tasks:
        video = _find_video(fam_dir / task / subdir, view, episode)
        if video is None:
            missing.append(task)
            print(f"  [skip] {task}: no '{view}' ep{episode} video", flush=True)
            continue
        frame = _last_frame(video)
        if frame is None:
            missing.append(task)
            print(f"  [skip] {task}: video decoded 0 frames", flush=True)
            continue
        out_png = snap_dir / f"{task}.png"
        imageio.imwrite(out_png, frame)
        written.append(task)
        print(f"  [ok]   {task} -> snapshots/{task}.png", flush=True)

    summary = {
        "family": family,
        "view": view,
        "n_tasks": len(tasks),
        "written": len(written),
        "missing_video": missing,
        "snapshots_dir": str(snap_dir),
    }
    print(
        f"=== {family}: {len(written)}/{len(tasks)} snapshots -> {snap_dir}"
        + (f"  (missing video: {missing})" if missing else ""),
        flush=True,
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract a per-task last-frame review PNG (opposite view) for a bench family."
    )
    ap.add_argument("--family", required=True)
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--view", default="opposite_side_front", help="review-video view stem to pull the frame from")
    ap.add_argument("--subdir", default="base", help="subdir under each task holding the videos")
    args = ap.parse_args()
    extract_review_snapshots(
        args.family, args.out_root, episode=args.episode, view=args.view, subdir=args.subdir
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
