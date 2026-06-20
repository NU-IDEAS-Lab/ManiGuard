"""Per-task review montage — tile every demo of a task into one MP4 for efficient review.

Each trajectory contributes a side-by-side pair ``[left_shoulder | wrist]`` (a "group"),
arranged GROUPS_PER_ROW groups per row. Cells keep their native 256² pixels (the source is
already low-res — no spatial downscale); size is controlled by lowering the frame rate
(``--stride``). All cells play in lockstep; a shorter clip FREEZES on its last frame until the
longest in the montage finishes (then the whole MP4 loops in a player).

Labels are minimal: each group is tagged with its trajectory index (top-left); the whole
montage is titled with the task. One MP4 per task, auto-split into ``_p0/_p1`` if a task has
more than ``MAX_ROWS`` rows of demos. Streaming (parallel-decode) so memory stays ~10 MB even
at full 256² resolution.

  conda activate behavior
  PYTHONPATH=$HOME/project/ManiGuard python -m maniguard.data.datagen.review \
      --dataset scale_test --family clutter --task task_0000      # one task
  ... --dataset scale_test --family clutter --all                  # every task in the family
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from maniguard.data.datagen import reader
from maniguard.data.datagen.primitives.record import _close_video, _open_video, _write_frame

CELL = 256                  # native cell px (no downscale — source is already low-res)
CELLS_PER_ROW = 10          # always 10 cells/row (5 groups with wrist, 10 without) — tidy
MAX_ROWS = 5                # rows per file before splitting into _p0/_p1
TITLE_H = 26
LABEL_PAD = (4, 2)
LSHOULDER = "image_left_shoulder.mp4"
WRIST = "wrist_image.mp4"


class _Cell:
    """One video stream decoded lazily; freezes on its last frame once exhausted."""

    def __init__(self, path):
        import av
        self.cont = None
        self.it = iter(())
        if path and Path(path).exists():
            try:
                self.cont = av.open(str(path))
                self.it = self.cont.decode(video=0)
            except Exception:  # noqa: BLE001
                self.cont = None
        self.last = np.zeros((CELL, CELL, 3), dtype=np.uint8)
        self.done = self.cont is None

    def advance(self) -> None:
        if self.done:
            return
        try:
            fr = next(self.it)
            img = fr.to_ndarray(format="rgb24")
            if img.shape[:2] != (CELL, CELL):
                from PIL import Image
                img = np.asarray(Image.fromarray(img).resize((CELL, CELL), Image.BILINEAR))
            self.last = img.astype(np.uint8)
        except StopIteration:
            self.done = True


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """Draw a small text tag with a dark background in the image's top-left corner."""
    from PIL import Image, ImageDraw

    pim = Image.fromarray(img)
    d = ImageDraw.Draw(pim)
    px, py = LABEL_PAD
    tw = 7 * len(text) + 2 * px
    d.rectangle([0, 0, tw, 14 + py], fill=(0, 0, 0))
    d.text((px, py), text, fill=(255, 255, 0))
    return np.asarray(pim)


def _title_bar(width: int, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    bar = Image.new("RGB", (width, TITLE_H), (20, 20, 20))
    ImageDraw.Draw(bar).text((6, 6), text, fill=(255, 255, 255))
    return np.asarray(bar)


def _montage_one(traj_dirs, title_text: str, out_path: Path, stride: int, wrist: bool) -> None:
    """Write one montage MP4. Each group = [left_shoulder (| wrist)]; groups/row chosen so a
    row is always CELLS_PER_ROW cells wide (5 groups with wrist, 10 without)."""
    files = [LSHOULDER, WRIST] if wrist else [LSHOULDER]
    cells_per_group = len(files)
    groups_per_row = CELLS_PER_ROW // cells_per_group

    groups = []                              # each = (list[_Cell], traj_idx)
    for d in traj_dirs:
        m = reader.load_meta(d)
        idx = m.get("traj", Path(d).name).replace("traj_", "")
        groups.append(([_Cell(Path(d) / f) for f in files], idx))

    n = len(groups)
    n_rows = (n + groups_per_row - 1) // groups_per_row
    group_w = cells_per_group * CELL
    row_w = groups_per_row * group_w
    title = _title_bar(row_w, title_text)
    blank = np.zeros((CELL, group_w, 3), dtype=np.uint8)

    writer = None
    t = 0
    all_cells = [c for g in groups for c in g[0]]
    while not all(c.done for c in all_cells):
        for c in all_cells:
            c.advance()
        if t % stride == 0:
            rows = []
            for r in range(n_rows):
                cols = []
                for gc in range(groups_per_row):
                    gi = r * groups_per_row + gc
                    if gi < n:
                        cs, idx = groups[gi]
                        img = np.concatenate([c.last for c in cs], axis=1)
                        cols.append(_label(img, idx))
                    else:
                        cols.append(blank)
                rows.append(np.concatenate(cols, axis=1))
            canvas = np.concatenate([title] + rows, axis=0)
            if writer is None:
                h, w = canvas.shape[:2]
                writer = _open_video(out_path, max(1, 30 // stride), h, w)
            _write_frame(writer, canvas)
        t += 1
    if writer is not None:
        _close_video(writer)
        sz = out_path.stat().st_size / 1e6
        print(f"[review] {out_path}  ({n} trajs, {n_rows} rows, {sz:.1f} MB)", flush=True)
    else:
        print(f"[review] {title_text!r}: no frames decoded — skipped", flush=True)


def montage_task(dataset: str, family: str, task: str, *, stride: int = 3,
                 wrist: bool = True, root=reader.ROOT) -> None:
    base = Path(root) / dataset / family / task
    trajs = sorted(p for p in base.glob("traj_*") if (p / "traj.hdf5").exists())
    if not trajs:
        print(f"[review] no trajs under {base}")
        return
    cells_per_group = 2 if wrist else 1
    per_file = MAX_ROWS * (CELLS_PER_ROW // cells_per_group)   # 25 with wrist, 50 without
    tag = "lsw" if wrist else "ls"                             # left_shoulder+wrist / left_shoulder
    chunks = [trajs[i:i + per_file] for i in range(0, len(trajs), per_file)]
    for ci, chunk in enumerate(chunks):
        part = "" if len(chunks) == 1 else f"_p{ci}"
        title = f"{family} / {task}  ({len(trajs)} trajs)" + (f"  part {ci}" if part else "")
        _montage_one(chunk, title, base / f"_review_{tag}{part}.mp4", stride, wrist)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="demos")
    ap.add_argument("--family", default="clutter")
    ap.add_argument("--task", default=None, help="one task (e.g. task_0000); omit with --all")
    ap.add_argument("--all", action="store_true", help="every task in the family")
    ap.add_argument("--stride", type=int, default=3, help="keep every Nth frame (3 => 10fps)")
    ap.add_argument("--no-wrist", dest="wrist", action="store_false",
                    help="left_shoulder only (10 per row); default includes wrist (5 pairs/row)")
    a = ap.parse_args()

    if a.all:
        base = reader.ROOT / a.dataset / a.family
        tasks = sorted(p.name for p in base.glob("task_*") if p.is_dir())
        print(f"[review] {len(tasks)} tasks under {base}")
        for tk in tasks:
            montage_task(a.dataset, a.family, tk, stride=a.stride, wrist=a.wrist)
    elif a.task:
        montage_task(a.dataset, a.family, a.task, stride=a.stride, wrist=a.wrist)
    else:
        ap.error("pass --task <name> or --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
