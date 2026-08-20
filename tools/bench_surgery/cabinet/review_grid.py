"""Tile each variant level's per-task review STILLS into ONE montage PNG (one image per level)
for fast visual QC of a whole bench family. Reuses the bench review_snapshots frame extraction
(last frame = the settled idle-step state), then lays the 35 task stills out in a labelled grid.

Output: ``<out-dir>/<family>_<level>_<view>_grid.png`` — one per level (base/target/location/env/
language). Each cell is the task's last review frame with a ``task_NNNN`` tag; a title bar names
the level. Missing/undecodable clips become a grey "MISSING" cell so the grid stays 1:1 with tasks.

Usage:
  python -m tools.bench_surgery.cabinet.review_grid --family cabinet_pickup --view left_shoulder
  python -m tools.bench_surgery.cabinet.review_grid --family cabinet_pickup --view opposite_side_front --levels base
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from maniguard.data.bench_builder.review_snapshots import _find_video, _last_frame

OUT_ROOT_DEFAULT = "outputs/lerobot_datasets/maniguard-bench"
LEVELS = ("base", "target", "location", "env", "language")
TASK_RE = re.compile(r"^task_\d+$")
CELL = 256
PAD = 3
TITLE_H = 34


def _font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _cell(frame, label: str) -> Image.Image:
    if frame is None:
        img = Image.new("RGB", (CELL, CELL), (60, 60, 60))
        d = ImageDraw.Draw(img)
        d.text((CELL // 2 - 40, CELL // 2 - 8), "MISSING", fill=(220, 80, 80), font=_font(18))
    else:
        arr = np.asarray(frame, dtype=np.uint8)
        img = Image.fromarray(arr).convert("RGB")
        if img.size != (CELL, CELL):
            img = img.resize((CELL, CELL))
        d = ImageDraw.Draw(img)
    # task label: white text on a black box, top-left
    f = _font(18)
    tw = d.textlength(label, font=f)
    d.rectangle([0, 0, tw + 10, 24], fill=(0, 0, 0))
    d.text((5, 3), label, fill=(255, 255, 255), font=f)
    return img


def build_grid(family: str, out_root: str, view: str, episode: int, level: str,
               cols: int, out_dir: Path) -> Path:
    fam_dir = Path(out_root) / family
    tasks = sorted(d.name for d in fam_dir.iterdir() if d.is_dir() and TASK_RE.match(d.name))
    cells = []
    for t in tasks:
        v = _find_video(fam_dir / t / level, view, episode)
        frame = _last_frame(v) if v is not None else None
        cells.append(_cell(frame, t.replace("task_", "t")))
    rows = (len(cells) + cols - 1) // cols
    W = cols * CELL + (cols + 1) * PAD
    H = TITLE_H + rows * CELL + (rows + 1) * PAD
    canvas = Image.new("RGB", (W, H), (24, 24, 24))
    d = ImageDraw.Draw(canvas)
    d.text((8, 7), f"{family}  —  {level}  ({view})  [{len(tasks)} tasks, last-frame]",
           fill=(255, 255, 255), font=_font(20))
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        x = PAD + c * (CELL + PAD)
        y = TITLE_H + PAD + r * (CELL + PAD)
        canvas.paste(cell, (x, y))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{family}_{level}_{view}_grid.png"
    canvas.save(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="cabinet_pickup")
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    ap.add_argument("--view", default="left_shoulder")
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--levels", nargs="*", default=list(LEVELS))
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--out-dir", default=None, help="default: <out-root>/<family>/review_grids")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.out_root) / args.family / "review_grids"
    for level in args.levels:
        out = build_grid(args.family, args.out_root, args.view, args.episode, level, args.cols, out_dir)
        print(f"[grid] {level} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
