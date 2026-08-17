#!/usr/bin/env python
"""Build a SmolVLA-ready copy of a 5-cam datagen LeRobot dataset.

Runs in the **lerobot env** (same env as ``lerobot-train`` — SmolVLA is
LeRobot-native, so prepare and train share one environment, unlike the GR00T path).

    python tools/smolvla_sft/prepare_dataset.py \
        --src <5cam_lerobot_dir> --out <smolvla_dir> [--external-cam left]

The datagen export uses flat, non-standard keys (``image_*`` / ``state`` /
``actions``) and 5 cameras. ``lerobot-train`` derives the SmolVLA policy's
features purely from standard key PREFIXES (``observation.images.*`` / ``observation.state``
/ ``action``) and would consume every camera present, so this step produces a
2-cam, standard-keyed LeRobot v2.1 copy (the source is left untouched):

  - keep 2 views: the chosen overview ``image_<external_cam>`` -> ``observation.images.top``
    and ``wrist_image`` -> ``observation.images.wrist`` (drop the other 3 overviews);
  - rename numerics: ``state`` -> ``observation.state``, ``actions`` -> ``action``
    (drop the redundant ``actions_commanded``);
  - videos are PASSTHROUGH: an H.264 source mp4 is copied as-is; an AV1 source is
    transcoded to H.264 (libx264) so LeRobot's video backend decodes it everywhere
    (same reasoning as the GR00T path — some boxes' FFmpeg can't decode AV1).

The mapping is defined once in ``maniguard.smolvla_sft.embodiment`` (the single
source of truth, imported here). Rebuilt via ``LeRobotDataset.create`` (not manual
parquet/meta surgery) so LeRobot writes all metadata/stats correctly itself; only
the two kept videos are placed by hand (passthrough technique verified in
``maniguard.data.datagen.to_lerobot``).

Idempotent at dataset granularity: a fully-built ``out`` is skipped; a partial
``out`` must be removed and retried (LeRobot cannot resume a half-written dataset).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from maniguard.data.datagen import data_format

# Reuse the verified lerobot-0.3.3 passthrough patches (no PNG writes, offline
# version check, mp4-aware stats sampling) + the frame counter.
from maniguard.data.datagen.to_lerobot import _frame_count, _passthrough_images
from maniguard.smolvla_sft import embodiment as emb


def _video_codec(path: Path) -> str | None:
    import av

    try:
        with av.open(str(path)) as c:
            return c.streams.video[0].codec_context.name
    except Exception:
        return None


def _transcode_to_h264(src_mp4: Path, dst_mp4: Path, crf: int) -> None:
    """Decode src (any codec, incl. AV1) and re-encode to H.264 via PyAV.
    Mirrors tools/gr00t_sft/prepare_dataset.py (kept inline so each path stays
    self-contained)."""
    import av

    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_mp4.with_suffix(".tmp.mp4")
    with av.open(str(src_mp4)) as inp:
        ins = inp.streams.video[0]
        with av.open(str(tmp), mode="w") as out:
            outs = out.add_stream("libx264", rate=ins.average_rate or 30)
            outs.width = ins.codec_context.width
            outs.height = ins.codec_context.height
            outs.pix_fmt = "yuv420p"
            outs.options = {"crf": str(crf)}
            for frame in inp.decode(ins):
                for pkt in outs.encode(frame):
                    out.mux(pkt)
            for pkt in outs.encode():  # flush
                out.mux(pkt)
    tmp.replace(dst_mp4)


def _place_video(src_mp4: Path, dst_mp4: Path, crf: int) -> int:
    """Passthrough one source video into the new dataset under its renamed key:
    copy if already H.264, else transcode AV1 -> H.264. Returns the frame count."""
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    if _video_codec(src_mp4) == "h264":
        shutil.copy2(src_mp4, dst_mp4)
    else:
        _transcode_to_h264(src_mp4, dst_mp4, crf)
    return _frame_count(dst_mp4)


def _new_features(resolution: int) -> dict:
    """SmolVLA 2-cam standard-keyed feature schema."""

    def _img():
        return {"dtype": "video", "shape": (resolution, resolution, 3),
                "names": ["height", "width", "channel"]}

    return {
        emb.OVERVIEW_KEY: _img(),
        emb.WRIST_KEY: _img(),
        emb.STATE_KEY: {"dtype": "float32", "shape": (emb.STATE_DIM,),
                        "names": [f"joint_{i}" for i in range(emb.STATE_DIM)]},
        emb.ACTION_KEY: {"dtype": "float32", "shape": (emb.ACTION_DIM,),
                         "names": [f"joint_{i}" for i in range(emb.ACTION_DIM)]},
    }


def prepare(src: Path, out: Path, external_cam: str, repo_id: str, crf: int) -> dict:
    if not (src / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"{src}/meta/info.json not found — is this a LeRobot v2.1 dataset?")

    src_info = json.loads((src / "meta" / "info.json").read_text())
    fps = int(src_info["fps"])
    n_expected = int(src_info["total_episodes"])

    # Idempotency: a complete out is a no-op; a partial out must be cleared.
    if (out / "meta" / "info.json").is_file():
        done = int(json.loads((out / "meta" / "info.json").read_text())["total_episodes"])
        if done == n_expected:
            print(f"[prepare] already complete ({done} episodes), skip: {out}")
            return {"repo_id": repo_id, "episodes": done, "root": str(out), "skipped": True}
        raise FileExistsError(
            f"{out} is a partial dataset ({done}/{n_expected}); LeRobot cannot resume — rm it and retry."
        )

    rmap = emb.rename_map(external_cam)              # flat -> standard (validates external_cam)
    src_overview = f"image_{external_cam}"           # -> observation.images.top
    print(f"[prepare] {src.name}: {n_expected} episodes, external_cam={external_cam} "
          f"({src_overview}->{emb.OVERVIEW_KEY}, wrist_image->{emb.WRIST_KEY})")

    # task_index -> task string (version-independent read of tasks.jsonl).
    tasks: dict[int, str] = {}
    for line in (src / "meta" / "tasks.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            tasks[int(r["task_index"])] = r["task"]

    res = data_format.RESOLUTION
    dummy = {k: np.zeros((res, res, 3), dtype=np.uint8) for k in (emb.OVERVIEW_KEY, emb.WRIST_KEY)}

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    with _passthrough_images():
        # Read source (offline version check patched); numerics only, no video decode.
        src_repo = src_info.get("repo_id") or "local/src"
        src_ds = LeRobotDataset(src_repo, root=src)
        hf = src_ds.hf_dataset.select_columns(["state", "actions", "task_index", "episode_index"])

        dst = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=_new_features(res),
                                    root=out, robot_type=src_info.get("robot_type", "franka"),
                                    use_videos=True)

        # Group frames by source episode_index (parquet is ordered ep-then-frame).
        cur_ep, rows, kept = None, [], 0

        def flush(ep: int, rows: list[dict]) -> None:
            nonlocal kept
            new_ep = dst.meta.total_episodes
            # passthrough the two kept videos under their renamed keys
            v_over = _place_video(
                Path(src_ds.root) / src_ds.meta.get_video_file_path(ep, src_overview),
                Path(dst.root) / dst.meta.get_video_file_path(new_ep, emb.OVERVIEW_KEY), crf)
            v_wrist = _place_video(
                Path(src_ds.root) / src_ds.meta.get_video_file_path(ep, "wrist_image"),
                Path(dst.root) / dst.meta.get_video_file_path(new_ep, emb.WRIST_KEY), crf)
            if not (v_over == v_wrist == len(rows)):
                raise ValueError(f"ep {ep}: frame mismatch state={len(rows)} "
                                 f"top={v_over} wrist={v_wrist}")
            task = tasks[int(rows[0]["task_index"])]
            for row in rows:
                dst.add_frame({
                    **dummy,
                    emb.STATE_KEY: np.asarray(row["state"], dtype=np.float32),
                    emb.ACTION_KEY: np.asarray(row["actions"], dtype=np.float32),
                }, task=task)
            dst.save_episode()
            kept += 1
            if kept % 100 == 0 or kept == n_expected:
                print(f"[prepare]   {kept}/{n_expected} episodes")

        for row in hf:
            ep = int(row["episode_index"])
            if cur_ep is not None and ep != cur_ep:
                flush(cur_ep, rows)
                rows = []
            cur_ep, _ = ep, rows.append(row)
        if rows:
            flush(cur_ep, rows)

    summary = {"repo_id": repo_id, "episodes": kept, "root": str(out),
               "external_cam": external_cam, "skipped": False}
    print(f"[prepare] DONE {summary}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source 5-cam datagen LeRobot dir (untouched)")
    ap.add_argument("--out", required=True, help="output 2-cam SmolVLA LeRobot dir")
    ap.add_argument("--repo-id", required=True, help="repo_id stamped into the new dataset meta")
    ap.add_argument("--external-cam", default=emb.DEFAULT_EXTERNAL_CAM,
                    choices=emb.EXTERNAL_CAM_CHOICES,
                    help=f"which overview -> observation.images.top (default {emb.DEFAULT_EXTERNAL_CAM})")
    ap.add_argument("--crf", type=int, default=18, help="libx264 CRF for AV1->H.264 (default 18)")
    args = ap.parse_args()
    prepare(Path(args.src), Path(args.out), args.external_cam, args.repo_id, args.crf)


if __name__ == "__main__":
    main()
