#!/usr/bin/env python
"""Build a GR00T-ready copy of a ManiGuard joint LeRobot dataset.

Run inside the Isaac-GR00T uv venv:

    python tools/gr00t_sft/prepare_dataset.py --src <av1_lerobot_dir> --out <gr00t_dir>

Steps (the source dataset is left untouched):
  1. copy ``meta/`` + ``data/`` (parquet) from src to out;
  2. **transcode every video AV1 -> H.264** (libx264) from src to out via PyAV.
     Quest's module FFmpeg (6.1) can load torchcodec but cannot DECODE AV1
     ("Could not push packet to decoder: Function not implemented"); H.264 decodes
     everywhere. PyAV (a gr00t dep) bundles an FFmpeg with both the AV1 decoder
     and libx264, so no conda/system FFmpeg is needed for the transcode;
  3. write ``out/meta/modality.json`` (the 3-cam joint config from
     ``maniguard/gr00t_sft/maniguard_embodiment.py``, mapping GR00T's keys onto the
     export's existing ``state`` / ``actions`` / ``image_*`` names via ``original_key``);
  4. run ``gr00t.data.stats`` to produce ``meta/stats.json`` + ``meta/relative_stats.json``.

Idempotent-ish: an already-transcoded ``out`` video (codec == h264) is skipped, and
``gr00t.data.stats`` skips stats that already exist. Training then decodes the H.264
copy with the stock ``torchcodec`` backend (load it via ``module load ffmpeg/6.1``).
"""

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

# tools/gr00t_sft/prepare_dataset.py -> repo root -> the embodiment config file.
_REPO = Path(__file__).resolve().parents[2]
_CFG = _REPO / "maniguard" / "gr00t_sft" / "maniguard_embodiment.py"


def _load_embodiment_config():
    """Exec the self-contained config file: registers NEW_EMBODIMENT, returns module."""
    spec = importlib.util.spec_from_file_location("maniguard_embodiment", _CFG)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load embodiment config from {_CFG}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _video_codec(path: Path) -> str | None:
    import av

    try:
        with av.open(str(path)) as c:
            return c.streams.video[0].codec_context.name
    except Exception:
        return None


def transcode_to_h264(src_mp4: Path, dst_mp4: Path, crf: int) -> None:
    """Decode src (any codec, incl. AV1) and re-encode to H.264 via PyAV."""
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


def prepare(src: Path, out: Path, crf: int, mod) -> None:
    if not (src / "meta").is_dir():
        raise FileNotFoundError(f"{src}/meta not found — is this a LeRobot v2 dataset?")
    out.mkdir(parents=True, exist_ok=True)

    # 1. copy meta + parquet data (small; videos handled separately).
    for sub in ("meta", "data"):
        if (src / sub).is_dir():
            shutil.copytree(src / sub, out / sub, dirs_exist_ok=True)

    # 2. transcode every video AV1 -> H.264.
    src_videos = src / "videos"
    mp4s = sorted(src_videos.rglob("*.mp4")) if src_videos.is_dir() else []
    print(f"[prepare] processing {len(mp4s)} videos (transcode AV1->H.264, crf={crf}; copy if already H.264) ...")
    for i, src_mp4 in enumerate(mp4s, 1):
        dst_mp4 = out / src_mp4.relative_to(src)
        if dst_mp4.exists() and _video_codec(dst_mp4) == "h264":
            continue  # idempotent: already done
        dst_mp4.parent.mkdir(parents=True, exist_ok=True)
        if _video_codec(src_mp4) == "h264":
            shutil.copy2(src_mp4, dst_mp4)  # already H.264 (e.g. clutter) — no re-encode
        else:
            transcode_to_h264(src_mp4, dst_mp4, crf)
        if i % 25 == 0 or i == len(mp4s):
            print(f"[prepare]   {i}/{len(mp4s)}")

    # 3. modality.json
    (out / "meta" / "modality.json").write_text(json.dumps(mod.MODALITY_JSON, indent=4) + "\n")
    print(f"[prepare] wrote {out / 'meta' / 'modality.json'}")

    # 4. stats (NEW_EMBODIMENT config already registered via _load_embodiment_config)
    from gr00t.data.stats import main as stats_main
    from gr00t.data.types import EmbodimentTag

    print(f"[prepare] generating stats for {out} ...")
    stats_main(str(out), EmbodimentTag.NEW_EMBODIMENT)
    print(f"[prepare] done: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="source LeRobot dataset dir (AV1, untouched)")
    ap.add_argument("--out", required=True, help="output GR00T-ready dataset dir (H.264)")
    ap.add_argument("--crf", type=int, default=18, help="libx264 CRF, lower=higher quality (default 18)")
    args = ap.parse_args()

    mod = _load_embodiment_config()  # register NEW_EMBODIMENT once
    prepare(Path(args.src), Path(args.out), args.crf, mod)


if __name__ == "__main__":
    main()
