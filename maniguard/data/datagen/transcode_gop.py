#!/usr/bin/env python
"""Re-encode dataset videos with a configurable keyframe interval using PyAV.

The default GOP is 10. In-place mode writes and verifies a temporary video tree,
then keeps the original videos as videos.gop-orig. --out writes a new dataset and
copies data/ and meta/ unchanged. Each video must retain its frame count.

Only videos are re-encoded. Numeric arrays and metadata remain unchanged;
image statistics are not recomputed after the lossy transcode.

Usage:
    python -m maniguard.data.datagen.transcode_gop <dataset_dir> [--out <dst>] [--gop 10] [--crf 18] [--nproc N]
"""
import argparse
import glob
import os
import shutil
import sys
from multiprocessing import Pool

import av

# Leave headroom by default (thermals / machine responsiveness); override with --nproc.
DEFAULT_NPROC = max(1, (os.cpu_count() or 8) - 8)


def frame_count(path):
    with av.open(path) as c:
        return sum(1 for _ in c.decode(c.streams.video[0]))


def transcode_one(job):
    inp, outp, gop, crf = job
    try:
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        ic = av.open(inp)
        istream = ic.streams.video[0]
        rate = istream.average_rate or 30
        oc = av.open(outp, mode="w")
        ostream = oc.add_stream("h264", rate, options={"crf": str(crf)})
        ostream.width = istream.width
        ostream.height = istream.height
        ostream.pix_fmt = "yuv420p"
        ostream.codec_context.gop_size = gop
        nin = 0
        for frame in ic.decode(istream):
            nin += 1
            arr = frame.to_ndarray(format="rgb24")
            vframe = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for pkt in ostream.encode(vframe):
                oc.mux(pkt)
        for pkt in ostream.encode():
            oc.mux(pkt)
        oc.close()
        ic.close()
        nout = frame_count(outp)
        if nout != nin:
            return (inp, f"FRAME MISMATCH in={nin} out={nout}")
        return (inp, None)
    except Exception as e:  # noqa: BLE001
        return (inp, f"ERR {e!r}")


def transcode_tree(src_videos, dst_videos, gop, crf, nproc):
    vids = sorted(glob.glob(os.path.join(src_videos, "**", "*.mp4"), recursive=True))
    jobs = [(v, os.path.join(dst_videos, os.path.relpath(v, src_videos)), gop, crf) for v in vids]
    fails, done = [], 0
    with Pool(nproc) as p:
        for inp, err in p.imap_unordered(transcode_one, jobs, chunksize=4):
            done += 1
            if err:
                fails.append((inp, err))
                print(f"  FAIL {os.path.relpath(inp, src_videos)}: {err}", flush=True)
            if done % 200 == 0:
                print(f"  ...{done}/{len(vids)} fails={len(fails)}", flush=True)
    nnew = len(glob.glob(os.path.join(dst_videos, "**", "*.mp4"), recursive=True))
    return len(vids), nnew, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--out", default=None, help="write a NEW dataset here (src untouched); omit for in-place swap")
    ap.add_argument("--gop", type=int, default=10)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--nproc", type=int, default=DEFAULT_NPROC)
    a = ap.parse_args()

    src = os.path.join(a.dataset, "videos")
    if not os.path.isdir(src):
        sys.exit(f"ERROR: no {src}")

    if a.out:
        if os.path.exists(a.out):
            sys.exit(f"ERROR: --out {a.out} already exists -- remove or pick a fresh path")
        print(f"[transcode] SRC {a.dataset} -> OUT {a.out}  gop={a.gop} crf={a.crf} nproc={a.nproc}", flush=True)
        os.makedirs(a.out)
        for sub in ("data", "meta"):
            s = os.path.join(a.dataset, sub)
            if os.path.isdir(s):
                shutil.copytree(s, os.path.join(a.out, sub))
                print(f"[copy] {sub}/", flush=True)
        n, nn, fails = transcode_tree(src, os.path.join(a.out, "videos"), a.gop, a.crf, a.nproc)
        print(f"[transcode] videos {nn}/{n} fails={len(fails)}", flush=True)
        if fails or nn != n:
            sys.exit(f"[FAIL] {a.out} incomplete (fails={len(fails)}, {nn}/{n}) -- inspect/remove before using")
        print(f"[DONE] {a.out} (data/ meta/ copied, videos/ GOP={a.gop}; images/ skipped; src untouched)", flush=True)
    else:
        tmp = os.path.join(a.dataset, f"videos.new.{os.getpid()}")
        orig = os.path.join(a.dataset, "videos.gop-orig")
        if os.path.exists(orig):
            sys.exit(f"ERROR: {orig} exists (previous run?) -- resolve first")
        print(f"[transcode] IN-PLACE {a.dataset}  gop={a.gop} crf={a.crf} nproc={a.nproc}", flush=True)
        n, nn, fails = transcode_tree(src, tmp, a.gop, a.crf, a.nproc)
        print(f"[transcode] videos {nn}/{n} fails={len(fails)}", flush=True)
        if fails or nn != n:
            sys.exit(f"[ABORT] not swapping (fails={len(fails)}, {nn}/{n}). temp at {tmp}")
        os.rename(src, orig)
        os.rename(tmp, src)
        print(f"[DONE] new videos -> {src}   |   originals kept -> {orig}", flush=True)


if __name__ == "__main__":
    main()
