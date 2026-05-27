#!/usr/bin/env python3
"""Re-export the combined pnp-clutter SFT dataset with ABSOLUTE JOINT actions.

Reuses the already-rendered MP4s from ``lerobot_pnp_full_combined`` (no
re-render) and swaps the eef-delta state/actions for joint state/actions
recovered from the source rollout HDF5s
(:func:`maniguard.data.lerobot.joint_actions.extract_joint_trajectory`).

Episode -> source HDF5 mapping comes from the combined dataset's
``meta/source_provenance.jsonl``:
  * pnp_clutter_first25 (448): ``source_rollout`` is the exact HDF5 path.
  * pnp_sft_prior_n10_combined reliable (622): reconstruct
    ``{task_name}_g{grasp_index}/variant_{variant_idx:02d}/rollout.hdf5``.
  * unreliable (``_provenance_reliable: False``, ~113): skipped.

Only episodes whose source HDF5 exists AND whose length matches the combined
parquet (so the reused MP4 aligns frame-for-frame) are exported.

Usage:
  python -m maniguard.data.lerobot.reexport_joint_actions \
    --combined outputs/lerobot_pnp_full_combined \
    --out outputs/lerobot_pnp_clutter_joint \
    --repo-id IDEAS-Lab-Northwestern/sentinel-pnp-clutter-joint \
    [--limit 5]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from maniguard.data.lerobot.joint_actions import extract_joint_trajectory
from maniguard.data.lerobot.lerobot_writer import (
    LeRobotEpisodeWriter,
    create_or_open_dataset,
    episode_prompt,
    lerobot_features_joint,
)

PROMPT_TEMPLATE = "pick up the {target_clean} in the middle of the table and place it at the green goal"


def resolve_source_hdf5(prov: dict) -> str | None:
    sd = prov.get("source_dataset")
    if sd == "pnp_clutter_first25":
        return prov.get("source_rollout")
    if sd == "pnp_sft_prior_n10_combined":
        t, g, v = prov.get("task_name"), prov.get("grasp_index"), prov.get("variant_idx")
        if t is None or v is None:
            return None
        base = f"outputs/pnp_sft_prior_n10_combined/{t}" + ("" if g is None else f"_g{g}")
        return f"{base}/variant_{int(v):02d}/rollout.hdf5"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", default="outputs/lerobot_pnp_full_combined")
    ap.add_argument("--out", default="outputs/lerobot_pnp_clutter_joint")
    ap.add_argument("--repo-id", default="IDEAS-Lab-Northwestern/sentinel-pnp-clutter-joint")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--resolution", type=int, default=256)
    args = ap.parse_args()

    combined = Path(args.combined)
    info = json.loads((combined / "meta" / "info.json").read_text())
    cs = info.get("chunks_size", 1000)
    prov = [json.loads(l) for l in (combined / "meta" / "source_provenance.jsonl").read_text().splitlines()]

    def pq_path(ei):
        return combined / "data" / f"chunk-{ei // cs:03d}" / f"episode_{ei:06d}.parquet"

    def vid_path(ei, key):
        return combined / "videos" / f"chunk-{ei // cs:03d}" / key / f"episode_{ei:06d}.mp4"

    # Build the list of exportable episodes (reliable + length-aligned).
    plan = []
    skipped = {"unreliable": 0, "no_source": 0, "missing_file": 0, "len_mismatch": 0}
    for d in prov:
        if not d.get("_provenance_reliable"):
            skipped["unreliable"] += 1
            continue
        ei = d["episode_index"]
        src = resolve_source_hdf5(d)
        if not src:
            skipped["no_source"] += 1
            continue
        if not Path(src).exists():
            skipped["missing_file"] += 1
            continue
        n_pq = pq.read_table(pq_path(ei)).num_rows
        with h5py.File(src, "r") as f:
            n_h5 = f["data/demo_0"]["action"].shape[0]
        if n_pq != n_h5:
            skipped["len_mismatch"] += 1
            continue
        plan.append((ei, src, d.get("target_name", "object")))
    print(f"[reexport] exportable={len(plan)}  skipped={skipped}", flush=True)
    if args.limit:
        plan = plan[: args.limit]
        print(f"[reexport] limited to {len(plan)} episodes", flush=True)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    ds = create_or_open_dataset(
        repo_id=args.repo_id, root=out, fps=args.fps, resolution=args.resolution,
        apply_passthrough=True, features=lerobot_features_joint(args.resolution),
    )

    for i, (ei, src, target) in enumerate(plan):
        traj = extract_joint_trajectory(src)
        js, ja = traj["joint_state"], traj["joint_action"]
        writer = LeRobotEpisodeWriter(ds)
        for key in ("image_left", "image_right", "wrist_image"):
            shutil.copy(vid_path(ei, key), writer.target_mp4_paths[key])
        for t in range(len(js)):
            writer.add_step(js[t], ja[t])
        prompt = episode_prompt(target, PROMPT_TEMPLATE)
        writer.commit(prompt)
        if i % 50 == 0 or i == len(plan) - 1:
            print(f"  [{i+1}/{len(plan)}] src_ep={ei} T={len(js)} prompt={prompt!r}", flush=True)

    print(f"\n[reexport] done: {len(plan)} episodes -> {out}", flush=True)


if __name__ == "__main__":
    main()
