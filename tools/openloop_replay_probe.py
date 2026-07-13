#!/usr/bin/env python3
"""Open-loop replay probe for an openpi policy checkpoint.

Feeds an SFT episode's *recorded* observations back to a running policy server and
compares the predicted action against the recorded (ground-truth) action. This isolates
**did the checkpoint fit its training data?** (an open-loop question) from **does it work
in the closed loop?** (what the OmniGibson eval measures). A policy can ace this probe and
still fail the eval — that gap is closed-loop drift / collapse, not undertraining.

How it works
------------
For ~N evenly-spaced steps of one SFT episode it sends the recorded observation
(``observation/image_left`` = the external overview, ``observation/wrist_image``,
``observation/state``, ``prompt``) to the server and reads back the predicted action chunk.
It compares the chunk's first action to the recorded action and reports the absolute error
**normalized by each action dim's std** over the episode — a scale-free "how far off is the
prediction, in units of how much that joint actually varies".

Interpretation
--------------
* normalized error ~1-3%  -> the policy reproduces its training actions: it **fit the data**.
  Any eval failure is therefore closed-loop (drift / collapse / distribution coverage), NOT
  undertraining. (Companion diagnostic to the engagement metric — see
  ``docs/evaluation/engagement_metric.md``.)
* normalized error high    -> the checkpoint never fit this episode: undertrained, wrong
  dataset, or a mismatched observation mapping (e.g. wrong ``--external-cam``).

Usage::

    # 1. serve the checkpoint on :8000 in a separate process, e.g.
    #    python -m maniguard.serve.openpi_native \
    #        --config pi05_base_jar_transport_joint_2cam_lora \
    #        --checkpoint outputs/eval_ckpts/jar/2160 --port 8000
    #
    # 2. run the probe against an SFT episode of THAT checkpoint's dataset
    #    (run it while the server is idle — it competes with the eval client for the GPU):
    python tools/openloop_replay_probe.py outputs/lerobot_datasets/sim-jar-transport-30-joint-3cam
    python tools/openloop_replay_probe.py <dataset> --episode 15            # a different episode
    python tools/openloop_replay_probe.py <cabinet_dataset> --external-cam right

Notes
-----
* ``--external-cam`` MUST match the checkpoint's train/eval external view (the dataset's
  left or right overview fed into the server's single external slot). Only **cabinet** uses
  ``right`` (its left overview is low-quality); the other five families use ``left``.
  Picking the wrong one feeds an out-of-distribution view and inflates the error.
* Pick ``--episode`` to cover each trained operand/content (e.g. one per teleop'd object).
  Use ``meta/tasks.jsonl`` -> the episode's ``task_index`` tells you which prompt it is.
* Run with the eval client's interpreter (the ``behavior`` conda env), which has
  ``openpi_client`` + ``pandas`` + ``imageio``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from openpi_client import websocket_client_policy as wcp


def parse_args():
    p = argparse.ArgumentParser(
        description="Open-loop replay probe: compare a checkpoint's predicted vs recorded "
        "SFT actions to test whether it fit its training data.")
    p.add_argument("dataset", type=Path,
                   help="SFT lerobot dataset dir (the one the served checkpoint was trained on), "
                        "e.g. outputs/lerobot_datasets/sim-jar-transport-30-joint-3cam")
    p.add_argument("--episode", "-e", type=int, default=0,
                   help="episode index to replay (default: 0)")
    p.add_argument("--external-cam", choices=["left", "right"], default="left",
                   help="which dataset overview camera feeds the server's external slot; MUST "
                        "match the checkpoint's train external_cam. cabinet=right, others=left "
                        "(default: left)")
    p.add_argument("--samples", "-n", type=int, default=30,
                   help="number of evenly-spaced steps to sample from the episode (default: 30)")
    p.add_argument("--host", default="127.0.0.1", help="policy server host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000, help="policy server port (default: 8000)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="print per-sampled-step errors, not just the summary")
    return p.parse_args()


def main():
    args = parse_args()
    ds = args.dataset
    if not ds.exists():
        raise SystemExit(f"dataset not found: {ds}")
    ep = args.episode
    chunk = f"chunk-{ep // 1000:03d}"

    # --- recorded state + actions (lerobot parquet columns: 'state', 'actions') ---
    pq = ds / "data" / chunk / f"episode_{ep:06d}.parquet"
    if not pq.exists():
        raise SystemExit(f"episode parquet not found: {pq}")
    df = pd.read_parquet(pq)
    task_index = int(df["task_index"].iloc[0])
    states = np.stack([np.asarray(x, np.float32) for x in df["state"].values])
    acts = np.stack([np.asarray(x, np.float32) for x in df["actions"].values])

    # --- prompt for this episode (by its task_index) ---
    tasks = {
        json.loads(line)["task_index"]: json.loads(line)["task"]
        for line in (ds / "meta" / "tasks.jsonl").read_text().splitlines() if line.strip()
    }
    prompt = tasks[task_index]

    # --- recorded camera streams (external overview + wrist) ---
    ext_name = f"image_{args.external_cam}"
    ext = iio.imread(ds / "videos" / chunk / ext_name / f"episode_{ep:06d}.mp4")
    wrist = iio.imread(ds / "videos" / chunk / "wrist_image" / f"episode_{ep:06d}.mp4")

    T = len(df)
    print(f"dataset = {ds.name}")
    print(f"episode {ep}  task_index={task_index}  ({T} steps)  external_cam={args.external_cam}")
    print(f"prompt  = {prompt!r}")
    print(f"server  = {args.host}:{args.port}")

    policy = wcp.WebsocketClientPolicy(host=args.host, port=args.port)
    step = max(1, T // args.samples)
    errs = []
    for t in range(0, T, step):
        obs = {
            "observation/image_left": ext[t],      # the server's single external slot
            "observation/wrist_image": wrist[t],
            "observation/state": states[t],
            "prompt": prompt,
        }
        res = policy.infer(obs)
        pred = np.asarray(res["actions"], np.float32)
        first = pred[0] if pred.ndim == 2 else pred   # compare the chunk's first action
        e = np.abs(first - acts[t])
        errs.append(e)
        if args.verbose:
            print(f"  t={t:4d}  mean|pred-rec|={e.mean():.3f}  max={e.max():.3f}")

    errs = np.asarray(errs)
    std = acts.std(0)
    norm = errs.mean(0) / (std + 1e-6)
    print("\n=== SUMMARY ===")
    print("mean|pred-rec| / dim :", np.round(errs.mean(0), 3))
    print("action std / dim     :", np.round(std, 3))
    print("normalized err / dim :", np.round(norm, 2))
    print(f"OVERALL  mean abs err = {errs.mean():.3f}   normalized = {norm.mean():.2f}  "
          f"({'fit the data (<~0.05)' if norm.mean() < 0.05 else 'check obs mapping / undertrained'})")


if __name__ == "__main__":
    main()
