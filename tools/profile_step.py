"""Profile env.step() to attribute wall-time to Python packages.

Builds the same OG env that ``sentinel.rl.algorithms.ppo`` trains on, runs
N random-action steps under cProfile, then aggregates time by top-level
package (``omnigibson``, ``sentinel``, ``torch``, ``omni.*``, ``pxr``,
``numpy``, …). Prints both the per-package roll-up and the top
individual functions, so the bottleneck stands out from the noise.

Usage example (same shape as PPO entry):
    python tools/profile_step.py \
        --category goblet --model nawrfs \
        --diagnostics-file datasets/.../task_0022/base/diagnostics.jsonl \
        --grasp-dataset-dir outputs/grasp_datasets/task0022_inscene \
        --num-steps 200 --output-dir outputs/profiles/task0022
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time
from collections import defaultdict
from pathlib import Path


def _package_of(filename: str) -> str:
    """Map a stats record's filename to a coarse package name.

    The cProfile filename is an absolute path. We bucket by the deepest
    directory whose basename names a recognisable package (``omnigibson``,
    ``sentinel``, ``torch``, ``omni.physx``, …) and otherwise fall back to
    one of two coarse labels (``stdlib``, ``builtin``).
    """
    if filename in ("~", "", "<built-in>"):
        return "builtin"
    parts = filename.replace("\\", "/").split("/")

    # Known top-level packages we care about — pick the *innermost* match so
    # ``.../site-packages/omnigibson/.../envs/env_base.py`` resolves to
    # ``omnigibson`` not the outer ``site-packages``.
    known = (
        "sentinel", "omnigibson", "bddl", "omni",
        "pxr", "usd", "isaacsim", "isaac_sim",
        "stable_baselines3", "wandb",
        "torch", "numpy", "scipy", "gymnasium", "gym",
        "trimesh", "yaml", "h5py", "matplotlib", "PIL", "cv2",
    )
    deepest = None
    for p in parts:
        if p in known:
            deepest = p
    if deepest is not None:
        # Distinguish omni.physx / omni.isaac / omni.kit / omni.usd within omni
        if deepest == "omni":
            try:
                idx = parts.index("omni")
                sub = parts[idx + 1] if idx + 1 < len(parts) else ""
                return f"omni.{sub}" if sub else "omni"
            except ValueError:
                return "omni"
        return deepest

    if "site-packages" in parts or "dist-packages" in parts:
        return "other-thirdparty"
    if filename.startswith("<frozen") or "/python3." in filename:
        return "stdlib"
    return "other"


def _aggregate_by_package(stats: pstats.Stats) -> list[tuple[str, float, float, int]]:
    """Roll up (tottime, cumtime, ncalls) by package_of(filename)."""
    bucket_tot: dict[str, float] = defaultdict(float)
    bucket_cum: dict[str, float] = defaultdict(float)
    bucket_calls: dict[str, int] = defaultdict(int)
    for (filename, _line, _func), (cc, _nc, tt, ct, _callers) in stats.stats.items():
        pkg = _package_of(filename)
        bucket_tot[pkg] += tt   # tottime: time IN this function only (no callees)
        bucket_cum[pkg] += ct   # cumtime: includes callees — sums are inflated
        bucket_calls[pkg] += cc
    rows = [(pkg, bucket_tot[pkg], bucket_cum[pkg], bucket_calls[pkg])
            for pkg in bucket_tot]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Reuse the same env-args group PPO uses so flags line up 1:1.
    from sentinel.rl.cli.common import add_env_args, validate_env_args
    add_env_args(parser)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=10,
                        help="Steps to run before turning on the profiler — "
                             "lets JIT/compile + lazy imports settle.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/profiles/step"))
    parser.add_argument("--top", type=int, default=25,
                        help="Number of individual functions to print after "
                             "the per-package roll-up.")
    args = parser.parse_args()
    validate_env_args(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build the same env wrapper PPO uses; we want apples-to-apples timing.
    from sentinel.rl.envs.wrappers import build_vec_env
    # Force num_envs=1 so we profile the inner og env without SB3 vec overhead.
    args.num_envs = 1
    vec_env = build_vec_env(args, out_dir=args.output_dir, verbose=True)

    print(f"\n[profile] resetting env…", flush=True)
    obs = vec_env.reset()

    # Sample random actions ahead of time — same shape PPO would deliver.
    import numpy as np
    action_space = vec_env.action_space
    actions = np.stack([action_space.sample() for _ in range(args.num_steps)])

    # ---------------------------------------------------------------- warmup
    print(f"[profile] warming up {args.warmup_steps} steps…", flush=True)
    for i in range(args.warmup_steps):
        vec_env.step(actions[i % args.num_steps][None])

    # --------------------------------------------------------------- profile
    print(f"[profile] profiling {args.num_steps} steps…", flush=True)
    prof = cProfile.Profile()
    t0 = time.time()
    prof.enable()
    for i in range(args.num_steps):
        vec_env.step(actions[i][None])
    prof.disable()
    dt = time.time() - t0

    fps = args.num_steps / dt
    print(f"\n[profile] {args.num_steps} steps in {dt:.2f}s → {fps:.1f} FPS", flush=True)

    # Dump raw stats so the user can re-explore with `python -m pstats`.
    raw_path = args.output_dir / "step.prof"
    prof.dump_stats(str(raw_path))
    print(f"[profile] raw stats: {raw_path}", flush=True)

    # ---------------------------------------------------------------- report
    stats = pstats.Stats(prof).strip_dirs()
    rows = _aggregate_by_package(pstats.Stats(prof))  # use full filenames

    print("\n=== Per-package wall time (tottime = self-only, no callees) ===")
    print(f"{'package':<24} {'tottime(s)':>11} {'%total':>7} {'cumtime(s)':>11} {'ncalls':>10}")
    print("-" * 70)
    total_tot = sum(r[1] for r in rows) or 1e-9
    for pkg, tot, cum, ncalls in rows:
        pct = 100.0 * tot / total_tot
        if pct < 0.1:
            continue
        print(f"{pkg:<24} {tot:>11.3f} {pct:>6.1f}% {cum:>11.3f} {ncalls:>10d}")

    print(f"\n=== Top {args.top} individual functions by tottime ===")
    stats.sort_stats("tottime").print_stats(args.top)

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
