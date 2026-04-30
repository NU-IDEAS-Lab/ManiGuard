#!/usr/bin/env python3
"""Proprio-only PPO on the grasp-reset PickAndLiftTask.

This is an A/B entry point for state-based training on benchmark scene files
whose saved robot config may be RGB-only. It leaves the original dataset scene
and ``sentinel.rl.algorithms.ppo`` path untouched by writing a per-run scene
copy under ``--output-dir`` with robot ``obs_modalities`` forced to
``["proprio"]``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from pathlib import Path

from sentinel.rl.cli.common import (
    add_env_args, add_training_args, add_video_args, add_wandb_args,
    validate_env_args,
)


_KIT_LOG_DIR = (
    Path(os.environ.get("CONDA_PREFIX", "/workspace/miniconda3/envs/behavior"))
    / "lib/python3.11/site-packages/isaacsim/kit/logs/Kit/OmniGibson/3.8"
)
_KIT_LOG_PATTERNS = (
    "Imported scene",
    "Starting ppo",
    "Training done",
    "Waiting for RtPso",
    "Attaching rgb to render product",
    "Traceback",
    "Segmentation",
)
_KIT_ERROR_RE = re.compile(r"\[(Error|Fatal)\]|\bERROR\b")


def parse_args():
    p = argparse.ArgumentParser(
        description="Proprio-only PPO on PickAndLiftTask with grasp-reset."
    )
    add_env_args(p)
    add_training_args(p)
    add_wandb_args(p)
    add_video_args(p)
    return p.parse_args()


def _log_stage(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [ppo_proprio] {message}", flush=True)


class _KitLogTailer:
    """Print selected OmniGibson Kit log milestones while long startup runs."""

    def __init__(self, *, since: float):
        self._since = since
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._path: Path | None = None
        self._pos = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _latest_log(self) -> Path | None:
        if not _KIT_LOG_DIR.exists():
            return None
        logs = [
            path for path in _KIT_LOG_DIR.glob("kit_*.log")
            if path.stat().st_mtime >= self._since - 5
        ]
        if not logs:
            return None
        return max(logs, key=lambda path: path.stat().st_mtime)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                path = self._latest_log()
                if path is not None and path != self._path:
                    self._path = path
                    self._pos = path.stat().st_size
                    _log_stage(f"Kit log: {path}")

                if self._path is not None:
                    with self._path.open("r", encoding="utf-8", errors="replace") as f:
                        f.seek(self._pos)
                        for line in f:
                            if "[py stdout]" in line or "[py stderr]" in line:
                                continue
                            if (
                                any(pattern in line for pattern in _KIT_LOG_PATTERNS)
                                or _KIT_ERROR_RE.search(line)
                            ):
                                print(f"[Kit] {line.rstrip()}", flush=True)
                        self._pos = f.tell()
            except Exception as exc:  # noqa: BLE001 - progress helper only
                _log_stage(f"Kit log tail disabled after error: {exc}")
                return

            self._stop.wait(2.0)


def _make_proprio_scene_copy(scene_file: Path, out_dir: Path) -> tuple[Path, int]:
    """Copy ``scene_file`` into ``out_dir`` with robot obs set to proprio."""
    source = Path(scene_file).resolve()
    if not source.exists():
        raise SystemExit(f"--scene-file does not exist: {source}")

    override_dir = out_dir / "scene_overrides"
    override_dir.mkdir(parents=True, exist_ok=True)
    target = override_dir / f"{source.stem}.proprio{source.suffix}"

    data = json.loads(source.read_text())
    init_info = data.get("objects_info", {}).get("init_info", {})
    patched = 0
    for obj_key, info in init_info.items():
        args = info.get("args")
        if not isinstance(args, dict):
            continue
        if not obj_key.startswith("robot_") and "controller_config" not in args:
            continue
        if "obs_modalities" in args:
            args["obs_modalities"] = ["proprio"]
            patched += 1

    if patched == 0:
        raise RuntimeError(
            f"{source}: no robot init_info entry with args.obs_modalities found"
        )

    target.write_text(json.dumps(data))
    return target, patched


def main():
    start_time = time.time()
    _log_stage("parsing CLI and validating env args")
    args = parse_args()
    validate_env_args(args)

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    _log_stage("creating proprio scene override")
    source_scene_file = Path(args.scene_file).resolve()
    proprio_scene_file, patched_robots = _make_proprio_scene_copy(
        source_scene_file, out
    )
    args.scene_file = proprio_scene_file
    print(f"  source scene:           {source_scene_file}", flush=True)
    print(f"  proprio scene:          {proprio_scene_file}", flush=True)
    print(f"  patched robot obs:      {patched_robots}", flush=True)

    _log_stage("starting Kit log watcher")
    kit_tailer = _KitLogTailer(since=start_time)
    kit_tailer.start()

    # OG macros must be set before omnigibson is imported (which happens inside
    # build_vec_env via sb3_vec).
    _log_stage("importing OmniGibson and setting macros")
    from omnigibson.macros import gm
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_FLATCACHE = True
    if os.environ.get("OMNIGIBSON_HEADLESS", "0") == "1":
        gm.HEADLESS = True

    _log_stage("importing PPO / SB3 helpers")
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed

    from sentinel.rl.envs.wrappers import build_vec_env
    from sentinel.rl.training.trainer import run_training

    _log_stage(f"building OmniGibson vec env (num_envs={args.num_envs})")
    vec_env = build_vec_env(args, out_dir=out)
    _log_stage("vec env ready")
    set_random_seed(args.seed)

    tensorboard_log_dir = out / f"tb_{time.strftime('%Y%m%d-%H%M%S')}"
    _log_stage(f"tensorboard dir: {tensorboard_log_dir}")

    if args.resume_from is not None:
        if not args.resume_from.exists():
            raise SystemExit(f"--resume-from does not exist: {args.resume_from}")
        _log_stage("loading PPO checkpoint")
        print(f"  resuming from:          {args.resume_from}", flush=True)
        model = PPO.load(
            str(args.resume_from),
            env=vec_env,
            tensorboard_log=str(tensorboard_log_dir),
            device="cuda",
        )
        print(f"  timesteps at ckpt:      {model.num_timesteps:,}", flush=True)
    else:
        _log_stage("constructing fresh PPO model")
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=str(tensorboard_log_dir),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            device="cuda",
            seed=args.seed,
        )

    try:
        _log_stage("entering shared training loop")
        run_training(
            model,
            args,
            algo_name="ppo",
            extra_wandb_config={
                "obs_mode": "proprio",
                "source_scene_file": str(source_scene_file),
                "patched_scene_file": str(proprio_scene_file),
                "patched_robot_obs_count": patched_robots,
            },
        )
    finally:
        kit_tailer.stop()


if __name__ == "__main__":
    main()
