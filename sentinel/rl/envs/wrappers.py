"""SB3 wrappers for OmniGibson environments.

Two layers live here:

* ``SentinelSB3VectorEnvironment`` — subclass of upstream
  ``omnigibson.envs.sb3_vec_env.SB3VectorEnvironment`` that fixes two
  numpy/torch bridging bugs in the upstream adapter (detailed below).
  This is the low-level OG-vec → SB3-vec adapter; construct it once per
  training run when ``num_envs > 1``.

* ``build_vec_env(args, out_dir)`` — high-level constructor that maps
  parsed CLI args to a ready-to-use SB3 ``VecEnv``. Handles the single-env
  (DummyVecEnv) vs multi-env (SentinelSB3VectorEnvironment) branch, the
  scene-file / runtime-spawn split, and Monitor wrapping. Every algorithm
  entry point (PPO, PPO-Lag, FOCOPS, CUPS, …) calls this instead of
  re-implementing the wiring.

The two pieces are co-located because they solve the same problem at
different abstraction levels: turning OG into something SB3 can train on.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn

import omnigibson as og
from omnigibson.envs.sb3_vec_env import (
    SB3VectorEnvironment, last_stepped_env, last_stepped_time,
)

# Side-effect: registers PickAndLiftTask (and any future sentinel tasks) with
# OG's task registry so ``type: "PickAndLiftTask"`` resolves in the config
# produced by ``grasp_reset_scene.build_config``.
import sentinel.rl.tasks  # noqa: F401


# ---------------------------------------------------------------- low-level
class SentinelSB3VectorEnvironment(SB3VectorEnvironment):
    """Drop-in replacement for upstream ``SB3VectorEnvironment``.

    Two fixes vs upstream:

    1. ``step_async`` stored the raw SB3 numpy action; downstream task/reward
       code expects torch (e.g. ``GraspReward`` does ``th.abs(action)``). We
       convert np → torch at the boundary.
    2. ``step_wait`` returned ``th.clone(self.buf_rews / buf_dones)``, but
       those buffers are numpy arrays allocated by SB3's DummyVecEnv. We
       return ``np.copy`` so SB3 sees the expected type.
    """

    def step_async(self, actions) -> None:
        with og.sim.render_on_step(self.render_on_step):
            global last_stepped_env, last_stepped_time
            if last_stepped_env != self:
                assert (
                    last_stepped_time is None or self.last_reset_time > last_stepped_time
                ), "You must call reset() before using a different environment."
                last_stepped_env = self
                last_stepped_time = time.time()

            if not isinstance(actions, th.Tensor):
                actions = th.as_tensor(actions)
            self.actions = actions
            for i, action in enumerate(actions):
                self.envs[i]._pre_step(action)

    def step_wait(self) -> VecEnvStepReturn:
        with og.sim.render_on_step(self.render_on_step):
            og.sim.step()

            for env_idx in range(self.num_envs):
                obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[
                    env_idx
                ]._post_step(self.actions[env_idx])
                self.buf_dones[env_idx] = terminated or truncated
                self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

                if self.buf_dones[env_idx]:
                    self.buf_infos[env_idx]["terminal_observation"] = obs
                    obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
                self._save_obs(env_idx, obs)

            return (
                self._obs_from_buf(),
                np.copy(self.buf_rews),
                np.copy(self.buf_dones),
                copy.deepcopy(self.buf_infos),
            )


# --------------------------------------------------------------- high-level
def _load_diagnostics(diagnostics_file: Path) -> dict:
    """Read the first JSON line from a diagnostics.jsonl file."""
    import json as _json

    with diagnostics_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return _json.loads(line)
    raise SystemExit(f"Empty diagnostics file: {diagnostics_file}")


def build_vec_env(args, *, out_dir: Path, verbose: bool = True):
    """Build an SB3-ready VecEnv from parsed CLI args.

    Expects ``args`` to have the attributes added by
    ``sentinel.rl.cli.common.add_env_args`` and ``--output-dir``. Callers are
    responsible for ``validate_env_args(args)`` beforehand.

    Args:
        args: argparse ``Namespace`` (category, model, target_name,
            grasp_dataset_dir, scene_file, num_envs, reset_mode,
            arm_controller).
        out_dir: path for monitor.csv etc. — directory is created if missing.
        verbose: print booting banners.

    Returns:
        ``vec_env``: a ``DummyVecEnv(Monitor(og_env))`` when num_envs == 1, or
        ``VecMonitor(SentinelSB3VectorEnvironment)`` otherwise. Both expose
        the SB3 ``VecEnv`` API that ``PPO(...)`` (and future constrained
        variants) consume directly.
    """
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    from sentinel.rl.envs.grasp_reset_scene import build_config

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    goal_region_spec = None
    diagnostics_file = getattr(args, "diagnostics_file", None)
    if diagnostics_file is not None:
        diag = _load_diagnostics(Path(diagnostics_file))
        goal_region_spec = diag.get("goal_region")
        if goal_region_spec is None:
            raise SystemExit(
                "diagnostics.jsonl has no 'goal_region' key. "
                "Only benchmark tasks with sphere goal regions are supported."
            )
        if args.target_name is None:
            args.target_name = goal_region_spec.get("target_name")

    target_name = args.target_name or f"target_{args.category}_{args.model}"
    grasp_path = args.grasp_dataset_dir / f"grasps_{args.category}_{args.model}.pt"
    if not grasp_path.exists():
        if diagnostics_file is None:
            raise SystemExit(f"Missing grasp dataset: {grasp_path}")
        grasp_path = None
        if verbose:
            print(f"  grasp dataset not found, running without grasp reset",
                  flush=True)

    cfg = build_config(
        target_name=target_name,
        category=args.category,
        model=args.model,
        grasp_dataset_path=grasp_path,
        scene_file=args.scene_file,
        grasp_reset_mode=args.reset_mode,
        arm_controller=args.arm_controller,
        goal_region_spec=goal_region_spec,
    )
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] Booting OG "
              f"({args.num_envs} env{'s' if args.num_envs > 1 else ''})...",
              flush=True)

    if args.num_envs == 1:
        env = og.Environment(configs=cfg)
        if verbose:
            print(f"  obs_space:    {env.observation_space}", flush=True)
            print(f"  action_space: {env.action_space}", flush=True)
        monitor_path = str(out_dir / "monitor.csv")
        vec_env = DummyVecEnv([lambda: Monitor(env, filename=monitor_path)])
    else:
        raw = SentinelSB3VectorEnvironment(
            num_envs=args.num_envs, config=cfg, render_on_step=False,
        )
        if verbose:
            print(f"  obs_space:    {raw.observation_space}", flush=True)
            print(f"  action_space: {raw.action_space}", flush=True)
        vec_env = VecMonitor(raw, filename=str(out_dir / "monitor.csv"))

    return vec_env


__all__ = ["SentinelSB3VectorEnvironment", "build_vec_env"]
