"""SB3 callbacks for sentinel RL."""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

log = logging.getLogger(__name__)


class SafetyCallback(BaseCallback):
    """Record per-step LTL violations from env info dicts into the SB3 logger."""

    def _on_step(self) -> bool:
        try:
            infos = self.locals.get("infos")
            if infos:
                violations = sum(1 for info in infos if info.get("ltl_violation"))
                self.logger.record("safety/ltl_violations", violations)
        except Exception as exc:
            log.warning("sb3 callback info dict access failed: %s", exc)
            pass
        return True


class GCCallback(BaseCallback):
    """Run ``gc.collect()`` every N env steps.

    OmniGibson + Isaac Sim Replicator's per-step ``annotator.get_data()`` path
    leaves Python-side references that can accumulate between cycles. Forcing a
    full GC sweep periodically may reclaim cycle-held buffers before the host
    runs out of RAM. No-op if the underlying leak is C-level (SDK internals).
    """

    def __init__(self, collect_every: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.collect_every = collect_every

    def _on_step(self) -> bool:
        # self.n_calls increments once per env.step across the whole vec batch
        if self.n_calls % self.collect_every == 0:
            collected = gc.collect()
            if self.verbose:
                self.logger.record("gc/objects_collected", collected)
        return True


class ViewerVideoCallback(BaseCallback):
    """Periodically record a clip from OG's viewer camera.

    Every ``save_freq`` env-steps, starts grabbing ``og.sim.viewer_camera`` RGB
    frames for ``clip_length`` steps, then writes the clip to
    ``<save_dir>/rollout_<timesteps>.mp4``. Uses the shared viewer camera (not
    a robot-attached RGB sensor), so no slowdown from adding RGB to
    ``obs_modalities`` — the viewer is always rendering anyway.

    Multi-env note: there's exactly one viewer camera. In a tiled multi-scene
    setup it observes whichever scene tile the viewer is aimed at; by default
    that's env 0. The captured video shows env 0's rollout.

    Failure handling: any exception during frame capture aborts the current
    clip and logs a warning; training keeps running.
    """

    def __init__(self, save_freq: int, clip_length: int, save_dir: Path,
                 fps: int = 30, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.clip_length = int(clip_length)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self._recording = False
        self._frames: list = []
        self._clip_start_step = 0
        # Track last save-interval so we handle num_timesteps jumping over the
        # exact save_freq boundary (vec-env bumps timesteps by num_envs per
        # _on_step call, so ``num_timesteps % save_freq == 0`` is unreliable).
        self._last_saved_interval = -1

    def _grab_frame(self):
        import omnigibson as og
        # ``render_on_step=False`` on SentinelSB3VectorEnvironment means the
        # viewport renderer doesn't tick during training steps, so
        # ``viewer_camera.get_obs()`` returns stale frames (and a saved clip
        # ends up being ~200× the same image). Force a render first — same
        # pattern as sentinel.task_generation.pipeline_common uses.
        og.sim.render()
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
        # OG returns a torch tensor on GPU; move to CPU numpy H×W×{3,4}.
        if hasattr(rgb, "cpu"):
            rgb = rgb.cpu().numpy()
        import numpy as np
        arr = np.asarray(rgb)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]  # drop alpha
        return arr.astype("uint8")

    def _save_clip(self):
        import imageio
        path = self.save_dir / f"rollout_{self._clip_start_step:08d}.mp4"
        imageio.mimwrite(str(path), self._frames, fps=self.fps, quality=8,
                         macro_block_size=1)
        if self.verbose:
            self.logger.record("video/clip_saved_step", self._clip_start_step)
        log.info("ViewerVideoCallback: wrote %s (%d frames)",
                 path, len(self._frames))

    def _on_step(self) -> bool:
        # Use num_timesteps (env-step count) rather than n_calls (vec-step
        # count) so ``--video-freq 100000`` fires every 100k env steps — the
        # scale users see in training logs.
        current_interval = int(self.num_timesteps) // self.save_freq
        if not self._recording and current_interval > self._last_saved_interval:
            self._recording = True
            self._frames = []
            self._clip_start_step = int(self.num_timesteps)
            self._last_saved_interval = current_interval
            log.info("ViewerVideoCallback: starting clip at step %d",
                     self._clip_start_step)

        if self._recording:
            try:
                self._frames.append(self._grab_frame())
            except Exception as exc:  # noqa: BLE001 — never let video kill training
                log.warning("ViewerVideoCallback: frame grab failed (%s); "
                            "aborting clip", exc)
                self._recording = False
                self._frames = []
                return True

            if len(self._frames) >= self.clip_length:
                try:
                    self._save_clip()
                except Exception as exc:  # noqa: BLE001
                    log.warning("ViewerVideoCallback: save failed: %s", exc)
                self._recording = False
                self._frames = []
        return True


class PeriodicEvalCallback(BaseCallback):
    """Run K deterministic eval episodes every N env-steps, log to SB3 logger.

    Why ``_on_rollout_end`` instead of ``_on_step``:
        SB3's on-policy algorithms (PPO, PPO-Lag, ...) track
        ``self._last_obs`` across rollout boundaries. If we call
        ``vec_env.reset()`` inside ``_on_step`` mid-rollout, ``_last_obs``
        goes stale and corrupts the rollout buffer. Firing in
        ``_on_rollout_end`` — after the rollout is complete but before
        ``model.train()`` — is safe: the rollout buffer is already frozen,
        and when SB3 starts the next rollout it pulls the new obs from the
        env directly.

    Downside of sharing ``self.training_env`` rather than a dedicated eval
    env: the training env's underlying OG simulation has to absorb the eval
    ``reset()`` calls. For OG's grasp-reset setup that's cheap (cached-mode
    reset is a joint teleport, ~100 ms). For a separate-env design we'd
    need a second ``og.Environment`` in the same process, which OG's global
    ``og.sim`` singleton doesn't clearly support.
    """

    def __init__(self, eval_freq: int, n_eval_episodes: int,
                 deterministic: bool = True, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic = bool(deterministic)
        self._last_eval_interval = -1

    def _on_step(self) -> bool:  # no-op; real work is in _on_rollout_end
        return True

    def _on_rollout_end(self) -> None:
        current_interval = int(self.num_timesteps) // self.eval_freq
        if current_interval <= self._last_eval_interval:
            return
        self._last_eval_interval = current_interval

        # Lazy import: avoids a circular path callbacks ← algorithms.eval at
        # module load time (algorithms.eval imports envs.wrappers which imports
        # tasks which... only safe once the training stack is fully initialized).
        from sentinel.rl.algorithms.eval import evaluate_policy

        t0 = time.time()
        metrics = evaluate_policy(
            self.model,
            self.training_env,
            n_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
        )
        dt = time.time() - t0
        for k, v in metrics.items():
            self.logger.record(f"eval/{k}", v)
        self.logger.record("eval/wall_time_s", float(dt))
        if self.verbose:
            log.info(
                "PeriodicEvalCallback @ step %d: mean_reward=%.2f "
                "success_rate=%.2f mean_cost=%.2f (%.1fs)",
                int(self.num_timesteps),
                metrics["mean_reward"],
                metrics["success_rate"],
                metrics["mean_cost"],
                dt,
            )
