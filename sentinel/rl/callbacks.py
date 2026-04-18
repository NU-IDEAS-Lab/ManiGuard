"""SB3 callbacks for sentinel RL."""

from __future__ import annotations

import gc

from stable_baselines3.common.callbacks import BaseCallback


class SafetyCallback(BaseCallback):
    """Record per-step LTL violations from env info dicts into the SB3 logger."""

    def _on_step(self) -> bool:
        try:
            infos = self.locals.get("infos")
            if infos:
                violations = sum(1 for info in infos if info.get("ltl_violation"))
                self.logger.record("safety/ltl_violations", violations)
        except Exception:
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
