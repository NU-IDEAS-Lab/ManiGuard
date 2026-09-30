"""Generate grasp and motion variants from deterministic draw indices.

For each reachable grasp, sample standoff, lateral approach offset, and lift
multiplier. Draw zero uses the nominal values. Later draws use a seed derived
from grasp ID and draw index; the engine also receives that seed for planning.
Family skeletons consume the parameters relevant to their motion sequence.
"""
from __future__ import annotations

import numpy as np

from maniguard.data.datagen.executor.contracts import SampleParams


class VariationSampler:
    def __init__(self, *, n_per_grasp: int = 3, base_standoff: float = 0.10,
                 min_clearance: float = 0.03, jitter_xy: float = 0.015,
                 jitter_standoff: float = 0.03, lift_mult_range=(1.0, 1.5)):
        self.n_per_grasp = int(n_per_grasp)
        self.base_standoff = float(base_standoff)
        self.min_clearance = float(min_clearance)
        self.jitter_xy = float(jitter_xy)
        self.jitter_standoff = float(jitter_standoff)
        self.lift_mult_range = tuple(lift_mult_range)

    def _params(self, c, k: int) -> SampleParams:
        """Build SampleParams using a 32-bit seed derived from (grasp ID, draw index). Draw zero uses nominal values; later draws sample approach offsets and lift height. The seed derivation is deterministic but does not guarantee globally unique seed values."""
        vseed = int(np.random.SeedSequence([int(c.id), int(k)]).generate_state(1)[0])
        rng = np.random.default_rng(vseed)
        lo, hi = self.lift_mult_range
        if k == 0:
            dx, dy, standoff, mult = 0.0, 0.0, self.base_standoff, 1.0
        else:
            dx, dy = (float(v) for v in rng.uniform(-self.jitter_xy, self.jitter_xy, 2))
            standoff = self.base_standoff + float(rng.uniform(0.0, self.jitter_standoff))
            mult = float(rng.uniform(lo, hi))
        return SampleParams(seed=vseed, draw_index=int(k), standoff_m=standoff,
                            min_clearance_m=self.min_clearance, lift_clearance_mult=mult,
                            jitter={"above_xy": (dx, dy)})

    def variants(self, cands):
        """Bounded: ``(grasp, params)`` over reachable grasps x ``n_per_grasp`` draws."""
        for c in cands:
            if getattr(c, "reachable", True):
                for k in range(self.n_per_grasp):
                    yield c, self._params(c, k)

    def variants_stream(self, cands, start_k: int = 0):
        """Open-ended: round-robin reachable grasps, draw start_k, start_k+1, ... forever. The driver
        breaks once it has collected its target number of successes (or hits its attempt cap) and
        persists the resume cursor (start_k lets a top-up continue past the draws it already tried).
        Spreads draws evenly across grasps so diversity doesn't pile onto one grasp."""
        import itertools
        reach = [c for c in cands if getattr(c, "reachable", True)]
        if not reach:
            return                              # NO reachable grasp -> yield nothing (else itertools.count()
            #                                     spins forever with an empty inner loop = a hard CPU hang
            #                                     that blocks the whole sweep). The driver then ends 0/target.
        for k in itertools.count(int(start_k)):
            for c in reach:
                yield c, self._params(c, k)
