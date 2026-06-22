"""VariationSampler — the scaling / diversity engine (§4.3).

Yields ``(GraspCand, SampleParams)`` variants = reachable grasps x per-grasp jittered draws,
so ONE base task produces many DIFFERENT demos (data robustness). Generic across families.

v1 diversity levers (all consumed by the skeleton's ``derive_segments`` — no engine change):
  * **grasp**    — each reachable annotated grasp
  * **standoff** — pre-grasp standoff distance along the approach axis
  * **above_xy** — lateral offset of the pre-grasp approach point

Draw ``k=0`` per grasp is canonical (no jitter); ``k>0`` are RNG-jittered (deterministic per
``(grasp_id, k)`` — no wall-clock/global RNG). cuRobo-seed diversity and in-goal-sphere
placement variety are future levers (need ``solve_segment`` seed plumbing / an engine goal
offset); for now grasp x jitter gives the spread.
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
        """SampleParams for grasp ``c`` draw ``k``. UNIQUE master seed ``grasp_id*1000 + k``
        drives ALL randomness: jitter + lift-height here, and the engine's cuRobo trajopt
        (``torch.manual_seed``) with the same value — one seed per variant, every variant
        differs. draw 0 = canonical (no jitter, lift exactly 1.0× clearance); draw>0 jitters
        waypoints + randomizes lift height in ``lift_mult_range`` × clearance."""
        vseed = int(c.id) * 1000 + k
        rng = np.random.default_rng(vseed)
        lo, hi = self.lift_mult_range
        if k == 0:
            dx, dy, standoff, mult = 0.0, 0.0, self.base_standoff, 1.0
        else:
            dx, dy = (float(v) for v in rng.uniform(-self.jitter_xy, self.jitter_xy, 2))
            standoff = self.base_standoff + float(rng.uniform(0.0, self.jitter_standoff))
            mult = float(rng.uniform(lo, hi))
        return SampleParams(seed=vseed, standoff_m=standoff, min_clearance_m=self.min_clearance,
                            lift_clearance_mult=mult, jitter={"above_xy": (dx, dy)})

    def variants(self, cands):
        """Bounded: ``(grasp, params)`` over reachable grasps x ``n_per_grasp`` draws."""
        for c in cands:
            if getattr(c, "reachable", True):
                for k in range(self.n_per_grasp):
                    yield c, self._params(c, k)

    def variants_stream(self, cands):
        """Open-ended: round-robin reachable grasps, draw 0,1,2,... forever. The driver breaks
        once it has collected its target number of successes (or hits its attempt cap). Spreads
        draws evenly across grasps so diversity doesn't pile onto one grasp."""
        import itertools
        reach = [c for c in cands if getattr(c, "reachable", True)]
        if not reach:
            return                              # NO reachable grasp -> yield nothing (else itertools.count()
            #                                     spins forever with an empty inner loop = a hard CPU hang
            #                                     that blocks the whole sweep). The driver then ends 0/target.
        for k in itertools.count():
            for c in reach:
                yield c, self._params(c, k)
