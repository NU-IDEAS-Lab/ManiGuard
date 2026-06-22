"""cuRobo grasp scoring — rank the annotated grasp candidates by reachability so the driver
tries the reachable, high-quality ones first and skips hopeless ones (the §4.3 "use all
USABLE grasps, prefer high score" pre-filter). Generic across families.

Score = can cuRobo reach this grasp's PRE-GRASP standoff (a collision-checked plan from the
current pose, target dropped), weighted by how cleanly it converged (low pos/rot error, not
salvaged). Unreachable grasps get ``reachable=False`` and are skipped by the sampler.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from maniguard.data.datagen.primitives.curobo_seg import solve_segment


def score_grasps(world, robot, target, cands, *, standoff_m: float = 0.10,
                 timeout: float = 3.0, plan_tries: int = 2) -> list:
    """Set ``score`` + ``reachable`` on each GraspCand and return them sorted (reachable
    first, then high score). The standoff plan mirrors the engine's pre_grasp, so a grasp
    that scores reachable will almost certainly clear the demo's first segment. Each grasp is
    retried ``plan_tries`` times before declared unreachable: this old cuRobo fork's solve is
    stochastic, so a single unlucky solve would wrongly drop a reachable grasp (e.g. every grasp
    of a task scoring unreachable -> no variants -> 0 demos)."""
    import torch as th

    init_q = robot.get_joint_positions()
    world.update_obstacles(ignore_objects=[target])      # target dropped (open gripper encloses it)
    for c in cands:
        R = Rot.from_quat(c.eef_quat).as_matrix()
        appr = R[:, 2]
        appr = appr / (np.linalg.norm(appr) + 1e-9)
        standoff = np.asarray(c.eef_pos, float) - standoff_m * appr
        res = None
        for _attempt in range(max(1, plan_tries)):       # retry: a fresh solve explores new seeds
            res = solve_segment(
                world.motion_gen, robot,
                th.as_tensor(standoff, dtype=th.float32),
                th.as_tensor(np.asarray(c.eef_quat, float), dtype=th.float32),
                init_q, timeout=timeout, label=f"score:g{c.id}")
            if res is not None:
                break
        if res is None:
            c.reachable, c.score = False, -1.0
        else:
            err = (res.pos_err or 0.0) + (res.rot_err or 0.0)
            c.reachable = True
            c.score = float(1.0 / (1.0 + err) - (0.1 if res.salvaged else 0.0))
        print(f"[grasp_select] g{c.id} ({c.approach}): reachable={c.reachable} "
              f"score={c.score:.3f}", flush=True)
    cands.sort(key=lambda c: (c.reachable, c.score), reverse=True)
    return cands
