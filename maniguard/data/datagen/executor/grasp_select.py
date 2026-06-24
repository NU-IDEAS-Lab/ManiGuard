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

# Joint-limit margin (rad) the chosen grasp's wrist must keep at the pre-grasp standoff. Below this
# the wrist starts too near a limit and the subsequent straight lift/carry stalls (singularity-
# adjacent). Tuned from sim collection (see the grasp-singularity spec/plan); 0.2 rad ≈ 11.5°.
MARGIN_FLOOR = 0.2


def roll_variants(eef_quat_xyzw):
    """The two physically-equivalent parallel-jaw eef orientations: the given quat and the
    same quat rolled 180 deg about its OWN approach axis (eef +Z). The fingers swap but the
    grasp point and approach direction are unchanged. Returns ``(q0, q180)`` xyzw."""
    q0 = np.asarray(eef_quat_xyzw, dtype=float)
    q180 = (Rot.from_quat(q0) * Rot.from_rotvec([0.0, 0.0, np.pi])).as_quat()  # local +Z roll
    return q0, q180


def joint_margin(q, lower, upper):
    """Smallest distance (rad) from any joint to its nearest limit; negative if out of range."""
    q = np.asarray(q, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    return float(np.min(np.minimum(q - lower, upper - q)))


def score_grasps(world, robot, target, cands, *, standoff_m: float = 0.10,
                 timeout: float = 3.0, plan_tries: int = 2,
                 roll_disambig: bool = True, margin_floor: float = MARGIN_FLOOR) -> list:
    """Set ``score`` + ``reachable`` (+ ``margin``/``chosen_roll``/``chosen_quat``) on each GraspCand
    and return them sorted (reachable first, then highest joint-margin, then highest score). The
    standoff plan mirrors the engine's pre_grasp, so a grasp that scores reachable will almost
    certainly clear the demo's first segment. Each grasp is retried ``plan_tries`` times before
    declared unreachable: this old cuRobo fork's solve is stochastic, so a single unlucky solve
    would wrongly drop a reachable grasp (e.g. every grasp of a task scoring unreachable -> no
    variants -> 0 demos).

    ``roll_disambig`` (default on): a parallel-jaw grasp is physically identical under a 180° roll
    about its approach axis, but the annotation stores only one roll. We IK BOTH roll variants at the
    standoff, read the planned arm config, and keep the roll whose worst joint sits FARTHEST from a
    limit (``margin = min(q - lower, upper - q)``). A grasp whose best roll still keeps the wrist
    within ``margin_floor`` of a limit is dropped (``reachable=False``): the subsequent straight
    lift/carry would push that wrist past its limit and stall (the palm-flip side-grasp singularity).
    The winning roll is recorded so execution reaches it (cabinet threads ``chosen_roll`` to the grasp
    segments). Set ``roll_disambig=False`` to keep the legacy single-quat, no-floor behaviour (used for
    the drawer-handle grasps, which have their own contact gates)."""
    import torch as th

    init_q = robot.get_joint_positions()
    world.update_obstacles(ignore_objects=[target])      # target dropped (open gripper encloses it)
    arm_idx = robot.arm_control_idx[robot.default_arm]
    lower = np.asarray(robot.joint_lower_limits)[arm_idx]
    upper = np.asarray(robot.joint_upper_limits)[arm_idx]
    for c in cands:
        variants = roll_variants(c.eef_quat) if roll_disambig else (np.asarray(c.eef_quat, float),)
        best = None                                       # (margin, roll_idx, quat, res)
        for roll_idx, quat in enumerate(variants):
            appr = Rot.from_quat(quat).as_matrix()[:, 2]
            appr = appr / (np.linalg.norm(appr) + 1e-9)
            standoff = np.asarray(c.eef_pos, float) - standoff_m * appr
            res = None
            for _attempt in range(max(1, plan_tries)):    # retry: a fresh solve explores new seeds
                res = solve_segment(
                    world.motion_gen, robot,
                    th.as_tensor(standoff, dtype=th.float32),
                    th.as_tensor(np.asarray(quat, float), dtype=th.float32),
                    init_q, timeout=timeout, label=f"score:g{c.id}r{roll_idx}")
                if res is not None:
                    break
            if res is None:
                continue
            q = res.arm_traj[-1].detach().cpu().numpy().reshape(-1)   # final-waypoint arm config (7,)
            m = joint_margin(q, lower, upper)
            if best is None or m > best[0]:
                best = (m, roll_idx, np.asarray(quat, float), res)
        if best is None:
            c.reachable, c.score, c.margin, c.chosen_roll, c.chosen_quat = (
                False, -1.0, float("-inf"), False, np.asarray(c.eef_quat, float))
        else:
            m, roll_idx, quat, res = best
            err = (res.pos_err or 0.0) + (res.rot_err or 0.0)
            c.margin, c.chosen_roll, c.chosen_quat = m, bool(roll_idx), quat
            c.reachable = (m >= margin_floor) if roll_disambig else True
            c.score = float(1.0 / (1.0 + err) - (0.1 if res.salvaged else 0.0))
        print(f"[grasp_select] g{c.id} ({c.approach}): reachable={c.reachable} "
              f"margin={c.margin:.3f} roll={c.chosen_roll} score={c.score:.3f}", flush=True)
    cands.sort(key=lambda c: (c.reachable, c.margin, c.score), reverse=True)
    return cands
