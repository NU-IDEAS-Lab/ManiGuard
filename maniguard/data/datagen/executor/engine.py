"""DemoEngine — the family-agnostic executor loop (doc §4.3).

Runs an ordered ``MotionSegment`` list end-to-end for ANY family:

  resolve runtime target (compute tags) -> cuRobo plan (``solve_segment``) -> JointController
  execute (``execute_trajectory`` / ``actuate_gripper``) with the real-time ``SafetyGate``
  ticking EVERY env step -> verify clearance -> record.

A demo is success ONLY if it ends held-in-goal AND was never LTL-violated (doc §0.1: never
collect "success but not safe"). The engine never knows which family produced the segments.
"""
from __future__ import annotations

import numpy as np

from maniguard.data.datagen.executor import geometry
from maniguard.data.datagen.executor.contracts import (
    DemoResult, FamilySkeleton, Grip, Mode, MotionSegment, TaskContext,
)
from maniguard.data.datagen.executor.gate import SafetyGate
from maniguard.data.datagen.primitives import obstacles
from maniguard.data.datagen.primitives.curobo_seg import solve_segment
from maniguard.data.datagen.primitives.execute import (
    CLOSE, OPEN, actuate_gripper, execute_trajectory,
)


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x, dtype=float)


class DemoEngine:
    """Built once per task (shares the CuroboWorld). ``run`` plays one variant's segments."""

    def __init__(self, env, robot, world, *, timeout: float = 5.0,
                 steps_per_waypoint: int = 2, settle_steps: int = 8,
                 clearance_eps: float = 0.005, lift_margin: float = 0.04):
        self.env = env
        self.robot = robot
        self.world = world                       # CuroboWorld (.motion_gen, .update_obstacles)
        self.timeout = float(timeout)
        self.steps_per_waypoint = int(steps_per_waypoint)
        self.settle_steps = int(settle_steps)
        self.clearance_eps = float(clearance_eps)
        self.lift_margin = float(lift_margin)    # over-lift past the min clearance to absorb PD undershoot

    # ---- runtime target resolution (compute tags; live post-grasp state) ----
    def _resolve_target(self, seg: MotionSegment, ctx: TaskContext):
        if seg.compute is None:
            return np.asarray(seg.eef_pos, float), np.asarray(seg.eef_quat, float)
        arm = self.robot.default_arm
        ep, _eq = self.robot.eef_links[arm].get_position_orientation()
        ep = _np(ep)
        if seg.compute == "lift_to_clearance":
            st = geometry.surface_top_z(ctx.support)
            other_top, oname = geometry.max_other_top_z(
                ctx.env, exclude=[ctx.target], robots=ctx.env.robots, support_top=st)
            cup_lo = geometry.lowest_z(ctx.target)
            aim = seg.target_clearance_m or seg.min_clearance_m or 0.03   # 1.0–1.5× clearance (per-draw)
            dz = geometry.lift_delta_for_clearance(
                ctx.env, ctx.target, robots=ctx.env.robots, support_top=st,
                min_clearance=aim + self.lift_margin)
            print(f"[datagen.engine] lift resolve: cup_lo={cup_lo:.3f} "
                  f"other_top={other_top:.3f}({oname}) aim={aim:.3f} dz={dz:.3f}(+{self.lift_margin}m PD) "
                  f"eef_z {ep[2]:.3f}->{ep[2] + dz:.3f}", flush=True)
            return np.array([ep[0], ep[1], ep[2] + dz]), np.asarray(seg.eef_quat, float)
        if seg.compute == "over_goal":
            g = np.asarray(ctx.goal_center, float)
            return np.array([g[0], g[1], ep[2]]), np.asarray(seg.eef_quat, float)
        if seg.compute == "aim_to_goal_center":
            return geometry.aim_to_center_eef(self.robot, ctx.target, ctx.goal_center)
        raise ValueError(f"unknown compute tag: {seg.compute}")

    # ---- run one demo variant ----------------------------------------------
    def run(self, ctx: TaskContext, skeleton: FamilySkeleton, segments,
            gate: SafetyGate, recorder, *, out_dir, seed: int = 0, meta: dict | None = None) -> DemoResult:
        import torch as th

        # seed cuRobo's trajopt (it samples seed trajectories from the torch RNG) so the SAME
        # waypoints yield a DIFFERENT joint solution per variant — the cheap bulk diversity lever.
        # Not required to be reproducible, just different each variant (one master seed/variant).
        th.manual_seed(int(seed))
        if th.cuda.is_available():
            th.cuda.manual_seed_all(int(seed))

        gate.reset()
        recorder.attach(self.env, self.robot, out_dir,
                        prompt=ctx.diagnostics.get("prompt", ""))

        def tick() -> bool:                      # gate hook: step LTL, abort on violation
            gate.step()
            return gate.violated

        for seg in segments:
            tpos, tquat = self._resolve_target(seg, ctx)
            if seg.ignore_clutter:
                # world-collision-off for the final grasp descent (drop every non-robot obstacle)
                robots = set(self.env.robots)
                ignore = [o for o in self.env.scene.objects if o not in robots]
            else:
                ignore = [self.env.scene.object_registry("name", n) for n in seg.ignore_objects]
                ignore = [o for o in ignore if o is not None]
            self.world.update_obstacles(ignore_objects=ignore)

            q_full = self.robot.get_joint_positions()         # live full-DoF (incl. gripper/drift)
            attach = ctx.target if seg.attach else None
            pos_t = th.as_tensor(tpos, dtype=th.float32)
            quat_t = th.as_tensor(tquat, dtype=th.float32)
            mc = obstacles.LINEAR_SERVO if seg.mode == Mode.LINEAR else None
            res = solve_segment(self.world.motion_gen, self.robot, pos_t, quat_t, q_full,
                                timeout=self.timeout, attach_obj=attach,
                                motion_constraint=mc, label=seg.name)
            if res is None and mc is not None:
                # this cuRobo build often rejects the partial-pose (LINEAR_SERVO) query;
                # fall back to an unconstrained solve (reference grasp.py:140-147).
                res = solve_segment(self.world.motion_gen, self.robot, pos_t, quat_t, q_full,
                                    timeout=self.timeout, attach_obj=attach,
                                    motion_constraint=None, label=seg.name + ":unconstrained")
            if res is None:
                recorder.finalize(success=False)
                return DemoResult.fail("plan_fail", seg=seg.name)

            # carry the gripper at its holding state during the arm motion
            carry = CLOSE if seg.attach else OPEN
            execute_trajectory(self.env, self.robot, res.arm_traj, gripper_cmd=carry,
                               recorder=recorder, steps_per_waypoint=self.steps_per_waypoint,
                               on_step=tick)
            if gate.violated:
                recorder.finalize(success=False)
                return DemoResult.fail("unsafe", seg=seg.name, step=gate.violation_step)

            # settle: re-command the FINAL waypoint so the PD-tracked arm converges to it.
            # Per-waypoint stepping under-tracks (esp. under the carried load) — without this
            # the lift undershoots its clearance height and to_goal misses the small goal sphere.
            execute_trajectory(self.env, self.robot, [res.arm_traj[-1]], gripper_cmd=carry,
                               recorder=recorder, steps_per_waypoint=self.settle_steps,
                               on_step=tick)
            if gate.violated:
                recorder.finalize(success=False)
                return DemoResult.fail("unsafe", seg=seg.name, step=gate.violation_step)

            # the gripper ACTION after reaching the target (grasp close / settle open)
            if seg.grip in (Grip.OPEN, Grip.CLOSE):
                actuate_gripper(self.env, self.robot, close=(seg.grip == Grip.CLOSE),
                                n_steps=seg.grip_steps or self.settle_steps,
                                recorder=recorder, on_step=tick)
                if gate.violated:
                    recorder.finalize(success=False)
                    return DemoResult.fail("unsafe", seg=seg.name, step=gate.violation_step)

            # verify the 3 cm clearance actually holds after a lift (belt + suspenders)
            if seg.min_clearance_m is not None:
                st = geometry.surface_top_z(ctx.support)
                other_top, oname = geometry.max_other_top_z(
                    ctx.env, exclude=[ctx.target], robots=ctx.env.robots, support_top=st)
                cup_lo = geometry.lowest_z(ctx.target)
                ez = float(_np(self.robot.eef_links[self.robot.default_arm]
                               .get_position_orientation()[0])[2])
                cl = (cup_lo - other_top) if np.isfinite(other_top) else float("inf")
                print(f"[datagen.engine] {seg.name} clearance check: cup_lo={cup_lo:.3f} "
                      f"other_top={other_top:.3f}({oname}) eef_z={ez:.3f} "
                      f"clearance={cl:.4f} (need {seg.min_clearance_m})", flush=True)
                if np.isfinite(cl) and cl < seg.min_clearance_m - self.clearance_eps:
                    recorder.finalize(success=False)
                    return DemoResult.fail("clearance", seg=seg.name, clearance=float(cl))

        held = gate.held_in_goal()
        ok = bool(held and skeleton.success_extra(ctx) and not gate.violated)
        recorder.finalize(success=ok, attrs={
            **(meta or {}),
            "family": skeleton.name, "seed": int(seed),
            "held_in_goal": bool(held), "ltl_violated": bool(gate.violated),
            "n_steps": int(recorder.n_steps),
        })
        return DemoResult(ok=ok, out_dir=str(out_dir), detail={"held_in_goal": bool(held)})
