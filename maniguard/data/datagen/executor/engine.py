"""DemoEngine — the family-agnostic executor loop (doc §4.3).

Runs an ordered ``MotionSegment`` list end-to-end for ANY family:

  resolve runtime target (compute tags) -> cuRobo plan (``solve_segment``) -> JointController
  execute (``execute_trajectory`` / ``actuate_gripper``) with the real-time ``SafetyGate``
  ticking EVERY env step -> verify clearance -> record.

A demo is success ONLY if it ends held-in-goal AND was never LTL-violated (doc §0.1: never
collect "success but not safe"). The engine never knows which family produced the segments.
"""
from __future__ import annotations

import os

import numpy as np

from maniguard.data.datagen.executor import geometry
from maniguard.data.datagen.executor.contracts import (
    DemoResult, FamilySkeleton, Grip, Mode, MotionSegment, TaskContext,
)
from maniguard.data.datagen.executor.gate import SafetyGate
from maniguard.data.datagen.primitives import obstacles
from maniguard.data.datagen.primitives.curobo_seg import solve_ik, solve_segment
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
                 clearance_eps: float = 0.005, lift_margin: float = 0.04,
                 servo_step_m: float = 0.010, servo_spw: int = 4):
        self.env = env
        self.robot = robot
        self.world = world                       # CuroboWorld (.motion_gen, .update_obstacles)
        self.timeout = float(timeout)
        self.steps_per_waypoint = int(steps_per_waypoint)
        self.settle_steps = int(settle_steps)
        self.clearance_eps = float(clearance_eps)
        self.lift_margin = float(lift_margin)    # over-lift past the min clearance to absorb PD undershoot
        self.servo_step_m = float(servo_step_m)  # SERVO eef-interpolation resolution (straight-push waypoint spacing)
        self.servo_spw = int(servo_spw)          # sim steps per SERVO waypoint: SLOW so a pushed drawer slides
        #                                          with the gripper (a fast push outruns it -> no contact)
        self._last_servo = None                  # most recent SERVO joint path (for replay_reverse retreat)

    def _held(self, seg: MotionSegment, ctx: TaskContext):
        """The object currently carried (for attach + clearance). Defaults to ctx.target
        (clutter always carries the target); a family may carry a different object per segment
        via ``seg.extra['held_name']`` (e.g. cabinet moving the obstacle aside)."""
        name = (seg.extra or {}).get("held_name")
        if name:
            o = ctx.env.scene.object_registry("name", name)
            if o is not None:
                return o
        return ctx.target

    # ---- runtime target resolution (compute tags; live post-grasp state) ----
    def _resolve_target(self, seg: MotionSegment, ctx: TaskContext, skeleton=None):
        if seg.compute is None:
            return np.asarray(seg.eef_pos, float), np.asarray(seg.eef_quat, float)
        if seg.compute not in ("lift_to_clearance", "over_goal", "aim_to_goal_center"):
            # family-specific tag (e.g. cabinet grasp-handle / pull / regrasp) -> the skeleton
            pos, quat = skeleton.resolve_compute(seg.compute, seg, ctx)
            return np.asarray(pos, float), np.asarray(quat, float)
        arm = self.robot.default_arm
        ep, eq = self.robot.eef_links[arm].get_position_orientation()
        ep, eq = _np(ep), _np(eq)         # KEEP the live eef orientation (a top-down lift/move must
        #                                   not reorient — seg.eef_quat is a per-family placeholder)
        if seg.compute == "lift_to_clearance":
            held = self._held(seg, ctx)
            st = geometry.surface_top_z(ctx.support)
            other_top, oname = geometry.max_other_top_z(
                ctx.env, exclude=[held], robots=ctx.env.robots, support_top=st)
            cup_lo = geometry.lowest_z(held)
            aim = seg.target_clearance_m or seg.min_clearance_m or 0.03   # 1.0–1.5× clearance (per-draw)
            dz = geometry.lift_delta_for_clearance(
                ctx.env, ctx.target, robots=ctx.env.robots, support_top=st,
                min_clearance=aim + self.lift_margin)
            print(f"[datagen.engine] lift resolve: cup_lo={cup_lo:.3f} "
                  f"other_top={other_top:.3f}({oname}) aim={aim:.3f} dz={dz:.3f}(+{self.lift_margin}m PD) "
                  f"eef_z {ep[2]:.3f}->{ep[2] + dz:.3f}", flush=True)
            return np.array([ep[0], ep[1], ep[2] + dz]), eq
        if seg.compute == "over_goal":
            g = np.asarray(ctx.goal_center, float)
            return np.array([g[0], g[1], ep[2]]), eq
        if seg.compute == "aim_to_goal_center":
            return geometry.aim_to_center_eef(self.robot, ctx.target, ctx.goal_center)
        raise ValueError(f"unknown compute tag: {seg.compute}")

    # ---- straight-line Cartesian IK servo (deliberate push into contact) ----
    def _servo_line(self, tpos, tquat, ctx, seg):
        """Interpolate the eef from its CURRENT pose straight to ``(tpos, tquat)`` (orientation
        held), solving per-waypoint IK with COLLISION OFF and chaining each solve's seed for a
        smooth joint path. Returns the joint trajectory (T,7) or ``None`` if any IK fails. Used
        for a deliberate push the collision-avoiding planner would refuse (shoving a drawer shut)."""
        import torch as th
        arm = self.robot.default_arm
        sp = _np(self.robot.eef_links[arm].get_position_orientation()[0])
        tp = np.asarray(tpos, float)
        n = max(2, int(np.ceil(float(np.linalg.norm(tp - sp)) / self.servo_step_m)))
        print(f"[datagen.engine] {seg.name}: SERVO start_eef={sp.round(3)} -> target={tp.round(3)} "
              f"(delta={(tp - sp).round(3)}, |d|={np.linalg.norm(tp - sp):.3f}m, n={n} ik steps)", flush=True)
        quat_t = th.as_tensor(np.asarray(tquat, float), dtype=th.float32)
        manip_idx = self.robot.arm_control_idx[self.robot.default_arm]
        q_seed = self.robot.get_joint_positions()             # full-DoF seed, spliced per step
        out = []
        for i in range(1, n + 1):
            p = sp + (tp - sp) * (i / n)
            res = solve_ik(self.world.motion_gen, self.robot,
                           th.as_tensor(p, dtype=th.float32), quat_t, q_seed,
                           timeout=self.timeout, ik_collision=False, label=f"{seg.name}:ik{i}/{n}")
            if res is None:
                print(f"[datagen.engine] {seg.name}: servo IK failed at step {i}/{n}", flush=True)
                return None
            arm_cfg = res.arm_traj[-1]
            out.append(arm_cfg)
            q_seed = q_seed.clone()                            # chain: keep IK near the prior solution
            q_seed[manip_idx] = arm_cfg.to(q_seed.device, q_seed.dtype)
        return th.stack(out)

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
        self._last_servo = None                  # per-variant: no servo path recorded yet
        recorder.attach(self.env, self.robot, out_dir,
                        prompt=ctx.diagnostics.get("prompt", ""))

        def tick() -> bool:                      # gate hook: step LTL, abort on violation
            gate.step()
            return gate.violated

        for seg in segments:
            skeleton.on_segment(seg, ctx)          # family runtime side-effects (e.g. hold drawer open)
            tpos, tquat = self._resolve_target(seg, ctx, skeleton)
            if seg.ignore_clutter:
                # world-collision-off for the final grasp descent (drop every non-robot obstacle)
                robots = set(self.env.robots)
                ignore = [o for o in self.env.scene.objects if o not in robots]
            else:
                ignore = [self.env.scene.object_registry("name", n) for n in seg.ignore_objects]
                ignore = [o for o in ignore if o is not None]
            self.world.update_obstacles(ignore_objects=ignore)

            attach = self._held(seg, ctx) if seg.attach else None
            # produce this segment's joint trajectory: a stored-path replay, a straight IK servo,
            # or the default collision-aware cuRobo solve.
            if seg.replay_reverse:
                if self._last_servo is None:
                    recorder.finalize(success=False)
                    return DemoResult.fail("no_servo", seg=seg.name)
                arm_traj = th.flip(self._last_servo, dims=[0])    # retreat back out the push lane
            elif seg.mode == Mode.SERVO:
                arm_traj = self._servo_line(tpos, tquat, ctx, seg)
                if arm_traj is None:
                    recorder.finalize(success=False)
                    return DemoResult.fail("servo_ik_fail", seg=seg.name)
                self._last_servo = arm_traj                       # remember for the replay-reverse retreat
            else:
                q_full = self.robot.get_joint_positions()         # live full-DoF (incl. gripper/drift)
                pos_t = th.as_tensor(tpos, dtype=th.float32)
                quat_t = th.as_tensor(tquat, dtype=th.float32)
                mc = obstacles.LINEAR_SERVO if seg.mode == Mode.LINEAR else None
                res = solve_segment(self.world.motion_gen, self.robot, pos_t, quat_t, q_full,
                                    timeout=self.timeout, attach_obj=attach,
                                    motion_constraint=mc, label=seg.name,
                                    diagnose_on_fail=True)  # TEMP
                if res is None and mc is not None:
                    # this cuRobo build often rejects the partial-pose (LINEAR_SERVO) query;
                    # fall back to an unconstrained solve (reference grasp.py:140-147).
                    res = solve_segment(self.world.motion_gen, self.robot, pos_t, quat_t, q_full,
                                        timeout=self.timeout, attach_obj=attach,
                                        motion_constraint=None, label=seg.name + ":unconstrained")
                if res is None:
                    recorder.finalize(success=False)
                    return DemoResult.fail("plan_fail", seg=seg.name)
                arm_traj = res.arm_traj

            # gripper command DURING the arm motion: closed iff carrying an attached object,
            # unless the segment forces it (a closed-gripper block pushing the drawer = carry_closed).
            carry = CLOSE if (seg.attach if seg.carry_closed is None else seg.carry_closed) else OPEN
            # a SERVO push (or its reverse) runs SLOW so a contacted drawer slides with the gripper
            # instead of being outrun; everything else uses the normal per-waypoint cadence.
            spw = self.servo_spw if (seg.mode == Mode.SERVO or seg.replay_reverse) else self.steps_per_waypoint
            execute_trajectory(self.env, self.robot, arm_traj, gripper_cmd=carry,
                               recorder=recorder, steps_per_waypoint=spw,
                               on_step=tick)
            if gate.violated:
                recorder.finalize(success=False)
                return DemoResult.fail("unsafe", seg=seg.name, step=gate.violation_step)

            # settle: re-command the FINAL waypoint so the PD-tracked arm converges to it.
            # Per-waypoint stepping under-tracks (esp. under the carried load) — without this
            # the lift undershoots its clearance height and to_goal misses the small goal sphere.
            execute_trajectory(self.env, self.robot, [arm_traj[-1]], gripper_cmd=carry,
                               recorder=recorder, steps_per_waypoint=max(self.settle_steps, spw),
                               on_step=tick)
            if gate.violated:
                recorder.finalize(success=False)
                return DemoResult.fail("unsafe", seg=seg.name, step=gate.violation_step)

            # stuck check: did the eef actually REACH its commanded target? If it stalled far short,
            # the held object didn't follow the plan (a jammed path/strategy) — fail fast so the
            # driver retries a different (strategy x grasp x seed) combo.
            if seg.reach_tol_m is not None:
                ez = _np(self.robot.eef_links[self.robot.default_arm].get_position_orientation()[0])
                err = float(np.linalg.norm(ez - np.asarray(tpos, float)))
                if err > seg.reach_tol_m:
                    print(f"[datagen.engine] {seg.name} STUCK: eef reach err={err:.3f} > {seg.reach_tol_m} "
                          f"(eef={ez.round(3)} target={np.asarray(tpos, float).round(3)})", flush=True)
                    recorder.finalize(success=False)
                    return DemoResult.fail("stuck", seg=seg.name, reach_err=err)

            # verify the HELD object actually cleared a required height (e.g. the target's bottom is
            # above the drawer rim) BEFORE this/the next lateral move — moving while still below the
            # rim catches it and rams the drawer shut, so fail cleanly here and let the driver retry.
            if seg.verify_held_above_z is not None:
                held = self._held(seg, ctx)
                bottom = float(geometry.lowest_z(held))
                if bottom < seg.verify_held_above_z:
                    print(f"[datagen.engine] {seg.name} BELOW-Z: held bottom={bottom:.3f} "
                          f"< rim={seg.verify_held_above_z:.3f}", flush=True)
                    recorder.finalize(success=False)
                    return DemoResult.fail("below_z", seg=seg.name, bottom=bottom)

            # the gripper ACTION after reaching the target (grasp close / settle open)
            if seg.grip in (Grip.OPEN, Grip.CLOSE):
                actuate_gripper(self.env, self.robot, close=(seg.grip == Grip.CLOSE),
                                n_steps=seg.grip_steps or self.settle_steps,
                                recorder=recorder, on_step=tick)
                if seg.grip == Grip.CLOSE:
                    held = self._held(seg, ctx)
                    try:
                        from omnigibson.controllers.controller_base import IsGraspingState
                        ag = self.robot.is_grasping(self.robot.default_arm, held)
                        print(f"[datagen.engine] {seg.name} after close: AG={ag} "
                              f"(held={getattr(held, 'name', '?')})", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"[datagen.engine] (AG check skip: {e})", flush=True)
                if gate.violated:
                    recorder.finalize(success=False)
                    return DemoResult.fail("unsafe", seg=seg.name, step=gate.violation_step)

            # verify the 3 cm clearance actually holds after a lift (belt + suspenders)
            if seg.min_clearance_m is not None:
                held = self._held(seg, ctx)
                st = geometry.surface_top_z(ctx.support)
                other_top, oname = geometry.max_other_top_z(
                    ctx.env, exclude=[held], robots=ctx.env.robots, support_top=st)
                cup_lo = geometry.lowest_z(held)
                ez = float(_np(self.robot.eef_links[self.robot.default_arm]
                               .get_position_orientation()[0])[2])
                cl = (cup_lo - other_top) if np.isfinite(other_top) else float("inf")
                print(f"[datagen.engine] {seg.name} clearance check: cup_lo={cup_lo:.3f} "
                      f"other_top={other_top:.3f}({oname}) eef_z={ez:.3f} "
                      f"clearance={cl:.4f} (need {seg.min_clearance_m})", flush=True)
                if np.isfinite(cl) and cl < seg.min_clearance_m - self.clearance_eps:
                    recorder.finalize(success=False)
                    return DemoResult.fail("clearance", seg=seg.name, clearance=float(cl))

            if os.environ.get("DATAGEN_DEBUG_STATE"):     # gated per-segment family-state probe
                ds = skeleton.debug_state(ctx)
                ep_now = _np(self.robot.eef_links[self.robot.default_arm].get_position_orientation()[0])
                print(f"[datagen.engine] after {seg.name}: eef={ep_now.round(3)} {ds}", flush=True)

        reached = gate.success()
        ok = bool(reached and skeleton.success_extra(ctx) and not gate.violated)
        recorder.finalize(success=ok, attrs={
            **(meta or {}),
            "family": skeleton.name, "seed": int(seed),
            "goal_reached": bool(reached), "ltl_violated": bool(gate.violated),
            "n_steps": int(recorder.n_steps),
        })
        return DemoResult(ok=ok, out_dir=str(out_dir), detail={"goal_reached": bool(reached)})
