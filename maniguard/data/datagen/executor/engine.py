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

FLARE_TOL = 0.5          # rad: max |panda_joint3| (j2 = arm index 2) before a config is "elbow-flared".
#                          PROVISIONAL — locked in S1 from the flip(>=1 rad)-vs-clean(<=0.3 rad) j2 gap.
PIN_L2_PERTURB = (0.3, -0.3, 0.6, -0.6, 0.9, -0.9)   # rad j2 seed perturbations for the dormant L2 ladder.


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x, dtype=float)


class DemoEngine:
    """Built once per task (shares the CuroboWorld). ``run`` plays one variant's segments."""

    def __init__(self, env, robot, world, *, timeout: float = 5.0,
                 steps_per_waypoint: int = 2, settle_steps: int = 8, rest_settle_steps: int = 45,
                 clearance_eps: float = 0.005, lift_margin: float = 0.04,
                 servo_step_m: float = 0.010, servo_spw: int = 4,
                 max_steps: int = 3600, plan_tries: int = 2):
        self.env = env
        self.robot = robot
        self.world = world                       # CuroboWorld (.motion_gen, .update_obstacles)
        self.timeout = float(timeout)
        self.steps_per_waypoint = int(steps_per_waypoint)
        self.settle_steps = int(settle_steps)
        self.rest_settle_steps = int(rest_settle_steps)  # monitored end-of-rollout settle-to-rest steps
        self.clearance_eps = float(clearance_eps)
        self.lift_margin = float(lift_margin)    # over-lift past the min clearance to absorb PD undershoot
        self.servo_step_m = float(servo_step_m)  # SERVO eef-interpolation resolution (straight-push waypoint spacing)
        self.servo_spw = int(servo_spw)          # sim steps per SERVO waypoint: SLOW so a pushed drawer slides
        #                                          with the gripper (a fast push outruns it -> no contact)
        self.max_steps = int(max_steps)          # backstop: abort + fail "timeout" if a rollout exceeds this many
        #                                          recorded steps (30 fps → 3600 ≈ 2 min). A healthy cabinet demo is
        #                                          ~2400; this only catches a pathological runaway, never a normal run.
        self.plan_tries = max(1, int(plan_tries))  # cuRobo FREE/LINEAR plan attempts per segment: trajopt samples
        #                                          seed trajectories from the (advancing) torch RNG, so a retry
        #                                          explores NEW seeds → clears this old fork's intermittent plan_fail
        self._last_servo = None                  # most recent SERVO joint path (for replay_reverse retreat)
        self._path_buf = None                    # accumulating joint path (for replay_reverse_path return-to-home)

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
        for a deliberate push the collision-avoiding planner would refuse (shoving a drawer shut).

        DATAGEN_LOG_SERVO: per-waypoint flare/Delta trace + (for place_across) the candidate no-flare
        ``q_pin`` flare. Phase A only logs (the chained seed still drives); the pin is wired in Phase B."""
        import torch as th
        arm = self.robot.default_arm
        sp = _np(self.robot.eef_links[arm].get_position_orientation()[0])
        tp = np.asarray(tpos, float)
        step = seg.servo_step_m if seg.servo_step_m is not None else self.servo_step_m   # per-segment override
        n = max(2, int(np.ceil(float(np.linalg.norm(tp - sp)) / step)))
        print(f"[datagen.engine] {seg.name}: SERVO start_eef={sp.round(3)} -> target={tp.round(3)} "
              f"(delta={(tp - sp).round(3)}, |d|={np.linalg.norm(tp - sp):.3f}m, n={n} ik steps)", flush=True)
        quat_t = th.as_tensor(np.asarray(tquat, float), dtype=th.float32)
        manip_idx = self.robot.arm_control_idx[self.robot.default_arm]

        log_servo = bool(os.environ.get("DATAGEN_LOG_SERVO"))
        if log_servo and seg.name == "place_across":            # S1 endpoint no-flare probe (no pin)
            ai = self.robot.arm_control_idx[arm]
            cand = self._build_pin_seed(tpos, tquat)
            print("[servo] place_across PROBE: " + ("no no-flare endpoint config (L3)" if cand is None
                  else f"q_pin flare={geometry.arm_flare(_np(cand[ai])):.3f} "
                       f"q_pin={np.round(_np(cand[ai]), 3)}"), flush=True)

        q_seed = self.robot.get_joint_positions()             # full-DoF seed, spliced per step
        out, prev_cfg = [], None
        for i in range(1, n + 1):
            p = sp + (tp - sp) * (i / n)
            res = solve_ik(self.world.motion_gen, self.robot,
                           th.as_tensor(p, dtype=th.float32), quat_t, q_seed,
                           timeout=self.timeout, ik_collision=False, label=f"{seg.name}:ik{i}/{n}")
            if res is None:
                print(f"[datagen.engine] {seg.name}: servo IK failed at step {i}/{n}", flush=True)
                return None
            arm_cfg = res.arm_traj[-1]
            if log_servo:
                self._log_servo_wp(seg, i, n, arm_cfg, prev_cfg, p)
            out.append(arm_cfg); prev_cfg = arm_cfg
            q_seed = q_seed.clone()                            # chain: keep IK near the prior solution
            q_seed[manip_idx] = arm_cfg.to(q_seed.device, q_seed.dtype)
        if log_servo:
            self._log_flare(seg.name)
        return th.stack(out)

    def _build_pin_seed(self, tpos, tquat):
        """No-flare reference arm config (full-DoF tensor) at the segment endpoint (tpos, held tquat),
        or None (=> servo_ik_fail). L1: tailored no-flare seed (geometry.noflare_seed) -> solve_ik,
        accept iff arm_flare < FLARE_TOL. L2 (dormant, only when L1 flares): perturb j2, re-solve, pick
        min flare. L3: None. Adds +1 solve_ik (L1) over the carry's ~20-40; bounded by self.timeout."""
        import torch as th
        arm = self.robot.default_arm
        ai = self.robot.arm_control_idx[arm]
        q_full = self.robot.get_joint_positions()
        bp, bq = self.robot.get_position_orientation()        # robot base world pose
        base_yaw = geometry.quat_yaw(_np(bq))
        pos_t = th.as_tensor(np.asarray(tpos, float), dtype=th.float32)
        quat_t = th.as_tensor(np.asarray(tquat, float), dtype=th.float32)

        def _solve(seed_arm):
            seed_full = q_full.clone()
            seed_full[ai] = th.as_tensor(seed_arm, dtype=seed_full.dtype, device=seed_full.device)
            res = solve_ik(self.world.motion_gen, self.robot, pos_t, quat_t, seed_full,
                           timeout=self.timeout, ik_collision=False, label="place_across:pin")
            if res is None:
                return None
            out = seed_full.clone()
            out[ai] = res.arm_traj[-1].to(out.device, out.dtype)
            return out

        seed = geometry.noflare_seed(_np(q_full[ai]), _np(bp), base_yaw, np.asarray(tpos, float)[:2])
        q_pin = _solve(seed)                                   # L1
        if q_pin is not None and geometry.arm_flare(_np(q_pin[ai])) < FLARE_TOL:
            return q_pin
        best, best_flare = None, FLARE_TOL                     # L2 (dormant)
        for dj2 in PIN_L2_PERTURB:
            s = seed.copy(); s[2] = float(dj2)
            cand = _solve(s)
            if cand is None:
                continue
            f = geometry.arm_flare(_np(cand[ai]))
            if f < best_flare:
                best, best_flare = cand, f
        return best                                            # None => L3

    # ---- debug instrumentation: per-segment joint diagnostics (DATAGEN_LOG_JOINTS) ----
    def _arm_limits(self):
        ai = self.robot.arm_control_idx[self.robot.default_arm]
        lo = np.asarray(self.robot.joint_lower_limits)[ai]
        hi = np.asarray(self.robot.joint_upper_limits)[ai]
        return ai, lo, hi

    def _log_cmd_joints(self, seg, arm_traj):
        """COMMANDED arm trajectory's closest approach to a joint limit — shows whether the IK/plan
        ITSELF drives a joint to its limit (a joint-limit singularity) on this segment."""
        _, lo, hi = self._arm_limits()
        traj = _np(arm_traj)
        if traj.ndim == 1:
            traj = traj.reshape(1, -1)
        margins = np.minimum(traj - lo, hi - traj)             # (T,7) dist to nearest limit per joint
        per_wp_min = margins.min(axis=1)
        wi = int(np.argmin(per_wp_min))
        print(f"[joints] {seg.name} CMD n={len(traj)}: final_min_margin={margins[-1].min():+.3f}"
              f"(j{int(np.argmin(margins[-1]))}) traj_min_margin={per_wp_min[wi]:+.3f}"
              f"(j{int(np.argmin(margins[wi]))})@wp{wi}; final_q={np.round(traj[-1], 3)}", flush=True)

    def _log_achieved_joints(self, seg, arm_traj, tpos):
        """ACHIEVED arm config after execution vs the COMMANDED final waypoint. A large per-joint
        tracking error on a near-limit joint = the controller saturating against that limit (the
        stall); eef_err shows the Cartesian shortfall (the held object that 'didn't lift')."""
        ai, lo, hi = self._arm_limits()
        q_now = np.asarray(self.robot.get_joint_positions())[ai]
        cmd = _np(arm_traj[-1]).reshape(-1)
        margins = np.minimum(q_now - lo, hi - q_now)
        jm = int(np.argmin(margins))
        terr = np.abs(q_now - cmd)
        jt = int(np.argmax(terr))
        ep = _np(self.robot.eef_links[self.robot.default_arm].get_position_orientation()[0])
        eerr = float(np.linalg.norm(ep - np.asarray(tpos, float)))
        print(f"[joints] {seg.name} DONE: min_margin={margins[jm]:+.3f}(j{jm}) "
              f"max_track_err={terr[jt]:.3f}(j{jt}) eef_err={eerr:.3f}; achieved_q={np.round(q_now, 3)}", flush=True)

    def _log_contacts(self, seg):
        """Which NON-robot scene bodies the robot is touching at the END of this segment. A contact
        with the cabinet / another object during a free approach is the collision that stalled it
        (explains a large eef_err with healthy joint margins = obstruction, not a singularity)."""
        try:
            contacts = self.robot.contact_list()
        except Exception as e:  # noqa: BLE001
            print(f"[contacts] {seg.name}: contact_list failed: {e}", flush=True)
            return
        rname = str(self.robot.name)

        def _obj(b):                                     # /World/scene_0/<object>/<link>/...
            parts = b.strip("/").split("/")
            if "scene_0" in parts:
                i = parts.index("scene_0")
                return parts[i + 1] if i + 1 < len(parts) else parts[-1]
            return parts[1] if len(parts) >= 2 else b

        hit: dict[str, int] = {}
        for c in contacts:
            b0, b1 = str(getattr(c, "body0", "")), str(getattr(c, "body1", ""))
            rlink = b0 if rname in b0 else (b1 if rname in b1 else None)
            other = b1 if rname in b0 else (b0 if rname in b1 else None)
            if rlink is None or other is None or rname in other:
                continue
            # robot-link : object  — distinguishes a fingertip grip from a forearm collision
            key = f"{_obj(other)}<->{rlink.strip('/').split('/')[-1]}"
            hit[key] = hit.get(key, 0) + 1
        print(f"[contacts] {seg.name}: {dict(sorted(hit.items())) or 'NONE'}", flush=True)

    # ---- place_across pinned-seed diagnostics (DATAGEN_LOG_SERVO) -----------
    def _log_flare(self, tag):
        """live Cartesian out-of-plane elbow flare from FK link reads (panda_link2 shoulder ->
        panda_link4 elbow vs the shoulder->eef reach plane)."""
        arm = self.robot.default_arm
        sh = _np(self.robot.links["panda_link2"].get_position_orientation()[0])
        el = _np(self.robot.links["panda_link4"].get_position_orientation()[0])
        ee = _np(self.robot.eef_links[arm].get_position_orientation()[0])
        off = geometry.elbow_lateral_offset(sh[:2], el[:2], ee[:2])
        lat = "degen" if off is None else f"{off:.3f}"
        print(f"[servo] {tag}: elbow_lat={lat} elbow_z={el[2]:.3f}", flush=True)

    def _log_servo_wp(self, seg, i, n, arm_cfg, prev_cfg, p):
        """per-waypoint trace: flare(=|j2|), |Delta| from the prior waypoint + argmax joint, j3
        (panda_joint4) margin to its straight limit -0.0698, commanded on-line eef-z."""
        c = _np(arm_cfg).reshape(-1)
        dmax, dj = 0.0, -1
        if prev_cfg is not None:
            d = np.abs(c - _np(prev_cfg).reshape(-1)); dmax, dj = float(d.max()), int(d.argmax())
        print(f"[servo] {seg.name} wp{i}/{n}: flare={geometry.arm_flare(c):.3f} "
              f"dmax={dmax:.3f}(j{dj}) j3_marg={(-0.0698 - c[3]):+.3f} eef_z={float(p[2]):.3f} "
              f"q={np.round(c, 3)}", flush=True)

    def _write_trace_line(self, ctx):
        """[DATAGEN_TRACE] one per-step full-state row -> out_dir/trace.jsonl: current segment, arm joints
        q + joint velocities qd, and the target object's pose + linear/angular velocity. Lets us traceback
        per-step motion smoothness (joint accel spikes) + object tip onset (angular velocity) for ANY
        rollout, kept or failed. Flushed per line so a failed rollout's trace is complete."""
        import json
        ai = self.robot.arm_control_idx[self.robot.default_arm]
        q = _np(self.robot.get_joint_positions())[ai]
        try:
            qd = _np(self.robot.get_joint_velocities())[ai]
        except Exception:  # noqa: BLE001
            qd = np.zeros_like(q)
        rec = {"seg": self._cur_seg, "s": int(getattr(self, "_trace_i", 0)),
               "q": [round(float(v), 4) for v in q], "qd": [round(float(v), 4) for v in qd]}
        self._trace_i = int(getattr(self, "_trace_i", 0)) + 1
        tgt = getattr(ctx, "target", None)
        if tgt is not None:
            try:
                p, quat = tgt.get_position_orientation()
                rec["op"] = [round(float(v), 4) for v in _np(p)]
                rec["oq"] = [round(float(v), 4) for v in _np(quat)]
                rec["olv"] = [round(float(v), 3) for v in _np(tgt.get_linear_velocity())]
                rec["oav"] = [round(float(v), 3) for v in _np(tgt.get_angular_velocity())]
            except Exception:  # noqa: BLE001
                pass
        self._trace_f.write(json.dumps(rec) + "\n")
        self._trace_f.flush()

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
        self._path_buf = None                    # per-variant: no relocate path accumulated yet
        self._timeout = False                    # tripped if the rollout exceeds self.max_steps
        self._cur_seg = None                     # current segment name, labels the DATAGEN_TRACE rows
        recorder.attach(self.env, self.robot, out_dir,
                        prompt=ctx.diagnostics.get("prompt", ""))

        self._trace_f = None                     # DATAGEN_TRACE: per-step full-state dump for traceback
        self._trace_i = 0                         # per-rollout step counter for the trace rows
        if os.environ.get("DATAGEN_TRACE"):      # (joints q+qdot + target pose/lin+ang vel), per rollout
            self._trace_f = open(os.path.join(str(out_dir), "trace.jsonl"), "w")  # noqa: SIM115

        def tick() -> bool:                      # per-step hook: LTL + step-limit (+ trace), abort on either
            gate.step()
            if self._trace_f is not None:
                self._write_trace_line(ctx)
            if gate.violated:
                return True
            if recorder.n_steps >= self.max_steps:
                self._timeout = True
                return True
            return False

        carry = OPEN                             # last-segment gripper cmd; held during the end settle
        for seg in segments:
            if self._timeout:                      # step-limit tripped in a prior segment's settle/gripper
                recorder.finalize(success=False)
                return DemoResult.fail("timeout", seg=seg.name, n_steps=int(recorder.n_steps))
            self._cur_seg = seg.name               # label the DATAGEN_TRACE rows by segment
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
            if seg.path_begin:
                self._path_buf = []                              # start accumulating the relocate's joint path
            # produce this segment's joint trajectory: a stored-path replay, a straight IK servo,
            # or the default collision-aware cuRobo solve.
            if seg.replay_reverse_path:
                if not self._path_buf:
                    recorder.finalize(success=False)
                    return DemoResult.fail("no_path", seg=seg.name)
                rev = th.flip(th.cat(self._path_buf, dim=0), dims=[0])   # retrace the WHOLE relocate back toward HOME
                k = max(1, int(seg.replay_frac * rev.shape[0]))   # replay_frac of it -> stop ~ (1-frac) into the path
                arm_traj = rev[:k]
                self._path_buf = None
            elif seg.replay_reverse:
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
                res = None
                n_tries = seg.plan_tries if seg.plan_tries is not None else self.plan_tries
                for attempt in range(n_tries):             # retry: cuRobo trajopt is stochastic, a fresh
                    tag = seg.name if attempt == 0 else f"{seg.name}:try{attempt + 1}"   # call explores new seeds
                    res = solve_segment(self.world.motion_gen, self.robot, pos_t, quat_t, q_full,
                                        timeout=self.timeout, attach_obj=attach,
                                        motion_constraint=mc, label=tag, ik_rot_relax=seg.rot_relax,
                                        ik_pos_relax=seg.pos_relax, diagnose_on_fail=True)
                    if res is None and mc is not None:
                        # this cuRobo build often rejects the partial-pose (LINEAR_SERVO) query;
                        # fall back to an unconstrained solve (reference grasp.py:140-147).
                        res = solve_segment(self.world.motion_gen, self.robot, pos_t, quat_t, q_full,
                                            timeout=self.timeout, attach_obj=attach,
                                            motion_constraint=None, label=tag + ":unconstrained",
                                            ik_rot_relax=seg.rot_relax, ik_pos_relax=seg.pos_relax)
                    if res is not None:
                        break
                if res is None:
                    recorder.finalize(success=False)
                    return DemoResult.fail("plan_fail", seg=seg.name)
                arm_traj = res.arm_traj

            # accumulate this segment's joint path so a later replay_reverse_path can retrace it (skip the
            # replay segment itself, which already consumed + cleared the buffer).
            if self._path_buf is not None and not seg.replay_reverse_path:
                self._path_buf.append(arm_traj)

            if os.environ.get("DATAGEN_LOG_JOINTS"):
                self._log_cmd_joints(seg, arm_traj)

            # gripper command DURING the arm motion: closed iff carrying an attached object,
            # unless the segment forces it (a closed-gripper block pushing the drawer = carry_closed).
            carry = CLOSE if (seg.attach if seg.carry_closed is None else seg.carry_closed) else OPEN
            # a SERVO push (or its reverse) runs SLOW so a contacted drawer slides with the gripper
            # instead of being outrun; everything else uses the normal per-waypoint cadence.
            base_spw = seg.servo_spw if seg.servo_spw is not None else self.servo_spw   # per-segment override
            spw = base_spw if (seg.mode == Mode.SERVO or seg.replay_reverse or seg.replay_reverse_path) else self.steps_per_waypoint
            execute_trajectory(self.env, self.robot, arm_traj, gripper_cmd=carry,
                               recorder=recorder, steps_per_waypoint=spw,
                               on_step=tick)
            if self._timeout:
                print(f"[datagen.engine] {seg.name} TIMEOUT: {recorder.n_steps} steps > {self.max_steps}", flush=True)
                recorder.finalize(success=False)
                return DemoResult.fail("timeout", seg=seg.name, n_steps=int(recorder.n_steps))
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

            if os.environ.get("DATAGEN_LOG_JOINTS"):
                self._log_achieved_joints(seg, arm_traj, tpos)
                self._log_contacts(seg)

            # stuck check: did the eef actually REACH its commanded target? If it stalled far short,
            # the held object didn't follow the plan (a jammed path/strategy) — fail fast so the
            # driver retries a different (strategy x grasp x seed) combo.
            if seg.reach_tol_m is not None:
                ez = _np(self.robot.eef_links[self.robot.default_arm].get_position_orientation()[0])
                tp = np.asarray(tpos, float)
                # reach_xy_only: a carry segment whose eef Z droops under the held load (the compliant wrist
                # sags at a high/extended pose) but for which XY-reaching-the-target IS the functional need —
                # the held-above-rim Z is gated separately by verify_held_above_z. Don't fail on benign Z droop.
                err = float(np.linalg.norm((ez - tp)[:2] if seg.reach_xy_only else (ez - tp)))
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

        if self._timeout:                         # a step-limit tripped inside the LAST segment's settle/
            recorder.finalize(success=False)      # gripper has no next-iteration check -- fail cleanly here
            return DemoResult.fail("timeout", seg=seg.name, n_steps=int(recorder.n_steps))

        # Monitored settle-to-rest: step the env (arm + gripper held at their final command) with the
        # gate ticking EVERY step, so post-release physics the segment loop never covered -- an object
        # toppling once the gripper's masked-upright AG hold ends, or the drawer/object settling -- is SEEN
        # by the LTL gate (45deg upright + dropped) BEFORE the success+safety verdict is locked. Without
        # this, a topple completing in the unmonitored gap after the last segment was mislabeled a clean
        # success. Not recorded (the demo still ends on the last segment's frame); only the gate advances,
        # and success() is re-checked below on the SETTLED state (also catches a drawer springing back open).
        if self.rest_settle_steps > 0:
            self._cur_seg = "rest_settle"
            actuate_gripper(self.env, self.robot, close=(carry == CLOSE),
                            n_steps=self.rest_settle_steps, recorder=None, on_step=tick)

        reached = gate.success()
        ok = bool(reached and skeleton.success_extra(ctx) and not gate.violated)
        recorder.finalize(success=ok, attrs={
            **(meta or {}),
            "family": skeleton.name, "seed": int(seed),
            "goal_reached": bool(reached), "ltl_violated": bool(gate.violated),
            "n_steps": int(recorder.n_steps),
        })
        if self._trace_f is not None:
            self._trace_f.close()
        return DemoResult(ok=ok, out_dir=str(out_dir), detail={"goal_reached": bool(reached)})
