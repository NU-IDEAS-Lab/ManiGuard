"""Cabinet FIRSTHALF skeleton — the shipped cabinet demo, stopped once the drawer is open.

The sim counterpart of the real-robot ``higher-firsthalf`` setup, which ends after the
blocking object is moved aside and the drawer is pulled open (the full-horizon real policy
is ``higherZ``). Pairing the two lets the sim2real comparison use the same task horizon on
both sides.

The full cabinet demo is four phases::

    1  relocate blockers          ┐ kept
    2  open drawer                ┘ kept   <- ends at `handle_back_open`
    3  place target in drawer      dropped
    4  close drawer                dropped

This subclass does NOT re-derive phases 1-2. It calls ``CabinetSkeleton.derive_segments``
and truncates, so the kept prefix goes through the *same code path with the same RNG draws*
as the full-horizon demo: for a given ``(grasp, draw_index)`` the firsthalf demo's blocker
placement, open distance, and jitter are identical to the full one's. The two collections
are therefore matched pairs differing only in horizon — which re-deriving the phases here,
or truncating recorded trajectories after the fact, would not give.

Truncating a RECORDED trajectory was the alternative and does not work: the recorder stores
no segment labels (state / actions / sim-state dumps only), the drawer joint is buried in a
serialized state blob, and the correct cut is not "drawer fully open" but after the gripper
releases the handle and backs off -- a boundary that the gripper signal cannot disambiguate,
since phase 2 ends OPEN and phase 3 opens again to pre-grasp. Cutting by construction, here,
is exact.

⚠️ Success and prompt are NOT defined here. The shipped task's goal is ``inside & closed``,
which a firsthalf demo can never satisfy, so every demo would be discarded by the gate. Run
this family WITH the matching horizon-override table, which substitutes the goal and the
instruction together::

    python -m maniguard.data.datagen.driver \\
        --task-dir <bench>/cabinet_pickup/task_0019/base --family cabinet_firsthalf \\
        --horizon-override configs/firsthalf/cabinet_task0019.json ...

LTL safety is inherited unchanged: cabinet's constraints are pure safety formulas
(``G (...)``), which stay meaningful on a truncated horizon.
"""
from __future__ import annotations

from maniguard.data.datagen.executor.contracts import GraspCand, MotionSegment, SampleParams, TaskContext
from maniguard.data.datagen.families.cabinet import CabinetSkeleton

# Last segment of phase 2 (`_open_drawer`): the gripper lifts straight up off the handle bar
# after releasing it. Emitted unconditionally, so it is a reliable cut point.
LAST_KEPT_SEGMENT = "handle_back_open"


class CabinetFirstHalfSkeleton(CabinetSkeleton):
    """Phases 1-2 of the cabinet demo (relocate blockers, open drawer)."""

    name = "cabinet_firsthalf"

    def derive_segments(self, ctx: TaskContext, target_grasp: GraspCand,
                        params: SampleParams) -> list[MotionSegment]:
        segs = super().derive_segments(ctx, target_grasp, params)
        if not segs:
            return segs                       # parent's "no room to relocate" early-out
        names = [s.name for s in segs]
        if LAST_KEPT_SEGMENT not in names:
            # Refuse to guess a cut point: silently keeping the whole sequence would collect
            # full-horizon demos under the firsthalf label.
            raise ValueError(
                f"{self.name}: no {LAST_KEPT_SEGMENT!r} segment to truncate at "
                f"(got {names}) -- the cabinet phase layout changed; update this skeleton."
            )
        cut = len(names) - 1 - names[::-1].index(LAST_KEPT_SEGMENT)   # last occurrence
        kept = segs[:cut + 1]
        print(f"[datagen.cab.firsthalf] truncated {len(segs)} -> {len(kept)} segments "
              f"(ends at {kept[-1].name!r}; dropped {names[cut + 1:]})", flush=True)
        return kept

    def demo_attrs(self, ctx: TaskContext) -> dict:
        """Record how far the drawer actually ended up open.

        The demo is COMMANDED to ``open_dist`` (a reachability search per task, derated by
        ``OPEN_DIST_SAFETY``), but what matters for scoring a policy is what the arm ACHIEVED,
        servo tracking error included. A firsthalf demo ends with the drawer open, so the
        end-of-demo joint position is that value directly -- unlike the full-horizon demo,
        which closes the drawer again in phase 4.

        The distribution of this field over the collected demos is what calibrates the eval
        threshold in ``configs/firsthalf/*.json``; recording it is the only reason a policy is
        later asked for a number rather than OmniGibson's boolean Open state."""
        P = self._prepare(ctx)
        cab = P["cab"]
        return {
            "open_joint_achieved": float(cab.get_joint_positions()[self._drawer_jidx(cab, ctx)]),
            "open_dist_commanded": float(P["open_dist"]),
            "drawer_stroke_m": float(ctx.diagnostics["cabinet_info"]["stroke_m"]),
        }
