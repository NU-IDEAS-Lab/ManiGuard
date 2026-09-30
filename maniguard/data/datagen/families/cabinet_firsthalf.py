"""Collect the cabinet sequence through blocker relocation and drawer opening.

Reuse CabinetSkeleton.derive_segments and truncate after handle_back_open,
including handle release and retreat. The prefix uses the same segment-generation
code and random draws as the full sequence.

Pass a matching --horizon-override table to replace the full task's goal and
instruction. Without it, the full inside-and-closed goal remains unsatisfied.
The task's LTL specification is unchanged.

Example:
    python -m maniguard.data.datagen.driver \
        --task-dir <bench>/cabinet_pickup/task_0019/base --family cabinet_firsthalf \
        --horizon-override configs/firsthalf/cabinet_task0019.json --target 1
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

        This field supports calibration of the numeric drawer-opening threshold in
        ``configs/firsthalf/*.json`` from collected demonstrations."""
        P = self._prepare(ctx)
        cab = P["cab"]
        return {
            "open_joint_achieved": float(cab.get_joint_positions()[self._drawer_jidx(cab, ctx)]),
            "open_dist_commanded": float(P["open_dist"]),
            "drawer_stroke_m": float(ctx.diagnostics["cabinet_info"]["stroke_m"]),
        }
