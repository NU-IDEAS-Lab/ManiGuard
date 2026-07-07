"""Generic Layer-2 contracts that decouple the family-agnostic executor from the
per-family manip skeletons.

The whole datagen executor is built around ONE idea: a family only answers "which
motion segments make up this task" (``FamilySkeleton.derive_segments``), and the
generic engine plans / executes / gates / records / scales them identically for every
family. Nothing here imports OmniGibson or cuRobo — these are plain data contracts so
they stay cheap to import and trivial to unit-test.

Layer map::

    primitives/ (L1)         scene, curobo_seg.solve_segment, execute, obstacles, record, cameras
    executor/   (L2 generic) contracts (this file) + engine + variation + grasp_select + gate + geometry
    families/   (L2 specific) clutter.py ...: ONLY implement FamilySkeleton
    driver.py   (L3)         orchestrate: task -> variants -> engine -> collect success+safe demos
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Mode(str, Enum):
    """How the engine produces a segment's joint trajectory."""

    FREE = "free"             # free-space, fully collision-free cuRobo plan to the eef target
    LINEAR = "linear_servo"   # LINEAR_SERVO: lock orientation + perpendicular position, free
    #                           ONLY the approach axis (top-down => "fix xy, move z straight")
    SERVO = "servo"           # straight-line Cartesian IK servo: interpolate the eef from the
    #                           current pose to the target (orientation held), per-waypoint IK with
    #                           COLLISION OFF — a deliberate straight push into contact (e.g. shoving
    #                           a drawer shut) that cuRobo's collision-avoiding planner would refuse.
    #                           The executed joint path is stored so a later segment can replay it.


class Grip(str, Enum):
    """Gripper action applied AFTER a segment reaches its target."""

    HOLD = "hold"
    OPEN = "open"
    CLOSE = "close"


@dataclass
class MotionSegment:
    """One motion the engine plans + executes as a unit. A family's ``derive_segments``
    returns an ordered list of these; the engine never knows which family produced them."""

    name: str                              # "approach" / "grasp" / "lift" / "transport" / "to_goal"
    eef_pos: np.ndarray                    # world eef target position (3,)
    eef_quat: np.ndarray                   # world eef target orientation, xyzw (4,)
    mode: Mode = Mode.FREE
    attach: bool = False                   # plan with the held target attached (avoid obstacles WITH it)
    ignore_objects: tuple[str, ...] = ()   # obstacle names to drop for THIS segment (e.g. the target during grasp approach)
    ignore_clutter: bool = False           # drop the support + ALL clutter (world-collision-off) for this segment's
    #                                        plan — for the final grasp descent into the cluttered, near-table region
    #                                        (a short controlled move; execution safety is the physics + AG + LTL gate)
    grip: Grip = Grip.HOLD                 # gripper action after the target is reached
    grip_steps: int = 0                    # settle/close/open steps for that gripper action
    min_clearance_m: float | None = None   # if set, engine VERIFIES clearance AFTER this segment (>= this floor)
    target_clearance_m: float | None = None  # lift AIM clearance (>= min_clearance_m; varies per draw for
    #                                          height diversity). None => aim at min_clearance_m.
    compute: str | None = None             # runtime-resolved eef target (None => use eef_pos/eef_quat as given).
    #                                        GENERIC tags the engine resolves inline from the LIVE state:
    #                                          "lift_to_clearance"  -> raise z until held lowest-z clears clutter + min_clearance
    #                                          "over_goal"          -> move xy over the goal, keep current height + orientation
    #                                          "aim_to_goal_center" -> translate so the held object CENTRE lands on the goal-sphere centre
    #                                        any OTHER tag is delegated to ``FamilySkeleton.resolve_compute`` (family-specific,
    #                                        e.g. cabinet's grasp-handle / pull-drawer / regrasp-moved-target).
    extra: dict = field(default_factory=dict)   # family payload for resolve_compute (grasp pose, standoff, open_dist, ...)
    replay_reverse: bool = False           # execute the REVERSED joint path of the most recent SERVO segment
    #                                        (retreat straight back out the way the push came in — that lane is
    #                                        guaranteed clear). The eef_pos/quat/compute are ignored for this segment.
    path_begin: bool = False               # start accumulating this + the following segments' joint trajectories
    #                                        into one buffer (the relocate path from ~HOME to the place spot).
    replay_reverse_path: bool = False      # execute the REVERSED accumulated path (``replay_frac`` of it) — retrace
    #                                        the whole relocate BACK to ~HOME via the exact joint path it took out, so
    #                                        the next FREE plan starts from the good config the first pick solved from
    #                                        (JointController is joint-space → reverse-replay is drift-free, no replan).
    replay_frac: float = 0.8               # fraction of the reversed path to replay (0.8 => return to ~20% = near HOME)
    carry_closed: bool | None = None       # gripper command DURING the motion: None => default (closed iff attach);
    #                                        True => hold CLOSED (e.g. a closed-gripper block shoving a drawer, no attach)
    reach_tol_m: float | None = None       # if set, after the segment VERIFY the eef reached its commanded target
    reach_xy_only: bool = False            # with reach_tol_m: check only the XY distance (ignore benign Z droop of the held load; the held-above-rim Z is gated by verify_held_above_z)
    #                                        within this tol; else fail "stuck" (the held object didn't follow the
    #                                        plan — a path/strategy that jammed → the driver retries another combo)
    verify_held_above_z: float | None = None  # if set, after the segment VERIFY the HELD object's lowest point is
    #                                        ABOVE this absolute world z; else fail "below_z". Guards a lift that
    #                                        must clear an obstacle's top BEFORE any lateral move (e.g. raise the
    #                                        target clear of the drawer rim before moving it over the cavity —
    #                                        moving while still below the rim catches the rim and rams the drawer)
    plan_tries: int | None = None          # per-segment override of the engine's cuRobo FREE/LINEAR plan retries
    #                                        (None => engine default). A hard segment (e.g. the close re-grasp)
    #                                        gets more shots; the rest keep the cheap default.
    servo_step_m: float | None = None      # SERVO only: per-segment eef-interpolation step (None => engine
    #                                        default). FINER = smaller per-step eef increment.
    servo_spw: int | None = None           # SERVO only: per-segment sim-steps-per-waypoint (None => engine
    #                                        default). servo_step_m small + servo_spw=1 => each sim step advances
    #                                        the eef one small uniform increment (the rigid controller slams only
    #                                        that tiny amount) => continuous UNIFORM-velocity glide instead of the
    #                                        default slam-1cm-then-idle stutter. Used for the gentle drawer close.
    rot_relax: float | None = None         # FREE/LINEAR only: temporarily widen cuRobo's IK rotation_threshold (rad)
    #                                        + the trajopt-salvage rot_tol for THIS segment's plan only (try/finally
    #                                        restored). Use when the goal orientation's reach-limited mismatch is on a
    #                                        PHYSICALLY SYMMETRIC axis (e.g. the roll-symmetric cabinet handle bar), so
    #                                        the few-degree miss is harmless: cuRobo would otherwise IK_FAIL at the
    #                                        far-reach open handle and never produce a trajectory to grasp it.
    pos_relax: float | None = None         # FREE/LINEAR only: same lever for cuRobo's IK position_threshold (m). At a
    #                                        STANDOFF pre-grasp the arm at its reach envelope lands a few mm short
    #                                        (measured 5.2-5.8 mm > the 5 mm gate); that residual is absorbed by the
    #                                        following SERVO which re-aims from the LIVE target pose. Pair with rot_relax
    #                                        — at the far reach EITHER axis can bind first (whichever does => IK_FAIL).
    reach_fallback: bool = False            # transport only: if the precise (eef→goal / object-centre→goal-centre)
    #                                        plan fails on a FAR goal, the engine may relax to the closest-to-goal
    #                                        placement that is IK-reachable AND still leaves the held object
    #                                        intersecting the goal sphere (pull the eef back toward the robot;
    #                                        optionally add an upright-preserving world-Z yaw). See engine
    #                                        ._reach_fallback_transport. The precise plan stays the normal path.
    free_fallback: bool = False             # SERVO only: if the straight-line IK servo can't reach the target
    #                                        (servo_ik_fail), fall back to a collision-aware cuRobo FREE solve with
    #                                        an orientation-hold constraint (obstacles.UPRIGHT_HOLD). Lets a
    #                                        transport run as an orientation-safe SERVO gate by default, only
    #                                        invoking cuRobo (which can tilt the held object) when the servo
    #                                        genuinely can't reach a far goal. See engine dispatch.
    require_attach: bool = False           # after this segment's CLOSE grip, VERIFY AG actually attached the
    #                                        held object (is_grasping == 1); else fail "no_attach" fast instead
    #                                        of blindly carrying nothing through the rest of the rollout
    no_salvage: bool = False                # require a genuinely-successful (collision-free) cuRobo solve —
    #                                        skip the endpoint-tol salvage (which can keep a colliding path).
    #                                        The stack target transport sets this so a winding salvaged path
    #                                        never knocks the just-built re-stack pile.


@dataclass
class TaskContext:
    """Everything parsed once per base task, shared by sampler / skeleton / engine.
    ``env`` / ``robot`` / ``target`` / ``support`` are opaque OmniGibson handles."""

    env: Any
    robot: Any
    target: Any                            # the grasp/transport target object
    target_key: str                        # "category/model" — the annotation DB key
    target_name: str                       # scene object name
    goal_center: np.ndarray                # world (3,) goal-sphere centre
    goal_radius: float                     # goal-sphere radius (m)
    support: Any | None = None             # support-surface object (the table)
    diagnostics: dict = field(default_factory=dict)
    grasp_record: dict = field(default_factory=dict)   # this target's annotation entry (its grasps[])


@dataclass
class GraspCand:
    """A candidate grasp = a world eef-target pose (from the annotation DB) plus the
    cuRobo reachability score filled in by ``grasp_select``."""

    id: int                                # annotation grasp id
    eef_pos: np.ndarray                    # world (3,)
    eef_quat: np.ndarray                   # world xyzw (4,)
    approach: str = "top_down"             # metadata (top_down / side)
    score: float = 0.0                     # cuRobo reachability / IK-cost / collision score (higher = better)
    reachable: bool = True
    margin: float = float("-inf")          # joint-limit margin (rad) of the chosen roll variant; -inf = unreachable
    chosen_roll: bool = False              # True if the 180°-about-approach roll variant won (vs the annotated quat)
    chosen_quat: np.ndarray | None = None  # world xyzw of the chosen roll variant (selection-time, for place IK checks)
    is_top_down: bool = False              # geometric: approach axis within TOP_DOWN_MAX_TILT_DEG of straight-down
    wrist_open: bool = False               # geometric: approach leans toward the cabinet -> wrist/arm on the drawer-OPEN side, clear of the cabinet (relocate only; needs prefer_wrist_dir)


@dataclass
class SampleParams:
    """The per-demo diversity knobs the VariationSampler draws; consumed by
    ``derive_segments`` to vary waypoints while staying safe + reachable."""

    seed: int = 0                          # the variant's master seed: cuRobo trajopt (torch.manual_seed)
    #                                        in the engine + jitter/lift draws in the sampler all derive from it
    draw_index: int = 0                    # the draw index k (grasp-independent) this variant came from; the
    #                                        driver persists max(draw_index)+1 as the resume cursor next_draw
    standoff_m: float = 0.10               # pre-grasp standoff along the approach axis
    min_clearance_m: float = 0.03          # required clearance floor over clutter during transport
    lift_clearance_mult: float = 1.0       # lift AIM = min_clearance_m × this (1.0–1.5 for height diversity)
    jitter: dict[str, Any] = field(default_factory=dict)   # {"above_xy": (dx, dy), ...}


@dataclass
class DemoResult:
    """Outcome of one engine run on one variant."""

    ok: bool
    fail_stage: str | None = None          # which segment / why it failed
    detail: dict = field(default_factory=dict)
    out_dir: str | None = None

    @classmethod
    def fail(cls, stage: str, **detail: Any) -> DemoResult:
        return cls(ok=False, fail_stage=stage, detail=detail)


class FamilySkeleton(ABC):
    """The ONLY thing each family implements. Pure planning logic — no cuRobo, no
    execution, no recording (the engine owns all of those)."""

    name: str = "base"

    @abstractmethod
    def grasp_candidates(self, ctx: TaskContext) -> list[GraspCand]:
        """The target's candidate grasp eef-targets in WORLD frame. clutter: look up the
        target's annotation grasps and transform object-local -> world via the live pose."""

    @abstractmethod
    def derive_segments(self, ctx: TaskContext, grasp: GraspCand,
                        params: SampleParams) -> list[MotionSegment]:
        """Turn (target, chosen grasp, goal, obstacles) into the ordered MotionSegment list
        = this family's manip skeleton."""

    def success_extra(self, ctx: TaskContext) -> bool:
        """Extra family-specific success terms beyond the generic goal check. Default: none
        (clutter). e.g. lid overrides this with a lid_attached check."""
        return True

    def resolve_compute(self, tag: str, seg: "MotionSegment", ctx: TaskContext):
        """Resolve a FAMILY-SPECIFIC ``compute`` tag to a world ``(eef_pos, eef_quat)`` from the
        LIVE state (the engine handles the generic tags itself and delegates the rest here).
        Default: unsupported. Cabinet overrides this for grasp-handle / pull / regrasp."""
        raise ValueError(f"{type(self).__name__} has no resolve_compute for tag {tag!r}")

    def select_grasps(self, ctx: TaskContext, world: Any, robot: Any) -> None:
        """Optional per-task hook (called once by the driver after the CuroboWorld is built,
        before the variant loop) to pre-select reachable AUXILIARY grasps the family resolves
        internally — e.g. cabinet's drawer-handle / obstacle grasps, which are NOT the target
        grasp the sampler iterates and so cannot be filtered by the driver's grasp scoring.
        Default: no-op (clutter has only the target grasp, already scored by the driver)."""
        return None

    def variation_knobs(self, ctx: TaskContext) -> dict[str, Any]:
        """Which waypoints / ranges may jitter for diversity. Default: engine defaults."""
        return {}

    def score_drop_extra(self, ctx: TaskContext) -> list:
        """Extra scene objects the driver should ALSO drop from the collision world when it scores the
        TARGET grasp (``score_grasps`` accepts a LIST target = multi-drop). Default: none. A family whose
        target is BURIED at scoring time but EXPOSED by execution time (stack: the bottom object under the
        pile, which is uncovered before it is grasped) returns the objects that will be gone by then —
        else the still-covered target scores 0-reachable and the sampler yields no variants."""
        return []

    def score_margin_floor(self) -> float | None:
        """Family override for ``score_grasps``' joint-limit margin floor (rad). Default ``None`` =
        the shared ``MARGIN_FLOOR`` — other families are untouched. A family whose few viable side
        grasps land just under the shared floor on edge placements (jar: best grasp ~0.17 after a
        yaw surgery) may relax it slightly rather than yield 0 attempts."""
        return None

    def relocate_prefer_top_down(self) -> bool:
        """Whether the driver should rank the TARGET's Phase-1 relocate grasps top-down-first (a
        straight vertical relocate lift stalls a side grasp's wrist; a top-down grasp lifts cleanly).
        Default False (clutter: goal-region placement, no such lift constraint). Cabinet returns True."""
        return False

    def grasping_mode(self) -> str:
        """OmniGibson AG mode baked into the datagen env for this family. Default ``"assisted"``
        (magnetize only an object held BETWEEN both fingers — realistic force-closure). A family
        whose target is un-graspable by force closure (e.g. a tabletop slab wider than the gripper
        in both horizontal axes, which an edge-pinch only shoves away) overrides to ``"sticky"``
        (magnetize on first single-finger contact), so the demo teaches the task SEQUENCE rather
        than stalling on an impossible grasp. Keep eval's grasp mode consistent with this."""
        return "assisted"

    def relocate_open_dir(self, ctx: TaskContext):
        """World-frame xy unit vector the relocate grasp's WRIST should trail toward, so the arm body
        stays clear of an obstruction when picking objects in front of it. The grasp scorer prefers
        grasps whose approach leans OPPOSITE this (toward the obstruction), putting the wrist on this
        side. Default None (no azimuth preference). Cabinet returns the drawer-OPEN direction: picking
        from the open side keeps the arm off the closed cabinet (else cuRobo can't reach + the SERVO
        descend jams on the cabinet). Returns ``np.ndarray | None``."""
        return None

    def debug_state(self, ctx: TaskContext) -> str:
        """Optional one-line family state string the engine prints after each segment when
        ``DATAGEN_DEBUG_STATE`` is set (e.g. cabinet's live drawer-joint value). Default: none."""
        return ""

    def on_segment(self, seg: "MotionSegment", ctx: TaskContext) -> None:
        """Optional side-effect hook the engine calls just BEFORE planning each segment — for runtime
        state a family must toggle mid-rollout that isn't a pose (e.g. cabinet stiffening the drawer
        joint to hold it open while the arm reaches over it, then softening it for the deliberate
        close). Default: no-op."""
        return None
