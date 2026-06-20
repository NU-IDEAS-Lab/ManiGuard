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
    """How cuRobo plans a segment."""

    FREE = "free"             # free-space, fully collision-free plan to the eef target
    LINEAR = "linear_servo"   # LINEAR_SERVO: lock orientation + perpendicular position, free
    #                           ONLY the approach axis (top-down => "fix xy, move z straight")


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
    #                                        the engine resolves these from the LIVE post-grasp state via geometry:
    #                                          "lift_to_clearance"  -> raise z until held lowest-z clears clutter + min_clearance
    #                                          "over_goal"          -> move xy over the goal, keep current height + orientation
    #                                          "aim_to_goal_center" -> translate so the held object CENTRE lands on the goal-sphere centre


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


@dataclass
class SampleParams:
    """The per-demo diversity knobs the VariationSampler draws; consumed by
    ``derive_segments`` to vary waypoints while staying safe + reachable."""

    seed: int = 0                          # the variant's master seed: cuRobo trajopt (torch.manual_seed)
    #                                        in the engine + jitter/lift draws in the sampler all derive from it
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
        """Extra family-specific success terms beyond the generic held-in-goal. Default:
        none (clutter). e.g. lid overrides this with a lid_attached check."""
        return True

    def variation_knobs(self, ctx: TaskContext) -> dict[str, Any]:
        """Which waypoints / ranges may jitter for diversity. Default: engine defaults."""
        return {}
