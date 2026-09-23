"""Configure the cuRobo collision world and segment constraints.

CuroboWorld creates the motion generator and updates obstacles with optional
object exclusions. The gripper-collision context manager disables selected links
when the planner exposes that API; otherwise it warns once and changes nothing.
LINEAR_SERVO and UPRIGHT_HOLD provide partial-pose weights for motion planning.
The mimic-joint patch fills missing finger positions when reindexing joint state.
"""
from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager

# Franka panda gripper links — disabled from the collision world while the fingers
# close around a target. Names must match the URDF cuRobo loaded for the Franka.
GRIPPER_COLLISION_LINKS = ("panda_hand", "panda_leftfinger", "panda_rightfinger")

# Partial-pose-hold weights (orientation x3, position x3); the trailing 0.0 frees the
# approach axis → a pure linear servo. Pass as solve_segment(motion_constraint=...).
LINEAR_SERVO = [0.1, 0.1, 0.1, 0.1, 0.1, 0.0]

# Orientation-only hold: lock the eef ROTATION (x3), free ALL position (x3). Used for the
# t_transport FREE fallback so the held object keeps ~its grasp (upright) orientation THROUGHOUT the
# cuRobo path — an unconstrained FREE plan is free to route through configs that tilt the rigidly-held
# object >45deg mid-path, tripping the per-step LTL upright check. Best-effort: this cuRobo build may
# reject the partial-pose query (solve_segment then retries unconstrained).
UPRIGHT_HOLD = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


_MIMIC_PATCHED = False
_TOGGLE_WARNED = False


def _install_mimic_patch() -> None:
    """Make cuRobo's mimic-joint reindex tolerant of a missing finger joint.

    cuRobo (Stanford fork) reindexes the joint state against every mimic source
    joint; Franka's ``panda_finger_joint1`` is a mimic source but OG's joint state
    into ``compute_trajectories`` doesn't always carry the finger DOFs (the
    MultiFingerGripperController owns them), so the reindex raises
    ``ValueError: 'panda_finger_joint1' is not in list`` and the whole plan errors.
    Fill missing finger joints with 0.04 m (fully open — open-gripper collision
    spheres are the correct geometry; a closed gripper's are too thin) instead of
    raising. Idempotent.
    """
    global _MIMIC_PATCHED
    if _MIMIC_PATCHED:
        return
    import torch as th
    from curobo.types.state import JointState

    _orig_reindex = JointState.inplace_reindex
    _fills = {"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04}

    def inplace_reindex_tolerant(self, joint_names):
        if self.joint_names is None:
            raise ValueError("joint names are not specified in JointState")
        missing = [j for j in joint_names if j not in self.joint_names]
        if not missing:
            return _orig_reindex(self, joint_names)
        device = self.position.device
        fill_shape = list(self.position.shape)
        fill_shape[-1] = 1
        for j in missing:
            col = th.full(fill_shape, _fills.get(j, 0.0), device=device,
                          dtype=self.position.dtype)
            self.position = th.cat([self.position, col], dim=-1)
        for attr in ("velocity", "acceleration", "jerk"):
            val = getattr(self, attr, None)
            if val is not None:
                zeros = th.zeros(list(self.position.shape[:-1]) + [len(missing)],
                                 device=device, dtype=val.dtype)
                setattr(self, attr, th.cat([val, zeros], dim=-1))
        self.joint_names = list(self.joint_names) + missing
        return _orig_reindex(self, joint_names)

    JointState.inplace_reindex = inplace_reindex_tolerant
    _MIMIC_PATCHED = True


class CuroboWorld:
    """The cuRobo motion generator + its obstacle world + constraint levers.

    Build once per scene (kernel warm-up is the expensive part), then reuse its
    ``motion_gen`` across every ``solve_segment`` call::

        world = CuroboWorld(env, robot)
        world.update_obstacles()                       # load the scene
        res = solve_segment(world.motion_gen, robot, ...)
        with world.gripper_collision_disabled():       # clamp around a target
            res = solve_segment(world.motion_gen, robot, ..., motion_constraint=LINEAR_SERVO)
    """

    def __init__(self, env, robot, *, enable_head_tracking: bool = False):
        from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection
        from omnigibson.action_primitives.starter_semantic_action_primitives import (
            StarterSemanticActionPrimitives,
        )

        _install_mimic_patch()
        self.env = env
        self.robot = robot
        self._primitives = StarterSemanticActionPrimitives(
            env, robot, enable_head_tracking=enable_head_tracking)
        self.motion_gen = self._primitives._motion_generator
        self._raw_mg = self.motion_gen.mg[CuRoboEmbodimentSelection.DEFAULT]

    def update_obstacles(self, ignore_objects: Sequence = ()) -> None:
        """(Re)load the scene into the collision world, ignoring ``ignore_objects``
        (e.g. a just-grasped target whose held pose would otherwise read as a
        self-collision). Call after the scene changes (grasp/release)."""
        self.motion_gen.update_obstacles(ignore_objects=list(ignore_objects))

    @contextmanager
    def gripper_collision_disabled(self):
        """Temporarily disable gripper-link collision checks when the planner exposes toggle_link_collision, restoring them on exit. If the API is unavailable, warn once and leave collision settings unchanged."""
        global _TOGGLE_WARNED
        raw = self._raw_mg
        toggleable = hasattr(raw, "toggle_link_collision")
        if not toggleable and not _TOGGLE_WARNED:
            print("[datagen.obstacles] WARNING: this cuRobo build has no "
                  "toggle_link_collision; gripper_collision_disabled() is a NO-OP. "
                  "Use update_obstacles(ignore_objects=...) / attach_obj instead.",
                  flush=True)
            _TOGGLE_WARNED = True
        if toggleable:
            raw.toggle_link_collision(list(GRIPPER_COLLISION_LINKS), False)
        try:
            yield
        finally:
            if toggleable:
                raw.toggle_link_collision(list(GRIPPER_COLLISION_LINKS), True)
