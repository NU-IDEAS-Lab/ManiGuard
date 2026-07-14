"""Runtime patches that add ManiGuard-specific behavior to upstream OmniGibson.

The refactor/omnigibson branch is incrementally peeling ManiGuard's code out of
the vendored ``OmniGibson/`` tree so OmniGibson can be consumed as an upstream
dependency. This module is the central place where the *remaining* runtime
modifications live. Two kinds of patches are applied:

1. **Post-load hook on ``omnigibson.object_states``** — the moment the
   ``omnigibson.object_states`` subpackage finishes executing its ``__init__``,
   ManiGuard injects three extra names: ``Dropped`` / ``Upright`` (new state
   classes defined under :mod:`maniguard.object_states`) plus ``Grasped`` as a
   backwards-compatible alias of upstream ``IsGrasping``. This runs *during*
   ``import omnigibson`` so downstream modules that reference
   ``object_states.Grasped`` / ``object_states.Upright`` at module load time
   (e.g. ``omnigibson.utils.bddl_utils``) see the attributes they expect.

2. **Eager class / function patches** — applied after ``import omnigibson``
   completes:

   * ``omnigibson.object_states.factory._DEFAULT_STATE_SET`` gains the two new
     states so any object can be annotated with them.
   * ``omnigibson.termination_conditions.grasp_goal.GraspGoal`` learns an
     optional ``hold_steps`` counter.
   * ``omnigibson.reward_functions.grasp_reward.GraspReward`` falls back to an
     alternative link when robots lack ``torso_lift_link``.
   * ``omnigibson.utils.sampling_utils.draw_debug_markers`` is replaced with a
     tensor-dtype/device-safe variant.
   * ``omnigibson.utils.bddl_utils.SUPPORTED_PREDICATES`` gains four new keys
     (``upright`` / ``dropped`` / ``grasped`` / ``stashed``) registered from
     :mod:`maniguard.utils.bddl_predicates`.

Set ``MANIGUARD_SKIP_OMNIGIBSON_PATCH=1`` in the environment to opt out.

Two upstream files still carry ManiGuard modifications on this branch
(``utils/bddl_utils.py``, ``tasks/grasp_task.py``). Extracting them requires
either upstream PRs or a full sys.modules override of the module and is
tracked as follow-up work.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import os
import sys

_HOOK_INSTALLED = False
_EAGER_APPLIED = False


class _ObjectStatesPostLoadFinder(importlib.abc.MetaPathFinder):
    """Wraps the loader for ``omnigibson.object_states`` so maniguard can inject
    extra state classes the moment the package finishes loading."""

    TARGET = "omnigibson.object_states"

    def __init__(self) -> None:
        self._in_progress = False

    def find_spec(self, fullname, path, target=None):  # noqa: D401
        if fullname != self.TARGET or self._in_progress:
            return None

        # Re-enter the import system to locate the real spec without recursing
        # through this finder.
        self._in_progress = True
        try:
            real_spec = None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                find = getattr(finder, "find_spec", None)
                if find is None:
                    continue
                try:
                    found = find(fullname, path, target)
                except Exception:
                    found = None
                if found is not None:
                    real_spec = found
                    break
            if real_spec is None:
                return None

            original_loader = real_spec.loader

            class _WrappedLoader(importlib.abc.Loader):
                def create_module(self, spec):  # noqa: D401
                    if hasattr(original_loader, "create_module"):
                        return original_loader.create_module(spec)
                    return None

                def exec_module(self, module):  # noqa: D401
                    original_loader.exec_module(module)
                    _inject_state_aliases(module)

            real_spec.loader = _WrappedLoader()
            return real_spec
        finally:
            self._in_progress = False


def _inject_state_aliases(object_states_module) -> None:
    """Add Dropped / Upright / Grasped attrs to ``omnigibson.object_states``."""
    from maniguard.object_states.dropped import Dropped
    from maniguard.object_states.upright import Upright

    if hasattr(object_states_module, "IsGrasping") and not hasattr(
        object_states_module, "Grasped"
    ):
        object_states_module.Grasped = object_states_module.IsGrasping
    if not hasattr(object_states_module, "Dropped"):
        object_states_module.Dropped = Dropped
    if not hasattr(object_states_module, "Upright"):
        object_states_module.Upright = Upright

    all_list = getattr(object_states_module, "__all__", None)
    if isinstance(all_list, list):
        for name in ("Dropped", "Upright", "Grasped"):
            if name not in all_list:
                all_list.append(name)

    # Also tag the submodule so downstream code that already imported
    # ``omnigibson.object_states.robot_related_states`` can see ``Grasped``.
    rrs = sys.modules.get("omnigibson.object_states.robot_related_states")
    if rrs is not None and hasattr(rrs, "IsGrasping") and not hasattr(rrs, "Grasped"):
        rrs.Grasped = rrs.IsGrasping


def _install_import_hook() -> None:
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return
    sys.meta_path.insert(0, _ObjectStatesPostLoadFinder())
    _HOOK_INSTALLED = True

    # If omnigibson.object_states was already imported before maniguard loaded,
    # the hook will never fire. Patch retroactively in that case.
    already = sys.modules.get("omnigibson.object_states")
    if already is not None:
        _inject_state_aliases(already)


def _extend_factory_lists() -> None:
    import omnigibson.object_states.factory as factory
    from maniguard.object_states.dropped import Dropped
    from maniguard.object_states.upright import Upright

    factory._DEFAULT_STATE_SET = frozenset(
        set(factory._DEFAULT_STATE_SET) | {Dropped, Upright}
    )


def _patch_grasp_goal() -> None:
    from omnigibson.termination_conditions.grasp_goal import GraspGoal

    if getattr(GraspGoal, "_maniguard_patched", False):
        return

    original_init = GraspGoal.__init__

    def __init__(self, obj_name, hold_steps: int = 1):
        original_init(self, obj_name)
        self.hold_steps = max(1, int(hold_steps))
        self._held_steps = 0

    def _step(self, task, env, action):
        self.obj = (
            env.scene.object_registry("name", self.obj_name) if self.obj is None else self.obj
        )
        robot = env.robots[0]
        obj_in_hand = robot._ag_obj_in_hand[robot.default_arm]
        is_grasping = obj_in_hand == self.obj
        if is_grasping:
            self._held_steps += 1
        else:
            self._held_steps = 0
        return self._held_steps >= self.hold_steps

    original_reset = GraspGoal.reset

    def reset(self, task, env):
        original_reset(self, task, env)
        self._held_steps = 0

    GraspGoal.__init__ = __init__
    GraspGoal._step = _step
    GraspGoal.reset = reset
    GraspGoal._maniguard_patched = True


def _patch_grasp_reward() -> None:
    from omnigibson.reward_functions.grasp_reward import GraspReward

    if getattr(GraspReward, "_maniguard_patched", False):
        return

    def _robot_center(robot):
        if "torso_lift_link" in robot.links:
            return robot.links["torso_lift_link"].get_position_orientation()[0]
        root_link = getattr(robot, "root_link", None)
        if root_link is not None:
            return root_link.get_position_orientation()[0]
        return next(iter(robot.links.values())).get_position_orientation()[0]

    original_step = GraspReward._step

    def _step(self, task, env, action):
        try:
            return original_step(self, task, env, action)
        except KeyError as exc:
            if "torso_lift_link" not in str(exc):
                raise
            # Retry with a robot-center that doesn't assume torso_lift_link.
            import math
            import omnigibson.utils.transform_utils as T

            robot = env.robots[0]
            robot_center = _robot_center(robot)
            obj_center = self.obj.get_position_orientation()[0]
            dist = T.l2_distance(robot_center, obj_center)
            return math.exp(-dist) * self.dist_coeff

    GraspReward._step = _step
    GraspReward._maniguard_patched = True


def _patch_sampling_utils() -> None:
    import omnigibson.utils.sampling_utils as sampling_utils

    if getattr(sampling_utils, "_maniguard_patched", False):
        return

    import torch as th

    def draw_debug_markers(hit_positions, radius=0.01):
        from omnigibson.utils.ui_utils import draw_line

        hit_positions = th.as_tensor(hit_positions, dtype=th.float32)
        color = th.cat(
            [
                th.rand(3, dtype=hit_positions.dtype, device=hit_positions.device),
                th.ones(1, dtype=hit_positions.dtype, device=hit_positions.device),
            ]
        )
        unit_axes = th.eye(3, dtype=hit_positions.dtype, device=hit_positions.device)
        color_rgba = tuple(float(x) for x in color.detach().cpu().tolist())
        for vec in hit_positions:
            for dim in range(3):
                start_point = vec + unit_axes[dim] * radius
                end_point = vec - unit_axes[dim] * radius
                draw_line(
                    tuple(float(x) for x in start_point.detach().cpu().tolist()),
                    tuple(float(x) for x in end_point.detach().cpu().tolist()),
                    color_rgba,
                )

    sampling_utils.draw_debug_markers = draw_debug_markers
    sampling_utils._maniguard_patched = True


def _patch_franka_longfinger() -> None:
    import omnigibson.robots.franka as franka_mod
    from omnigibson.utils.asset_utils import get_dataset_path

    franka_cls = franka_mod.FrankaPanda
    if getattr(franka_cls, "_maniguard_longfinger_patched", False):
        return

    panda_bundle = getattr(franka_mod, "FRANKA_PANDA_BUNDLE", "franka_panda")
    longfinger_bundle = "franka_panda_longfinger"
    setattr(franka_mod, "FRANKA_PANDA_LONGFINGER_BUNDLE", longfinger_bundle)

    orig_usd_path = franka_cls.usd_path
    orig_urdf_path = franka_cls.urdf_path
    orig_curobo_path = franka_cls.curobo_path

    def _franka_panda_asset_bundle(self):
        dataset_root = get_dataset_path("omnigibson-robot-assets")
        longfinger_dir = os.path.join(dataset_root, f"models/franka/{longfinger_bundle}")
        if getattr(self, "end_effector", None) == "gripper" and os.path.isdir(longfinger_dir):
            return longfinger_bundle
        return panda_bundle

    def _usd_path(self):
        dataset_root = get_dataset_path("omnigibson-robot-assets")
        if getattr(self, "model_name", None) == panda_bundle:
            bundle_name = self._franka_panda_asset_bundle
            return os.path.join(dataset_root, f"models/franka/{bundle_name}/usd/{bundle_name}.usda")
        return orig_usd_path.fget(self)

    def _urdf_path(self):
        assert getattr(self, "_model_name", None) == panda_bundle, (
            f"Only {panda_bundle} has urdf currently. Got: {getattr(self, '_model_name', None)}"
        )
        bundle_name = self._franka_panda_asset_bundle
        return os.path.join(
            get_dataset_path("omnigibson-robot-assets"),
            f"models/franka/{bundle_name}/urdf/{bundle_name}.urdf",
        )

    def _curobo_path(self):
        assert getattr(self, "_model_name", None) == panda_bundle, (
            f"Only {panda_bundle} is currently supported for curobo. Got: {getattr(self, '_model_name', None)}"
        )
        bundle_name = self._franka_panda_asset_bundle
        return os.path.join(
            get_dataset_path("omnigibson-robot-assets"),
            f"models/franka/{bundle_name}/curobo/{bundle_name}_description_curobo_default.yaml",
        )

    franka_cls._franka_panda_asset_bundle = property(_franka_panda_asset_bundle)
    franka_cls.usd_path = property(_usd_path)
    franka_cls.urdf_path = property(_urdf_path)
    franka_cls.curobo_path = property(_curobo_path)

    # AG endpoints are hardcoded in FrankaPanda.__init__ (franka.py:118-123)
    # at the original short-finger position (z=0.045, ~83% along a 54mm
    # finger). With the long-finger asset (~150mm finger), z=0.045 sits at
    # ~30% — the rays span the finger ROOT, not the tip, so `is_grasping`
    # never fires when the operator closes around the actual tip contact.
    # Override the assisted_grasp_*_points properties to return 4 points
    # along the finger (root → tip) when the longfinger bundle is in use;
    # AG fires if any start/end ray pair triggers, so denser = better
    # coverage. Patching properties (not __init__) sidesteps the
    # save_init_info sig.bind decorator at python_utils.py:66.
    import torch as _th
    from omnigibson.robots.manipulation_robot import GraspingPoint as _GraspingPoint

    # 1 × 4 (= 4 points per finger). The x sweep we tried earlier was
    # measurably slower (12×12 = 144 ray pairs every step) and wasn't the
    # gating factor anyway — AG mostly fails on the "two fingers in
    # contact" requirement, not on raycast coverage.
    LONG_AG_X = (0.0,)
    LONG_AG_Z = (0.045, 0.085, 0.120, 0.140)

    def _longfinger_start_points(arm):
        return [_GraspingPoint(link_name="panda_rightfinger",
                               position=_th.tensor([x, 0.001, z]))
                for z in LONG_AG_Z for x in LONG_AG_X]

    def _longfinger_end_points(arm):
        return [_GraspingPoint(link_name="panda_leftfinger",
                               position=_th.tensor([x, 0.001, z]))
                for z in LONG_AG_Z for x in LONG_AG_X]

    def _is_longfinger_in_use(self):
        if getattr(self, "end_effector", None) != "gripper":
            return False
        try:
            return self._franka_panda_asset_bundle == longfinger_bundle
        except Exception:  # noqa: BLE001
            return False

    def _patched_ag_start(self):
        if _is_longfinger_in_use(self):
            return {self.default_arm: _longfinger_start_points(self.default_arm)}
        return {self.default_arm: self._ag_start_points}

    def _patched_ag_end(self):
        if _is_longfinger_in_use(self):
            return {self.default_arm: _longfinger_end_points(self.default_arm)}
        return {self.default_arm: self._ag_end_points}

    franka_cls._assisted_grasp_start_points = property(_patched_ag_start)
    franka_cls._assisted_grasp_end_points = property(_patched_ag_end)
    franka_cls._maniguard_longfinger_patched = True


def _register_bddl_predicates() -> None:
    from maniguard.utils.bddl_predicates import register_maniguard_predicates

    register_maniguard_predicates()


def _patch_attached_to_disable_collision() -> None:
    """Enable upstream's commented-out
    ``_disable_collision_between_child_and_parent`` call inside
    ``AttachedTo._attach``.

    Once a FixedJoint binds two bodies, their collision pairs no longer
    need PhysX contact resolution — the joint constraint is the only
    relevant interaction. Leaving collision enabled creates a
    multi-constraint reconciliation problem at the moment of attach
    (lid mesh ↔ container-rim mesh + lid root ↔ container F-link
    FixedJoint + neighbors). PhysX's solver applies impulses that, for
    light containers like a steamer basket holding a potato, send the
    whole assembly flying.

    Upstream OG has the disable-collision call commented out (see
    ``omnigibson/object_states/attached_to.py:320-321``). We restore it
    via a runtime wrapper around ``_attach``.
    """
    from omnigibson.object_states.attached_to import AttachedTo

    if getattr(AttachedTo._attach, "_maniguard_disable_collision_patched", False):
        return

    original = AttachedTo._attach

    def _attach_patched(self, other, child_link, parent_link,
                        joint_type=None, can_joint_break=True):
        result = original(self, other, child_link, parent_link,
                          joint_type=joint_type, can_joint_break=can_joint_break)
        # Hot-path collision-filter add: SKIP the upstream stop/play
        # cycle (``_disable_collision_between_child_and_parent`` does
        # ``og.sim.stop() + ... + og.sim.play()`` which crashes when
        # invoked from within an in-flight ``sim.step``: try_snap calls
        # ``set_value`` → ``_attach`` → wrapper, all inside one
        # ``og.sim.step`` from the settle loop). The filtered-collision
        # add takes effect for new contacts going forward without a
        # sim restart.
        if other in self.parents_disabled_collisions:
            return result
        self.parents_disabled_collisions.add(other)
        for child_link_obj in self.obj.links.values():
            for parent_link_obj in other.links.values():
                child_link_obj.add_filtered_collision_pair(parent_link_obj)
        return result

    _attach_patched._maniguard_disable_collision_patched = True
    AttachedTo._attach = _attach_patched


def _patch_attachable_for_f_link_objects() -> None:
    """Auto-inject the ``attachable`` ability into any object whose USD
    has an F (female attachment) meta-link.

    BEHAVIOR-1K's category taxonomy declares ``attachable`` for the M
    side (lid/cap) but not the F side (tupperware/canister/kettle/etc.),
    even though those categories carry the F-meta-link prims required
    for the coupling. Without ``attachable`` declared, the F-side object
    gets no ``AttachedTo`` state at load time → ``set_value(container,
    True)`` from the M side fails inside ``_get_parent_candidates``
    (which reads ``other.states[AttachedTo]``).

    We patch ``StatefulObject.prepare_object_states`` so that, just
    before state instantiation, we scan the object's links for any
    ``is_meta_link AND meta_link_id.endswith("F")`` and add
    ``"attachable"`` to ``self._abilities`` if missing. By this point
    the USD is loaded and links are introspectable.
    """
    from omnigibson.objects.stateful_object import StatefulObject

    if getattr(StatefulObject.prepare_object_states,
               "_maniguard_attachable_patched", False):
        return

    original = StatefulObject.prepare_object_states

    def prepare_object_states_patched(self):
        for link in self.links.values():
            if not getattr(link, "is_meta_link", False):
                continue
            # is_meta_link short-circuit makes meta_link_id safe to read.
            # M side too: the taxonomy declares ``attachable`` for lid but NOT for cap
            # (cap/* carries the M meta-link yet gets no AttachedTo state -> LidSnapper
            # discovers zero pairs on every bottle-cap task). Any object with an
            # attachment meta-link (either polarity) gets the ability.
            if link.meta_link_id.endswith(("F", "M")) \
                    and "attachable" not in self._abilities:
                self._abilities["attachable"] = {}
                break
        return original(self)

    prepare_object_states_patched._maniguard_attachable_patched = True
    StatefulObject.prepare_object_states = prepare_object_states_patched


def _patch_create_joint_skip_render() -> None:
    """Skip ``og.sim.render()`` inside ``create_joint`` when all four
    local-pose args are provided by the caller.

    Root cause of the AG-induced ``AttributeError: 'NoneType' object has
    no attribute 'view'`` crash:

    ``omnigibson/utils/usd_utils.py:create_joint`` calls ``og.sim.render()``
    after defining the joint to "auto-fill the local pose before
    overwriting it". When this runs inside a physics callback (the AG fire
    path: ``_handle_assisted_grasping → _establish_grasp_rigid →
    create_joint``), the render dispatches timeline events that invalidate
    the robot's articulation handles
    (``ArticulationView._is_initialized = False`` via
    ``_invalidate_physics_handle_callback``). Immediately after the joint
    is created, ``_establish_grasp_rigid`` calls
    ``self.get_joint_positions()`` which now returns ``None`` because the
    view was de-initialised → AttributeError → simulator segfault.

    The render is a no-op for any caller (AG included) that already
    provides explicit ``joint_frame_*`` arguments, because those values
    immediately overwrite whatever the render auto-filled. So we wrap
    ``create_joint`` and skip the render in that case. When some pose
    args are omitted (e.g. legacy callers relying on the default), we
    fall back to the original behaviour.

    Upstream OG already flags this as fragile (see the in-source comment
    on the offending line about ``multi_gpu``); this patch makes the
    same protection apply on single-GPU as well.
    """
    import omnigibson.utils.usd_utils as usd_utils

    orig_create_joint = usd_utils.create_joint

    def create_joint_safe(
        prim_path,
        joint_type,
        body0=None,
        body1=None,
        enabled=True,
        exclude_from_articulation=False,
        joint_frame_in_parent_frame_pos=None,
        joint_frame_in_parent_frame_quat=None,
        joint_frame_in_child_frame_pos=None,
        joint_frame_in_child_frame_quat=None,
        break_force=None,
        break_torque=None,
    ):
        all_poses_provided = (
            joint_frame_in_parent_frame_pos is not None
            and joint_frame_in_parent_frame_quat is not None
            and joint_frame_in_child_frame_pos is not None
            and joint_frame_in_child_frame_quat is not None
        )
        if not all_poses_provided:
            return orig_create_joint(
                prim_path=prim_path,
                joint_type=joint_type,
                body0=body0,
                body1=body1,
                enabled=enabled,
                exclude_from_articulation=exclude_from_articulation,
                joint_frame_in_parent_frame_pos=joint_frame_in_parent_frame_pos,
                joint_frame_in_parent_frame_quat=joint_frame_in_parent_frame_quat,
                joint_frame_in_child_frame_pos=joint_frame_in_child_frame_pos,
                joint_frame_in_child_frame_quat=joint_frame_in_child_frame_quat,
                break_force=break_force,
                break_torque=break_torque,
            )

        # Inlined replacement of the original body, omitting the render()
        # call that invalidates articulation handles when invoked inside
        # a physics callback. Layout mirrors the upstream function so this
        # remains a faithful drop-in.
        import omnigibson as og
        import omnigibson.lazy as lazy
        from omnigibson.utils.constants import JointType
        assert JointType.is_valid(joint_type=joint_type), (
            f"Invalid joint specified for creation: {joint_type}"
        )
        assert body0 is not None or body1 is not None, (
            "At least either body0 or body1 must be specified when creating a joint!"
        )

        joint = getattr(lazy.pxr.UsdPhysics, joint_type).Define(og.sim.stage, prim_path)
        if body0 is not None:
            assert lazy.isaacsim.core.utils.prims.is_prim_path_valid(body0), (
                f"Invalid body0 path specified: {body0}"
            )
            joint.GetBody0Rel().SetTargets([lazy.pxr.Sdf.Path(body0)])
        if body1 is not None:
            assert lazy.isaacsim.core.utils.prims.is_prim_path_valid(body1), (
                f"Invalid body1 path specified: {body1}"
            )
            joint.GetBody1Rel().SetTargets([lazy.pxr.Sdf.Path(body1)])

        joint_prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(prim_path)
        lazy.pxr.PhysxSchema.PhysxJointAPI.Apply(joint_prim)

        # Skip og.sim.render() — caller's explicit values overwrite
        # everything it would auto-fill, and the render invalidates
        # articulation handles when called inside a physics callback.

        joint_prim.GetAttribute("physics:localPos0").Set(
            lazy.pxr.Gf.Vec3f(*joint_frame_in_parent_frame_pos.tolist())
        )
        joint_prim.GetAttribute("physics:localRot0").Set(
            lazy.pxr.Gf.Quatf(*joint_frame_in_parent_frame_quat[[3, 0, 1, 2]].tolist())
        )
        joint_prim.GetAttribute("physics:localPos1").Set(
            lazy.pxr.Gf.Vec3f(*joint_frame_in_child_frame_pos.tolist())
        )
        joint_prim.GetAttribute("physics:localRot1").Set(
            lazy.pxr.Gf.Quatf(*joint_frame_in_child_frame_quat[[3, 0, 1, 2]].tolist())
        )

        if break_force is not None:
            joint_prim.GetAttribute("physics:breakForce").Set(break_force)
        if break_torque is not None:
            joint_prim.GetAttribute("physics:breakTorque").Set(break_torque)
        joint_prim.GetAttribute("physics:jointEnabled").Set(enabled)
        joint_prim.GetAttribute("physics:excludeFromArticulation").Set(exclude_from_articulation)

        return joint_prim

    usd_utils.create_joint = create_joint_safe

    # ``manipulation_robot.py`` does ``from omnigibson.utils.usd_utils
    # import create_joint`` at module load, so it captures the unpatched
    # reference if it's imported before this patch runs. Force-import and
    # rebind so the AG fire path always uses the safe version.
    import omnigibson.robots.manipulation_robot as manip
    manip.create_joint = create_joint_safe


def _patch_throttle_assisted_grasping(interval: int) -> None:
    """Throttle ``_handle_assisted_grasping`` to fire every Nth substep call.

    AG is wired into ``deploy_control`` which fires per physics substep — at
    ``physics_frequency=300`` / ``action_frequency=30`` that's 10 fires per
    env step, and each full-raycast fire costs ~2 ms (longfinger geometry,
    16 raycasts per fire). For RL where the policy keeps fingers closed,
    most of those substep fires are redundant — the AG state can only
    change as fast as the policy emits new actions.

    With ``interval=N``, the wrapped method short-circuits on
    ``(N-1)/N`` calls and runs the original logic on every Nth call. Set
    ``interval=10`` to fire once per env step at the standard
    300/30 cadence. ``interval=1`` (or 0) disables the throttle.

    State is kept on the robot instance (``_maniguard_ag_call_counter``),
    so each robot has its own counter — multiple robots / multiple envs
    don't interfere.
    """
    if interval <= 1:
        return  # no-op — let AG fire every substep

    from omnigibson.robots.manipulation_robot import ManipulationRobot

    if getattr(ManipulationRobot, "_maniguard_ag_throttle_patched", False):
        return  # already patched (idempotent across re-imports)

    original_handle = ManipulationRobot._handle_assisted_grasping

    def _throttled_handle(self):
        # Skip unless this is the Nth substep tick.
        counter = getattr(self, "_maniguard_ag_call_counter", 0) + 1
        self._maniguard_ag_call_counter = counter
        if counter % interval != 0:
            return
        original_handle(self)

    ManipulationRobot._handle_assisted_grasping = _throttled_handle
    ManipulationRobot._maniguard_ag_throttle_patched = True
    ManipulationRobot._maniguard_ag_throttle_interval = interval


def apply_ag_throttle_from_env() -> None:
    """Read ``MANIGUARD_AG_SUBSTEP_INTERVAL`` and install the throttle.

    Called from wrappers.build_vec_env so the patch lands after OG is
    imported but before env construction. Reading from env (rather than
    forcing every caller through a Python API) keeps the patch out of OG's
    import order — and lets profile scripts opt in via shell.
    """
    raw = os.environ.get("MANIGUARD_AG_SUBSTEP_INTERVAL")
    if not raw:
        return
    try:
        interval = int(raw)
    except ValueError:
        return
    _patch_throttle_assisted_grasping(interval)


def _apply_eager_patches() -> None:
    global _EAGER_APPLIED
    if _EAGER_APPLIED:
        return
    _extend_factory_lists()
    _patch_grasp_goal()
    _patch_grasp_reward()
    _patch_sampling_utils()
    _patch_create_joint_skip_render()
    if not os.environ.get("MANIGUARD_SKIP_LONGFINGER"):
        _patch_franka_longfinger()
    apply_ag_throttle_from_env()
    _register_bddl_predicates()
    _patch_attachable_for_f_link_objects()
    _patch_attached_to_disable_collision()
    _EAGER_APPLIED = True


def apply() -> None:
    """Install the ManiGuard OmniGibson patches. Idempotent."""
    if os.environ.get("MANIGUARD_SKIP_OMNIGIBSON_PATCH"):
        return

    _install_import_hook()

    try:
        importlib.import_module("omnigibson")
    except ImportError:
        # Lightweight consumers (e.g. pure curation scripts) can use maniguard
        # without OmniGibson installed; skip the eager patches in that case.
        return

    _apply_eager_patches()


__all__ = ["apply"]
