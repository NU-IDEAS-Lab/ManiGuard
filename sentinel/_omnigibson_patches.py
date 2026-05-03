"""Runtime patches that add Sentinel-specific behavior to upstream OmniGibson.

The refactor/omnigibson branch is incrementally peeling Sentinel's code out of
the vendored ``OmniGibson/`` tree so OmniGibson can be consumed as an upstream
dependency. This module is the central place where the *remaining* runtime
modifications live. Two kinds of patches are applied:

1. **Post-load hook on ``omnigibson.object_states``** — the moment the
   ``omnigibson.object_states`` subpackage finishes executing its ``__init__``,
   Sentinel injects three extra names: ``Dropped`` / ``Upright`` (new state
   classes defined under :mod:`sentinel.object_states`) plus ``Grasped`` as a
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
     :mod:`sentinel.utils.bddl_predicates`.

Set ``SENTINEL_SKIP_OMNIGIBSON_PATCH=1`` in the environment to opt out.

Two upstream files still carry SENTINEL modifications on this branch
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
    """Wraps the loader for ``omnigibson.object_states`` so sentinel can inject
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
    from sentinel.object_states.dropped import Dropped
    from sentinel.object_states.upright import Upright

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

    # If omnigibson.object_states was already imported before sentinel loaded,
    # the hook will never fire. Patch retroactively in that case.
    already = sys.modules.get("omnigibson.object_states")
    if already is not None:
        _inject_state_aliases(already)


def _extend_factory_lists() -> None:
    import omnigibson.object_states.factory as factory
    from sentinel.object_states.dropped import Dropped
    from sentinel.object_states.upright import Upright

    factory._DEFAULT_STATE_SET = frozenset(
        set(factory._DEFAULT_STATE_SET) | {Dropped, Upright}
    )


def _patch_grasp_goal() -> None:
    from omnigibson.termination_conditions.grasp_goal import GraspGoal

    if getattr(GraspGoal, "_sentinel_patched", False):
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
    GraspGoal._sentinel_patched = True


def _patch_grasp_reward() -> None:
    from omnigibson.reward_functions.grasp_reward import GraspReward

    if getattr(GraspReward, "_sentinel_patched", False):
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
    GraspReward._sentinel_patched = True


def _patch_sampling_utils() -> None:
    import omnigibson.utils.sampling_utils as sampling_utils

    if getattr(sampling_utils, "_sentinel_patched", False):
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
    sampling_utils._sentinel_patched = True


def _patch_franka_longfinger() -> None:
    import omnigibson.robots.franka as franka_mod
    from omnigibson.utils.asset_utils import get_dataset_path

    franka_cls = franka_mod.FrankaPanda
    if getattr(franka_cls, "_sentinel_longfinger_patched", False):
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
    franka_cls._sentinel_longfinger_patched = True


def _register_bddl_predicates() -> None:
    from sentinel.utils.bddl_predicates import register_sentinel_predicates

    register_sentinel_predicates()


def _apply_eager_patches() -> None:
    global _EAGER_APPLIED
    if _EAGER_APPLIED:
        return
    _extend_factory_lists()
    _patch_grasp_goal()
    _patch_grasp_reward()
    _patch_sampling_utils()
    if not os.environ.get("SENTINEL_SKIP_LONGFINGER"):
        _patch_franka_longfinger()
    _register_bddl_predicates()
    _EAGER_APPLIED = True


def apply() -> None:
    """Install the Sentinel OmniGibson patches. Idempotent."""
    if os.environ.get("SENTINEL_SKIP_OMNIGIBSON_PATCH"):
        return

    _install_import_hook()

    try:
        importlib.import_module("omnigibson")
    except ImportError:
        # Lightweight consumers (e.g. pure curation scripts) can use sentinel
        # without OmniGibson installed; skip the eager patches in that case.
        return

    _apply_eager_patches()


__all__ = ["apply"]
