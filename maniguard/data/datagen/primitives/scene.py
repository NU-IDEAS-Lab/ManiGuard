"""Reconstruct an empty-scene OmniGibson environment from a frozen task.

Load the support, task objects, robot pose, and goal marker from diagnostics and
the scene snapshot. external_sensors supplies camera configurations and
pre_build_hooks runs setup callbacks before environment construction.

Use the shared environment and task-runtime helpers. Dry tasks use CPU dynamics
with flatcache; particle tasks selected by task_needs_gpu_dynamics use GPU
dynamics with flatcache disabled.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from maniguard.data.datagen.primitives.task_io import (
    build_object_cfg,
    identify_task_objects,
    load_diagnostics_row,
    load_scene_info,
)


@dataclass
class SceneBundle:
    """Everything a Layer-2 family skeleton needs after the scene is built."""

    env: Any
    og: Any
    robot: Any
    surface: Any                 # the support-surface object (task_names[0])
    diagnostics: dict
    scene_info: dict
    task_names: list             # [support, obj1, obj2, ...]
    goal_spec: Any | None        # GoalRegionSpec, or None if the task has no goal


def _needs_gpu_dynamics(diag: dict) -> bool:
    """Use GPU dynamics for declared spill systems or physical liquid selections.

    Lid labels without a spill specification do not imply simulated particles.
    """
    from maniguard.data.scene.liquid_state import spill_systems
    if spill_systems(diag):
        return True
    if diag.get("lid_info"):
        return False
    return bool((diag.get("selection") or {}).get("system_name"))


def task_needs_gpu_dynamics(task_dir: str | Path, episode: int = 1) -> bool:
    """Whether this base task needs GPU dynamics, read from its dumped diagnostics row. Pure file
    read (no ``omnigibson`` import) so the driver can call it BEFORE :func:`init_omnigibson`, which
    must set the gm macro before ``import omnigibson``."""
    return _needs_gpu_dynamics(load_diagnostics_row(Path(task_dir), episode))


def init_omnigibson(headless: bool = True, needs_gpu_dynamics: bool = False):
    """Set OmniGibson macros before initialization and return the imported module. Use CPU dynamics with flatcache for dry tasks and GPU dynamics without flatcache for particle tasks."""
    from omnigibson.macros import gm

    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    gm.USE_GPU_DYNAMICS = needs_gpu_dynamics
    gm.ENABLE_FLATCACHE = not needs_gpu_dynamics
    if headless:
        gm.HEADLESS = True

    import omnigibson as og

    return og


def scene_from_task_dir(
    task_dir: str | Path,
    episode: int = 1,
    *,
    grasping_mode: str = "assisted",
    external_sensors: Sequence[dict] | None = None,
    pre_build_hooks: Sequence[Callable[[], None]] = (),
    settle_steps: int = 10,
) -> SceneBundle:
    """Build the empty-scene env for one base task and return a :class:`SceneBundle`.

    ``grasping_mode`` is forwarded to the robot config (``"assisted"`` default;
    families with thin objects may pass ``"sticky"``). ``external_sensors`` /
    ``pre_build_hooks`` are the camera seams (see module docstring).
    """
    import omnigibson as og
    import torch as th

    from maniguard.envs.frozen_task_runtime import (
        build_env_config,
        extract_scene_robot_setup,
    )
    from maniguard.utils.goal_region import GoalRegionSpec, spawn_goal_region_marker

    task_dir = Path(task_dir)
    diagnostics = load_diagnostics_row(task_dir, episode)
    scene_info = load_scene_info(task_dir, episode)
    task_names = identify_task_objects(scene_info, diagnostics)
    print(f"[datagen.scene] {task_dir.name}: {len(task_names)} task objects "
          f"(surface={task_names[0]})", flush=True)

    # each object keeps its snapshot fixed_base (surface + furniture like a cabinet are fixed,
    # manipulable objects free); the surface is forced fixed as a safety.
    object_cfgs = [build_object_cfg(task_names[0], scene_info, fixed_base=True)]
    object_cfgs += [build_object_cfg(n, scene_info) for n in task_names[1:]]

    robot_setup = extract_scene_robot_setup(scene_info)
    if robot_setup is None:
        raise RuntimeError(f"No robot found in scene snapshot for {task_dir.name}")

    # Use joint_position_raw with 30 Hz action and rendering frequencies.
    # Forward the family-selected grasping mode to the robot configuration.
    env_cfg = build_env_config(
        scene_info,
        diagnostics,
        controller_preset="joint_position_raw",
        grasping_mode=grasping_mode,
        action_frequency=30,
        rendering_frequency=30,
    )
    # pnp wants a plain empty Scene with explicit object cfgs, NOT the snapshot's
    # furnished InteractiveTraversableScene.
    env_cfg["scene"] = {"type": "Scene"}
    env_cfg["objects"] = object_cfgs
    # Camera streams are configured by the cameras primitive.
    if external_sensors is not None:
        env_cfg["env"]["external_sensors"] = list(external_sensors)

    for hook in pre_build_hooks:
        hook()

    env = og.Environment(configs=env_cfg)
    env.reset()

    # env.reset can perturb spawn poses — re-apply the dumped poses (and articulated joint
    # state, e.g. a cabinet drawer's initial open fraction).
    reg = scene_info["state"]["registry"]["object_registry"]
    for cfg in object_cfgs:
        obj = env.scene.object_registry("name", cfg["name"])
        if obj is None:
            continue
        obj.set_position_orientation(
            position=th.tensor(cfg["position"], dtype=th.float32),
            orientation=th.tensor(cfg["orientation"], dtype=th.float32),
        )
        jp = reg.get(cfg["name"], {}).get("joint_pos")
        if jp and getattr(obj, "joints", None) and len(obj.joints) == len(jp):
            obj.set_joint_positions(th.tensor(jp, dtype=th.float32))
        if hasattr(obj, "keep_still"):
            obj.keep_still()

    robot = env.robots[0]
    if robot_setup.get("position") is not None:
        robot.set_position_orientation(
            position=th.tensor(robot_setup["position"], dtype=th.float32),
            orientation=th.tensor(robot_setup["orientation"], dtype=th.float32),
        )
    if hasattr(robot, "keep_still"):
        robot.keep_still()
    # Explicit-object reconstruction does not restore the snapshot's systems.
    from maniguard.data.scene.liquid_state import restore_liquid_systems, spill_systems
    restore_liquid_systems(
        env, scene_info.get("state", {}).get("registry", {}).get("system_registry", {}),
        spill_systems(diagnostics),
    )
    og.sim.step()

    goal_spec = None
    gr_payload = diagnostics.get("goal_region")
    if gr_payload is not None:
        goal_spec = GoalRegionSpec.from_json(gr_payload)
        spawn_goal_region_marker(env, goal_spec)
        og.sim.step()

    for _ in range(max(0, int(settle_steps))):
        og.sim.step()

    if spill_systems(diagnostics):
        # Keep restored liquid in subsequent env.reset() calls as well.
        env.scene.update_initial_file()
    surface = env.scene.object_registry("name", task_names[0])
    return SceneBundle(
        env=env,
        og=og,
        robot=robot,
        surface=surface,
        diagnostics=diagnostics,
        scene_info=scene_info,
        task_names=task_names,
        goal_spec=goal_spec,
    )
