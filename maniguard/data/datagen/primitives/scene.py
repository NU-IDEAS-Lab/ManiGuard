"""Build the empty-scene OmniGibson env from a base-task dump — Layer-1 primitive.

Replaces the monolithic ``_build_env`` of the reference pnp script
(``maniguard/data/curobo/pick_and_place_from_dataset.py``): this builds ONLY the
scene — floor + fixed support surface + every spawned task object at its dumped
pose + the Franka at its dump pose + the ``goal_region`` marker — settled and
ready to step. Cameras (4 bench third-person + injected wrist) and recording are
layered on top by the cameras (P8) / record (P9) primitives through two seams:

  * ``external_sensors``  — the external VisionSensor config list to drop into
    ``env_cfg["env"]["external_sensors"]`` (P8 passes the bench 4-view config).
  * ``pre_build_hooks``   — callables run just before ``og.Environment(...)`` (P8
    passes the wrist-camera monkeypatch installer, which must patch the FrankaPanda
    class before the robot is loaded).

Env infra (``build_env_config``, ``extract_scene_robot_setup``, goal-region
helpers) is imported from the shared ``maniguard.envs`` / ``maniguard.utils``
trees — NOT the old curobo reference tree. Dump parsing is the local ``task_io``
primitive. GPU dynamics is OFF for dry tasks (the curobo + JointController pipeline
is CPU/obs-bound; GPU PhysX is slower and NaN-prone here — see
project_gpu_physx_rl_not_faster), matching the reference pnp init; liquid/particle
tasks (diagnostics ``selection.system_name``) auto-enable it — see
:func:`task_needs_gpu_dynamics` / :func:`init_omnigibson`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

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
    """True if the task carries a PhysX particle/fluid system that only simulates under the GPU
    dynamics pipeline. Clutter-liquid tasks declare ``selection.system_name`` (e.g. ``"water"``);
    under the default CPU pipeline the fluid particles deterministically NaN-segfault at water-system
    init. Mirrors ``bench_builder.finalize_base._needs_gpu_dynamics`` (the canonical bench gating).

    lid_transport: liquid-mode tasks carry a LEGACY ``system_name="water"`` in the diag but the
    bench scenes bake NO particles at all ("filled" is narrative) — GPU dynamics there is pure
    risk (boot segfault seen on task_0023) with zero benefit. Gate them out on ``lid_info``."""
    if diag.get("lid_info"):
        return False
    return bool((diag.get("selection") or {}).get("system_name"))


def task_needs_gpu_dynamics(task_dir: str | Path, episode: int = 1) -> bool:
    """Whether this base task needs GPU dynamics, read from its dumped diagnostics row. Pure file
    read (no ``omnigibson`` import) so the driver can call it BEFORE :func:`init_omnigibson`, which
    must set the gm macro before ``import omnigibson``."""
    return _needs_gpu_dynamics(load_diagnostics_row(Path(task_dir), episode))


def init_omnigibson(headless: bool = True, needs_gpu_dynamics: bool = False):
    """Set datagen OmniGibson macros, then import + return ``omnigibson``.

    MUST be called once before :func:`scene_from_task_dir` (gm macros take effect only before
    ``import omnigibson``). Dry tasks run GPU-dynamics OFF + flatcache ON (the curobo + JointController
    pipeline is CPU/obs-bound; GPU PhysX is slower and NaN-prone here — see project_gpu_physx_rl_not_faster),
    mirroring the reference pnp init. Liquid/particle tasks (``needs_gpu_dynamics=True``, detected per
    task via :func:`task_needs_gpu_dynamics`) REQUIRE GPU dynamics + flatcache OFF or the PhysX water
    system NaN-segfaults at init — matching the bench_builder finalize + source liquid pipelines.
    """
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
    import torch as th
    import omnigibson as og

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

    # JointController, joint_position_raw preset = RIGID Isaac position drive (isaac_kp=1e7),
    # MATCHING eval/teleop/bench (configs/eval/*_joint.yaml + robot_pose.BENCH_CONTROLLER_PRESET).
    # The old joint_position_impedance preset (soft, pos_kp=50) drooped the held load ~0.07 m at arm
    # extension and could not track the descent wrist-swing, failing the cabinet place; eval is rigid
    # anyway, so datagen-rigid is ALSO more train/eval-consistent (recorded action = next-achieved q).
    # action/render at 30 Hz; assisted grasp by default.
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
    # Camera streams are owned by the cameras primitive (P8).
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
    og.sim.step()

    goal_spec = None
    gr_payload = diagnostics.get("goal_region")
    if gr_payload is not None:
        goal_spec = GoalRegionSpec.from_json(gr_payload)
        spawn_goal_region_marker(env, goal_spec)
        og.sim.step()

    for _ in range(max(0, int(settle_steps))):
        og.sim.step()

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
