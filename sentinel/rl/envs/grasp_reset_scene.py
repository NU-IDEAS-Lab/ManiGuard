"""OG environment config for the empty-scene grasp-reset setup.

Builds a minimal Scene with one ``FrankaMounted`` at origin + a fixed
breakfast_table + a single ``DatasetObject`` target. Intended for tasks that
reset the agent into a saved grasp pose on the target (see
``sentinel.rl.grasps.reset.GraspDatasetResetter``): the scene is stable,
reproducible, and doesn't depend on a pre-generated benchmark scene dir.

The config is used by both the rollout smoke test (``rollout_test``) and the
PPO training entry (``training.ppo_grasp_reset``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


# Arm controller specs. Only ``name`` is supplied — OG's
# ``_generate_controller_config`` merges with the robot's defaults
# (``_default_arm_joint_controller_configs`` gives
# motor_type="position", use_delta_commands=True, use_impedances=False).
#
# JointController bypasses the operational-space Jacobian inversion entirely,
# sidestepping the ``numpy.linalg.LinAlgError: Matrix is singular`` crash OSC
# hits when PPO exploration drives the arm into near-singular configurations
# (fully extended elbow, wrist lock). Default arm controller flipped to joint
# on 2026-04-21 after the 49k-step crash.
#
# ``command_output_limits`` override: JointController's default output range
# in delta mode is the FULL joint-position limit (±~2.9 rad for Franka) — so
# a PPO sample at std≈1 with ``action_normalize=True`` saturates to ±2.9 rad
# per 1/30 s step, whipping the arm and dropping the mug instantly. We clamp
# to ±0.05 rad/step per joint so each step's motion is on the order of the
# OSC default (±0.2 m, ±0.5 rad pose delta) in joint-space terms.
_JOINT_ARM_CTRL = {
    "name": "JointController",
    "command_output_limits": (-0.05, 0.05),
}
_OSC_ARM_CTRL = {"name": "OperationalSpaceController"}


def _patch_scene_for_joint_controller(scene_file: Path) -> Path:
    """Produce a sibling scene JSON with arm_0 swapped to JointController.

    Scene files bake ``controller_config`` into ``objects_info.init_info`` at
    save time, so loading the unmodified file yields an OSC-controlled arm
    regardless of what we pass via env config. We rewrite out-of-band (leaves
    the original snapshot untouched) and cache the result as a sibling file,
    refreshed whenever the source is newer.

    Also clears the per-controller goal state: OSC's ``target_ori_mat`` goal
    shape won't deserialize into JointController, so we reset the saved
    ``controllers.arm_0`` entry to ``goal_is_valid=False, goal=None``.
    """
    patched = scene_file.with_suffix(".joint.json")
    if patched.exists() and patched.stat().st_mtime >= scene_file.stat().st_mtime:
        return patched
    data = json.loads(scene_file.read_text())

    init_info = data.get("objects_info", {}).get("init_info", {})
    swapped = 0
    for obj_key, info in init_info.items():
        if not obj_key.startswith("robot_"):
            continue
        ctrl_cfg = info.get("args", {}).get("controller_config", {})
        if "arm_0" in ctrl_cfg:
            ctrl_cfg["arm_0"] = dict(_JOINT_ARM_CTRL)
            swapped += 1
    if swapped == 0:
        raise RuntimeError(
            f"{scene_file}: no robot_* entry with controller_config.arm_0 found"
        )

    reg = data.get("state", {}).get("registry", {}).get("object_registry", {})
    for obj_key, obj_state in reg.items():
        if not obj_key.startswith("robot_"):
            continue
        controllers = obj_state.get("controllers")
        if isinstance(controllers, dict) and "arm_0" in controllers:
            controllers["arm_0"] = {"goal_is_valid": False, "goal": None}

    patched.write_text(json.dumps(data))
    return patched


def build_config(
    target_name: str,
    category: str,
    model: str,
    grasp_dataset_path: Path | str,
    *,
    scene_file: Optional[Path | str] = None,
    obs_modalities: Optional[list[str]] = None,
    visualize_goal: bool = True,
    goal_offset: tuple[float, float, float] = (0.0, 0.0, 0.15),
    success_radius: float = 0.05,
    max_steps: int = 200,
    flatten_obs: bool = True,
    flatten_action: bool = True,
    action_frequency: int = 30,
    physics_frequency: int = 300,
    table_category: str = "breakfast_table",
    table_model: str = "rjgmmy",
    table_position: tuple[float, float, float] = (0.8, 0.0, 0.41),
    target_position: tuple[float, float, float] = (0.7, 0.0, 0.87),
    grasp_reset_pose_range_b: Optional[dict] = None,
    grasp_reset_mode: str = "cached",
    arm_controller: str = "joint",
) -> dict:
    """Assemble the OG Environment config dict for the grasp-reset setup.

    Args:
        target_name: name used to refer to the target object in the scene.
            When ``scene_file`` is set, this MUST match the name used in the
            saved scene (e.g. ``target_mug`` for the ``mug_into_bowl`` scene).
        category, model: BEHAVIOR-1K dataset asset identifiers for the target.
            Ignored when ``scene_file`` is set (the target comes from the
            saved scene); kept in the signature for logging parity.
        grasp_dataset_path: path to ``grasps_<cat>_<model>.pt`` used by the
            task's per-reset ``GraspDatasetResetter``.
        scene_file: optional path to a pre-baked scene JSON (as produced by
            ``env.scene.save``). When provided, the scene USD comes from the
            file — robot and target must both be in it. Required for
            ``num_envs > 1`` because OG's scene-tiling trips on empty scenes.
        obs_modalities: robot observation modalities. Defaults to
            ``["proprio"]`` — minimal and fast. Pass ``["rgb", "proprio"]``
            when training from pixels. Ignored when ``scene_file`` is set
            (robot modalities come from the saved scene).
        visualize_goal: spawn the translucent goal-marker sphere.
        goal_offset: (3,) task goal = target init pos + this offset.
        success_radius: goal-region tolerance (m).
        max_steps: task timeout.
        flatten_obs, flatten_action: whether OG should flatten dict spaces
            (required for SB3 MlpPolicy; MultiInputPolicy handles dicts).
        table_*, target_position: scene geometry — match the defaults used at
            grasp-collection time so IK of saved grasps stays reachable.
        grasp_reset_pose_range_b: optional per-reset body-frame perturbation
            forwarded to the ``GraspDatasetResetter``. Only meaningful with
            ``grasp_reset_mode="ik"``.
        grasp_reset_mode: ``"cached"`` (default, skips online cuRobo IK and
            therefore supports ``num_envs > 1``) or ``"ik"`` (full online
            cuRobo IK, required for per-reset pose randomization).
        arm_controller: ``"joint"`` (default) uses JointController
            (position / delta / no impedances) — no Jacobian inversion, so
            PPO can't crash the sim on singular arm configs. ``"osc"`` uses
            OperationalSpaceController for comparison / legacy runs; note
            this is fragile under RL exploration (see 2026-04-21 crash).
    """
    if arm_controller not in ("joint", "osc"):
        raise ValueError(f"arm_controller must be 'joint' or 'osc', got {arm_controller!r}")
    arm_ctrl_cfg = _JOINT_ARM_CTRL if arm_controller == "joint" else _OSC_ARM_CTRL

    if obs_modalities is None:
        obs_modalities = ["proprio"]

    # Two spawn modes:
    #
    # 1. ``scene_file`` set — load a pre-baked scene (robot + table + target
    #    all in one saved USD state). Needed for ``num_envs > 1``: OG's
    #    scene-tiling code computes the world AABB of the scene prim at
    #    ``idx != 0`` and trips ``decompose_mat``'s perspective check on an
    #    empty scene. Pre-baked scenes have well-formed transforms, so tiling
    #    works. The caller is responsible for ensuring ``target_name`` matches
    #    the object named in the saved scene (e.g. ``target_mug``).
    #
    # 2. ``scene_file`` None — runtime-spawn robot + objects. Objects go
    #    through the task's ``objects_config`` so GraspTask._load places them
    #    in the SCENE frame via ``set_position_orientation(..., frame="scene")``;
    #    this keeps ``num_envs=1`` working cleanly but still breaks tiling at
    #    ``num_envs >= 2``.
    if scene_file is not None:
        # When using JointController, rewrite the scene's baked-in
        # controller_config to a sibling file (no-op if already cached).
        scene_path = Path(scene_file)
        if arm_controller == "joint":
            scene_path = _patch_scene_for_joint_controller(scene_path)
        scene_cfg = {"type": "Scene", "scene_file": str(scene_path)}
        robots_cfg: list = []
        objects_config: list = []
    else:
        scene_cfg = {"type": "Scene"}
        robots_cfg = [dict(
            type="FrankaMounted",
            name="agent_0",
            obs_modalities=list(obs_modalities),
            action_type="continuous",
            action_normalize=True,
            grasping_mode="physical",
            self_collisions=True,
            position=[0.0, 0.0, 0.0],
            orientation=[0.0, 0.0, 0.0, 1.0],
            controller_config={
                "arm_0": dict(arm_ctrl_cfg),
                "gripper_0": {"name": "MultiFingerGripperController"},
            },
        )]
        objects_config = [
            dict(
                type="DatasetObject",
                name="support_table",
                category=table_category,
                model=table_model,
                position=list(table_position),
                orientation=[0.0, 0.0, 0.0, 1.0],
                fixed_base=True,
            ),
            dict(
                type="DatasetObject",
                name=target_name,
                category=category,
                model=model,
                position=list(target_position),
                orientation=[0.0, 0.0, 0.0, 1.0],
            ),
        ]

    return dict(
        env={
            "action_frequency": action_frequency,
            "physics_frequency": physics_frequency,
            "flatten_action_space": flatten_action,
            "flatten_obs_space": flatten_obs,
        },
        scene=scene_cfg,
        robots=robots_cfg,
        objects=[],
        task=dict(
            type="PickAndLiftTask",
            obj_name=target_name,
            goal_offset=list(goal_offset),
            success_radius=success_radius,
            visualize_goal=visualize_goal,
            objects_config=objects_config,
            termination_config={"max_steps": max_steps},
            grasp_dataset_path=str(grasp_dataset_path),
            grasp_reset_pose_range_b=grasp_reset_pose_range_b,
            grasp_reset_mode=grasp_reset_mode,
        ),
    )


__all__ = ["build_config"]
