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

from pathlib import Path
from typing import Optional


def build_config(
    target_name: str,
    category: str,
    model: str,
    grasp_dataset_path: Path | str,
    *,
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
) -> dict:
    """Assemble the OG Environment config dict for the grasp-reset setup.

    Args:
        target_name: name used to refer to the target object in the scene.
        category, model: BEHAVIOR-1K dataset asset identifiers for the target.
        grasp_dataset_path: path to ``grasps_<cat>_<model>.pt`` used by the
            task's per-reset ``GraspDatasetResetter``.
        obs_modalities: robot observation modalities. Defaults to
            ``["proprio"]`` — minimal and fast. Pass ``["rgb", "proprio"]``
            when training from pixels.
        visualize_goal: spawn the translucent goal-marker sphere.
        goal_offset: (3,) task goal = target init pos + this offset.
        success_radius: goal-region tolerance (m).
        max_steps: task timeout.
        flatten_obs, flatten_action: whether OG should flatten dict spaces
            (required for SB3 MlpPolicy; MultiInputPolicy handles dicts).
        table_*, target_position: scene geometry — match the defaults used at
            grasp-collection time so IK of saved grasps stays reachable.
        grasp_reset_pose_range_b: optional per-reset body-frame perturbation
            forwarded to the ``GraspDatasetResetter``.
    """
    if obs_modalities is None:
        obs_modalities = ["proprio"]

    return dict(
        env={
            "action_frequency": action_frequency,
            "physics_frequency": physics_frequency,
            "flatten_action_space": flatten_action,
            "flatten_obs_space": flatten_obs,
        },
        scene=dict(type="Scene"),
        robots=[dict(
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
                # OSC pose-delta — grasp collector + resetter both target this
                # controller's action semantics.
                "arm_0": {"name": "OperationalSpaceController"},
                "gripper_0": {"name": "MultiFingerGripperController"},
            },
        )],
        objects=[
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
        ],
        task=dict(
            type="PickAndLiftTask",
            obj_name=target_name,
            goal_offset=list(goal_offset),
            success_radius=success_radius,
            visualize_goal=visualize_goal,
            objects_config=[],
            termination_config={"max_steps": max_steps},
            grasp_dataset_path=str(grasp_dataset_path),
            grasp_reset_pose_range_b=grasp_reset_pose_range_b,
        ),
    )


__all__ = ["build_config"]
