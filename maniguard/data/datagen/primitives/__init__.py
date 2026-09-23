"""Shared primitives for scripted demonstration collection.

Modules:
  task_io: parse diagnostics and scene snapshots.
  scene: reconstruct the task environment.
  curobo_seg: solve motion segments and inverse kinematics.
  grasp_obb: sample grasp poses from object geometry.
  execute: replay joint trajectories and actuate the gripper.
  obstacles: configure the planner's collision world and constraints.
  cameras: configure the four external views and wrist camera.
  record: write RAW videos, joint trajectories, simulation states, and metadata.

LeRobot conversion is provided separately by datagen.to_lerobot.
"""
