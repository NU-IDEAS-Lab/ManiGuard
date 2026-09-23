"""Robot initialization constants for benchmark construction.

BENCH_INIT_QPOS contains the seven arm joints followed by the two gripper
joints. Builders save this pose with a 2 cm base offset above the selected
support plane. Controller and grasping constants select the construction
defaults; evaluation and collection can supply their own runtime settings."""

# 7-DOF arm == OmniGibson FrankaPanda _default_robot_model_joint_pos (arm slice):
# J2=-1.3 (shoulder back), J4=-2.87 (elbow folded), J6=2.0 (wrist down -> gripper points
# down for the wrist-cam top-down view), J7=0.75 (gripper plane orthogonal to the arm plane).
BENCH_INIT_ARM_QPOS = [0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75]

# parallel-jaw gripper, fully open
BENCH_INIT_GRIPPER_QPOS = [0.04, 0.04]

# full 9-DOF (arm + gripper), order matching FrankaPanda's joints for set_joint_positions
BENCH_INIT_QPOS = BENCH_INIT_ARM_QPOS + BENCH_INIT_GRIPPER_QPOS

# Mount the robot base 0.02 m above the selected support plane:
# base_z = support_top + ROBOT_MOUNT_OFFSET.
ROBOT_MOUNT_OFFSET = 0.02

# Construction defaults. Controller dictionaries are defined in
# maniguard.envs.frozen_task_runtime.CONTROLLER_PRESETS.
# joint_position_raw accepts absolute joint targets in radians without
# clipping them to [-1, 1], which would truncate several initial-pose joints.
# exact controller GELLO teleop collection used, so it matches the training distribution.
BENCH_CONTROLLER_PRESET = "joint_position_raw"
# Default grasping mode saved by construction. Runtime configuration may
# select sticky, assisted, or physical grasping to match the experiment.
BENCH_GRASPING_MODE = "assisted"
