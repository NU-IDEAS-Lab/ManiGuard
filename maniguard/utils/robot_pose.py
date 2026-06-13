"""Canonical robot init pose for ManiGuard-Bench tasks.

Every bench task's FrankaPanda + long-finger gripper starts from ONE natural folded
"ready" pose — the OmniGibson FrankaPanda default pose (the natural extended posture shown
in the BEHAVIOR-1K Franka docs). The wrist J6 folds the gripper to point down, so the wrist
camera (mounted under the eef, along the gripper approach axis) looks down-and-forward at the
workspace; and J7=0.75 leaves the two-finger gripper plane orthogonal to the arm-links plane
(not skewed) — both visually confirmed.

Baked into every base snapshot, so it is simultaneously the snapshot state, the
rollout/teleop start, and the eval reset (one source of truth). This fixes the earlier
"deformed init pose" eval problem, where base snapshots carried inconsistent poses across
families (clutter/lid/dusty splayed near-zero, jar/cabinet/stack already this default).
See the design doc section 2, rule #5. Teleop should import from here so the collection
start and the bench init can never drift.
"""

# 7-DOF arm == OmniGibson FrankaPanda _default_robot_model_joint_pos (arm slice):
# J2=-1.3 (shoulder back), J4=-2.87 (elbow folded), J6=2.0 (wrist down -> gripper points
# down for the wrist-cam top-down view), J7=0.75 (gripper plane orthogonal to the arm plane).
BENCH_INIT_ARM_QPOS = [0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75]

# parallel-jaw gripper, fully open
BENCH_INIT_GRIPPER_QPOS = [0.04, 0.04]

# full 9-DOF (arm + gripper), order matching FrankaPanda's joints for set_joint_positions
BENCH_INIT_QPOS = BENCH_INIT_ARM_QPOS + BENCH_INIT_GRIPPER_QPOS

# Robot mount height: the FrankaPanda base sits ROBOT_MOUNT_OFFSET above the support
# surface's top plane, i.e. base_z = support_top + ROBOT_MOUNT_OFFSET. 0.02 m is the
# clearance the jar/cabinet pipelines already use (--robot-on-surface-clearance-m); the
# bench enforces this ONE formula uniformly across all 6 families so the robot<->table
# relationship — and therefore the robot-frame camera framing + reachability — is
# consistent family-to-family (absolute z still varies with each table's height).
ROBOT_MOUNT_OFFSET = 0.02

# --- Bench SOFTWARE-layer config declarations (design doc §2, layer ②). ----------------
# These are the bench's SINGLE DECLARATION of the canonical source defaults for the two
# operational "knobs". The controller DICTIONARY definitions stay in the one shared
# registry maniguard.envs.frozen_task_runtime.CONTROLLER_PRESETS (never duplicated here);
# the bench only declares WHICH preset is canonical. Downstream eval/teleop keep their own
# per-run override params and may switch knobs per experiment — but the same family's
# collection and eval must agree. These declarations are baked into every base snapshot so
# the saved robot config is uniform + correct, and are inert downstream (eval reloads
# controllers / sets grasping_mode after load).
#
# joint_position_raw: raw-radian joint position, NO command clipping — required because
# pose A has joints at |q|>1 rad (e.g. -2.87, 2.0) that the clipped "joint_position" preset
# (command_input_limits="default" -> clamps to (-1,1)) would silently mangle. It is also the
# exact controller GELLO teleop collection used, so it matches the training distribution.
BENCH_CONTROLLER_PRESET = "joint_position_raw"

# assisted: the neutral default grasp mode. A few families (esp. the joint-trained ones,
# whose teleop distribution is often "sticky") may override to "sticky" at run time; that is
# an operational per-family choice made consistently across collection + eval, not here.
BENCH_GRASPING_MODE = "assisted"
