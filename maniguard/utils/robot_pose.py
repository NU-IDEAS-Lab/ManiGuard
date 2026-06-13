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
