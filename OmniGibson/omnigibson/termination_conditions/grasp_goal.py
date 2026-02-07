from omnigibson.termination_conditions.termination_condition_base import SuccessCondition


class GraspGoal(SuccessCondition):
    """
    GraspGoal (success condition)
    """

    def __init__(self, obj_name, hold_steps: int = 1):
        self.obj_name = obj_name
        self.obj = None
        self.hold_steps = max(1, int(hold_steps))
        self._held_steps = 0

        # Run super init
        super().__init__()

    def _step(self, task, env, action):
        self.obj = env.scene.object_registry("name", self.obj_name) if self.obj is None else self.obj
        robot = env.robots[0]
        obj_in_hand = robot._ag_obj_in_hand[robot.default_arm]
        is_grasping = obj_in_hand == self.obj
        if is_grasping:
            self._held_steps += 1
        else:
            self._held_steps = 0
        return self._held_steps >= self.hold_steps

    def reset(self, task, env):
        super().reset(task, env)
        self._held_steps = 0