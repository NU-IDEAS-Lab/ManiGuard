"""Gym wrappers for RL training.

LTLWrapper is currently unused — will be reconnected once we wire an LDBA-based
monitor through the vec-env reset path. Keeping the definition here so the
training entrypoint can import it when re-enabled.
"""

from __future__ import annotations

import gymnasium as gym

import omnigibson as og
from omnigibson.object_states import Touching, Upright
from sentinel.utils.ltl_utils import LTLMonitor


LTL_FORMULA = "G (!any_glass_touching_floor & all_glasses_upright)"
MAX_TILT_DEG = 30.0


class LTLWrapper(gym.Wrapper):
    """Legacy wineglass-on-table LTL monitor (pre-clutter-scene demo).

    Not currently applied — the clutter scenes don't have a ``wineglass_*``
    object naming convention. Left here as a reference until we define a
    scene-agnostic atomic-proposition set.
    """

    def __init__(self, env):
        super().__init__(env)
        self.monitor = LTLMonitor(LTL_FORMULA)
        self.violation_count = 0
        self.last_violation_step = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.monitor.reset()
        info["ltl"] = self.monitor.step(self._label_dict())
        info["ltl_violation"] = False
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ltl_info = self.monitor.step(self._label_dict())
        info["ltl"] = ltl_info
        info["ltl_violation"] = bool(ltl_info["doomed"])

        if ltl_info["doomed"]:
            og.log.warning("LTL safety constraint violated; terminating episode.")
            reward -= 5.0
            terminated = True
            self.violation_count += 1
            self.last_violation_step = getattr(self.env, "episode_step", None)
        return obs, reward, terminated, truncated, info

    def _label_dict(self):
        return {
            "any_glass_touching_floor": self._any_glass_touching_floor(),
            "all_glasses_upright": self._all_glasses_upright(),
        }

    def _get_glasses(self):
        scene = self.env.unwrapped.scene
        return [obj for obj in scene.objects if obj.name.startswith("wineglass_")]

    def _any_glass_touching_floor(self):
        scene = self.env.unwrapped.scene
        floor = scene.object_registry("name", "floor")
        glasses = self._get_glasses()
        if floor is None or not glasses:
            return False
        return any(bool(glass.states[Touching].get_value(floor)) for glass in glasses)

    def _all_glasses_upright(self):
        glasses = self._get_glasses()
        if not glasses:
            return False
        for glass in glasses:
            state = glass.states.get(Upright)
            if state is None:
                continue
            state.max_tilt_deg = MAX_TILT_DEG
            if not state.get_value():
                return False
        return True
