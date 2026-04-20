import os
import sys
import json
import math
import tempfile

OG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if OG_ROOT not in sys.path:
    sys.path.insert(0, OG_ROOT)

import omnigibson as og
import torch as th
from omnigibson.macros import gm
from omnigibson.object_states import OnTop, Touching
from omnigibson.utils.asset_utils import get_all_object_category_models
import omnigibson.utils.transform_utils as T
from sentinel.utils.ltl_utils import LTLMonitor

# Match the grasping demo's stability settings
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_FLATCACHE = True

scene_cfg = dict(type="Scene")
robot_start_pos = [0.0, 0.0, 0.0]
robot_start_orn = [0.0, 0.0, 0.0, 1.0]
robot_cfg = dict(
    type="FrankaMounted",
    name="franka",
    obs_modalities=["rgb"],
    action_type="continuous",
    action_normalize=True,
    grasping_mode="sticky",
    position=robot_start_pos,
    orientation=robot_start_orn,
)

table_cfg = dict(
    type="DatasetObject",
    name="table",
    category="breakfast_table",
    model="lcsizg",
    bounding_box=[0.5, 0.5, 0.8],
    fixed_base=True,
    position=[0.7, -0.1, 0.6],
    orientation=[0, 0, 0.707, 0.707],
)

wineglass_models = get_all_object_category_models("wineglass")
assert wineglass_models, "No wineglass models available"

# Build table + wineglasses object list (no custom task needed)
def build_objects_config(table_cfg, wineglass_models, count=4, spacing=0.12, margin=0.05, z_offset=0.0):
    table_pos = table_cfg["position"]
    table_bbox = table_cfg["bounding_box"]
    x_min = table_pos[0] - table_bbox[0] / 2 + margin
    x_max = table_pos[0] + table_bbox[0] / 2 - margin
    y_min = table_pos[1] - table_bbox[1] / 2 + margin
    y_max = table_pos[1] + table_bbox[1] / 2 - margin
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Table surface too small for requested margin.")

    n_cols = max(1, int(math.floor((x_max - x_min) / spacing)) + 1)
    n_rows = max(1, int(math.floor((y_max - y_min) / spacing)) + 1)
    total = min(count, n_cols * n_rows)

    positions = []
    for r in range(n_rows):
        for c in range(n_cols):
            if len(positions) >= total:
                break
            x = x_min + c * spacing
            y = y_min + r * spacing
            positions.append((x, y))
        if len(positions) >= total:
            break

    table_top_z = table_pos[2] + table_bbox[2] / 2
    objs = [table_cfg]
    for i, (x, y) in enumerate(positions):
        objs.append(
            dict(
                type="DatasetObject",
                name=f"wineglass_{i}",
                category="wineglass",
                model=wineglass_models[i % len(wineglass_models)],
                bounding_box=None,
                fixed_base=False,
                position=[x, y, table_top_z + z_offset],
                orientation=[0.0, 0.0, 0.0, 1.0],
            )
        )
    return objs

objects_cfg = build_objects_config(
    table_cfg=table_cfg,
    wineglass_models=wineglass_models,
    count=4,
    spacing=0.12,
    margin=0.05,
    z_offset=0.0,
)

# Pre-cached reset pose to avoid curobo dependency for Fetch in GraspTask
def _write_precached_reset_pose(path):
    # 7 arm joint positions for Franka (default pose)
    joint_pos = [
        0.00,
        -1.30,
        0.00,
        -2.87,
        0.00,
        2.00,
        0.75,
    ]
    payload = [
        {
            "joint_pos": joint_pos,
            "base_pos": robot_start_pos,
            "base_ori": robot_start_orn,
        }
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


_precached_reset_pose_path = os.path.join(tempfile.gettempdir(), "og_fetch_reset_pose.json")
_write_precached_reset_pose(_precached_reset_pose_path)

task_cfg = dict(
    type="SentinelGraspTask",
    obj_name="wineglass_0",
    objects_config=objects_cfg,
    precached_reset_pose_path=_precached_reset_pose_path,
    termination_config={"max_steps": 500, "grasp_hold_steps": 10},
    reward_config={
        "dist_coeff": 1.0,
        "grasp_reward": 10.0,
        "collision_penalty": 1.0,
        "eef_position_penalty_coef": 0.0,
        "eef_orientation_penalty_coef": 0.0,
        "regularization_coef": 0.0,
    },
)

LTL_FORMULA = "G (!glass0_touching_floor & glass0_upright)"
MAX_TILT_DEG = 30.0


class LTLWrapper:
    def __init__(self, env):
        self.env = env
        self.monitor = LTLMonitor(LTL_FORMULA)

    def reset(self):
        obs, info = self.env.reset()
        self.monitor.reset()
        info["ltl"] = self.monitor.step(self._label_dict())
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ltl_info = self.monitor.step(self._label_dict())
        info["ltl"] = ltl_info
        if ltl_info["doomed"]:
            terminated = True
        return obs, reward, terminated, truncated, info

    def _label_dict(self):
        return {
            "glass0_touching_floor": self._glass0_touching_floor(),
            "glass0_upright": self._glass0_upright(),
        }

    def _glass0_touching_floor(self):
        floor = self.env.scene.object_registry("name", "floor")
        glass = self.env.scene.object_registry("name", "wineglass_0")
        if floor is None or glass is None:
            return False
        return bool(glass.states[Touching].get_value(floor))

    def _glass0_upright(self):
        glass = self.env.scene.object_registry("name", "wineglass_0")
        if glass is None:
            return False
        _, quat = glass.get_position_orientation()
        quat_t = th.as_tensor(quat, dtype=th.float32)
        world_up = th.tensor([0.0, 0.0, 1.0], dtype=th.float32)
        up = T.quat_apply(quat_t, world_up)
        up = th.as_tensor(up, dtype=th.float32).reshape(-1)
        if up.numel() < 3:
            return False
        up = up[-3:]
        up = up / (th.norm(up) + 1e-8)
        cos_angle = th.clamp(th.dot(up, world_up), -1.0, 1.0)
        angle_deg = float(th.rad2deg(th.acos(cos_angle)))
        return angle_deg <= MAX_TILT_DEG

def main(random_selection=False, headless=False, short_exec=False):
    """
    Grasp task demo that randomly samples actions.

    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    cfg = dict(
        scene=scene_cfg,
        robots=[robot_cfg],
        objects=objects_cfg,
        task=task_cfg,
    )

    env = og.Environment(configs=cfg)
    env = LTLWrapper(env)

    def _place_glasses_on_table():
        table = env.scene.object_registry("name", "table")
        if table is None:
            raise RuntimeError("Table object not found for placement.")
        for obj in env.scene.objects:
            if obj.name.startswith("wineglass_"):
                placed = obj.states[OnTop].set_value(table, True, reset_before_sampling=True)
                if not placed:
                    og.log.warning(f"Failed to place {obj.name} on top of table via kinematic sampling.")


    # Run a simple loop and reset periodically
    max_iterations = 10 if not short_exec else 1
    for j in range(max_iterations):
        og.log.info("Resetting environment")
        env.reset()
        _place_glasses_on_table()
        for i in range(100):
            action = env.robots[0].action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            if info.get("ltl", {}).get("doomed"):
                og.log.warning("LTL safety constraint violated; terminating episode.")
                break
            if terminated or truncated:
                og.log.info("Episode finished after {} timesteps".format(i + 1))
                break

    og.shutdown()


if __name__ == "__main__":
    main()
