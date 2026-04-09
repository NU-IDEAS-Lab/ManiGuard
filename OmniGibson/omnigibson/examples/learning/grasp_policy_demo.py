"""
Example training code using stable-baselines3 PPO for the GraspTask demo scene.
This is a starting point and is not expected to converge without additional tuning.
"""

import os
import sys

OG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if OG_ROOT not in sys.path:
    sys.path.insert(0, OG_ROOT)

import argparse
import json
import math
import tempfile
import time

import omnigibson as og
import torch as th
from omnigibson.macros import gm
from omnigibson.object_states import OnTop, Touching, Upright
from omnigibson.utils.asset_utils import get_all_object_category_models
from omnigibson.utils.python_utils import meets_minimum_version
from omnigibson.utils.ltl_utils import LTLMonitor

try:
    import gymnasium as gym
    import torch as th
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.preprocessing import is_image_space
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage

except ModuleNotFoundError:
    og.log.error(
        "stable-baselines3 is not installed. "
        "Run the following command to install stable-baselines3:\n"
        "pip install stable-baselines3[extra]\n"
    )
    raise

assert meets_minimum_version(gym.__version__, "0.28.1"), "Please install/update gymnasium to version >= 0.28.1"

# Speed / stability settings
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)
        extractors = {}
        total_concat_size = 0
        feature_size = 128
        for key, subspace in observation_space.spaces.items():
            if "rgb" in key:
                og.log.info(f"obs {key} shape: {subspace.shape}")
                n_input_channels = subspace.shape[0]
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 4, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                test_tensor = th.zeros(subspace.shape)
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
                total_concat_size += feature_size
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded = []
        for key, extractor in self.extractors.items():
            encoded.append(extractor(observations[key]))
        return th.cat(encoded, dim=1)


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


def _write_precached_reset_pose(path, robot_start_pos, robot_start_orn):
    joint_pos = [0.00, -1.30, 0.00, -2.87, 0.00, 2.00, 0.75]
    payload = [
        {
            "joint_pos": joint_pos,
            "base_pos": robot_start_pos,
            "base_ori": robot_start_orn,
        }
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


class OnTopResetWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        table = self.env.unwrapped.scene.object_registry("name", "table")
        if table is not None:
            for obj in self.env.unwrapped.scene.objects:
                if obj.name.startswith("wineglass_"):
                    obj.states[OnTop].set_value(table, True, reset_before_sampling=True)
        return obs, info


LTL_FORMULA = "G (!any_glass_touching_floor & all_glasses_upright)"
MAX_TILT_DEG = 30.0


class LTLWrapper(gym.Wrapper):
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
            print("LTL info:", ltl_info)
            print("==================================   ==============================")
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


class SafetyCallback(BaseCallback):
    def _on_step(self) -> bool:
        try:
            infos = self.locals.get("infos")
            if infos:
                violations = sum(1 for info in infos if info.get("ltl_violation"))
                self.logger.record("safety/ltl_violations", violations)
        except Exception:
            pass
        return True


def make_env():
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

    objects_cfg = build_objects_config(
        table_cfg=table_cfg,
        wineglass_models=wineglass_models,
        count=4,
        spacing=0.12,
        margin=0.05,
        z_offset=0.0,
    )

    precached_reset_pose_path = os.path.join(tempfile.gettempdir(), "og_fetch_reset_pose.json")
    _write_precached_reset_pose(precached_reset_pose_path, robot_start_pos, robot_start_orn)

    task_cfg = dict(
        type="GraspTask",
        obj_name="wineglass_0",
        objects_config=objects_cfg,
        precached_reset_pose_path=precached_reset_pose_path,
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

    cfg = dict(
        scene=scene_cfg,
        robots=[robot_cfg],
        objects=objects_cfg,
        task=task_cfg,
        env=dict(flatten_action_space=True, flatten_obs_space=True),
    )

    env = og.Environment(configs=cfg)
    env = OnTopResetWrapper(env)
    env = LTLWrapper(env)
    return env


def make_vec_env():
    env = DummyVecEnv([make_env])
    env = VecMonitor(env)
    space = env.observation_space
    needs_transpose = False
    if hasattr(space, "spaces"):
        for subspace in space.spaces.values():
            if is_image_space(subspace):
                needs_transpose = True
                break
    else:
        needs_transpose = is_image_space(space)
    if needs_transpose:
        env = VecTransposeImage(env)
    return env


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate a PPO agent for GraspTask")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to PPO checkpoint to load")
    parser.add_argument("--eval", action="store_true", help="Evaluate the PPO agent from --checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume training from --checkpoint")
    parser.add_argument("--print-checkpoint-keys", action="store_true", help="Print checkpoint observation keys and exit")
    parser.add_argument(
        "--eval-during-training",
        action="store_true",
        help="Enable EvalCallback during training. Disabled by default because OmniGibson supports only one active sim.",
    )
    args = parser.parse_args()

    tensorboard_log_dir = os.path.join("log_dir", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    seed = 0

    env = make_vec_env()
    set_random_seed(seed)
    env.reset()

    policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor)

    if args.print_checkpoint_keys:
        assert args.checkpoint is not None, "If printing checkpoint keys, --checkpoint must be specified!"
        model = PPO.load(args.checkpoint)
        og.log.info("Checkpoint observation keys:")
        if hasattr(model.observation_space, "spaces"):
            for key in sorted(model.observation_space.spaces.keys()):
                og.log.info("  %s", key)
        else:
            og.log.info("  (non-dict observation space)")
        return

    if args.eval:
        assert args.checkpoint is not None, "If evaluating a PPO policy, --checkpoint must be specified!"
        model = PPO.load(args.checkpoint)
        eval_env = make_vec_env()
        og.log.info("Starting evaluation...")
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
        og.log.info("Finished evaluation!")
        og.log.info(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")
    else:
        if args.resume:
            assert args.checkpoint is not None, "If resuming, --checkpoint must be specified!"
            model = PPO.load(args.checkpoint, env=env, device="cuda")
        else:
            model = PPO(
                "MultiInputPolicy",
                env,
                verbose=1,
                tensorboard_log=tensorboard_log_dir,
                policy_kwargs=policy_kwargs,
                n_steps=20 * 10,
                batch_size=8,
                device="cuda",
            )
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=tensorboard_log_dir, name_prefix="grasp")
        safety_callback = SafetyCallback()
        callbacks = [checkpoint_callback, safety_callback]
        if args.eval_during_training:
            og.log.warning(
                "EvalCallback enabled during training. OmniGibson supports only one active sim; "
                "running eval in a separate process is recommended."
            )
            eval_env = make_vec_env()
            eval_callback = EvalCallback(eval_env=eval_env, eval_freq=1000, n_eval_episodes=20)
            callbacks.append(eval_callback)
        callback = CallbackList(callbacks)

        og.log.info("Starting training...")
        model.learn(total_timesteps=10000000, callback=callback, reset_num_timesteps=not args.resume)
        og.log.info("Finished training!")


if __name__ == "__main__":
    main()