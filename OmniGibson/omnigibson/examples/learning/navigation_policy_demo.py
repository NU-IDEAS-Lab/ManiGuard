"""
Example training code using stable-baselines3 PPO for one BEHAVIOR activity.
Note that due to the sparsity of the reward, this training code will not converge and achieve task success.
This only serves as a starting point that users can further build upon.
"""

import argparse
import os
import time
import datetime

import yaml
import numpy as np

import omnigibson as og
from omnigibson import example_config_path
from omnigibson.macros import gm
from omnigibson.utils.python_utils import meets_minimum_version
from omnigibson.learning.utils.obs_utils import create_video_writer, write_video

try:
    import gymnasium as gym
    import torch as th
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.utils import set_random_seed

except ModuleNotFoundError:
    og.log.error(
        "stable-baselines3 is not installed. "
        "Run the following command to install stable-baselines3:\n"
        "pip install stable-baselines3[extra]\n"
    )
    exit(1)

assert meets_minimum_version(gym.__version__, "0.28.1"), "Please install/update gymnasium to version >= 0.28.1"

# We don't need object states nor transitions rules, so we disable them now, and also enable flatcache for maximum speed
gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False
gm.ENABLE_FLATCACHE = True


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        super().__init__(observation_space, features_dim=1)
        extractors = {}
        self.step_index = 0
        self.img_save_dir = "img_save_dir"
        os.makedirs(self.img_save_dir, exist_ok=True)
        total_concat_size = 0
        feature_size = 128
        for key, subspace in observation_space.spaces.items():
            # For now, only keep RGB observations
            if "rgb" in key:
                og.log.info(f"obs {key} shape: {subspace.shape}")
                n_input_channels = subspace.shape[0]  # channel first
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

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        self.step_index += 1

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))

        feature = th.cat(encoded_tensor_list, dim=1)
        return feature


def main():
    # Parse args
    parser = argparse.ArgumentParser(description="Train or evaluate a PPO agent in BEHAVIOR")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Absolute path to desired PPO checkpoint to load for evaluation",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="If set, will evaluate the PPO agent found from --checkpoint",
    )

    parser.add_argument(
        "--video_record_freq",
        type=int,
        default=1000,
        help="Frequency (in steps) to record video during training",
    )

    args = parser.parse_args()
    tensorboard_log_dir = os.path.join("log_dir", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    prefix = ""
    seed = 0

    # Load config
    with open(f"{example_config_path}/turtlebot_nav.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Make sure flattened obs and action space is used
    cfg["env"]["flatten_action_space"] = True
    cfg["env"]["flatten_obs_space"] = True

    # Only use RGB obs
    cfg["robots"][0]["obs_modalities"] = ["rgb"]

    # If we're not eval, turn off the start / goal markers so the agent doesn't see them
    if not args.eval:
        cfg["task"]["visualize_goal"] = False

    env = og.Environment(configs=cfg)

    # If we're evaluating, hide the ceilings and enable camera teleoperation so the user can easily
    # visualize the rollouts dynamically
    if args.eval:
        ceiling = env.scene.object_registry("name", "ceilings")
        ceiling.visible = False
        og.sim.enable_viewer_camera_teleoperation()

    # Set the set
    set_random_seed(seed)
    env.reset()

    # Video recording setup (only for training if not eval, or both if desired)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_fpath = f"navigation_training_demo_{timestamp}.mp4"
    video_writer = create_video_writer(
        fpath=video_fpath,
        resolution=(720, 1280), # Viewer camera default resolution
        rate=30,
    )
    print(f"[DEBUG] Video recording started: {video_fpath}")

    # Wrap the env's step function to record video
    original_step = env.step
    step_counter = 0 # Define a counter to keep track of steps
    
    def step_with_recording(action):
        nonlocal step_counter
        step_ret = original_step(action)
        
        # Record video only if we hit the frequency
        if args.video_record_freq > 0 and step_counter % args.video_record_freq == 0:
             # Record viewer camera
            viewer_obs, _ = og.sim.viewer_camera.get_obs()
            if "rgb" in viewer_obs:
                rgb_img = viewer_obs["rgb"][..., :3]
                if hasattr(rgb_img, "cpu"):
                    rgb_img = rgb_img.cpu().numpy()
                write_video(rgb_img[None, ...], video_writer, mode="rgb")
        
        step_counter += 1
        return step_ret
    
    # Monkey patch the step function
    env.step = step_with_recording

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )

    os.makedirs(tensorboard_log_dir, exist_ok=True)

    if args.eval:
        assert args.checkpoint is not None, "If evaluating a PPO policy, @checkpoint argument must be specified!"
        model = PPO.load(args.checkpoint)
        og.log.info("Starting evaluation...")
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
        og.log.info("Finished evaluation!")
        og.log.info(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")

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
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=tensorboard_log_dir, name_prefix=prefix)
        eval_callback = EvalCallback(eval_env=env, eval_freq=1000, n_eval_episodes=20)
        callback = CallbackList([checkpoint_callback, eval_callback])

        og.log.debug(model.policy)
        og.log.info(f"model: {model}")

        og.log.info("Starting training...")
        model.learn(
            total_timesteps=10000000,
            callback=callback,
        )
        og.log.info("Finished training!")
        
        # Cleanup
        if video_writer:
            video_writer[0].close()
            print(f"[DEBUG] Video saved to {video_fpath}")


if __name__ == "__main__":
    main()
