import os

import yaml
import numpy as np

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import choose_from_options
from omnigibson.learning.utils.obs_utils import create_video_writer, write_video

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Generates a BEHAVIOR Task environment in an online fashion.

    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Ask the user whether they want online object sampling or not
    sampling_options = {
        False: "Use a pre-sampled cached BEHAVIOR activity scene",
        True: "Sample the BEHAVIOR activity in an online fashion",
    }
    should_sample = choose_from_options(
        options=sampling_options, name="online object sampling", random_selection=random_selection
    )

    # Load the pre-selected configuration and set the online_sampling flag
    config_filename = os.path.join(og.example_config_path, "r1pro_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["task"]["online_object_sampling"] = should_sample
    cfg["task"]["use_presampled_robot_pose"] = not should_sample

    # Load the environment
    env = og.Environment(configs=cfg)

    # Setup video recording
    # Use 1280x720 for viewer camera (default resolution in config)
    import datetime
    # Setup video output directory
    video_dir = os.path.dirname(__file__) + "/log_videos"
    os.makedirs(video_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_fpath = f"{video_dir}/behavior_demo_god_view_{timestamp}.mp4"
    
    video_writer = create_video_writer(
        fpath=video_fpath,
        resolution=(720, 1280),
        rate=30,
    )
    print(f"[DEBUG] Video recording started: {video_fpath}")

    # Move camera to a good position
    og.sim.viewer_camera.set_position_orientation(
        position=[1.6, 6.15, 1.5], orientation=[-0.2322, 0.5895, 0.7199, -0.2835]
    )

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    print("="*60)
    print("="*60)
    print("="*60)
    print("="*60)
    print(f"[DEBUG] [behavior_env_demo.py] So far so good. Ready to step the environment.")
    print("="*60)
    print("="*60)
    print("="*60)
    print("="*60)
    print("="*60)
    
    # Run a simple loop and reset periodically
    try:
        max_iterations = 3 if not short_exec else 1
        for j in range(max_iterations):
            print("="*60)
            print("="*60)
            print(f"[DEBUG] [behavior_env_demo.py] Iteration {j}: Resetting environment")
            print("="*60)
            print("="*60)
            
            og.log.info("Resetting environment")
            env.reset()
            for i in range(100):
                print("="*60)
                print("="*60)
                print(f"[DEBUG] [behavior_env_demo.py] Iteration {j}: Step {i}: Taking action")
                print("="*60)
                print("="*60)
                action = env.robots[0].action_space.sample()
                # breakpoint()
                state, reward, terminated, truncated, info = env.step(action * 0.2)

                # Record Viewer Camera (God View)
                # og.sim.viewer_camera points to the main viewer camera
                viewer_obs, _ = og.sim.viewer_camera.get_obs()
                if "rgb" in viewer_obs:
                    print(f"[DEBUG] [behavior_env_demo.py] Writing video for viewer camera")
                    
                    # Get RGB, remove alpha channel if present, add batch dimension
                    rgb_img = viewer_obs["rgb"][..., :3]
                    # (H, W, 3) -> (1, H, W, 3)
                    write_video(rgb_img[None, ...].cpu().numpy(), video_writer, mode="rgb")
                    print(f"[DEBUG] [behavior_env_demo.py] Video written for viewer camera")
                    print("="*60)
                    print("="*60)

                if terminated or truncated:
                    og.log.info("Episode finished after {} timesteps".format(i + 1))
                    break

    finally:
        # Always close the environment at the end
        if video_writer is not None:
            video_writer[0].close()
            print(f"[DEBUG] Video saved successfully to {video_fpath}")
        og.shutdown()


if __name__ == "__main__":
    # main()
    main(random_selection=True)
