import os

import yaml

import omnigibson as og
from omnigibson.utils.ui_utils import choose_from_options
from omnigibson.learning.utils.obs_utils import create_video_writer, write_video


def main(random_selection=False, headless=False, short_exec=False):
    """
    Prompts the user to select a type of scene and loads a turtlebot into it, generating a Point-Goal navigation
    task within the environment.

    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Load the config
    config_filename = os.path.join(og.example_config_path, "turtlebot_nav.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # check if we want to quick load or full load the scene
    load_options = {
        "Quick": "Only load the building assets (i.e.: the floors, walls, ceilings)",
        "Full": "Load all interactive objects in the scene",
    }
    load_mode = choose_from_options(options=load_options, name="load mode", random_selection=random_selection)
    if load_mode == "Quick":
        config["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]

    # Load the environment
    env = og.Environment(configs=config)

    # Setup video recording
    # Use 1280x720 for viewer camera (default resolution in config)
    import datetime
    # Setup video output directory
    video_dir = os.path.dirname(__file__) + "/log_videos"
    os.makedirs(video_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_fpath = f"{video_dir}/navigation_env_demo_god_view_{timestamp}.mp4"
    
    video_writer = create_video_writer(
        fpath=video_fpath,
        resolution=(720, 1280),
        rate=30,
    )
    print(f"[DEBUG] Video recording started: {video_fpath}")

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Run a simple loop and reset periodically
    max_iterations = 2 if not short_exec else 1
    try:
        for j in range(max_iterations):
            print("="*60)
            print("="*60)
            print(f"[DEBUG] [navigation_env_demo.py] Iteration {j}: Resetting environment")
            print("="*60)
            print("="*60)
            
            og.log.info("Resetting environment")
            env.reset()
            for i in range(2000):
                print("="*60)
                print("="*60)
                print(f"[DEBUG] [navigation_env_demo.py] Iteration {j}: Step {i}: Taking action")
                print("="*60)
                print("="*60)
                action = env.action_space.sample()
                state, reward, terminated, truncated, info = env.step(action)

                # Record Viewer Camera (God View)
                # og.sim.viewer_camera points to the main viewer camera
                viewer_obs, _ = og.sim.viewer_camera.get_obs()
                if "rgb" in viewer_obs:
                    print(f"[DEBUG] [navigation_env_demo.py] Writing video for viewer camera")
                    
                    # Get RGB, remove alpha channel if present, add batch dimension
                    rgb_img = viewer_obs["rgb"][..., :3]
                    # (H, W, 3) -> (1, H, W, 3)
                    write_video(rgb_img[None, ...].cpu().numpy(), video_writer, mode="rgb")
                    print(f"[DEBUG] [navigation_env_demo.py] Video written for viewer camera")
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
    main()
