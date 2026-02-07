import os
import time

import yaml

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.learning.utils.obs_utils import create_video_writer, write_video

NUM_STEPS = 100


def main(random_selection=False, headless=False, short_exec=False):
    # Load the config
    gm.RENDER_VIEWER_CAMERA = True
    gm.ENABLE_FLATCACHE = True
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_TRANSITION_RULES = False
    gm.ENABLE_OBJECT_STATES = False

    config_filename = os.path.join(og.example_config_path, "franka_vector_env.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "walls", "coffee_table"]

    # Load the environment
    vec_env = og.VectorEnvironment(5, config)
    
    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()
    
    # Setup video recording
    # Use 1280x720 for viewer camera (default resolution in config)
    import datetime
    # Setup video output directory
    video_dir = os.path.dirname(__file__) + "/log_videos"
    os.makedirs(video_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_fpath = f"{video_dir}/vector_env_demo_god_view_{timestamp}.mp4"
    
    video_writer = create_video_writer(
        fpath=video_fpath,
        resolution=(720, 1280),
        rate=30,
    )
    print(f"[DEBUG] Video recording started: {video_fpath}")

    max_iterations = 3 if not short_exec else 1
    for _ in range(max_iterations):
        start_time = time.time()
        for _ in range(NUM_STEPS):
            actions = []
            for e in vec_env.envs:
                actions.append(e.action_space.sample())
            vec_env.step(actions)

            # Record Viewer Camera (God View)
            # og.sim.viewer_camera points to the main viewer camera
            viewer_obs, _ = og.sim.viewer_camera.get_obs()
            if "rgb" in viewer_obs:
                print(f"[DEBUG] [vector_env_demo.py] Writing video for viewer camera")
                
                # Get RGB, remove alpha channel if present, add batch dimension
                rgb_img = viewer_obs["rgb"][..., :3]
                # (H, W, 3) -> (1, H, W, 3)
                write_video(rgb_img[None, ...].cpu().numpy(), video_writer, mode="rgb")
                print(f"[DEBUG] [vector_env_demo.py] Video written for viewer camera")

        step_time = time.time() - start_time
        fps = NUM_STEPS / step_time
        effective_fps = NUM_STEPS * len(vec_env.envs) / step_time
        print("fps", fps)
        print("effective fps", effective_fps)

    # Always close the environment at the end
    if video_writer is not None:
        video_writer[0].close()
        print(f"[DEBUG] Video saved successfully to {video_fpath}")
    og.shutdown()


if __name__ == "__main__":
    main()
