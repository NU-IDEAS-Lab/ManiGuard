#!/usr/bin/env python3
"""Record offline rollout videos for a proprio/object-position PPO checkpoint.

This entry point is intentionally eval-only: it loads a saved policy, runs
``model.predict()`` rollouts, writes viewer-camera MP4s and a metrics JSON, and
never calls ``learn()`` or writes training checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch as th

from sentinel.rl.algorithms.ppo_proprio import (
    _make_proprio_scene_copy,
    _validate_state_obs_space,
)
from sentinel.rl.algorithms.ppo_proprio_goal import _validate_privileged_obs_space
from sentinel.rl.cli.common import add_env_args, validate_env_args


def parse_args():
    p = argparse.ArgumentParser(
        description="Record eval rollout videos for a proprio PPO checkpoint."
    )
    add_env_args(p)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to a saved SB3 PPO checkpoint (.zip).")
    p.add_argument("--task-type", choices=["PickAndLiftTask", "PickAndLiftPrivilegedTask"],
                   default="PickAndLiftTask",
                   help="Task used by the checkpoint. ppo_proprio_goal uses "
                        "PickAndLiftPrivilegedTask with a 10D task::low_dim.")
    p.add_argument("--grasping-mode", choices=["physical", "assisted", "sticky"],
                   default=None,
                   help="Optional robot grasping_mode to write into the eval "
                        "scene override. Use sticky for sticky-trained checkpoints.")
    p.add_argument("--n-episodes", type=int, default=3,
                   help="Number of eval episodes to record.")
    p.add_argument("--max-steps-per-episode", type=int, default=220,
                   help="Hard cap for each recorded episode.")
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--camera-mode", choices=["viewer", "wrist"], default="viewer",
                   help="viewer: record the default viewer camera. wrist: move "
                        "the viewer camera near the EEF and point at the target.")
    p.add_argument("--wrist-distance", type=float, default=0.12,
                   help="Meters to place the wrist camera behind the EEF from "
                        "the target direction.")
    p.add_argument("--wrist-height", type=float, default=0.06,
                   help="Meters to lift the wrist camera above the EEF.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/rl_proprio_eval_video"))
    p.add_argument("--stochastic", action="store_true",
                   help="Use stochastic policy actions instead of deterministic.")
    return p.parse_args()


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [eval_video_proprio] {message}",
          flush=True)


def _configure_omnigibson() -> None:
    from omnigibson.macros import gm

    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_FLATCACHE = True
    if os.environ.get("OMNIGIBSON_HEADLESS", "0") == "1":
        gm.HEADLESS = True


def _unwrap_env(vec_env):
    env = vec_env.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env


def _look_at_quat(position: th.Tensor, target: th.Tensor) -> th.Tensor:
    import omnigibson.utils.transform_utils as T

    forward = target - position
    if th.norm(forward) < 1e-6:
        forward = th.tensor([1.0, 0.0, 0.0], dtype=th.float32)
    forward = forward / th.norm(forward)

    up_guess = th.tensor([0.0, 0.0, 1.0], dtype=th.float32)
    if th.abs(th.dot(forward, up_guess)) > 0.95:
        up_guess = th.tensor([0.0, 1.0, 0.0], dtype=th.float32)

    # USD cameras look down their local -Z axis. Build a world rotation whose
    # local -Z points toward target and whose local +Y is as upright as possible.
    cam_z = -forward
    cam_x = th.linalg.cross(up_guess, cam_z)
    cam_x = cam_x / th.norm(cam_x)
    cam_y = th.linalg.cross(cam_z, cam_x)
    cam_y = cam_y / th.norm(cam_y)
    rmat = th.stack([cam_x, cam_y, cam_z], dim=1)
    return T.mat2quat(rmat)


def _set_wrist_camera_pose(vec_env, *, target_name: str | None,
                           wrist_distance: float, wrist_height: float) -> None:
    import omnigibson as og

    env = _unwrap_env(vec_env)
    robot = env.robots[0]
    arm = robot.default_arm
    eef_pos = robot.get_eef_position(arm).detach().cpu().to(th.float32)

    target_pos = None
    if target_name is not None:
        obj = env.scene.object_registry("name", target_name)
        if obj is not None:
            target_pos = obj.get_position_orientation()[0].detach().cpu().to(th.float32)
    if target_pos is None:
        # Fallback: look a short distance forward from the EEF in world X.
        target_pos = eef_pos + th.tensor([0.25, 0.0, 0.0], dtype=th.float32)

    away_from_target = eef_pos - target_pos
    if th.norm(away_from_target) < 1e-6:
        away_from_target = th.tensor([1.0, 0.0, 0.0], dtype=th.float32)
    away_from_target = away_from_target / th.norm(away_from_target)

    cam_pos = (
        eef_pos
        + away_from_target * float(wrist_distance)
        + th.tensor([0.0, 0.0, float(wrist_height)], dtype=th.float32)
    )
    cam_quat = _look_at_quat(cam_pos, target_pos)
    og.sim.viewer_camera.set_position_orientation(position=cam_pos, orientation=cam_quat)


def _grab_viewer_frame(vec_env=None, args=None):
    if args is not None and args.camera_mode == "wrist":
        _set_wrist_camera_pose(
            vec_env,
            target_name=args.target_name,
            wrist_distance=args.wrist_distance,
            wrist_height=args.wrist_height,
        )

    import omnigibson as og

    og.sim.render()
    rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    arr = np.asarray(rgb)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype("uint8")


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), frames, fps=int(fps), quality=8,
                     macro_block_size=1)


def _rollout_episode(model, vec_env, *, deterministic: bool, max_steps: int,
                     video_path: Path, fps: int, args) -> dict:
    obs = vec_env.reset()
    frames = [_grab_viewer_frame(vec_env, args)]
    total_reward = 0.0
    total_cost = 0.0
    sim_fault_recoveries = 0
    final_info = {}
    done = False

    for step in range(1, max_steps + 1):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = vec_env.step(action)
        info = dict(infos[0])
        total_reward += float(rewards[0])
        total_cost += float(info.get("cost", 0.0))
        sim_fault_recoveries += int(bool(info.get("sim_fault_recovered")))
        frames.append(_grab_viewer_frame(vec_env, args))

        if bool(dones[0]):
            done = True
            final_info = info
            break

    if not done:
        final_info = {
            "TimeLimit.truncated": True,
            "eval_video_max_steps": int(max_steps),
        }

    truncated = bool(final_info.get("TimeLimit.truncated", False))
    success = bool(done and not truncated)
    _write_video(video_path, frames, fps)

    return {
        "video": str(video_path),
        "reward": float(total_reward),
        "cost": float(total_cost),
        "length": int(len(frames) - 1),
        "done": bool(done),
        "truncated": bool(truncated),
        "success": bool(success),
        "sim_fault_recoveries": int(sim_fault_recoveries),
    }


def _aggregate(episodes: list[dict]) -> dict:
    rewards = np.asarray([ep["reward"] for ep in episodes], dtype=np.float64)
    lengths = np.asarray([ep["length"] for ep in episodes], dtype=np.float64)
    costs = np.asarray([ep["cost"] for ep in episodes], dtype=np.float64)
    successes = np.asarray([ep["success"] for ep in episodes], dtype=np.float64)
    return {
        "n_episodes": int(len(episodes)),
        "mean_reward": float(rewards.mean()) if len(rewards) else 0.0,
        "std_reward": float(rewards.std()) if len(rewards) else 0.0,
        "min_reward": float(rewards.min()) if len(rewards) else 0.0,
        "max_reward": float(rewards.max()) if len(rewards) else 0.0,
        "mean_length": float(lengths.mean()) if len(lengths) else 0.0,
        "std_length": float(lengths.std()) if len(lengths) else 0.0,
        "success_rate": float(successes.mean()) if len(successes) else 0.0,
        "mean_cost": float(costs.mean()) if len(costs) else 0.0,
        "std_cost": float(costs.std()) if len(costs) else 0.0,
        "sim_fault_recoveries": int(sum(ep["sim_fault_recoveries"] for ep in episodes)),
    }


def main():
    args = parse_args()
    validate_env_args(args)

    if args.num_envs != 1:
        raise SystemExit("eval_video_proprio supports --num-envs 1 only.")
    if args.n_episodes < 1:
        raise SystemExit("--n-episodes must be >= 1.")
    if args.max_steps_per_episode < 1:
        raise SystemExit("--max-steps-per-episode must be >= 1.")
    if not args.checkpoint.exists():
        raise SystemExit(f"--checkpoint does not exist: {args.checkpoint}")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    _log("creating proprio scene override")
    source_scene_file = Path(args.scene_file).resolve()
    proprio_scene_file, patched_robots = _make_proprio_scene_copy(
        source_scene_file,
        out,
        grasping_mode=args.grasping_mode,
    )
    args.scene_file = proprio_scene_file
    args.task_type = args.task_type
    print(f"  source scene:           {source_scene_file}", flush=True)
    print(f"  proprio scene:          {proprio_scene_file}", flush=True)
    print(f"  patched robot obs:      {patched_robots}", flush=True)
    print(f"  task type:              {args.task_type}", flush=True)
    print(f"  grasping mode:          {args.grasping_mode or '<scene/default>'}", flush=True)

    _log("importing OmniGibson and building env")
    _configure_omnigibson()

    from stable_baselines3 import PPO

    from sentinel.rl.envs.wrappers import build_vec_env

    vec_env = build_vec_env(args, out_dir=out)
    if args.task_type == "PickAndLiftPrivilegedTask":
        _validate_privileged_obs_space(vec_env)
    else:
        _validate_state_obs_space(vec_env)

    _log(f"loading checkpoint: {args.checkpoint}")
    model = PPO.load(str(args.checkpoint), env=vec_env, device="cuda")
    print(f"  model timesteps:        {model.num_timesteps:,}", flush=True)

    deterministic = not args.stochastic
    _log(
        f"recording {args.n_episodes} "
        f"{'deterministic' if deterministic else 'stochastic'} episodes"
    )
    video_dir = out / "videos"
    episodes = []
    t0 = time.time()
    for ep_idx in range(args.n_episodes):
        video_path = video_dir / f"episode_{ep_idx:03d}.mp4"
        result = _rollout_episode(
            model,
            vec_env,
            deterministic=deterministic,
            max_steps=args.max_steps_per_episode,
            video_path=video_path,
            fps=args.video_fps,
            args=args,
        )
        episodes.append(result)
        print(
            f"  ep {ep_idx:03d}: reward={result['reward']:.3f} "
            f"len={result['length']} success={result['success']} "
            f"video={video_path}",
            flush=True,
        )

    metrics = _aggregate(episodes)
    out_json = out / (
        f"eval_video_{args.checkpoint.stem}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "source_scene_file": str(source_scene_file),
        "patched_scene_file": str(proprio_scene_file),
        "patched_robot_obs_count": int(patched_robots),
        "task_type": args.task_type,
        "grasping_mode": args.grasping_mode,
        "category": args.category,
        "model": args.model,
        "target_name": args.target_name,
        "deterministic": bool(deterministic),
        "camera_mode": args.camera_mode,
        "wrist_distance": float(args.wrist_distance),
        "wrist_height": float(args.wrist_height),
        "max_steps_per_episode": int(args.max_steps_per_episode),
        "video_fps": int(args.video_fps),
        "wall_time_s": float(time.time() - t0),
        "metrics": metrics,
        "episodes": episodes,
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n=== Eval video results ===", flush=True)
    for key, value in metrics.items():
        print(f"  {key:24s} {value}", flush=True)
    print(f"\nWrote metrics: {out_json}", flush=True)
    print(f"Wrote videos:  {video_dir}", flush=True)

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
