#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf
from omnigibson.learning.policies import WebsocketPolicy


REPO_ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = REPO_ROOT / "RLinf"
if str(RLINF_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Isaac Sim's pip runtime needs to initialize before omni / OmniGibson imports.
try:  # pragma: no cover - depends on the local Isaac runtime
    import isaacsim  # noqa: F401
except ImportError:
    isaacsim = None

from rlinf.envs.sentinel.sentinel_env import SentinelEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local Sentinel eval against a websocket Pi0.5 policy server."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Policy server host.")
    parser.add_argument("--port", type=int, default=8000, help="Policy server port.")
    parser.add_argument(
        "--benchmark-root",
        default=str(
            REPO_ROOT
            / "datasets/canonical_tabletop_20260407/benchmark_20260407_Curated_Full_Canonical"
        ),
        help="Frozen canonical scene root.",
    )
    parser.add_argument(
        "--activity-root",
        default=str(
            REPO_ROOT
            / "datasets/canonical_tabletop_20260407/benchmark_20260407_Curated_Full_Canonical_activity_definitions"
        ),
        help="Dataset-local activity definitions root.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs/local_4080_websocket_eval"),
        help="Directory for video/results artifacts.",
    )
    parser.add_argument(
        "--scene-name",
        action="append",
        dest="scene_names",
        help="Optional scene name override. Repeat to pin specific scenes.",
    )
    parser.add_argument("--max-scenes", type=int, default=1, help="Number of scenes to assign.")
    parser.add_argument(
        "--max-episode-steps", type=int, default=20, help="Short rollout horizon."
    )
    parser.add_argument("--seed-offset", type=int, default=0, help="Worker seed offset.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the Sentinel env in headless mode.",
    )
    parser.add_argument(
        "--reset-only",
        action="store_true",
        help="Reset the env once and exit before stepping.",
    )
    parser.add_argument(
        "--graceful-exit",
        action="store_true",
        help="Attempt normal interpreter shutdown instead of forcing a fast process exit.",
    )
    return parser.parse_args()


def build_env_cfg(args: argparse.Namespace):
    env_defaults = OmegaConf.load(
        REPO_ROOT / "RLinf/examples/embodiment/config/env/sentinel_franka_tabletop.yaml"
    )
    og_cfg = OmegaConf.load(
        REPO_ROOT
        / f"OmniGibson/omnigibson/configs/{env_defaults.base_config_name}.yaml"
    )

    cfg = OmegaConf.create(
        {
            "auto_reset": False,
            "ignore_terminations": True,
            "max_episode_steps": args.max_episode_steps,
            "video_cfg": {
                "save_video": True,
                "video_base_dir": str(Path(args.output_dir).resolve() / "sentinel_eval"),
            },
            "sentinel_cfg": OmegaConf.to_container(env_defaults.sentinel_cfg, resolve=True),
            "omnigibson_cfg": og_cfg,
        }
    )
    cfg.sentinel_cfg.benchmark_root = str(Path(args.benchmark_root).resolve())
    cfg.sentinel_cfg.activity_root = str(Path(args.activity_root).resolve())
    cfg.sentinel_cfg.max_scenes = args.max_scenes
    cfg.sentinel_cfg.headless = bool(args.headless)
    if args.scene_names:
        cfg.sentinel_cfg.scene_names = list(args.scene_names)
    return cfg


def finish_process(graceful_exit: bool) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    if not graceful_exit:
        # Isaac / OmniGibson can segfault during interpreter shutdown after a successful one-shot run.
        os._exit(0)


def main() -> None:
    args = parse_args()
    cfg = build_env_cfg(args)
    env = SentinelEnv(
        cfg=cfg,
        num_envs=1,
        seed_offset=args.seed_offset,
        total_num_processes=1,
        worker_info={"worker_id": 0},
        record_metrics=True,
    )

    obs, _ = env.reset()
    if args.reset_only:
        print(
            json.dumps(
                {
                    "scene_name": env._scene_specs[0].scene_name,
                    "prompt": env._scene_specs[0].prompt,
                    "main_image_shape": list(obs["main_images"].shape),
                    "wrist_image_shape": list(obs["wrist_images"].shape),
                    "state_shape": list(obs["states"].shape),
                },
                ensure_ascii=True,
            )
        )
        finish_process(args.graceful_exit)
        return

    policy = WebsocketPolicy(host=args.host, port=args.port)
    policy.reset()

    total_reward = 0.0
    final_step = 0
    final_done = False
    for step_idx in range(args.max_episode_steps):
        action = policy.forward(obs)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        obs, rewards, terminations, truncations, infos = env.step(action)
        total_reward += float(rewards[0].item())
        final_step = step_idx + 1
        final_done = bool(terminations[0].item() or truncations[0].item())
        if final_done:
            break

    print(
        json.dumps(
            {
                "scene_name": env._scene_specs[0].scene_name,
                "prompt": env._scene_specs[0].prompt,
                "episode_len": final_step,
                "done": final_done,
                "return": total_reward,
                "results_dir": str(Path(cfg.video_cfg.video_base_dir).resolve().parent),
            },
            ensure_ascii=True,
        )
    )
    finish_process(args.graceful_exit)


if __name__ == "__main__":
    main()
