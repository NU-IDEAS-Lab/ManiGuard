#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from omnigibson.learning.utils.array_tensor_utils import torch_to_numpy
from omnigibson.learning.utils.network_utils import WebsocketClientPolicy
from PIL import Image


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
        "--pi05-checkpoint-dir",
        default=str(REPO_ROOT / "checkpoints" / "RLinf-Pi05-LIBERO-SFT"),
        help="Local Pi0.5 checkpoint root used for loading optional reset-state statistics.",
    )
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
    parser.add_argument(
        "--action-frequency",
        type=int,
        default=20,
        help="Environment action/control frequency in Hz. openpi non-DROID Franka expects 20 Hz.",
    )
    parser.add_argument(
        "--rendering-frequency",
        type=int,
        default=20,
        help="Environment rendering frequency in Hz. Defaults to matching action frequency for artifact fidelity.",
    )
    parser.add_argument(
        "--physics-frequency",
        type=int,
        default=120,
        help="Environment physics frequency in Hz.",
    )
    parser.add_argument(
        "--open-loop-horizon",
        type=int,
        default=2,
        help="How many actions to consume from each predicted chunk before querying the policy again. "
        "Defaults to a small local-debug value for frequent replanning; pass an explicit larger value for benchmark-style runs.",
    )
    parser.add_argument(
        "--control-mode",
        choices=("receding_horizon", "receding_temporal"),
        default="receding_horizon",
        help="How to consume action chunks. `receding_horizon` executes chunk actions directly; "
        "`receding_temporal` ensembles overlapping replans to reduce zig-zag control chatter.",
    )
    parser.add_argument(
        "--temporal-ensemble-max",
        type=int,
        default=5,
        help="Maximum number of recent replans to ensemble when using `--control-mode receding_temporal`.",
    )
    parser.add_argument(
        "--ltl-violation-grace-steps",
        type=int,
        default=0,
        help="After the first LTL violation, continue for this many extra steps before terminating.",
    )
    parser.add_argument(
        "--terminate-on-ltl-violation",
        action="store_true",
        help="Terminate rollout on LTL violation after the configured grace steps.",
    )
    parser.add_argument(
        "--ignore-terminations",
        action="store_true",
        help="Ignore success / failure terminations and only stop on time limit.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable policy/review video recording for this run.",
    )
    parser.add_argument(
        "--save-action-trace",
        action="store_true",
        help="Write per-step state/action traces for debugging the policy-control contract.",
    )
    parser.add_argument(
        "--reset-joint-source",
        choices=("scene", "pi05_state_mean"),
        default="scene",
        help="Robot reset source: use the frozen scene robot pose, or override joints with Pi0.5 state mean.",
    )
    parser.add_argument(
        "--embodiment-profile",
        default="franka_tabletop_libero_v1",
        help="Sentinel embodiment profile name.",
    )
    parser.add_argument("--seed-offset", type=int, default=0, help="Worker seed offset.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the Sentinel env in headless mode.",
    )
    parser.add_argument(
        "--reset-settle-steps",
        type=int,
        default=10,
        help="How many simulator steps to wait after reset before the first policy query. "
        "This mirrors the stabilization wait used by upstream sim examples such as LIBERO.",
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


def _load_pi05_state_mean(checkpoint_dir: str) -> list[float]:
    norm_stats_path = Path(checkpoint_dir).resolve() / "assets" / "franka" / "norm_stats.json"
    norm_stats = json.loads(norm_stats_path.read_text(encoding="utf-8"))
    state_mean = norm_stats["norm_stats"]["state"]["mean"]
    if len(state_mean) != 8:
        raise ValueError(
            f"Expected 8D Pi0.5 Franka state mean in {norm_stats_path}, got {len(state_mean)}"
        )
    return [float(value) for value in state_mean]


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
            "ignore_terminations": bool(args.ignore_terminations),
            "max_episode_steps": args.max_episode_steps,
            "ltl_violation_grace_steps": args.ltl_violation_grace_steps,
            "terminate_on_ltl_violation": bool(args.terminate_on_ltl_violation),
            "video_cfg": {
                "save_video": not bool(args.no_video),
                "video_base_dir": str(Path(args.output_dir).resolve() / "sentinel_eval"),
            },
            "sentinel_cfg": OmegaConf.to_container(env_defaults.sentinel_cfg, resolve=True),
            "omnigibson_cfg": og_cfg,
        }
    )
    cfg.sentinel_cfg.benchmark_root = str(Path(args.benchmark_root).resolve())
    cfg.sentinel_cfg.activity_root = str(Path(args.activity_root).resolve())
    cfg.sentinel_cfg.max_scenes = args.max_scenes
    cfg.sentinel_cfg.embodiment_profile = args.embodiment_profile
    cfg.sentinel_cfg.headless = bool(args.headless)
    cfg.sentinel_cfg.robot_reset_source = args.reset_joint_source
    cfg.sentinel_cfg.reset_settle_steps = int(args.reset_settle_steps)
    cfg.omnigibson_cfg.env.action_frequency = int(args.action_frequency)
    cfg.omnigibson_cfg.env.rendering_frequency = int(args.rendering_frequency)
    cfg.omnigibson_cfg.env.physics_frequency = int(args.physics_frequency)
    if cfg.omnigibson_cfg.robots:
        cfg.omnigibson_cfg.robots[0].control_freq = int(args.action_frequency)
    cfg.sentinel_cfg.robot_reset_policy_state = (
        _load_pi05_state_mean(args.pi05_checkpoint_dir)
        if args.reset_joint_source == "pi05_state_mean"
        else None
    )
    if args.scene_names:
        cfg.sentinel_cfg.scene_names = list(args.scene_names)
    return cfg


def finish_process(graceful_exit: bool) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    if not graceful_exit:
        # Isaac / OmniGibson can segfault during interpreter shutdown after a successful one-shot run.
        os._exit(0)


def _to_list(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _to_list(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_list(item) for item in value]
    return value


def _controller_debug_payload(controller) -> dict | None:
    if controller is None:
        return None
    payload = {
        "class": controller.__class__.__name__,
        "command_dim": int(controller.command_dim),
        "control_dim": int(controller.control_dim),
        "control_type": int(controller.control_type),
        "command_input_limits": _to_list(controller.command_input_limits),
        "command_output_limits": _to_list(controller.command_output_limits),
        "dof_idx": _to_list(controller.dof_idx),
    }
    if hasattr(controller, "use_delta_commands"):
        payload["use_delta_commands"] = bool(controller.use_delta_commands)
    if hasattr(controller, "_mode"):
        payload["mode"] = str(controller._mode)
    if hasattr(controller, "_inverted"):
        payload["inverted"] = bool(controller._inverted)
    return payload


def _save_action_trace_meta(env: SentinelEnv, args: argparse.Namespace, output_dir: Path, obs: dict) -> Path:
    scene_name = env._scene_specs[0].scene_name
    robot = env.envs[0].robots[0]
    arm_name = f"arm_{robot.default_arm}"
    gripper_name = f"gripper_{robot.default_arm}"
    arm_indices = robot.arm_control_idx[robot.default_arm]
    gripper_indices = robot.gripper_control_idx[robot.default_arm]
    arm_joint_positions = robot.get_joint_positions()[arm_indices]
    gripper_joint_positions = robot.get_joint_positions()[gripper_indices]
    action_space = getattr(robot, "action_space", None)
    arm_joint_names = list(getattr(robot, "arm_joint_names", {}).get(robot.default_arm, []))
    gripper_joint_names = list(getattr(robot, "gripper_joint_names", {}).get(robot.default_arm, []))

    meta_path = output_dir / f"{scene_name}_action_trace_meta.json"
    robot_control_freq = float(getattr(robot, "control_freq", getattr(robot, "_control_freq", float("nan"))))
    meta_payload = {
        "scene_name": scene_name,
        "prompt": env._scene_specs[0].prompt,
        "policy_camera": env._camera_views[0],
        "max_episode_steps": int(args.max_episode_steps),
        "env_action_frequency_hz": int(args.action_frequency),
        "env_rendering_frequency_hz": int(args.rendering_frequency),
        "env_physics_frequency_hz": int(args.physics_frequency),
        "robot_control_frequency_hz": robot_control_freq,
        "control_mode": str(args.control_mode),
        "temporal_ensemble_max": int(args.temporal_ensemble_max),
        "policy_client_image_preprocess": {
            "method": "resize_with_pad",
            "target_hw": [224, 224],
        },
        "reset_joint_source": str(args.reset_joint_source),
        "reset_policy_state": _to_list(obs["states"][0]),
        "arm_joint_names": arm_joint_names,
        "gripper_joint_names": gripper_joint_names,
        "arm_joint_lower_limits": _to_list(robot.joint_lower_limits[arm_indices]),
        "arm_joint_upper_limits": _to_list(robot.joint_upper_limits[arm_indices]),
        "gripper_joint_lower_limits": _to_list(robot.joint_lower_limits[gripper_indices]),
        "gripper_joint_upper_limits": _to_list(robot.joint_upper_limits[gripper_indices]),
        "reset_arm_joint_positions": _to_list(arm_joint_positions),
        "reset_gripper_joint_positions": _to_list(gripper_joint_positions),
        "reset_gripper_joint_mean": float(torch.mean(gripper_joint_positions).item()),
        "reset_gripper_joint_sum": float(torch.sum(gripper_joint_positions).item()),
        "sensor_debug": _to_list(env._sensor_debug_infos[0]),
        "robot_action_dim": int(robot.action_dim),
        "robot_action_space_low": _to_list(action_space.low) if action_space is not None else None,
        "robot_action_space_high": _to_list(action_space.high) if action_space is not None else None,
        "controllers": {
            arm_name: _controller_debug_payload(robot._controllers.get(arm_name)),
            gripper_name: _controller_debug_payload(robot._controllers.get(gripper_name)),
        },
    }
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return meta_path


def _write_reset_debug_artifacts(env: SentinelEnv, output_dir: Path) -> dict:
    scene_name = env._scene_specs[0].scene_name
    policy_start_path = None
    wrist_start_path = None
    camera_path = output_dir / f"{scene_name}_reset_only_camera.json"

    if env._policy_start_frames[0] is not None:
        policy_start_path = output_dir / f"{scene_name}_reset_only_policy_start.png"
        imageio.imwrite(policy_start_path, env._policy_start_frames[0])
    if env._wrist_start_frames[0] is not None:
        wrist_start_path = output_dir / f"{scene_name}_reset_only_wrist_start.png"
        imageio.imwrite(wrist_start_path, env._wrist_start_frames[0])

    camera_payload = {
        "scene_name": scene_name,
        "policy_camera": env._camera_views[0],
        "sensor_debug": env._sensor_debug_infos[0],
    }
    camera_path.write_text(json.dumps(camera_payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return {
        "policy_start_path": str(policy_start_path) if policy_start_path is not None else None,
        "wrist_start_path": str(wrist_start_path) if wrist_start_path is not None else None,
        "camera_path": str(camera_path),
    }


def _trace_step_payload(
    step_idx: int,
    chunk_idx: int,
    chunk_len: int,
    policy_queries: int,
    requested_new_chunk: bool,
    state_before: torch.Tensor,
    raw_action: torch.Tensor,
    env_action: torch.Tensor,
    next_state: torch.Tensor,
    infos: dict,
    rewards: torch.Tensor,
    terminations: torch.Tensor,
    truncations: torch.Tensor,
    action_space_low,
    action_space_high,
    control_mode: str,
    temporal_active_sequences: int | None = None,
) -> dict:
    state_before = state_before.detach().cpu().to(torch.float32)
    raw_action = raw_action.detach().cpu().to(torch.float32)
    env_action = env_action.detach().cpu().to(torch.float32)
    next_state = next_state.detach().cpu().to(torch.float32)
    delta_from_state = raw_action - state_before

    low = torch.as_tensor(action_space_low, dtype=torch.float32) if action_space_low is not None else None
    high = torch.as_tensor(action_space_high, dtype=torch.float32) if action_space_high is not None else None
    exceeded_low = []
    exceeded_high = []
    compared_dims = 0
    dropped_action_dims = []
    if low is not None and high is not None:
        compared_dims = min(int(env_action.shape[0]), int(low.shape[0]), int(high.shape[0]))
        for idx, value in enumerate(raw_action[:compared_dims].tolist()):
            if value < float(low[idx]):
                exceeded_low.append({"dim": idx, "value": float(value), "low": float(low[idx])})
            if value > float(high[idx]):
                exceeded_high.append({"dim": idx, "value": float(value), "high": float(high[idx])})
        for idx, value in enumerate(raw_action[compared_dims:].tolist(), start=compared_dims):
            dropped_action_dims.append({"dim": idx, "value": float(value)})

    goal_status = infos.get("goal_status", [None])[0]
    return {
        "step_idx": int(step_idx),
        "chunk_idx": int(chunk_idx),
        "chunk_len": int(chunk_len),
        "policy_queries": int(policy_queries),
        "requested_new_chunk": bool(requested_new_chunk),
        "control_mode": str(control_mode),
        "temporal_active_sequences": (
            int(temporal_active_sequences) if temporal_active_sequences is not None else None
        ),
        "incoming_action_dim": int(raw_action.shape[0]),
        "robot_action_dim": int(compared_dims if low is not None and high is not None else env_action.shape[0]),
        "state_before": state_before.tolist(),
        "action": raw_action.tolist(),
        "action_sent_to_env": env_action.tolist(),
        "action_was_clipped": bool(not torch.equal(raw_action, env_action)),
        "action_clip_delta": (env_action - raw_action).tolist(),
        "action_minus_state": delta_from_state.tolist(),
        "next_state": next_state.tolist(),
        "reward": float(rewards[0].item()),
        "terminated": bool(terminations[0].item()),
        "truncated": bool(truncations[0].item()),
        "success": bool(infos["episode"]["success_once"][0].item()),
        "ltl_violation": bool(infos.get("ltl_violation", torch.tensor([False]))[0].item()),
        "ltl_violation_terminated": bool(
            infos.get("ltl_violation_terminated", torch.tensor([False]))[0].item()
        ),
        "goal_status": goal_status,
        "action_exceeded_action_space_low": exceeded_low,
        "action_exceeded_action_space_high": exceeded_high,
        "dropped_action_dims": dropped_action_dims,
    }


def _resolve_open_loop_horizon(server_metadata: dict, requested_horizon: int | None) -> int:
    server_metadata = server_metadata or {}
    server_chunk = int(server_metadata.get("action_chunk", 1))
    if requested_horizon is None:
        return max(server_chunk, 1)
    return max(min(int(requested_horizon), server_chunk), 1)


def _clip_action_to_space(action: torch.Tensor, low, high) -> torch.Tensor:
    clipped = action.detach().cpu().to(torch.float32).clone()
    if low is None or high is None:
        return clipped

    low_t = torch.as_tensor(low, dtype=torch.float32)
    high_t = torch.as_tensor(high, dtype=torch.float32)
    dims = min(int(clipped.shape[0]), int(low_t.shape[0]), int(high_t.shape[0]))
    finite_low = torch.isfinite(low_t[:dims])
    finite_high = torch.isfinite(high_t[:dims])
    if torch.any(finite_low):
        clipped[:dims][finite_low] = torch.maximum(clipped[:dims][finite_low], low_t[:dims][finite_low])
    if torch.any(finite_high):
        clipped[:dims][finite_high] = torch.minimum(clipped[:dims][finite_high], high_t[:dims][finite_high])
    return clipped


def _resize_with_pad_uint8(image: np.ndarray, height: int, width: int) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)
    if image.shape[:2] == (height, width):
        return image

    cur_height, cur_width = image.shape[:2]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized = np.asarray(
        Image.fromarray(image).resize((resized_width, resized_height), resample=Image.BILINEAR)
    )
    out = np.zeros((height, width, image.shape[2]), dtype=np.uint8)
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    out[
        pad_h0 : pad_h0 + resized_height,
        pad_w0 : pad_w0 + resized_width,
    ] = resized
    return out


def _prepare_policy_obs(obs: dict, image_size: tuple[int, int] = (224, 224)) -> dict:
    policy_obs = torch_to_numpy(obs)
    prepared = dict(policy_obs)
    target_h, target_w = image_size
    prepared["main_images"] = np.stack(
        [_resize_with_pad_uint8(image, target_h, target_w) for image in np.asarray(policy_obs["main_images"])],
        axis=0,
    )
    prepared["wrist_images"] = np.stack(
        [_resize_with_pad_uint8(image, target_h, target_w) for image in np.asarray(policy_obs["wrist_images"])],
        axis=0,
    )
    return prepared


def _query_policy_chunk(policy: WebsocketClientPolicy, obs: dict) -> torch.Tensor:
    action_chunk = policy.act(obs).detach().cpu().to(torch.float32)
    if action_chunk.ndim == 1:
        action_chunk = action_chunk.unsqueeze(0)
    return action_chunk


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
    policy_camera = env._camera_views[0]
    trace_path = None
    trace_meta_path = None
    action_space_low = None
    action_space_high = None
    robot_action_space = env.envs[0].robots[0].action_space
    if args.save_action_trace:
        trace_dir = Path(cfg.video_cfg.video_base_dir).resolve().parent
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / "sentinel_action_trace.jsonl"
        if trace_path.exists():
            trace_path.unlink()
        trace_meta_path = _save_action_trace_meta(env, args, trace_dir, obs)
        action_space_low = np.asarray(robot_action_space.low, dtype=np.float32)
        action_space_high = np.asarray(robot_action_space.high, dtype=np.float32)
    if args.reset_only:
        reset_debug_paths = _write_reset_debug_artifacts(
            env, Path(cfg.video_cfg.video_base_dir).resolve().parent
        )
        reset_ltl = dict((env._reset_infos[0] or {}).get("ltl") or {})
        print(
            json.dumps(
                {
                    "scene_name": env._scene_specs[0].scene_name,
                    "prompt": env._scene_specs[0].prompt,
                    "main_image_shape": list(obs["main_images"].shape),
                    "wrist_image_shape": list(obs["wrist_images"].shape),
                    "state_shape": list(obs["states"].shape),
                    "reset_ltl_violation": bool(reset_ltl.get("doomed", False)),
                    "reset_ltl_state": reset_ltl.get("state"),
                    "policy_camera": policy_camera,
                    "sensor_debug": env._sensor_debug_infos[0],
                    "action_trace_meta": str(trace_meta_path) if trace_meta_path else None,
                    "reset_debug_artifacts": reset_debug_paths,
                },
                ensure_ascii=True,
            )
        )
        finish_process(args.graceful_exit)
        return

    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    policy.reset()
    server_metadata = policy.get_server_metadata() or {}
    open_loop_horizon = _resolve_open_loop_horizon(server_metadata, args.open_loop_horizon)
    current_action_chunk = None
    chunk_step_idx = 0
    policy_query_count = 0
    temporal_sequences: list[dict[str, object]] = []
    temporal_step_counter = 0

    total_reward = 0.0
    final_step = 0
    final_done = False
    final_terminated = False
    final_truncated = False
    final_success = False
    final_ltl_violation = False
    final_ltl_violation_terminated = False
    final_termination_reason = "running"
    final_ltl_violation_first_step = None
    final_goal_status = None
    reset_ltl = {}
    for step_idx in range(args.max_episode_steps):
        if step_idx == 0:
            reset_ltl = dict((env._reset_infos[0] or {}).get("ltl") or {})
        state_before = obs["states"][0].detach().cpu().clone()
        requested_new_chunk = (
            current_action_chunk is None
            or chunk_step_idx >= open_loop_horizon
            or chunk_step_idx >= int(current_action_chunk.shape[0])
        )
        temporal_active_sequences = None
        if args.control_mode == "receding_horizon":
            if requested_new_chunk:
                policy_obs = _prepare_policy_obs(obs, image_size=(224, 224))
                current_action_chunk = _query_policy_chunk(policy, policy_obs)
                chunk_step_idx = 0
                policy_query_count += 1

            raw_action = current_action_chunk[chunk_step_idx]
            active_chunk_idx = chunk_step_idx
            current_chunk_len = int(current_action_chunk.shape[0])
            chunk_step_idx += 1
        else:
            requested_new_chunk = (not temporal_sequences) or (
                temporal_step_counter % open_loop_horizon == 0
            )
            if requested_new_chunk:
                policy_obs = _prepare_policy_obs(obs, image_size=(224, 224))
                new_chunk = _query_policy_chunk(policy, policy_obs)
                temporal_sequences.append({"actions": new_chunk, "idx": 0})
                if len(temporal_sequences) > int(args.temporal_ensemble_max):
                    temporal_sequences = temporal_sequences[-int(args.temporal_ensemble_max) :]
                policy_query_count += 1

            current_actions = []
            for sequence in temporal_sequences:
                sequence_actions = sequence["actions"]
                sequence_idx = int(sequence["idx"])
                current_idx = min(sequence_idx, int(sequence_actions.shape[0]) - 1)
                current_actions.append(sequence_actions[current_idx])
            stacked_actions = torch.stack(current_actions, dim=0)
            weights = torch.exp(0.005 * torch.arange(stacked_actions.shape[0], dtype=torch.float32))
            weights = weights / torch.sum(weights)
            raw_action = torch.sum(stacked_actions * weights[:, None], dim=0)
            raw_action[-1] = stacked_actions[-1, -1]  # gripper: use latest replan
            active_chunk_idx = int(
                min(int(temporal_sequences[-1]["idx"]), int(temporal_sequences[-1]["actions"].shape[0]) - 1)
            )
            current_chunk_len = int(temporal_sequences[-1]["actions"].shape[0])
            temporal_active_sequences = len(temporal_sequences)

            next_sequences: list[dict[str, object]] = []
            for sequence in temporal_sequences:
                next_idx = int(sequence["idx"]) + 1
                sequence["idx"] = next_idx
                if next_idx < int(sequence["actions"].shape[0]):
                    next_sequences.append(sequence)
            temporal_sequences = next_sequences
            temporal_step_counter += 1

        env_action = _clip_action_to_space(
            raw_action,
            low=robot_action_space.low,
            high=robot_action_space.high,
        )
        action = env_action.unsqueeze(0)
        obs, rewards, terminations, truncations, infos = env.step(action)
        if trace_path is not None:
            trace_payload = _trace_step_payload(
                step_idx=step_idx + 1,
                chunk_idx=active_chunk_idx,
                chunk_len=current_chunk_len,
                policy_queries=policy_query_count,
                requested_new_chunk=requested_new_chunk,
                state_before=state_before,
                raw_action=raw_action,
                env_action=env_action,
                next_state=obs["states"][0],
                infos=infos,
                rewards=rewards,
                terminations=terminations,
                truncations=truncations,
                action_space_low=action_space_low,
                action_space_high=action_space_high,
                control_mode=args.control_mode,
                temporal_active_sequences=temporal_active_sequences,
            )
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(trace_payload, ensure_ascii=True) + "\n")
        total_reward += float(rewards[0].item())
        final_step = step_idx + 1
        final_terminated = bool(terminations[0].item())
        final_truncated = bool(truncations[0].item())
        final_success = bool(infos["episode"]["success_once"][0].item())
        final_ltl_violation = bool(infos.get("ltl_violation", torch.tensor([False]))[0].item())
        final_ltl_violation_terminated = bool(
            infos.get("ltl_violation_terminated", torch.tensor([False]))[0].item()
        )
        final_ltl_violation_first_step = infos.get("ltl_violation_first_step", [None])[0]
        final_goal_status = infos.get("goal_status", [None])[0]
        if final_success:
            final_termination_reason = "success"
        elif final_truncated:
            final_termination_reason = "time_limit"
        elif final_ltl_violation_terminated:
            final_termination_reason = "ltl_violation"
        elif final_terminated:
            final_termination_reason = "env_terminated"
        final_done = bool(final_terminated or final_truncated)
        if final_done:
            break

    print(
        json.dumps(
            {
                "scene_name": env._scene_specs[0].scene_name,
                "prompt": env._scene_specs[0].prompt,
                "episode_len": final_step,
                "done": final_done,
                "terminated": final_terminated,
                "truncated": final_truncated,
                "success": final_success,
                "ltl_violation": final_ltl_violation,
                "ltl_violation_terminated": final_ltl_violation_terminated,
                "ltl_violation_first_step": final_ltl_violation_first_step,
                "goal_status": final_goal_status,
                "termination_reason": final_termination_reason,
                "reset_ltl_violation": bool(reset_ltl.get("doomed", False)),
                "reset_ltl_state": reset_ltl.get("state"),
                "policy_camera": policy_camera,
                "policy_server_metadata": server_metadata,
                "env_action_frequency_hz": int(args.action_frequency),
                "env_rendering_frequency_hz": int(args.rendering_frequency),
                "env_physics_frequency_hz": int(args.physics_frequency),
                "robot_control_frequency_hz": float(
                    getattr(env.envs[0].robots[0], "control_freq", getattr(env.envs[0].robots[0], "_control_freq", float("nan")))
                ),
                "control_mode": str(args.control_mode),
                "open_loop_horizon": open_loop_horizon,
                "temporal_ensemble_max": int(args.temporal_ensemble_max),
                "video_enabled": not bool(args.no_video),
                "action_trace": str(trace_path) if trace_path else None,
                "action_trace_meta": str(trace_meta_path) if trace_meta_path else None,
                "return": total_reward,
                "results_dir": str(Path(cfg.video_cfg.video_base_dir).resolve().parent),
            },
            ensure_ascii=True,
        )
    )
    finish_process(args.graceful_exit)


if __name__ == "__main__":
    main()
