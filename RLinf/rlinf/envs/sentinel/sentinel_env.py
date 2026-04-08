from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from omnigibson.macros import gm

from rlinf.envs.sentinel.registry import (
    SentinelSceneSpec,
    build_runtime_scene_info,
    build_scene_registry,
    slice_scene_registry_for_worker,
)
from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor

gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = True


class SentinelEnv(gym.Env):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.record_metrics = record_metrics
        self.auto_reset = cfg.auto_reset
        self.ignore_terminations = cfg.ignore_terminations
        self.video_cfg = cfg.video_cfg
        self.sentinel_cfg = cfg.sentinel_cfg
        self._is_start = True

        gm.HEADLESS = bool(self.sentinel_cfg.get("headless", False))
        gm.USE_GPU_DYNAMICS = bool(self.sentinel_cfg.get("use_gpu_dynamics", False))
        gm.ENABLE_FLATCACHE = not gm.USE_GPU_DYNAMICS

        self._elapsed_steps = torch.zeros(self.num_envs, dtype=torch.long)
        self.success_once = torch.zeros(self.num_envs, dtype=torch.bool)
        self.returns = torch.zeros(self.num_envs, dtype=torch.float32)
        self._frames = [[] for _ in range(self.num_envs)]
        self._episode_indices = [0 for _ in range(self.num_envs)]

        self._scene_registry = build_scene_registry(
            benchmark_root=self.sentinel_cfg.benchmark_root,
            activity_root=self.sentinel_cfg.activity_root,
            scene_names=self.sentinel_cfg.get("scene_names"),
            max_scenes=self.sentinel_cfg.get("max_scenes"),
        )
        self._scene_specs = slice_scene_registry_for_worker(
            registry=self._scene_registry,
            num_envs=self.num_envs,
            seed_offset=self.seed_offset,
        )

        self._results_dir = Path(self.video_cfg.video_base_dir).resolve().parent
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._results_path = self._results_dir / "sentinel_eval_results.jsonl"
        self._prompt_log_path = self._results_dir / "sentinel_prompts.jsonl"
        self._runtime_scene_root = self._results_dir / "_runtime_scene_cache"
        self._runtime_scene_root.mkdir(parents=True, exist_ok=True)

        self._init_envs()

    def _init_envs(self):
        import omnigibson as og

        if og.sim is not None:
            og.sim.stop()

        self._patch_activity_root()
        base_cfg = OmegaConf.to_container(self.cfg.omnigibson_cfg, resolve=True)
        self.envs = []
        for spec in self._scene_specs:
            env_cfg = self._make_env_cfg(base_cfg, spec)
            self.envs.append(og.Environment(configs=env_cfg, in_vec_env=True))

        og.sim.play()
        for env in self.envs:
            env.post_play_load()

    def _make_env_cfg(self, base_cfg: dict, spec: SentinelSceneSpec) -> dict:
        env_cfg = copy.deepcopy(base_cfg)
        env_cfg["scene"]["scene_model"] = spec.scene_name
        env_cfg["scene"]["scene_file"] = self._prepare_runtime_scene_file(spec)
        env_cfg["scene"]["scene_instance"] = None
        with open(spec.problem_file, "r", encoding="utf-8") as handle:
            env_cfg["task"]["predefined_problem"] = handle.read()
        env_cfg["task"]["activity_name"] = spec.activity_name
        env_cfg["task"]["activity_definition_id"] = 0
        env_cfg["task"]["activity_instance_id"] = 0
        env_cfg["task"]["online_object_sampling"] = False
        env_cfg["task"]["use_presampled_robot_pose"] = False
        return env_cfg

    def _prepare_runtime_scene_file(self, spec: SentinelSceneSpec) -> str:
        runtime_scene_dir = self._runtime_scene_root / spec.scene_name
        runtime_scene_dir.mkdir(parents=True, exist_ok=True)
        runtime_scene_path = runtime_scene_dir / "scene_ep1.runtime.json"

        with open(spec.scene_file, "r", encoding="utf-8") as handle:
            scene_info = json.load(handle)
        with open(spec.diagnostics_file, "r", encoding="utf-8") as handle:
            diagnostics = json.loads(next(line for line in handle if line.strip()))
        with open(spec.problem_file, "r", encoding="utf-8") as handle:
            problem_text = handle.read()

        runtime_scene_info = build_runtime_scene_info(
            scene_info=scene_info,
            diagnostics=diagnostics,
            problem_text=problem_text,
        )
        with runtime_scene_path.open("w", encoding="utf-8") as handle:
            json.dump(runtime_scene_info, handle, ensure_ascii=True, indent=2)
        return str(runtime_scene_path)

    @property
    def device(self):
        return "cpu"

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _patch_activity_root(self):
        import bddl.activity
        import bddl.config
        import omnigibson.tasks.behavior_task as behavior_task
        import omnigibson.utils.bddl_utils as bddl_utils

        activity_root = str(Path(self.sentinel_cfg.activity_root).resolve())
        repo_root = Path(__file__).resolve().parents[4]
        activity_root_path = Path(activity_root)
        domain_src_dir = repo_root / "bddl3" / "bddl" / "activity_definitions"
        for domain_name in ("domain_igibson.bddl", "domain_omnigibson.bddl"):
            domain_src = domain_src_dir / domain_name
            domain_dst = activity_root_path / domain_name
            if domain_src.is_file() and not domain_dst.exists():
                os.symlink(domain_src, domain_dst)

        bddl.config.ACTIVITY_CONFIGS_PATH = activity_root
        bddl.activity.ACTIVITY_CONFIGS_PATH = activity_root
        behavior_task.ACTIVITY_CONFIGS_PATH = activity_root

        activities = sorted(
            p.name for p in activity_root_path.iterdir() if p.is_dir() and (p / "problem0.bddl").is_file()
        )
        bddl_utils.BEHAVIOR_ACTIVITIES.clear()
        bddl_utils.BEHAVIOR_ACTIVITIES.extend(activities)
        behavior_task.BEHAVIOR_ACTIVITIES.clear()
        behavior_task.BEHAVIOR_ACTIVITIES.extend(activities)

    def _object_by_name(self, env, name: str | None):
        if not name:
            return None
        try:
            return env.scene.object_registry("name", name)
        except Exception:
            return None

    def _build_camera_pose(self, eye, lookat):
        import omnigibson.utils.transform_utils as T

        eye = np.asarray(eye, dtype=np.float32)
        lookat = np.asarray(lookat, dtype=np.float32)
        forward = lookat - eye
        forward = forward / max(np.linalg.norm(forward), 1e-6)
        up_guess = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(np.dot(forward, up_guess)) > 0.98:
            up_guess = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up_guess)
        right = right / max(np.linalg.norm(right), 1e-6)
        up = np.cross(forward, right)
        up = up / max(np.linalg.norm(up), 1e-6)
        rotation = np.stack([right, up, forward], axis=1)
        quat = T.mat2quat(torch.tensor(rotation, dtype=torch.float32))
        return torch.tensor(eye, dtype=torch.float32), quat.to(torch.float32)

    def _apply_policy_camera(self, env, spec: SentinelSceneSpec):
        from omnigibson.task_generation.pipeline_common import _support_relative_video_views

        robot = env.robots[0]
        target_obj = self._object_by_name(env, spec.target_object_name)
        support_obj = self._object_by_name(env, spec.support_object_name)
        if target_obj is None:
            return

        views = _support_relative_video_views(
            robot=robot,
            target_obj=target_obj,
            support_obj=support_obj,
            active_objects_by_inst=None,
        )
        canonical_view = next((view for view in views if view.get("canonical")), views[0])
        sensor = env.external_sensors["external_sensor0"]
        position, orientation = self._build_camera_pose(
            eye=canonical_view["eye"],
            lookat=canonical_view["lookat"],
        )
        sensor.set_position_orientation(position=position, orientation=orientation, frame="parent")

    def _extract_obs(self, env, raw_obs, spec: SentinelSceneSpec):
        external = raw_obs.get("external", {})
        main_rgb = external["external_sensor0"]["rgb"].to(torch.uint8)[..., :3]

        robot = env.robots[0]
        robot_obs = raw_obs[robot.name]
        wrist_rgb = None
        for sensor_name, sensor_obs in robot_obs.items():
            if not isinstance(sensor_obs, dict) or "rgb" not in sensor_obs:
                continue
            lowered = sensor_name.lower()
            if any(token in lowered for token in ("wrist", "realsense", "d405", "hand")):
                wrist_rgb = sensor_obs["rgb"].to(torch.uint8)[..., :3]
                break
        if wrist_rgb is None:
            for sensor_name, sensor_obs in robot_obs.items():
                if isinstance(sensor_obs, dict) and "rgb" in sensor_obs:
                    wrist_rgb = sensor_obs["rgb"].to(torch.uint8)[..., :3]
                    break
        if wrist_rgb is None:
            wrist_rgb = torch.zeros_like(main_rgb)

        arm_positions = robot.get_joint_positions()[robot.arm_control_idx[robot.default_arm]]
        gripper_positions = robot.get_joint_positions()[robot.gripper_control_idx[robot.default_arm]]
        gripper_scalar = torch.mean(gripper_positions).reshape(1)
        state = torch.cat([arm_positions.to(torch.float32), gripper_scalar.to(torch.float32)], dim=0)

        return {
            "main_images": main_rgb,
            "wrist_images": wrist_rgb,
            "state": state,
            "task_description": spec.prompt,
        }

    def _wrap_obs(self, obs_items):
        return {
            "main_images": torch.stack([item["main_images"] for item in obs_items], dim=0),
            "wrist_images": torch.stack([item["wrist_images"] for item in obs_items], dim=0),
            "states": torch.stack([item["state"] for item in obs_items], dim=0),
            "task_descriptions": [item["task_description"] for item in obs_items],
        }

    def _record_metrics(self, rewards, terminations, infos):
        info_list = []
        for env_idx, (reward, terminated, info) in enumerate(zip(rewards, terminations, infos)):
            self.returns[env_idx] += float(reward)
            self.success_once[env_idx] = self.success_once[env_idx] | bool(terminated)
            info_list.append(
                {
                    "success_once": self.success_once[env_idx].clone(),
                    "return": self.returns[env_idx].clone(),
                    "episode_len": self.elapsed_steps[env_idx].clone(),
                    "reward": self.returns[env_idx].clone()
                    / max(int(self.elapsed_steps[env_idx].item()), 1),
                }
            )
        return {"episode": to_tensor(list_of_dict_to_dict_of_list(info_list))}

    def _append_jsonl(self, path: Path, payload: dict):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _save_episode_artifacts(self, env_idx: int, terminated: bool, truncated: bool):
        if self.video_cfg.save_video and self._frames[env_idx]:
            scene_name = self._scene_specs[env_idx].scene_name
            video_path = self._results_dir / f"{scene_name}_episode_{self._episode_indices[env_idx]}.mp4"
            imageio.mimsave(video_path, self._frames[env_idx], fps=10)
        spec = self._scene_specs[env_idx]
        self._append_jsonl(
            self._results_path,
            {
                "scene_name": spec.scene_name,
                "activity_name": spec.activity_name,
                "target_synset": spec.target_synset,
                "success_once": bool(self.success_once[env_idx]),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "episode_len": int(self.elapsed_steps[env_idx].item()),
                "return": float(self.returns[env_idx].item()),
            },
        )
        self._append_jsonl(
            self._prompt_log_path,
            {
                "scene_name": spec.scene_name,
                "prompt": spec.prompt,
            },
        )
        self._frames[env_idx] = []
        self._episode_indices[env_idx] += 1

    def _reset_metrics(self, env_idx: int | None = None):
        if env_idx is None:
            self._elapsed_steps.zero_()
            self.success_once.zero_()
            self.returns.zero_()
            return
        self._elapsed_steps[env_idx] = 0
        self.success_once[env_idx] = False
        self.returns[env_idx] = 0.0

    def reset(self):
        import omnigibson as og

        obs_items = []
        for env_idx, (env, spec) in enumerate(zip(self.envs, self._scene_specs)):
            raw_obs, _ = env.reset()
            self._apply_policy_camera(env, spec)
            og.sim.step()
            for _ in range(2):
                og.sim.render()
            raw_obs, _ = env.get_obs()
            obs_items.append(self._extract_obs(env, raw_obs, spec))
        self._reset_metrics()
        return self._wrap_obs(obs_items), {}

    def step(self, actions=None):
        obs_items = []
        rewards = []
        terminations = []
        truncations = []
        info_payloads = []
        for env_idx, (env, spec, action) in enumerate(zip(self.envs, self._scene_specs, actions)):
            raw_obs, reward, terminated, truncated, info = env.step(action)
            if self.video_cfg.save_video:
                frame = raw_obs["external"]["external_sensor0"]["rgb"][..., :3].cpu().numpy().astype(np.uint8)
                self._frames[env_idx].append(frame)
            obs_items.append(self._extract_obs(env, raw_obs, spec))
            rewards.append(float(reward))
            terminations.append(bool(terminated))
            truncations.append(bool(truncated or self._elapsed_steps[env_idx] + 1 >= self.cfg.max_episode_steps))
            info_payloads.append(info)

        self._elapsed_steps += 1
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        terminations_t = torch.tensor(terminations, dtype=torch.bool)
        truncations_t = torch.tensor(truncations, dtype=torch.bool)
        infos = self._record_metrics(rewards_t, terminations_t, info_payloads)

        for env_idx, (terminated, truncated) in enumerate(zip(terminations_t, truncations_t)):
            if bool(terminated or truncated):
                self._save_episode_artifacts(env_idx, bool(terminated), bool(truncated))

        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = terminations_t.clone()
            terminations_t[:] = False

        return self._wrap_obs(obs_items), rewards_t, terminations_t, truncations_t, infos

    def chunk_step(self, chunk_actions):
        chunk_rewards = []
        chunk_terminations = []
        chunk_truncations = []
        infos = {}
        extracted_obs = None
        for step_idx in range(chunk_actions.shape[1]):
            extracted_obs, rewards, terminations, truncations, infos = self.step(
                chunk_actions[:, step_idx]
            )
            chunk_rewards.append(rewards)
            chunk_terminations.append(terminations)
            chunk_truncations.append(truncations)
        return (
            extracted_obs,
            torch.stack(chunk_rewards, dim=1),
            torch.stack(chunk_terminations, dim=1),
            torch.stack(chunk_truncations, dim=1),
            infos,
        )

    def flush_video(self):
        return None

    def update_reset_state_ids(self):
        return None
