"""Object-frame waypoint extraction and HDF5 serialization."""

from __future__ import annotations

import json
from typing import Any

import h5py
import numpy as np
import torch as th

import omnigibson.utils.transform_utils as T
from omnigibson.object_states import Open

from _common import apply_eef_z_offset, as_tensor, robot_action_offset


def find_subsequence(signals: list[dict[str, Any]], expected_sequence: list[str]) -> list[dict[str, Any]] | None:
    if not expected_sequence:
        return signals
    types = [signal["type"] for signal in signals]
    n = len(expected_sequence)
    for start in range(0, len(types) - n + 1):
        if types[start : start + n] == expected_sequence:
            return signals[start : start + n]
    return None


class WaypointExtractor:
    """Extract object-frame EEF trajectories around annotation signals."""

    def __init__(
        self,
        env,
        signals: list[dict[str, Any]],
        *,
        pick_min_start_distance: float,
        pick_max_distance: float,
        place_min_start_distance: float,
        place_max_distance: float,
        open_min_start_distance: float,
        open_max_distance: float,
        close_min_start_distance: float,
        close_max_distance: float,
        end_distance_threshold: float,
        max_frames: int | None = None,
        robot_id: int = 0,
        eef_z_offset: float = 0.0,
    ):
        self.env = env
        self.robot = env.robots[robot_id]
        self.action_offset = robot_action_offset(env, robot_id)
        self.signals = signals
        self.distance_params = {
            "pick": (pick_min_start_distance, pick_max_distance),
            "place": (place_min_start_distance, place_max_distance),
            "open": (open_min_start_distance, open_max_distance),
            "close": (close_min_start_distance, close_max_distance),
        }
        self.end_distance_threshold = end_distance_threshold
        self.max_frames = max_frames
        self.eef_z_offset = eef_z_offset
        self.waypoints: list[dict[str, Any]] = []
        self.frame_buffer: list[dict[str, Any]] = []
        self.signal_idx = 0
        self.current_frame_idx = 0

    def reset(self) -> None:
        self.waypoints = []
        self.frame_buffer = []
        self.signal_idx = 0
        self.current_frame_idx = 0

    def episode_start_callback(self, episode_id: int, env: Any) -> None:
        self.reset()

    def step(self, frame_idx: int, action: th.Tensor, env: Any) -> None:
        self.current_frame_idx = frame_idx
        eef_pos_robot, eef_quat_robot = self.robot.get_relative_eef_pose(arm="default")
        eef_pos_robot = as_tensor(eef_pos_robot)
        eef_quat_robot = as_tensor(eef_quat_robot)
        eef_pos_robot, eef_quat_robot = apply_eef_z_offset(eef_pos_robot, eef_quat_robot, self.eef_z_offset)
        robot_pos_world, robot_quat_world = self.robot.get_position_orientation()

        gripper_idx = self.robot.gripper_action_idx[self.robot.default_arm]
        gripper_action = float(action[self.action_offset + gripper_idx].item())

        object_poses: dict[str, dict[str, th.Tensor]] = {}
        link_poses: dict[tuple[str, str], dict[str, th.Tensor]] = {}
        for obj in env.scene.objects:
            if obj == self.robot:
                continue
            pos, quat = obj.get_position_orientation()
            object_poses[obj.name] = {"pos": as_tensor(pos), "quat": as_tensor(quat)}
            if Open in obj.states:
                for link_name, link in obj.links.items():
                    link_pos, link_quat = link.get_position_orientation()
                    link_poses[(obj.name, link_name)] = {"pos": as_tensor(link_pos), "quat": as_tensor(link_quat)}

        self.frame_buffer.append(
            {
                "frame_idx": frame_idx,
                "eef_pos_robot": eef_pos_robot,
                "eef_quat_robot": eef_quat_robot,
                "robot_pos_world": as_tensor(robot_pos_world),
                "robot_quat_world": as_tensor(robot_quat_world),
                "gripper_action": gripper_action,
                "object_poses": object_poses,
                "link_poses": link_poses,
            }
        )
        self._check_and_extract()

    def _signal_frame_and_reference(self, signal: dict[str, Any]) -> tuple[int, str, str | None]:
        signal_type = signal["type"]
        if signal_type == "pick":
            return int(signal["frame_idx"]), signal["object_name"], None
        if signal_type == "place":
            return int(signal["contact_frame_idx"]), signal["target_object_name"], None
        if signal_type in {"open", "close"}:
            return int(signal["release_frame_idx"]), signal["object_name"], signal.get("contact_link")
        raise ValueError(f"Unsupported signal type {signal_type!r}")

    def _check_and_extract(self) -> None:
        while self.signal_idx < len(self.signals):
            signal = self.signals[self.signal_idx]
            signal_frame, ref_obj_name, ref_link_name = self._signal_frame_and_reference(signal)
            if self.current_frame_idx < signal_frame:
                break
            signal_frame_data = next((frame for frame in self.frame_buffer if frame["frame_idx"] == signal_frame), None)
            if signal_frame_data is None:
                break
            _, max_distance = self.distance_params[signal["type"]]
            at_episode_end = self.max_frames is not None and self.current_frame_idx >= self.max_frames - 1
            if not self._end_condition_met(
                ref_obj_name,
                ref_link_name,
                signal["type"],
                max_distance,
                signal_frame_data,
            ):
                if not at_episode_end:
                    break
            self._extract_signal(signal, signal_frame, available_end_frame=self.current_frame_idx)
            self.signal_idx += 1

    def _eef_world_from_frame(self, frame: dict[str, Any]) -> th.Tensor:
        robot_t_world = T.pose2mat((frame["robot_pos_world"], frame["robot_quat_world"]))
        eef_t_robot = T.pose2mat((frame["eef_pos_robot"], frame["eef_quat_robot"]))
        eef_pos_world, _ = T.mat2pose(robot_t_world @ eef_t_robot)
        return eef_pos_world

    def _distance_from_signal(
        self,
        frame: dict[str, Any],
        signal_frame: dict[str, Any],
        signal_type: str,
        ref_obj_name: str,
        ref_link_name: str | None,
    ) -> float:
        eef_world = self._eef_world_from_frame(frame)
        signal_eef_world = self._eef_world_from_frame(signal_frame)
        if signal_type in {"pick", "place"}:
            return float(th.linalg.norm(eef_world - signal_eef_world).item())
        if ref_link_name is None:
            return 0.0
        link_key = (ref_obj_name, ref_link_name)
        link_signal = signal_frame["link_poses"].get(link_key, {}).get("pos")
        link_current = frame["link_poses"].get(link_key, {}).get("pos")
        if link_signal is None or link_current is None:
            return 0.0
        signal_z = abs(float(signal_eef_world[2] - link_signal[2]))
        current_z = abs(float(eef_world[2] - link_current[2]))
        return current_z - signal_z

    def _approach_distance_from_signal(
        self,
        frame: dict[str, Any],
        signal_frame: dict[str, Any],
        signal_type: str,
        ref_obj_name: str,
        ref_link_name: str | None,
    ) -> float:
        eef_world = self._eef_world_from_frame(frame)
        signal_eef_world = self._eef_world_from_frame(signal_frame)
        if signal_type in {"pick", "place"} or ref_link_name is None:
            return float(th.linalg.norm(eef_world - signal_eef_world).item())
        link_key = (ref_obj_name, ref_link_name)
        link_signal = signal_frame["link_poses"].get(link_key, {}).get("pos")
        link_current = frame["link_poses"].get(link_key, {}).get("pos")
        if link_signal is None or link_current is None:
            return 0.0
        current_dist = th.linalg.norm(eef_world - link_current).item()
        signal_dist = th.linalg.norm(signal_eef_world - link_signal).item()
        return float(current_dist - signal_dist)

    def _end_condition_met(
        self,
        ref_obj_name: str,
        ref_link_name: str | None,
        signal_type: str,
        max_distance: float,
        signal_frame: dict[str, Any],
    ) -> bool:
        latest = self.frame_buffer[-1]
        obj_pose = latest["object_poses"].get(ref_obj_name)
        if obj_pose is None:
            return False
        eef_world = self._eef_world_from_frame(latest)
        distance_to_ref_obj = float(th.linalg.norm(eef_world - obj_pose["pos"]).item())
        distance_from_signal = self._distance_from_signal(
            latest,
            signal_frame,
            signal_type,
            ref_obj_name,
            ref_link_name,
        )
        return distance_to_ref_obj > max_distance or distance_from_signal >= self.end_distance_threshold

    def _transform_eef_to_object_frame(
        self,
        frame: dict[str, Any],
        obj_pos: th.Tensor,
        obj_quat: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        robot_t_world = T.pose2mat((frame["robot_pos_world"], frame["robot_quat_world"]))
        eef_t_robot = T.pose2mat((frame["eef_pos_robot"], frame["eef_quat_robot"]))
        obj_t_world = T.pose2mat((obj_pos, obj_quat))
        eef_t_obj = T.pose_inv(obj_t_world) @ robot_t_world @ eef_t_robot
        return T.mat2pose(eef_t_obj)

    def _extract_signal(
        self,
        signal: dict[str, Any],
        signal_frame: int,
        available_end_frame: int | None = None,
    ) -> None:
        signal_type = signal["type"]
        min_start_distance, max_distance = self.distance_params[signal_type]
        _, ref_obj_name, ref_link_name = self._signal_frame_and_reference(signal)
        signal_frame_data = next((frame for frame in self.frame_buffer if frame["frame_idx"] == signal_frame), None)
        if signal_frame_data is None:
            print(f"[Waypoints] WARNING: signal frame {signal_frame} is unavailable; skipping")
            return
        if ref_obj_name not in signal_frame_data["object_poses"]:
            print(f"[Waypoints] WARNING: reference object {ref_obj_name!r} missing; skipping")
            return

        obj_pose_at_signal = signal_frame_data["object_poses"][ref_obj_name]
        candidate_frames = [frame for frame in self.frame_buffer if frame["frame_idx"] <= signal_frame]
        earliest_frame = min((frame["frame_idx"] for frame in candidate_frames), default=signal_frame)
        start_frame = None
        for frame in sorted(candidate_frames, key=lambda item: item["frame_idx"], reverse=True):
            distance = self._approach_distance_from_signal(
                frame,
                signal_frame_data,
                signal_type,
                ref_obj_name,
                ref_link_name,
            )
            if distance >= min_start_distance:
                start_frame = frame["frame_idx"]
                break
        if start_frame is None:
            start_frame = earliest_frame

        end_frame = available_end_frame if available_end_frame is not None else self.frame_buffer[-1]["frame_idx"]
        if self.max_frames is not None:
            end_frame = min(end_frame, self.max_frames - 1)

        waypoints = []
        for frame in sorted(self.frame_buffer, key=lambda item: item["frame_idx"]):
            if not (start_frame <= frame["frame_idx"] <= end_frame):
                continue
            eef_pos_obj, eef_quat_obj = self._transform_eef_to_object_frame(
                frame,
                obj_pose_at_signal["pos"],
                obj_pose_at_signal["quat"],
            )
            waypoints.append(
                {
                    "frame_idx": int(frame["frame_idx"]),
                    "eef_pos_obj": eef_pos_obj.cpu().numpy().tolist(),
                    "eef_quat_obj": eef_quat_obj.cpu().numpy().tolist(),
                    "gripper_action": float(frame["gripper_action"]),
                }
            )
            if frame["frame_idx"] > signal_frame:
                signal_distance = self._distance_from_signal(
                    frame,
                    signal_frame_data,
                    signal_type,
                    ref_obj_name,
                    ref_link_name,
                )
                approach_distance = self._approach_distance_from_signal(
                    frame,
                    signal_frame_data,
                    signal_type,
                    ref_obj_name,
                    ref_link_name,
                )
                if signal_distance >= self.end_distance_threshold or approach_distance > max_distance:
                    break

        subtask = {
            "type": signal_type,
            "signal_frame_idx": signal_frame,
            "reference_object": {
                "name": ref_obj_name,
                "category": signal.get("object_category" if signal_type == "pick" else "target_object_category", ""),
            },
            "extraction_params": {
                "min_start_distance": min_start_distance,
                "max_distance": max_distance,
                "end_distance_threshold": self.end_distance_threshold,
            },
            "actual_frames_before": int(signal_frame - start_frame),
            "actual_frames_after": int(
                max(0, (waypoints[-1]["frame_idx"] if waypoints else signal_frame) - signal_frame)
            ),
            "waypoints": waypoints,
        }
        if signal_type == "place":
            subtask["placed_object"] = {
                "name": signal.get("released_object_name", ""),
                "category": signal.get("released_object_category", ""),
            }
        self.waypoints.append(subtask)
        print(f"[Waypoints] {signal_type} frame={signal_frame} waypoints={len(waypoints)} ref={ref_obj_name}")

    def finalize(self) -> None:
        while self.signal_idx < len(self.signals):
            signal = self.signals[self.signal_idx]
            signal_frame, _, _ = self._signal_frame_and_reference(signal)
            last_frame = self.frame_buffer[-1]["frame_idx"] if self.frame_buffer else signal_frame
            self._extract_signal(signal, signal_frame, available_end_frame=last_frame)
            self.signal_idx += 1

    def get_waypoints(self) -> list[dict[str, Any]]:
        return self.waypoints


def save_waypoints_hdf5(all_waypoints: dict[str, list[dict[str, Any]]], path: str) -> None:
    with h5py.File(path, "w") as f:
        data_grp = f.create_group("data")
        data_grp.attrs["n_episodes"] = len(all_waypoints)
        for episode_key, subtasks in all_waypoints.items():
            ep_grp = data_grp.create_group(episode_key)
            ep_grp.attrs["n_subtasks"] = len(subtasks)
            for idx, subtask in enumerate(subtasks):
                subtask_grp = ep_grp.create_group(f"subtask_{idx}")
                for attr in ("type", "signal_frame_idx", "actual_frames_before", "actual_frames_after"):
                    subtask_grp.attrs[attr] = subtask[attr]
                subtask_grp.attrs["reference_object"] = json.dumps(subtask["reference_object"])
                subtask_grp.attrs["extraction_params"] = json.dumps(subtask["extraction_params"])
                if "placed_object" in subtask:
                    subtask_grp.attrs["placed_object"] = json.dumps(subtask["placed_object"])
                waypoints = subtask["waypoints"]
                subtask_grp.create_dataset(
                    "frame_indices",
                    data=np.asarray([w["frame_idx"] for w in waypoints], dtype=np.int32),
                )
                subtask_grp.create_dataset(
                    "eef_pos_obj",
                    data=np.asarray([w["eef_pos_obj"] for w in waypoints], dtype=np.float32),
                )
                subtask_grp.create_dataset(
                    "eef_quat_obj",
                    data=np.asarray([w["eef_quat_obj"] for w in waypoints], dtype=np.float32),
                )
                subtask_grp.create_dataset(
                    "gripper_action",
                    data=np.asarray([w["gripper_action"] for w in waypoints], dtype=np.float32),
                )


def load_waypoints_hdf5(path: str) -> dict[str, list[dict[str, Any]]]:
    all_waypoints: dict[str, list[dict[str, Any]]] = {}
    with h5py.File(path, "r") as f:
        for episode_key, ep_grp in f["data"].items():
            subtasks = []
            for idx in range(int(ep_grp.attrs["n_subtasks"])):
                subtask_grp = ep_grp[f"subtask_{idx}"]
                frame_indices = subtask_grp["frame_indices"][:]
                eef_pos = subtask_grp["eef_pos_obj"][:]
                eef_quat = subtask_grp["eef_quat_obj"][:]
                gripper = subtask_grp["gripper_action"][:]
                subtask = {
                    "type": subtask_grp.attrs["type"],
                    "signal_frame_idx": int(subtask_grp.attrs["signal_frame_idx"]),
                    "reference_object": json.loads(subtask_grp.attrs["reference_object"]),
                    "extraction_params": json.loads(subtask_grp.attrs["extraction_params"]),
                    "waypoints": [
                        {
                            "frame_idx": int(frame_indices[i]),
                            "eef_pos_obj": eef_pos[i].tolist(),
                            "eef_quat_obj": eef_quat[i].tolist(),
                            "gripper_action": float(gripper[i]),
                        }
                        for i in range(len(frame_indices))
                    ],
                }
                if "placed_object" in subtask_grp.attrs:
                    subtask["placed_object"] = json.loads(subtask_grp.attrs["placed_object"])
                subtasks.append(subtask)
            all_waypoints[episode_key] = subtasks
    return all_waypoints
