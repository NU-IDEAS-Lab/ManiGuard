"""Passive source-episode observer for simulator executor callbacks."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

from maniguard.visual.cameras import as_numpy
from maniguard.visual.profile import load_profile

from .provenance import runtime_provenance
from .schema import json_text, pack_record
from .snapshot import array_tree, capture_checkpoint
from .writer import EpisodeWriter


def _measured(call):
    try:
        return {"valid": True, "value": array_tree(call())}
    except Exception as exc:  # noqa: BLE001 -- preserve an unavailable sensor as explicit invalid data
        return {"valid": False, "error": f"{type(exc).__name__}: {exc}"}


def _action_bounds(values):
    """Represent unbounded Box dimensions without nonstandard JSON infinities."""
    values = as_numpy(values)
    if np.isnan(values).any():
        raise ValueError("NaN action-space boundary")
    return [float(value) if np.isfinite(value) else str(float(value)) for value in values]


class SourceEpisodeObserver:
    def __init__(self, cfg, scene):
        self.cfg, self.scene = cfg, scene
        self.profile = load_profile(cfg.source_profile)
        self.episode_id = uuid.uuid4().hex
        self.directory = Path(cfg.recording_output_dir) / self.episode_id
        self.writer = None
        self.pending_action = self.pending_observation = None
        self.proposal_id = -1
        self.invalid = False
        self.pending_boundary = -1
        self.constraint_monitors = {}

    def start(self, env, robot, observation, monitor, episode_seed, scene):
        self.env, self.robot, self.monitor = env, robot, monitor
        self.ap_names = sorted((scene.get("ltl_safety") or {}).get("propositions", {}))
        if monitor is not None:
            from maniguard.utils.ltl_utils import LTLMonitor

            constraints = (scene.get("ltl_safety") or {}).get("constraints", [])
            self.constraint_monitors = {c["id"]: LTLMonitor(c["ltl"]) for c in constraints}
            if len(self.constraint_monitors) != len(constraints):
                raise ValueError("Duplicate safety constraint identifiers")
        provenance = {}
        if self.cfg.recording_provenance:
            provenance = json.loads(Path(self.cfg.recording_provenance).read_text(encoding="utf-8"))
        metadata = {
            "episode_id": self.episode_id,
            "source_profile": self.profile,
            "camera_keys": self.profile["camera_keys"],
            "resolution": self.profile["resolution"],
            "rgb_encoding": self.profile["rgb_encoding"],
            "action_hz": self.cfg.action_frequency,
            "physics_hz": self.cfg.physics_frequency,
            "scene": scene,
            "episode_seed": episode_seed,
            "provenance": provenance,
            "runtime": runtime_provenance(),
            "configuration": vars(self.cfg),
            "ap_names": self.ap_names,
            "constraint_ap_names": {key: value.ap_list for key, value in self.constraint_monitors.items()},
            "render_settings": env._maniguard_source_capture.render_report,
            "command_semantics": "raw policy output; transformed executor output; applied clipped controller command",
            "snapshot_scope": "scene, controllers, assisted grasp, particles, monitor and simulator-process RNG; recorded-command replay",
        }
        if robot is not None:
            metadata["joint_names"] = list(robot.joints)
            metadata["controller_config"] = array_tree(robot._controller_config)
            metadata["controller_action_idx"] = array_tree(robot.controller_action_idx)
            metadata["joint_groups"] = array_tree({
                "arm": robot.arm_control_idx[robot.default_arm],
                "gripper": robot.gripper_control_idx[robot.default_arm],
            })
            metadata["object_names"] = [obj.name for obj in sorted(env.scene.objects, key=lambda obj: obj.name)]
        metadata["proposition_bindings"] = {}
        if monitor is not None and hasattr(monitor, "_resolver"):
            for name, definition in (scene.get("ltl_safety") or {}).get("propositions", {}).items():
                metadata["proposition_bindings"][name] = {}
                for field in ("over", "relative_to", "surface"):
                    patterns = definition.get(field)
                    if patterns:
                        objects = monitor._resolver.resolve_patterns([patterns] if isinstance(patterns, str) else patterns)
                        metadata["proposition_bindings"][name][field] = [
                            {"name": obj.name, "category": obj.category} for obj in objects.values()
                        ]
        self.writer = EpisodeWriter(self.directory, metadata, self.cfg.recording_max_bytes)
        scene_file = scene.get("scene_file")
        source_files = [("task_scene.json", scene_file)]
        if scene_file:
            diagnostics = Path(scene_file).with_name("diagnostics.jsonl")
            if diagnostics.is_file():
                source_files.append(("diagnostics.jsonl", diagnostics))
        for name, source in source_files:
            if source:
                source = Path(source)
                self.writer._check(source.stat().st_size)
                shutil.copyfile(source, self.directory / name)
                self.writer.append_event(
                    {"type": "source_file", "path": name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
                )
        self._append_observation(0, observation, monitor)

    def _physical(self, observation):
        from omnigibson.object_states import ContactBodies

        robot = self.robot
        record = {
            "robot": {
                "q": as_numpy(robot.get_joint_positions()),
                "dq": as_numpy(robot.get_joint_velocities()),
                "eef_pose_world": np.r_[as_numpy(robot.get_eef_position()), as_numpy(robot.get_eef_orientation())],
                "eef_pose_robot": np.r_[
                    as_numpy(robot.get_relative_eef_position()), as_numpy(robot.get_relative_eef_orientation())
                ],
                "policy_state": np.asarray(observation["states"]),
            }
        }
        objects = sorted(self.env.scene.objects, key=lambda obj: obj.name)
        if objects:
            record["objects"] = {
                "pose_world": np.stack(
                    [np.concatenate([as_numpy(x) for x in obj.get_position_orientation()]) for obj in objects]
                )
            }
        record["object_details"] = {
            obj.name: {
                "linear_velocity": _measured(obj.get_linear_velocity),
                "angular_velocity": _measured(obj.get_angular_velocity),
                "joints": {name: _measured(joint.get_state) for name, joint in obj.joints.items()},
                "contacts": _measured(
                    lambda obj=obj: sorted(link.prim_path for link in obj.states[ContactBodies].get_value())
                ),
            }
            for obj in objects
        }
        return record

    def _constraint_gt(self, labels):
        """Advance clause monitors using the exact same AP sample as the task monitor."""
        return {key: machine.step(labels) for key, machine in self.constraint_monitors.items()}

    def _append_observation(self, step, observation, monitor):
        source = self.env._maniguard_source_capture.last_observation
        if source["observation_id"] != step:
            raise ValueError("Source frame does not match the executor observation index")
        gt = (
            monitor.recording_state(step)
            if monitor is not None
            else {"valid": False, "observation": None, "error": None}
        )
        labels = (gt["observation"] or {}).get("ap", {})
        valid = gt["valid"] and set(labels) == set(self.ap_names)
        self.invalid |= monitor is not None and not valid
        record = dict(source)
        record.update(self._physical(observation))
        record["safety"] = {
            "monitor_valid": np.bool_(valid),
            "monitor_rejected": np.bool_((gt["observation"] or {}).get("doomed", False)),
        }
        if self.ap_names:
            record["safety"].update(
                ap_values=np.array([labels.get(key, False) for key in self.ap_names], bool),
                ap_valid=np.full(len(self.ap_names), valid, bool),
            )
        record["monitor_record"] = gt
        if valid and self.constraint_monitors:
            record["constraint_records"] = self._constraint_gt(labels)
            if any(row["doomed"] for row in record["constraint_records"].values()) != bool(
                record["safety"]["monitor_rejected"]
            ):
                self.invalid = True
                raise ValueError("Clause monitor verdicts disagree with the recorded task monitor")
        self.writer.append_observation(record)
        checkpoint = capture_checkpoint(self.env, monitor)
        state = checkpoint.pop("scene")
        self.writer.append_snapshot(step, state, checkpoint)
        self.last_scene_digest = hashlib.sha256(pack_record(state)).digest()

    def state_boundary(self, observation, reason):
        checkpoint = capture_checkpoint(self.env, self.monitor)
        digest = hashlib.sha256(pack_record(checkpoint["scene"])).digest()
        if digest == self.last_scene_digest:
            return
        capture = self.env._maniguard_source_capture
        record = dict(capture.capture(capture.observation_id))
        record.update(self._physical(observation))
        # The last monitor sample belongs to the preceding physical state.
        # Do not reuse it as GT for an instantaneous scripted mutation.
        record["safety"] = {"monitor_valid": False}
        self.pending_boundary = self.writer.append_boundary(record, checkpoint, reason)
        self.last_scene_digest = digest

    def proposal(self, step, chunk, planned_steps, action_space):
        if self.pending_action is not None:
            raise ValueError("New proposal arrived before the previous action outcome")
        self.proposal_id += 1
        if not 0 < planned_steps <= len(chunk):
            raise ValueError("Invalid planned proposal prefix")
        self.writer.append_event(
            {
                "type": "proposal",
                "proposal_id": self.proposal_id,
                "source_observation_id": step,
                "actions": np.asarray(chunk),
                "controller_action_shape": list(action_space.shape) if action_space is not None else None,
                "action_low": _action_bounds(action_space.low) if action_space is not None else None,
                "action_high": _action_bounds(action_space.high) if action_space is not None else None,
                "planned_steps": planned_steps,
                "monotonic_s": time.monotonic(),
            }
        )

    def before_action(self, step, offset, raw, transformed, command, *, issued=True):
        if self.pending_action is not None:
            raise ValueError("An action attempt is already pending")
        self.pending_action = {
            "source_observation_id": step,
            "target_observation_id": step + 1,
            "proposal_id": self.proposal_id,
            "proposal_offset": offset,
            "source_boundary_id": self.pending_boundary,
            "raw": np.array(raw, copy=True),
            "transformed": np.array(transformed, copy=True),
            "issued": bool(issued),
            "applied": np.array(command, copy=True),
            "duration_s": 1 / self.cfg.action_frequency,
        }
        self.writer.append_event({"type": "action_attempt", **self.pending_action, "monotonic_s": time.monotonic()})

    def applied(self, step):
        if self.pending_action is None or self.pending_action["source_observation_id"] != step:
            raise ValueError("Applied action has no matching attempt")
        self.writer.append_transition(self.pending_action)
        self.pending_action = None
        self.pending_boundary = -1

    def transition(self, step, observation):
        if self.pending_observation is not None:
            raise ValueError("Previous observation lacks its GT callback")
        self.pending_observation = (step, observation)

    def oracle(self, step, monitor):
        if self.pending_observation is None or self.pending_observation[0] != step:
            raise ValueError("GT callback does not match the post-action observation")
        self._append_observation(step, self.pending_observation[1], monitor)
        self.pending_observation = None

    def finish(self, result):
        if self.writer is None:
            self.directory.mkdir(parents=True, exist_ok=False)
            (self.directory / "initialization_failed.json").write_text(json_text(result) + "\n", encoding="utf-8")
            return
        if self.writer.closed:
            return
        complete = (
            result.get("status") == "completed"
            and not self.invalid
            and self.pending_action is None
            and self.pending_observation is None
            and self.writer.observations == self.writer.transitions + 1
            and not self.writer.failed
        )
        self.writer.finish("complete" if complete else "failed", result)


def create_observer(cfg, scene):
    return SourceEpisodeObserver(cfg, scene)
