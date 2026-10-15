"""Scripted demonstration adapter for the generic source episode observer."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from maniguard.data.datagen.primitives.execute import build_joint_action
from maniguard.data.datagen.primitives.record import _arm_gripper_state
from maniguard.visual.cameras import as_numpy

from .observer import SourceEpisodeObserver


class SourceRecorder:
    """Retain complete scripted attempts, including observed failed attempts."""

    def __init__(self, profile, scene, monitor, *, max_output_bytes):
        self.profile, self.scene, self.monitor = profile, scene, monitor
        self.max_output_bytes = max_output_bytes
        self.observer = None
        self.n_steps = 0
        self.recording_rest = False
        self.last_command = None
        self.attempt_metadata = {}

    def _observation(self):
        arm, grip = _arm_gripper_state(self.robot)
        return {"states": np.r_[arm, np.float32(grip)].astype(np.float32)}

    def attach(self, env, robot, out_dir, prompt=""):
        self.env, self.robot, self.out_dir = env, robot, Path(out_dir)
        self.n_steps, self.step_idx, self.plan_remaining, self.plan_offset = 0, 0, 0, 0
        self.recording_rest = False
        self.last_command = None
        cfg = SimpleNamespace(
            source_profile=self.profile,
            recording_output_dir=str(self.out_dir.parent),
            recording_max_bytes=self.max_output_bytes,
            recording_provenance=None,
            action_frequency=30,
            physics_frequency=120,
            random_policy=False,
            action_space="joint",
            state_mode="joint",
            policy="scripted joint trajectory",
        )
        self.observer = SourceEpisodeObserver(cfg, self.scene)
        self.observer.directory = self.out_dir
        env._maniguard_source_capture.capture(0)
        self.observer.start(
            env,
            robot,
            self._observation(),
            self.monitor,
            self.attempt_metadata.get("seed"),
            {**self.scene, "prompt": prompt, "attempt": self.attempt_metadata},
        )

    def record_plan(self, arm_traj, gripper_cmd, repeats):
        actions = np.stack([as_numpy(build_joint_action(self.robot, q, gripper_cmd)) for q in arm_traj])
        actions = np.repeat(actions, int(repeats), axis=0)
        self.observer.proposal(self.step_idx, actions, len(actions), None)
        self.plan_remaining, self.plan_offset = len(actions), 0

    def execute_step(self, action, on_step):
        command = as_numpy(action).copy()
        self.last_command = command.copy()
        if not self.plan_remaining:
            self.observer.proposal(self.step_idx, command[None], 1, None)
            self.plan_remaining, self.plan_offset = 1, 0
        self.observer.before_action(self.step_idx, self.plan_offset, command, command, command)
        self.env._maniguard_source_capture.step(action)
        self.observer.applied(self.step_idx)
        self.step_idx += 1
        if not self.recording_rest:
            self.n_steps += 1
        self.plan_offset += 1
        self.plan_remaining -= 1
        self.observer.transition(self.step_idx, self._observation())
        try:
            return bool(on_step is not None and on_step())
        finally:
            self.observer.oracle(self.step_idx, self.monitor)

    def passive_step(self, on_step):
        """Record a scripted settling interval with no newly issued command."""
        if self.last_command is None:
            raise ValueError("Cannot describe an active command before one was issued")
        self.state_boundary("scripted update before passive interval")
        self.observer.before_action(
            self.step_idx, -1, self.last_command, self.last_command, self.last_command, issued=False
        )
        self.env._maniguard_source_capture.hold_step()
        self.observer.applied(self.step_idx)
        self.step_idx += 1
        self.observer.transition(self.step_idx, self._observation())
        try:
            return bool(on_step is not None and on_step())
        finally:
            self.observer.oracle(self.step_idx, self.monitor)

    def finalize(self, success, attrs=None):
        self.observer.finish(
            {
                "status": "completed",
                "success": bool(success),
                "last_segment": getattr(self, "segment", None),
                **(attrs or {}),
            }
        )
        return self.out_dir if success else None

    def state_boundary(self, reason):
        self.observer.state_boundary(self._observation(), reason)
