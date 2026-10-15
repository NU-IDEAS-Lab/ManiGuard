from types import SimpleNamespace

import numpy as np


def test_source_attempt_identity_includes_grasp_and_resume_reads_failed_attempts(tmp_path):
    import json

    from maniguard.data.datagen.executor.resume import max_recorded_draw, source_attempt_name

    assert source_attempt_name(1, 0) != source_attempt_name(2, 0)
    root = tmp_path / source_attempt_name(2, 3)
    root.mkdir()
    (root / "episode.json").write_text(json.dumps({"status": "failed", "scene": {"attempt": {"draw_index": 3}}}))
    assert max_recorded_draw(tmp_path) == 3


def test_execution_uses_source_capture_once_and_ticks_gate_once():
    import torch

    from maniguard.data.datagen.primitives.execute import execute_trajectory

    calls = []
    robot = SimpleNamespace(
        default_arm="0",
        action_dim=8,
        arm_action_idx={"0": torch.arange(7)},
        gripper_action_idx={"0": torch.tensor([7])},
    )
    env = SimpleNamespace(step=lambda action: calls.append("legacy step"))

    class Recorder:
        def execute_step(self, action, on_step):
            calls.append("source step")
            return on_step()

    def gate():
        calls.append("gate")
        return False

    n = execute_trajectory(env, robot, np.zeros((2, 7)), recorder=Recorder(), on_step=gate)
    assert n == 2 and calls == ["source step", "gate", "source step", "gate"]


def test_passive_settling_is_not_a_new_command(monkeypatch):
    from maniguard.data.recording.datagen import SourceRecorder

    calls = []
    observer = SimpleNamespace(pending_action=None)

    def before(*args, issued=True):
        observer.pending_action = {"issued": issued}
        calls.append("before")

    observer.before_action = before
    observer.applied = lambda step: calls.append(("applied", observer.pending_action["issued"]))
    observer.transition = lambda *args: calls.append("observation")
    observer.oracle = lambda *args: calls.append("gt")
    rec = SourceRecorder(None, None, None, max_output_bytes=1000)
    rec.observer = observer
    rec.last_command = np.ones(8)
    rec.step_idx = 1
    rec.plan_remaining = 0
    rec.plan_offset = 0
    rec._observation = lambda: {"states": np.zeros(8)}
    rec.state_boundary = lambda reason: None
    rec.env = SimpleNamespace(_maniguard_source_capture=SimpleNamespace(hold_step=lambda: calls.append("physics")))
    rec.passive_step(lambda: calls.append("gate"))
    assert calls == ["before", "physics", ("applied", False), "observation", "gate", "gt"]


def test_lid_snap_routes_both_internal_steps_through_recorder(monkeypatch):
    import sys

    from maniguard.utils import lid_attach

    calls = []
    state = SimpleNamespace(attached=False)
    state.get_value = lambda other: state.attached

    def attach(other, yes, **kw):
        state.attached = yes
        calls.append(("attach", kw["can_joint_break"]))
        return yes

    state.set_value = attach
    kind = type("AttachedTo", (), {})
    lid = SimpleNamespace(name="lid", states={kind: state})
    container = SimpleNamespace(name="jar")
    snapper = object.__new__(lid_attach.LidSnapper)
    snapper._AttachedTo = kind
    snapper.pairs = [SimpleNamespace(lid=lid, container=container)]
    snapper._is_grasped = lambda *a: False
    snapper._lid_touching_container = lambda *a: (True, 1)
    monkeypatch.setattr(lid_attach, "reposition_lid_onto_F", lambda *a: True)
    monkeypatch.setitem(
        sys.modules, "omnigibson", SimpleNamespace(sim=SimpleNamespace(step=lambda: calls.append("unrecorded")))
    )
    assert snapper.try_snap(step_callback=lambda: calls.append("recorded")) == "lid"
    assert calls == ["recorded", ("attach", False), "recorded"]


def test_scripted_early_abort_retains_reason_in_episode_result(tmp_path):
    from maniguard.data.datagen.executor.contracts import FamilyAbort, MotionSegment
    from maniguard.data.datagen.executor.engine import DemoEngine

    recorded = []
    recorder = SimpleNamespace(
        n_steps=0,
        attach=lambda *a, **kw: None,
        finalize=lambda **kw: recorded.append(kw),
    )
    gate = SimpleNamespace(reset=lambda: None, violated=False)

    def abort(*args):
        raise FamilyAbort("wipe_incomplete", remaining=3)

    skeleton = SimpleNamespace(on_segment=abort)
    ctx = SimpleNamespace(diagnostics={}, target=None)
    engine = DemoEngine(None, None, None)
    result = engine.run(
        ctx, skeleton, [MotionSegment("wipe", None, None)], gate, recorder, out_dir=tmp_path
    )
    assert not result.ok
    assert recorded[0]["attrs"]["termination_reason"] == "wipe_incomplete"
    assert recorded[0]["attrs"]["failure_detail"] == {"seg": "wipe", "remaining": 3}
