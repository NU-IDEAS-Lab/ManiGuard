from types import SimpleNamespace

import numpy as np
from test_episode import observation


def test_observer_records_actual_executor_commands_not_proposal_tail(tmp_path, monkeypatch):
    from maniguard.data.recording.observer import SourceEpisodeObserver
    from maniguard.data.recording.reader import EpisodeReader
    from maniguard.visual.profile import load_profile

    profile = load_profile({"resolution": [16, 16]})
    cfg = SimpleNamespace(
        recording_output_dir=str(tmp_path),
        recording_max_bytes=10_000_000,
        source_profile=profile,
        recording_provenance=None,
        action_frequency=20,
        physics_frequency=120,
        random_policy=True,
    )
    obs = SourceEpisodeObserver(cfg, {"name": "fixture"})
    source = observation(0)
    source["camera_frames"] = {key: source["camera_frames"]["image_left"] for key in profile["camera_keys"]}
    env = SimpleNamespace(scene=SimpleNamespace(objects=[]), _maniguard_source_capture=SimpleNamespace(last_observation=source, render_report={}))
    monitor = SimpleNamespace(
        recording_state=lambda step: {
            "valid": True,
            "observation": {"step": step, "ap": {"upright": step == 0}, "doomed": step == 1, "state": step},
            "error": None,
        }
    )
    obs._physical = lambda policy: {"robot": {"q": np.zeros(9, np.float32)}}
    monkeypatch.setattr(
        "maniguard.data.recording.observer.capture_checkpoint", lambda *a: {"scene": {"x": 1}, "rng": {}}
    )
    expected_identity = obs.directory.name
    obs.directory = tmp_path / "attempt_000000_g0"
    obs.start(
        env,
        SimpleNamespace(joints={str(i): None for i in range(9)}, _controller_config={},
                        controller_action_idx={}, default_arm="0",
                        arm_control_idx={"0": np.arange(7)}, gripper_control_idx={"0": np.array([7, 8])}),
        {"states": np.zeros(8)},
        monitor,
        0,
        {"name": "fixture", "ltl_safety": {"propositions": {"upright": {}}}},
    )
    proposal = np.arange(70, dtype=np.float32).reshape(10, 7)
    obs.proposal(0, proposal, 2, SimpleNamespace(shape=(8,), low=np.r_[np.full(7, -np.inf), -1.], high=np.r_[np.full(7, np.inf), 1.]))
    raw, transformed, applied = proposal[0], np.ones(8), np.ones(8) * 0.5
    obs.before_action(0, 0, raw, transformed, applied)
    obs.applied(0)
    source = observation(1)
    source["camera_frames"] = {key: source["camera_frames"]["image_left"] for key in profile["camera_keys"]}
    env._maniguard_source_capture.last_observation = source
    obs.transition(1, {"states": np.ones(8)})
    obs.oracle(1, monitor)
    obs.finish({"status": "completed", "success": False})
    with EpisodeReader(obs.directory) as reader:
        reader.validate(decode_images=True)
        assert reader.metadata["episode_id"] == expected_identity
        assert reader.metadata["joint_groups"] == {"arm": list(range(7)), "gripper": [7, 8]}
        np.testing.assert_array_equal(reader.read_transition(0)["applied"], applied)
        event = next(e for e in reader.iter_events() if e["type"] == "proposal")
        assert len(event["actions"]) == 10 and event["planned_steps"] == 2
        assert event["action_low"] == ["-inf"] * 7 + [-1.]
        assert event["action_high"] == ["inf"] * 7 + [1.]
        assert reader.read_observation(1)["safety"]["ap_values"].tolist() == [False]


def test_constraint_gt_retains_temporal_first_rejection(tmp_path):
    from maniguard.data.recording.observer import SourceEpisodeObserver
    from maniguard.utils.ltl_utils import LTLMonitor

    observer = SourceEpisodeObserver.__new__(SourceEpisodeObserver)
    observer.constraint_monitors = {'order': LTLMonitor('support U closed'), 'upright': LTLMonitor('G upright')}
    rows = []
    for labels in [
        {'support': True, 'closed': False, 'upright': True},
        {'support': False, 'closed': False, 'upright': True},
        {'support': True, 'closed': True, 'upright': False},
    ]:
        rows.append(observer._constraint_gt(labels))
    assert not rows[0]['order']['doomed']
    assert rows[1]['order']['doomed'] and not rows[1]['upright']['doomed']
    assert rows[2]['order']['doomed'] and rows[2]['upright']['doomed']
