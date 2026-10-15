from types import SimpleNamespace

import numpy as np


def test_playback_capture_replaces_only_rgb_without_stepping():
    from maniguard.utils.camera_setup import SOURCE_EXTERNAL_SENSORS
    from maniguard.visual.cameras import playback_observation

    frames = {k: np.zeros((8, 8, 3), dtype=np.uint8) for k in [*SOURCE_EXTERNAL_SENSORS, 'wrist_image']}
    original = {f'external::{sensor}::rgb': 'old' for sensor in SOURCE_EXTERNAL_SENSORS.values()}
    original.update({'agent_0::eef_link:camera:0::rgb': 'old', 'agent_0::proprio': [1, 2]})
    calls = []
    capture = SimpleNamespace(observation_id=4, capture=lambda i: calls.append(i) or {'camera_frames': frames})
    result = playback_observation(original, 'agent_0', capture)
    assert calls == [5]
    assert result['agent_0::proprio'] is original['agent_0::proprio']
    assert result['external::cam_left::rgb'].shape == (8, 8, 3)
    assert result['agent_0::eef_link:camera:0::rgb'].shape == (8, 8, 3)
    assert original['external::cam_left::rgb'] == 'old'
