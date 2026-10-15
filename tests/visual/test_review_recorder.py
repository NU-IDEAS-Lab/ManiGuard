import json
from types import SimpleNamespace

import numpy as np

from maniguard.visual.profile import load_profile


def test_review_stream_has_five_views_and_no_extra_physics(tmp_path, monkeypatch):
    import maniguard.visual.capture as module
    from maniguard.envs.frozen_task_runtime import ReviewVideoRecorder
    profile = load_profile()
    events = []

    class Capture:
        def __init__(self, env, selection):
            self.profile = profile
            self.render_report = {}
        def capture(self, i):
            events.append(i)
            return {'camera_frames': {key: np.zeros((640, 640, 3), dtype=np.uint8) for key in profile['camera_keys']}}
        def close(self):
            events.append('close')

    monkeypatch.setattr(module, 'SourceCapture', Capture)
    with ReviewVideoRecorder(path=tmp_path, fps=20) as recorder:
        recorder.record(object(), SimpleNamespace(sim=None))
    assert events == [0, 'close']
    manifest = json.loads((tmp_path / 'source_preview_ep1/preview.json').read_text())
    assert manifest['complete'] and manifest['frame_count'] == 1
    assert len(list((tmp_path / 'source_preview_ep1').glob('*.mp4'))) == 5
