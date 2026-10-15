import numpy as np

from maniguard.data.datagen.to_lerobot import build_prompt_table, frame_rows


def test_build_prompt_table_dedups_first_seen():
    metas = [{"prompt": "A"}, {"prompt": "B"}, {"prompt": "A"}, {"prompt": "C"}, {"prompt": "B"}]
    unique, idx = build_prompt_table(metas)
    assert unique == ["A", "B", "C"]
    assert idx == [0, 1, 0, 2, 1]


def test_build_prompt_table_missing_prompt_is_empty_string():
    unique, idx = build_prompt_table([{"prompt": "A"}, {}, {"prompt": "A"}])
    assert unique == ["A", ""]
    assert idx == [0, 1, 0]


def test_frame_rows_shapes_and_values():
    traj = {
        "state": np.arange(2 * 8, dtype=np.float64).reshape(2, 8),
        "actions": np.ones((2, 8), dtype=np.float64),
        "actions_commanded": np.full((2, 8), 3.0, dtype=np.float64),
    }
    rows = frame_rows(traj)
    assert len(rows) == 2
    assert set(rows[0].keys()) == {"state", "actions", "actions_commanded"}
    assert rows[0]["state"].dtype == np.float32 and rows[0]["state"].shape == (8,)
    assert np.allclose(rows[1]["state"], np.arange(8, 16))
    assert np.allclose(rows[0]["actions"], 1.0) and np.allclose(rows[0]["actions_commanded"], 3.0)


def test_source_metadata_controls_video_contract():
    from maniguard.data.datagen.data_format import IMAGE_KEYS
    from maniguard.data.datagen.to_lerobot import video_contract
    assert video_contract([{}, {}]) == (256, 30, 'next_achieved')
    new = {'resolution':640, 'fps':20, 'video_keys':list(IMAGE_KEYS),
           'action_source':'applied', 'action_space':'joint', 'state_space':'joint'}
    assert video_contract([new, new]) == (640,20,'applied')
    import pytest
    with pytest.raises(ValueError, match='Mixed'): video_contract([{},new])
    with pytest.raises(ValueError, match='joint'): video_contract([{**new,'action_space':'eef'}])
    with pytest.raises(ValueError, match='views'): video_contract([{**new,'video_keys':['wrist_image']}])


def test_source_export_cannot_mislabel_eef_state_as_joint_state():
    from maniguard.data.datagen.to_lerobot import video_contract
    from maniguard.data.datagen.data_format import IMAGE_KEYS
    import pytest
    meta={'resolution':640, 'fps':20, 'video_keys':list(IMAGE_KEYS),
          'action_source':'applied', 'action_space':'joint', 'state_space':'eef_8d'}
    with pytest.raises(ValueError,match='joint state'):
        video_contract([meta])
