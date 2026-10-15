import numpy as np
import pytest


def test_stream_video_keeps_shape_count_and_rate(tmp_path):
    import av

    from maniguard.visual.video import StreamingVideoWriter

    path = tmp_path / "nested/view.mp4"
    with StreamingVideoWriter(path, fps=20, frame_hw=(16, 16)) as writer:
        for _ in range(4):
            writer.append(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.close()  # Idempotent cleanup, including callers' failure paths.
    with av.open(str(path)) as reader:
        assert reader.streams.video[0].average_rate == 20
        frames = list(reader.decode(video=0))
        assert len(frames) == 4
        assert all((frame.height, frame.width) == (16, 16) for frame in frames)


def test_partial_stream_flushes_on_error_without_retaining_input(tmp_path):
    import weakref

    import av

    from maniguard.visual.video import StreamingVideoWriter

    path = tmp_path / "partial.mp4"
    with pytest.raises(ValueError, match="shape"), StreamingVideoWriter(path, fps=30, frame_hw=(16, 16)) as writer:
        for _ in range(100):
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            ref = weakref.ref(frame)
            writer.append(frame)
            del frame
            assert ref() is None
        writer.append(np.zeros((8, 8, 3), dtype=np.uint8))
    with av.open(str(path)) as reader:
        assert len(list(reader.decode(video=0))) == 100
