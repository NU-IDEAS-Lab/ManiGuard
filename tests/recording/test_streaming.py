"""Exercise a long stream without keeping old image arrays alive."""

import gc
import weakref

from test_episode import metadata, observation, transition


def test_writer_does_not_retain_source_frames(tmp_path):
    from maniguard.data.recording import EpisodeReader, EpisodeWriter

    with EpisodeWriter(tmp_path / "episode", metadata(), 100_000_000) as writer:
        frame = observation(0)
        reference = weakref.ref(frame["camera_frames"]["image_left"])
        writer.append_observation(frame)
        del frame
        for i in range(400):
            writer.append_transition(transition(i))
            # Reuse a bounded RGB pattern; time and control indices keep increasing.
            frame = observation(i % 3)
            frame.update(observation_id=i + 1, sim_time_s=0.5 + 0.05 * (i + 1), physics_tick=60 + 6 * (i + 1))
            writer.append_observation(frame)
        del frame
        gc.collect()
        assert reference() is None
        writer.finish("complete", {})
    with EpisodeReader(tmp_path / "episode") as reader:
        assert reader.validate()["observations"] == 401
