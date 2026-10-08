"""Five-view source-quality previews of an already loaded benchmark scene."""

import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from PIL import Image

from .capture import SourceCapture


def render_preview(
    env, output, profile, *, frames=60, fps=30, mode="frozen", monitor=None, advance=None, before_capture=None
):
    """Create derived MP4 previews and one JPEG reference per source view.

    Frozen mode never steps physics. Idle mode keeps the simulator's existing
    controller goals, matching benchmark stability previews. Neither is a policy
    evaluation or an executed-command episode.
    """
    import av

    if mode not in ("frozen", "idle_step") or frames < 1:
        raise ValueError("Invalid preview mode or length")
    capture = SourceCapture(env, profile)
    if mode == "idle_step" and not np.isclose(capture.sim.get_sim_step_dt(), 1 / fps):
        raise ValueError("Preview FPS must match the simulator control rate")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "type": "scene_preview",
        "mode": mode,
        "profile": capture.profile,
        "render_settings": capture.render_report,
        "frames": [],
        "complete": False,
    }
    with ExitStack() as stack:
        writers = {}
        for key in capture.profile["camera_keys"]:
            container = stack.enter_context(av.open(str(output / f"{key}.mp4"), "w"))
            stream = container.add_stream("libx264", rate=int(fps))
            stream.height, stream.width = capture.profile["resolution"]
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "14"}
            writers[key] = (container, stream)
        for i in range(frames):
            if before_capture is not None:
                before_capture(i)
            if mode == "idle_step":
                with capture.sim.render_on_step(False):
                    if advance is None:
                        capture.sim.step()
                    else:
                        advance()
            record = capture.capture(i)
            if monitor is not None:
                monitor.step(i + 1)
            for key, rgb in record["camera_frames"].items():
                if i == 0:
                    Image.fromarray(rgb).save(output / f"{key}.jpg", quality=95, subsampling=0)
                container, stream = writers[key]
                for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                    container.mux(packet)
            report["frames"].append(
                {key: record[key] for key in ("observation_id", "sim_time_s", "physics_tick", "render_metadata")}
            )
        for container, stream in writers.values():
            for packet in stream.encode():
                container.mux(packet)
    report.update(complete=True, cameras=record["camera_metadata"])
    (output / "preview.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
