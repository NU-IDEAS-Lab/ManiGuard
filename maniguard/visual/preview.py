"""Five-view source-quality previews of an already loaded benchmark scene."""

import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from PIL import Image

from .capture import SourceCapture
from .video import StreamingVideoWriter


def render_preview(
    env, output, profile, *, frames=60, fps=30, mode="frozen", monitor=None, advance=None, before_capture=None, overwrite=False
):
    """Create derived MP4 previews and one JPEG reference per source view.

    Frozen mode never steps physics. Idle mode keeps the simulator's existing
    controller goals, matching benchmark stability previews. Neither is a policy
    evaluation or an executed-command episode.
    """
    if mode not in ("frozen", "idle_step") or frames < 1:
        raise ValueError("Invalid preview mode or length")
    capture = SourceCapture(env, profile)
    with ExitStack() as stack:
        stack.callback(capture.close)
        if mode == "idle_step" and not np.isclose(capture.sim.get_sim_step_dt(), 1 / fps):
            raise ValueError("Preview FPS must match the simulator control rate")
        output = Path(output) if output is not None else None
        if output is not None:
            output.mkdir(parents=True, exist_ok=overwrite)
        report = {
            "type": "scene_preview",
            "mode": mode,
            "profile": capture.profile,
            "render_settings": capture.render_report,
            "frames": [],
            "complete": False,
        }
        if output is not None:
            (output / "preview.json").write_text(json.dumps(report, indent=2) + "\n")
        writers = {}
        for key in capture.profile["camera_keys"] if output is not None else []:
            writers[key] = stack.enter_context(StreamingVideoWriter(
                output / f"{key}.mp4", fps=fps, frame_hw=capture.profile["resolution"]
            ))
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
            for key, writer in writers.items():
                rgb = record["camera_frames"][key]
                if i == 0:
                    Image.fromarray(rgb).save(output / f"{key}.jpg", quality=95, subsampling=0)
                writer.append(rgb)
            report["frames"].append(
                {key: record[key] for key in ("observation_id", "sim_time_s", "physics_tick", "render_metadata")}
            )
    report.update(complete=True, cameras=record["camera_metadata"])
    if output is not None:
        (output / "preview.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def preview_is_complete(output, selection="high_fidelity"):
    """Require an explicit complete manifest, the selected profile and every view."""
    from .profile import resolve_profile_selection

    output = Path(output)
    profile = resolve_profile_selection(selection)
    try:
        report = json.loads((output / "preview.json").read_text())
        return profile is not None and report.get("complete") is True and report.get("profile") == profile and all(
            (output / f"{key}.mp4").is_file() and (output / f"{key}.mp4").stat().st_size > 0
            for key in profile["camera_keys"]
        )
    except (OSError, ValueError):
        return False
