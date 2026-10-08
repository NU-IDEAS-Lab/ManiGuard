"""Offline derived trajectories from source episodes, with explicit action semantics."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np

from .reader import EpisodeReader
from .schema import json_text


def export_episode(source, output, *, cameras=None, state_field="policy_state", action_source="applied", fps=None):
    """Write selected MP4 views + numeric HDF5, leaving the source unchanged.

    Row i uses source observation i, before command i; the final observation is
    retained in the source. Video is a derived H.264 representation. No resizing
    or next-achieved-state substitution is implicit in this exporter.
    """
    import av

    output = Path(output)
    temporary = output.with_name(output.name + ".partial")
    if output.exists() or temporary.exists():
        raise FileExistsError(output)
    if action_source not in ("raw", "transformed", "applied"):
        raise ValueError("Unknown action source")
    with EpisodeReader(source) as reader:
        reader.validate()
        n = reader.metadata["transitions"]
        if not n:
            raise ValueError("Cannot export an episode with no executed commands")
        keys = reader.metadata["camera_keys"] if cameras is None else list(cameras)
        if not keys or len(set(keys)) != len(keys) or not set(keys) <= set(reader.metadata["camera_keys"]):
            raise ValueError("Invalid export camera selection")
        dt = np.diff(reader.h5["time/sim_time_s"][:])
        rate = reader.metadata.get("action_hz", round(1 / float(np.median(dt))))
        fps = rate if fps is None else fps
        if fps != rate or int(fps) != fps or not np.allclose(dt, 1 / fps, rtol=1e-5, atol=1e-7):
            raise ValueError("Export FPS must match the uniformly sampled source control rate")
        height, width = reader.metadata["resolution"]
        initial = reader.read_observation(0)["robot"][state_field]
        command = reader.read_transition(0)
        temporary.mkdir(parents=True)
        with ExitStack() as stack:
            h5 = stack.enter_context(h5py.File(temporary / "traj.hdf5", "w"))
            state = h5.create_dataset("state", shape=(n, *initial.shape), dtype=initial.dtype)
            actions = h5.create_dataset(
                "actions", shape=(n, *command[action_source].shape), dtype=command[action_source].dtype
            )
            issued = h5.create_dataset(
                "actions_commanded", shape=(n, *command["applied"].shape), dtype=command["applied"].dtype
            )
            h5.create_dataset(
                "command_issued",
                data=reader.h5["commands/issued"][:] if "issued" in reader.h5["commands"] else np.ones(n, bool),
            )
            h5.create_dataset("source_observation_id", data=np.arange(n, dtype="i8"))
            h5.create_dataset(
                "source_boundary_id",
                data=reader.h5["commands/source_boundary_id"][:]
                if "source_boundary_id" in reader.h5["commands"]
                else np.full(n, -1, dtype="i8"),
            )
            h5.create_dataset("target_observation_id", data=np.arange(1, n + 1, dtype="i8"))
            h5.create_dataset("sim_time_s", data=reader.h5["time/sim_time_s"][:n])
            streams = {}
            for key in keys:
                container = stack.enter_context(av.open(str(temporary / f"{key}.mp4"), mode="w"))
                stream = container.add_stream("libx264", rate=int(fps))
                stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
                stream.codec_context.gop_size = 10
                stream.options = {"crf": "14", "preset": "medium"}
                streams[key] = (container, stream)
            for i in range(n):
                row = reader.read_transition(i)
                boundary = int(row.get("source_boundary_id", -1))
                source_obs = (
                    reader.read_boundary(boundary, include_rgb=True)["observation"]
                    if boundary >= 0
                    else reader.read_observation(i, include_rgb=True)
                )
                state[i] = source_obs["robot"][state_field]
                actions[i], issued[i] = row[action_source], row["applied"]
                for key, (container, stream) in streams.items():
                    frame = av.VideoFrame.from_ndarray(source_obs["camera_frames"][key], format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
            for container, stream in streams.values():
                for packet in stream.encode():
                    container.mux(packet)
        metadata = {
            "source_episode_id": reader.metadata["episode_id"],
            "n_steps": n,
            "fps": int(fps),
            "resolution": height if height == width else [height, width],
            "video_keys": keys,
            "state_field": state_field,
            "state_space": reader.metadata.get("configuration", {}).get("state_mode")
            if state_field == "policy_state"
            else state_field,
            "action_source": action_source,
            "action_space": reader.metadata.get("configuration", {}).get("action_space")
            or (
                "joint"
                if action_source != "raw"
                and reader.metadata.get("configuration", {}).get("controller_preset") == "joint_position_raw"
                else None
            ),
            "alignment": "source observation i -> control interval i -> source observation i+1; command_issued marks passive holds",
            "success": reader.metadata.get("result", {}).get("success"),
            "prompt": reader.metadata.get("scene", {}).get("prompt", ""),
        }
        (temporary / "meta.json").write_text(json_text(metadata) + "\n", encoding="utf-8")
    temporary.rename(output)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--cameras", nargs="+")
    parser.add_argument("--state-field", default="policy_state")
    parser.add_argument("--action-source", choices=["raw", "transformed", "applied"], default="applied")
    export_episode(**vars(parser.parse_args()))
