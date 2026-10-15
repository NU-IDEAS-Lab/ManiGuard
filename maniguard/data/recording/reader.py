"""Offline random-access source reader; no simulator or model dependency."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .schema import SCHEMA_VERSION, decode_rgb, read_blob, unpack_record, validate_blobs


class EpisodeReader:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "episode.json").read_text(encoding="utf-8"))
        if self.metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported episode schema version")
        self.h5 = h5py.File(self.directory / "trajectory.hdf5", "r")
        self.snapshot_file = None

    def read_rgb(self, camera_key: str, observation_id: int):
        rgb = decode_rgb(self.read_rgb_bytes(camera_key, observation_id))
        if rgb.shape != (*self.metadata["resolution"], 3):
            raise ValueError("Decoded RGB dimensions do not match metadata")
        return rgb

    def read_rgb_bytes(self, camera_key: str, observation_id: int) -> bytes:
        """Return the original encoded frame for lossless downstream copying."""
        if camera_key not in self.metadata["camera_keys"]:
            raise KeyError(camera_key)
        return read_blob(self.h5[f"images/{camera_key}"], observation_id)

    def read_observation(self, index: int, *, include_rgb=False) -> dict:
        result = unpack_record(read_blob(self.h5["observation_metadata"], index))
        result.update({key: dataset[index] for key, dataset in self.h5["time"].items()})
        for section in ("robot", "objects", "safety"):
            if section in self.h5:
                result[section] = {key: dataset[index] for key, dataset in self.h5[section].items()}
        if include_rgb:
            result["camera_frames"] = {key: self.read_rgb(key, index) for key in self.metadata["camera_keys"]}
        return result

    def read_transition(self, index: int) -> dict:
        result = unpack_record(read_blob(self.h5["transition_metadata"], index))
        result.update({key: dataset[index] for key, dataset in self.h5["commands"].items()})
        return result

    def iter_events(self):
        with (self.directory / "events.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)

    def read_boundary(self, index, *, include_rgb=False):
        record = unpack_record(read_blob(self.h5["boundaries"], index))
        encoded = record["observation"].pop("encoded_rgb")
        if include_rgb:
            record["observation"]["camera_frames"] = {key: decode_rgb(blob) for key, blob in encoded.items()}
        return record

    def read_snapshot(self, observation_id: int):
        if self.snapshot_file is None:
            self.snapshot_file = h5py.File(self.directory / "snapshots.hdf5", "r")
        ids = self.snapshot_file["observation_id"][:]
        index = int(np.searchsorted(ids, observation_id))
        if index >= len(ids) or ids[index] != observation_id:
            raise KeyError(observation_id)
        record = unpack_record(read_blob(self.snapshot_file["records"], index))
        return record["state"], record["checkpoint"]

    def validate(self, *, decode_images=False, require_complete=True) -> dict:
        if require_complete and self.metadata["status"] != "complete":
            raise ValueError(f"Episode is {self.metadata['status']}, not complete")
        n, t = self.metadata["observations"], self.metadata["transitions"]
        if n != t + 1:
            raise ValueError("Episode lacks the final observation for its commands")
        if set(self.h5["images"]) != set(self.metadata["camera_keys"]):
            raise ValueError("Source camera set mismatch")
        for key in self.metadata["camera_keys"]:
            validate_blobs(self.h5[f"images/{key}"], n)
            if decode_images:
                for i in range(n):
                    self.read_rgb(key, i)
        validate_blobs(self.h5["observation_metadata"], n)
        for section in ("time", "robot", "objects", "safety"):
            if section in self.h5:
                for dataset in self.h5[section].values():
                    if len(dataset) != n:
                        raise ValueError(f"Wrong observation column length: {dataset.name}")
        ids = self.h5["time/observation_id"][:]
        if not np.array_equal(ids, np.arange(n)):
            raise ValueError("Non-contiguous observation indices")
        for key in ("sim_time_s", "physics_tick"):
            values = self.h5[f"time/{key}"][:]
            if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0):
                raise ValueError(f"Invalid observation time column: {key}")
        if t:
            validate_blobs(self.h5["transition_metadata"], t)
            for dataset in self.h5["commands"].values():
                if len(dataset) != t:
                    raise ValueError(f"Wrong command column length: {dataset.name}")
            for key, offset in (("source_observation_id", 0), ("target_observation_id", 1)):
                if not np.array_equal(self.h5[f"commands/{key}"][:], np.arange(t) + offset):
                    raise ValueError(f"Incorrect transition {key}")
        if self.metadata["snapshots"]:
            with h5py.File(self.directory / "snapshots.hdf5", "r") as snapshots:
                count = self.metadata["snapshots"]
                validate_blobs(snapshots["records"], count)
                ids = snapshots["observation_id"][:]
                if len(ids) != count or np.any(np.diff(ids) <= 0) or ids[0] < 0 or ids[-1] >= n:
                    raise ValueError("Invalid snapshot observation indices")
        for _ in self.iter_events():
            pass
        boundaries = self.metadata.get("boundaries", 0)
        if boundaries:
            validate_blobs(self.h5["boundaries"], boundaries)
            for i in range(boundaries):
                obs = self.read_boundary(i, include_rgb=decode_images)["observation"]
                parent = int(obs["observation_id"])
                if not 0 <= parent < n or obs["sim_time_s"] != self.h5["time/sim_time_s"][parent]:
                    raise ValueError("Boundary does not match an observation time")
        if t and "source_boundary_id" in self.h5["commands"]:
            for i, boundary in enumerate(self.h5["commands/source_boundary_id"]):
                if boundary < -1 or boundary >= boundaries:
                    raise ValueError("Invalid command boundary reference")
                if boundary >= 0 and self.read_boundary(int(boundary))["observation"]["observation_id"] != i:
                    raise ValueError("Command boundary belongs to another observation")
        return {
            "observations": n,
            "transitions": t,
            "snapshots": self.metadata["snapshots"],
            "images_decoded": bool(decode_images),
        }

    def close(self):
        self.h5.close()
        if self.snapshot_file is not None:
            self.snapshot_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
