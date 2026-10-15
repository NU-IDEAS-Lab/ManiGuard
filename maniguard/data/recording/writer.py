"""Single-owner episode writer with bounded frame memory and explicit finalization."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np

from .schema import append_blob, encode_rgb, json_text, pack_record, validate_metadata


class EpisodeWriter:
    def __init__(self, directory, metadata: dict, max_output_bytes: int):
        self.directory = Path(directory)
        self.metadata = validate_metadata(metadata)
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        self.budget = int(max_output_bytes)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.observations = self.transitions = self.snapshots = 0
        self.boundaries = 0
        self.closed = self.failed = False
        self.last_time = self.last_tick = None
        self.metadata.update(status="partial", observations=0, transitions=0, snapshots=0)
        self.metadata["boundaries"] = 0
        self._manifest()
        self.h5 = h5py.File(self.directory / "trajectory.hdf5", "w")
        self.snapshot_file = None
        self.events = (self.directory / "events.jsonl").open("w", encoding="utf-8")
        self.columns = {"observations": None, "commands": None}

    def _manifest(self):
        temporary = self.directory / "episode.json.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json_text(self.metadata) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.directory / "episode.json")

    def _check(self, additional=0):
        if self.closed or self.failed:
            raise RuntimeError("Episode writer is closed or failed")
        self.h5.flush()
        if self.snapshot_file is not None:
            self.snapshot_file.flush()
        self.events.flush()
        size = sum(path.stat().st_size for path in self.directory.iterdir() if path.is_file())
        if size + additional > self.budget:
            self.failed = True
            raise OSError(f"Episode output budget exceeded: {size + additional} > {self.budget}")

    def _append_columns(self, records: dict, owner: str, index: int):
        columns = {key: np.asarray(value) for key, value in records.items()}
        for key, array in columns.items():
            if array.dtype.kind not in "bifuc" or any(n == 0 for n in array.shape):
                raise ValueError(f"Numeric column {key} requires a nonempty numeric array or scalar")
        known = self.columns[owner]
        if known is not None:
            if columns.keys() != known.keys():
                raise ValueError("Numeric column set changed; use explicit validity fields for missing values")
            for key, array in columns.items():
                if (array.shape, array.dtype) != known[key]:
                    raise ValueError(f"Numeric column {key} changed shape or dtype")
        else:
            self.columns[owner] = {key: (array.shape, array.dtype) for key, array in columns.items()}
        for key, array in columns.items():
            if key not in self.h5:
                self.h5.create_dataset(
                    key, shape=(0, *array.shape), maxshape=(None, *array.shape), dtype=array.dtype, chunks=True
                )
            dataset = self.h5[key]
            dataset.resize((index + 1, *array.shape))
            dataset[index] = array

    def append_observation(self, record: dict):
        self._check()
        i = self.observations
        if record.get("observation_id") != i or self.transitions != i:
            raise ValueError("Observation must follow its incoming transition, starting at observation 0")
        time, tick = float(record["sim_time_s"]), int(record["physics_tick"])
        if not np.isfinite(time) or (i and (time <= self.last_time or tick <= self.last_tick)):
            raise ValueError("Observation time and physics ticks must increase")
        frames = record["camera_frames"]
        if set(frames) != set(self.metadata["camera_keys"]):
            raise ValueError("Observation camera set differs from metadata")
        encoded = {key: encode_rgb(frames[key], self.metadata) for key in self.metadata["camera_keys"]}
        columns = {
            "time/observation_id": np.int64(i),
            "time/sim_time_s": np.float64(time),
            "time/physics_tick": np.int64(tick),
        }
        for section in ("robot", "objects", "safety"):
            for key, value in record.get(section, {}).items():
                if "/" in key:
                    raise ValueError("Numeric field names cannot contain slashes")
                columns[f"{section}/{key}"] = value
        extra = {
            key: value
            for key, value in record.items()
            if key
            not in ("observation_id", "sim_time_s", "physics_tick", "camera_frames", "robot", "objects", "safety")
        }
        blob = pack_record(extra)
        self._check(sum(map(len, encoded.values())) + len(blob))
        try:
            self._append_columns(columns, "observations", i)
            for key, payload in encoded.items():
                append_blob(self.h5, f"images/{key}", payload)
            append_blob(self.h5, "observation_metadata", blob)
            self.h5.flush()
            self._check()
        except Exception:
            self.failed = True
            raise
        self.observations += 1
        self.last_time, self.last_tick = time, tick

    def append_transition(self, record: dict):
        self._check()
        i = self.transitions
        if (
            record.get("source_observation_id") != i
            or record.get("target_observation_id") != i + 1
            or self.observations != i + 1
        ):
            raise ValueError("Transition source/target must connect the latest observation to the next")
        for key in ("raw", "applied"):
            value = np.asarray(record[key])
            if value.ndim != 1 or not len(value) or not np.all(np.isfinite(value)):
                raise ValueError(f"{key} command must be a finite nonempty vector")
        numeric, extra = {}, {}
        for key, value in record.items():
            if "/" in key:
                raise ValueError("Transition field names cannot contain slashes")
            array = np.asarray(value) if not isinstance(value, dict) else None
            if array is not None and array.dtype.kind in "bifuc":
                numeric[f"commands/{key}"] = array
            else:
                extra[key] = value
        blob = pack_record(extra)
        self._check(len(blob) + sum(a.nbytes for a in numeric.values()))
        try:
            self._append_columns(numeric, "commands", i)
            append_blob(self.h5, "transition_metadata", blob)
            self.h5.flush()
            self._check()
        except Exception:
            self.failed = True
            raise
        self.transitions += 1

    def append_event(self, record: dict):
        payload = json_text(record) + "\n"
        self._check(len(payload.encode("utf-8")))
        try:
            self.events.write(payload)
            self.events.flush()
        except Exception:
            self.failed = True
            raise

    def append_boundary(self, observation, checkpoint, reason):
        """Instantaneous scripted state update, not a command or a physics tick."""
        if observation["observation_id"] != self.observations - 1 or self.transitions != self.observations - 1:
            raise ValueError("Boundary must precede the next control interval")
        if observation["sim_time_s"] != self.last_time or observation["physics_tick"] != self.last_tick:
            raise ValueError("Boundary cannot hide an unrecorded physics interval")
        obs = dict(observation)
        frames = obs.pop("camera_frames")
        obs["encoded_rgb"] = {key: encode_rgb(frames[key], self.metadata) for key in self.metadata["camera_keys"]}
        blob = pack_record({"observation": obs, "checkpoint": checkpoint, "reason": reason})
        self._check(len(blob))
        append_blob(self.h5, "boundaries", blob)
        self.h5.flush()
        index = self.boundaries
        self.boundaries += 1
        self.metadata["boundaries"] = self.boundaries
        self.append_event(
            {
                "type": "state_boundary",
                "boundary_id": index,
                "source_observation_id": obs["observation_id"],
                "reason": reason,
            }
        )
        return index

    def append_snapshot(self, observation_id: int, state: dict, checkpoint=None):
        if observation_id < 0 or observation_id >= self.observations:
            raise ValueError("Snapshot requires an existing observation")
        blob = pack_record({"state": state, "checkpoint": checkpoint})
        self._check(len(blob))
        try:
            if self.snapshot_file is None:
                self.snapshot_file = h5py.File(self.directory / "snapshots.hdf5", "w")
                self.snapshot_file.create_dataset(
                    "observation_id", shape=(0,), maxshape=(None,), dtype="i8", chunks=True
                )
            indices = self.snapshot_file["observation_id"]
            if len(indices) and observation_id <= indices[-1]:
                raise ValueError("Snapshot observation IDs must increase")
            append_blob(self.snapshot_file, "records", blob)
            indices.resize((len(indices) + 1,))
            indices[-1] = observation_id
            self.snapshot_file.flush()
            self._check()
        except Exception:
            self.failed = True
            raise
        self.snapshots += 1

    def finish(self, status: str, result: dict):
        if status not in ("complete", "partial", "failed"):
            raise ValueError("Episode status must be complete, partial or failed")
        if status == "complete":
            self._check()
            if self.observations != self.transitions + 1:
                raise ValueError("Complete episode requires T commands and T+1 observations")
        self.metadata.update(
            observations=self.observations, transitions=self.transitions, snapshots=self.snapshots, result=result
        )
        self.close()
        if status == "complete":
            from .reader import EpisodeReader

            with EpisodeReader(self.directory) as reader:
                reader.validate(require_complete=False)
        self.metadata["status"] = status
        self._manifest()

    def close(self):
        if self.closed:
            return
        try:
            self.h5.close()
            if self.snapshot_file is not None:
                self.snapshot_file.close()
            self.events.close()
        finally:
            self.closed = True
            self.metadata.update(observations=self.observations, transitions=self.transitions, snapshots=self.snapshots)
            self._manifest()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
