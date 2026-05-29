"""Minimal HDF5 state playback for annotation and waypoint extraction."""

from __future__ import annotations

import json
from typing import Any, Callable

import h5py
import numpy as np
import torch as th

import omnigibson as og

from _common import read_json_attr


StepCallback = Callable[[int, th.Tensor, Any], None]
EpisodeCallback = Callable[[int, Any], None]


def _as_tensor_dataset(dataset: h5py.Dataset) -> th.Tensor:
    arr = np.asarray(dataset)
    if arr.dtype.kind in {"S", "U", "O"}:
        raise TypeError(f"Dataset {dataset.name} is not numeric")
    return th.from_numpy(arr)


def _load_numeric_group(group: h5py.Group) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            data[key] = _load_numeric_group(item)
        else:
            data[key] = _as_tensor_dataset(item)
    return data


def _get_dataset(group: h5py.Group, *names: str) -> h5py.Dataset:
    for name in names:
        if name in group:
            return group[name]
    raise KeyError(f"None of {names!r} exists under {group.name}")


class MinimalPlaybackWrapper:
    """Replay an OG HDF5 trajectory by loading serialized sim states."""

    def __init__(
        self,
        env,
        input_hdf5: h5py.File,
        *,
        step_callback: StepCallback | None = None,
        episode_start_callback: EpisodeCallback | None = None,
    ):
        self.env = env
        self.input_hdf5 = input_hdf5
        self.scene_file = read_json_attr(input_hdf5["data"].attrs, "scene_file", default=None)
        self.step_callback = step_callback
        self.episode_start_callback = episode_start_callback

    def playback_episode(self, episode_id: int) -> None:
        data_grp = self.input_hdf5["data"]
        demo_key = f"demo_{episode_id}"
        if demo_key not in data_grp:
            raise KeyError(f"No episode {demo_key!r} in {self.input_hdf5.filename}")

        traj_grp = data_grp[demo_key]
        transitions = json.loads(traj_grp.attrs.get("transitions", "{}"))
        init_metadata = _load_numeric_group(traj_grp["init_metadata"]) if "init_metadata" in traj_grp else {}
        actions = _as_tensor_dataset(_get_dataset(traj_grp, "action", "actions"))
        states = _as_tensor_dataset(_get_dataset(traj_grp, "state", "states"))
        state_sizes = _as_tensor_dataset(_get_dataset(traj_grp, "state_size", "state_sizes"))

        if self.scene_file is not None:
            self.env.scene.restore(self.scene_file, update_initial_file=True)

        if init_metadata:
            with og.sim.stopped():
                for attr, values in init_metadata.items():
                    if len(values) != self.env.scene.n_objects:
                        raise ValueError(
                            f"init_metadata[{attr!r}] has {len(values)} values, "
                            f"but scene has {self.env.scene.n_objects} objects"
                        )
                for obj_idx, obj in enumerate(self.env.scene.objects):
                    for attr, values in init_metadata.items():
                        value = values[obj_idx]
                        setattr(obj, attr, value.item() if value.ndim == 0 else value)

        self.env.reset()
        for robot in self.env.robots:
            robot.control_enabled = False

        og.sim.load_state(states[0, : int(state_sizes[0])], serialized=True)
        if self.episode_start_callback is not None:
            self.episode_start_callback(episode_id, self.env)

        for frame_idx, (action, state, state_size) in enumerate(zip(actions, states[1:], state_sizes[1:])):
            if str(frame_idx) in transitions:
                pass
            og.sim.load_state(state[: int(state_size)], serialized=True)
            for obj in self.env.scene.objects:
                if hasattr(obj, "wake"):
                    obj.wake()
            if self.step_callback is not None:
                self.step_callback(frame_idx, action, self.env)
            og.sim.step()
