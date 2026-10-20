"""Render recorded physical states without re-executing policy commands."""

from __future__ import annotations

import json
from pathlib import Path

from .reader import EpisodeReader
from .snapshot import restore_scene_state


def controller_config_from_json(value, reference):
    """Restore controller JSON using the matching controller's native types.

    Upstream merges tuple-of-tensor limits before constructing controllers;
    preserving that structure also retains integer DOFs and bool masks.
    """
    import torch

    if isinstance(reference, torch.Tensor):
        return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if isinstance(value, dict):
        defaults = reference if isinstance(reference, dict) else {}
        return {key: controller_config_from_json(item, defaults.get(key)) for key, item in value.items()}
    if isinstance(value, list):
        if isinstance(reference, (list, tuple)) and len(value) == len(reference):
            return type(reference)(
                controller_config_from_json(item, default) for item, default in zip(value, reference)
            )
        return [controller_config_from_json(item, None) for item in value]
    return value



def resolve_playback_profile(metadata, selection=None):
    """Recorded settings are authoritative unless explicitly rerendering."""
    from maniguard.visual.profile import resolve_profile_selection
    profile = resolve_profile_selection(metadata["source_profile"] if selection is None else selection)
    return profile

def render_recorded_states(env, reader, output, *, profile=None, indices=None):
    """Use an already loaded matching scene; preserve original source times.

    This is state playback, not a deterministic rollout reconstruction. It does
    not step physics or run a policy/monitor. Material overrides are applied by
    the scene loader; dynamic particle and robot states come from snapshots.
    """
    import torch

    from maniguard.visual.cameras import configure_source_sensors
    from maniguard.visual.preview import render_preview

    reader.validate()
    ids = list(range(reader.metadata["observations"])) if indices is None else list(indices)
    if not ids or any(type(i) is not int or not 0 <= i < reader.metadata["observations"] for i in ids):
        raise ValueError("Playback needs existing observation indices")
    profile = resolve_playback_profile(reader.metadata, profile)
    sensors = configure_source_sensors(env, profile)
    source_times = []

    def restore(i):
        index = ids[i]
        state, _ = reader.read_snapshot(index)
        restore_scene_state(env, state)
        obs = reader.read_observation(index)
        for key, sensor in sensors.items():
            camera = obs["camera_metadata"][key]
            sensor.set_position_orientation(
                position=torch.tensor(camera["position_world_m"]),
                orientation=torch.tensor(camera["orientation_world_xyzw"]),
            )
        source_times.append(
            {"observation_id": index, "sim_time_s": float(obs["sim_time_s"]), "physics_tick": int(obs["physics_tick"])}
        )

    report = render_preview(
        env, output, profile, frames=len(ids), fps=reader.metadata["action_hz"], mode="frozen", before_capture=restore
    )
    report.update(
        type="recorded_state_playback",
        source_episode_id=reader.metadata["episode_id"],
        original_source_profile=reader.metadata["source_profile"],
        source_observations=source_times,
    )
    (Path(output) / "preview.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode")
    parser.add_argument("output")
    parser.add_argument(
        "--observations", type=int, nargs="+", required=True, help="Explicit indices keep state playback output bounded"
    )
    parser.add_argument("--source-profile", default=None, help="Explicit re-render override; default preserves recorded profile")
    args = parser.parse_args()
    source_path, output_path = Path(args.episode).resolve(), Path(args.output).resolve()
    if output_path == source_path or source_path in output_path.parents:
        parser.error("Playback output must be separate from its source episode")
    with EpisodeReader(args.episode) as reader:
        reader.validate()
        source = Path(args.episode)
        from maniguard.data.bench_builder.perturbation import apply_perturbation
        from maniguard.data.bench_builder.render import _build_og_config
        from maniguard.data.datagen.primitives.scene import init_omnigibson, task_needs_gpu_dynamics
        from maniguard.data.datagen.primitives.task_io import load_diagnostics_row
        from maniguard.visual.capture import initialize_source_simulator

        profile = resolve_playback_profile(reader.metadata, args.source_profile)
        og = init_omnigibson(needs_gpu_dynamics=task_needs_gpu_dynamics(source), source_profile=profile)
        diag = load_diagnostics_row(source, 1)
        cfg = _build_og_config(source / "task_scene.json", diag, profile["resolution"][0])
        cfg["env"].update(
            action_frequency=reader.metadata["action_hz"],
            rendering_frequency=60,
            physics_frequency=reader.metadata["physics_hz"],
        )
        initialize_source_simulator(reader.metadata["action_hz"], 60, reader.metadata["physics_hz"])
        env = og.Environment(configs=cfg)
        env.reset()
        apply_perturbation(env, diag)
        if reader.metadata.get("controller_config"):
            robot = env.robots[0]
            saved = reader.metadata["controller_config"]
            defaults = {key: robot._default_controller_config[key][config["name"]] for key, config in saved.items()}
            robot.reload_controllers(controller_config_from_json(saved, defaults))
        render_recorded_states(env, reader, args.output, indices=args.observations, profile=profile)
        og.shutdown()


if __name__ == "__main__":
    main()
