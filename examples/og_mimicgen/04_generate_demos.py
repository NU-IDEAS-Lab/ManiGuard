#!/usr/bin/env python3
"""Generate new demonstrations by replaying object-centric waypoints."""

from __future__ import annotations

import argparse
import random

import torch as th

import omnigibson as og
from omnigibson.envs import HDF5CollectionWrapper

from _common import (
    apply_eef_z_offset,
    apply_pose_randomizations,
    build_ik_delta_action,
    check_task_success,
    create_env,
    env_seed,
    linear_freespace_trajectory,
    load_json_or_yaml,
    make_full_action,
    maybe_reload_ik_controllers,
    parse_pose_randomization,
    prepare_output_path,
    shutdown_og,
    transform_waypoints_to_robot_frame,
    wake_scene_objects,
)
from _waypoints import load_waypoints_hdf5


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", required=True, help="env_config.json from stage 2")
    parser.add_argument("--waypoints", required=True, help="waypoints.hdf5 from stage 3")
    parser.add_argument("--output-hdf5", required=True, help="Path to write generated demos")
    parser.add_argument("--robot-id", type=int, default=0)
    parser.add_argument("--n-demos", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--action-frequency", type=float, default=30.0)
    parser.add_argument("--freespace-velocity", type=float, default=0.1)
    parser.add_argument("--eef-z-offset", type=float, default=0.0)
    parser.add_argument(
        "--randomize-object",
        action="append",
        default=[],
        metavar="OBJECT:DX,DY,DZ,DYAW",
        help="Object pose randomization applied after each reset. Can be repeated.",
    )
    parser.add_argument(
        "--reference-demo",
        default=None,
        help="Use a specific demo key instead of random source selection",
    )
    parser.add_argument("--save-failed", action="store_true", help="Save attempts even if task success is false")
    parser.add_argument("--no-controller-reload", action="store_true", help="Keep controllers from env_config")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _configure_generation_frequency(config: dict, action_frequency: float) -> dict:
    config = dict(config)
    env_cfg = dict(config.get("env", {}))
    env_cfg["action_frequency"] = float(action_frequency)
    env_cfg["rendering_frequency"] = float(action_frequency)
    env_cfg.setdefault("physics_frequency", 120)
    config["env"] = env_cfg
    return config


def _execute_pose_sequence(
    env,
    robot_id: int,
    positions: th.Tensor,
    quats: th.Tensor,
    gripper_cmds: th.Tensor,
) -> th.Tensor:
    robot = env.robots[robot_id]
    last_action = None
    for pos, quat, gripper_cmd in zip(positions, quats, gripper_cmds):
        robot_action = build_ik_delta_action(robot, pos, quat, float(gripper_cmd.item()))
        last_action = make_full_action(env, robot_id, robot_action)
        env.step(last_action)
    return last_action


def main() -> None:
    args = _build_parser().parse_args()
    env_seed(args.seed)
    output_path = prepare_output_path(args.output_hdf5, overwrite=args.overwrite)
    randomizations = [parse_pose_randomization(value) for value in args.randomize_object]
    all_waypoints = load_waypoints_hdf5(args.waypoints)
    episode_keys = sorted(all_waypoints.keys())
    if not episode_keys:
        raise RuntimeError(f"No waypoints found in {args.waypoints}")
    if args.reference_demo is not None and args.reference_demo not in all_waypoints:
        raise KeyError(f"--reference-demo {args.reference_demo!r} is not in {episode_keys}")

    config = _configure_generation_frequency(load_json_or_yaml(args.env_config), args.action_frequency)
    env = create_env(config)
    if not args.no_controller_reload:
        maybe_reload_ik_controllers(env, args.robot_id)
    env = HDF5CollectionWrapper(
        env=env,
        output_path=str(output_path),
        viewport_camera_path=None,
        overwrite=True,
        only_successes=not args.save_failed,
        flush_every_n_traj=10,
    )

    success_count = 0
    attempt_count = 0
    while success_count < args.n_demos and attempt_count < args.max_attempts:
        attempt_count += 1
        source_key = args.reference_demo or random.choice(episode_keys)
        subtasks = all_waypoints[source_key]
        print(f"[Generate] attempt={attempt_count} successes={success_count}/{args.n_demos} source={source_key}")

        env.reset()
        if randomizations:
            apply_pose_randomizations(env, randomizations)
            og.sim.step()

        failed = False
        last_action = None
        for subtask_idx, subtask in enumerate(subtasks):
            ref_name = subtask["reference_object"]["name"]
            ref_obj = env.scene.object_registry("name", ref_name, None)
            if ref_obj is None:
                print(f"[Generate] missing reference object {ref_name!r}; attempt failed")
                failed = True
                break
            robot = env.robots[args.robot_id]
            eef_pos_robot, eef_quat_robot, gripper_actions = transform_waypoints_to_robot_frame(
                subtask["waypoints"],
                ref_obj,
                robot,
            )
            eef_pos_robot, eef_quat_robot = apply_eef_z_offset(eef_pos_robot, eef_quat_robot, -args.eef_z_offset)

            start_pos = eef_pos_robot[0]
            start_quat = eef_quat_robot[0]
            freespace_pos, freespace_quat = linear_freespace_trajectory(
                robot,
                start_pos,
                start_quat,
                velocity=args.freespace_velocity,
                action_frequency=args.action_frequency,
            )
            approach_gripper = 1.0 if subtask["type"] in {"pick", "open", "close"} else 0.0
            freespace_gripper = th.full((len(freespace_pos),), approach_gripper, dtype=th.float32)
            try:
                last_action = _execute_pose_sequence(
                    env,
                    args.robot_id,
                    freespace_pos,
                    freespace_quat,
                    freespace_gripper,
                )
                last_action = _execute_pose_sequence(env, args.robot_id, eef_pos_robot, eef_quat_robot, gripper_actions)
            except Exception as exc:
                print(f"[Generate] subtask {subtask_idx} failed: {exc}")
                failed = True
                break

        wake_scene_objects(env)
        og.sim.step()
        task_success = False if failed else check_task_success(env, last_action)
        if task_success:
            success_count += 1
            print(f"[Generate] success {success_count}/{args.n_demos}")
        else:
            print("[Generate] task success check failed")

    print(f"[Generate] Saving generated HDF5 to {output_path}")
    env.save_data()
    print(f"[Generate] generated_successes={success_count} attempts={attempt_count}")
    shutdown_og()


if __name__ == "__main__":
    main()
