#!/usr/bin/env python3
"""Collect source demonstrations with OmniGibson SpaceMouse teleoperation."""

from __future__ import annotations

import argparse
import signal

import torch as th

import omnigibson.lazy as lazy
from omnigibson.envs import HDF5CollectionWrapper
from omnigibson.utils.ui_utils import KeyboardEventHandler

from _common import (
    create_env,
    load_json_or_yaml,
    make_full_action,
    maybe_reload_ik_controllers,
    prepare_output_path,
    shutdown_og,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="OmniGibson env config YAML/JSON")
    parser.add_argument("--output-hdf5", required=True, help="HDF5 path to write")
    parser.add_argument("--robot-id", type=int, default=0, help="Robot index to control")
    parser.add_argument("--steps", type=int, default=10000, help="Maximum teleop steps")
    parser.add_argument("--arm-speed-scaledown", type=float, default=0.04, help="SpaceMouse arm speed scale")
    parser.add_argument("--viewport-camera-path", default=None, help="Optional viewport camera path for HDF5 metadata")
    parser.add_argument(
        "--only-successes",
        action="store_true",
        help="Only persist episodes that OG reports successful",
    )
    parser.add_argument("--no-controller-reload", action="store_true", help="Keep controllers from the input config")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output HDF5")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_path = prepare_output_path(args.output_hdf5, overwrite=args.overwrite)

    cfg = load_json_or_yaml(args.config)
    env = create_env(cfg)
    env.reset()
    if not args.no_controller_reload:
        maybe_reload_ik_controllers(env, args.robot_id)

    recorder = HDF5CollectionWrapper(
        env=env,
        output_path=str(output_path),
        viewport_camera_path=args.viewport_camera_path,
        overwrite=True,
        only_successes=args.only_successes,
        flush_every_n_traj=1,
    )
    env = recorder
    robot = env.robots[args.robot_id]

    from omnigibson.utils.teleop_utils import TeleopSystem
    from telemoma.configs.base_config import teleop_config

    teleop_config.arm_left_controller = "spacemouse"
    teleop_config.arm_right_controller = "spacemouse"
    teleop_config.base_controller = "keyboard"
    teleop_config.interface_kwargs["spacemouse"] = {"arm_speed_scaledown": args.arm_speed_scaledown}

    teleop = TeleopSystem(config=teleop_config, robot=robot, show_control_marker=True)
    teleop.start()
    if "spacemouse" in teleop.interfaces:
        teleop.interfaces["spacemouse"].controllable_robot_parts = ["right"]

    exit_flag = {"quit": False}

    def request_quit() -> None:
        exit_flag["quit"] = True
        print("[Teleop] Quit requested")

    def reset_episode() -> None:
        print("[Teleop] Resetting episode")
        env.reset()

    KeyboardEventHandler.initialize()
    KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.Q, request_quit)
    KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.R, reset_episode)
    signal.signal(signal.SIGINT, lambda *_: request_quit())

    env.reset()
    prev_left_button = 0
    try:
        for step in range(args.steps):
            if exit_flag["quit"]:
                break
            for obj in env.scene.objects:
                if hasattr(obj, "wake"):
                    obj.wake()
            robot_action = th.as_tensor(teleop.get_action(teleop.get_obs()), dtype=th.float32)
            full_action = make_full_action(env, args.robot_id, robot_action)
            env.step(full_action)

            if "spacemouse" in teleop.interfaces:
                left_button = teleop.interfaces["spacemouse"].raw_data.buttons[0]
                if left_button and not prev_left_button:
                    reset_episode()
                prev_left_button = left_button

            if step % 100 == 0:
                print(f"[Teleop] step={step}")
    finally:
        print(f"[Teleop] Saving HDF5 to {output_path}")
        env.save_data()
        shutdown_og()


if __name__ == "__main__":
    main()
