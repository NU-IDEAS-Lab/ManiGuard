"""
SO-101 Leader Arm → Franka Teleop Demo

Teleoperates a FrankaMounted in OmniGibson using an SO-101 leader arm.
The SO-101 end-effector deltas are mapped to Franka IK targets.

Modes:
    1. Simple scene (default):
        python -m omnigibson.examples.teleoperation.so101_franka_demo

    2. Load a saved pipeline snapshot:
        python -m omnigibson.examples.teleoperation.so101_franka_demo \
            --snapshot outputs/pipeline_runs/empty_random_clutter_20260409_132528/scene_ep1.json

    3. Generate a fresh task scene (clutter/stack/transfer):
        python -m omnigibson.examples.teleoperation.so101_franka_demo --task clutter

Prerequisites:
    Terminal 1 (lerobot venv):
        python teleop_bridge/so101_server.py --mock          # mock mode
        python teleop_bridge/so101_server.py --port /dev/ttyACM0  # real hardware

    Terminal 2 (behavior conda):
        python -m omnigibson.examples.teleoperation.so101_franka_demo [--snapshot ...]
"""

import argparse
import json
import sys

import omnigibson as og
from omnigibson.teleop.so101_teleop import SO101TeleopAgent, SO101TeleopConfig


# ---------------------------------------------------------------------------
# Robot config helper
# ---------------------------------------------------------------------------

def _robot_cfg():
    """FrankaMounted with IK controller for teleop."""
    return {
        "type": "FrankaMounted",
        "obs_modalities": ["rgb"],
        "action_normalize": False,
        "grasping_mode": "assisted",
        "controller_config": {
            "arm_0": {
                "name": "InverseKinematicsController",
                "command_input_limits": None,
            },
            "gripper_0": {
                "name": "MultiFingerGripperController",
                "command_input_limits": (0.0, 1.0),
                "mode": "smooth",
            },
        },
    }


# ---------------------------------------------------------------------------
# Scene builders
# ---------------------------------------------------------------------------

def _build_simple_scene_cfg():
    """Simple table + a few objects for basic testing."""
    return dict(
        scene={"type": "Scene"},
        robots=[_robot_cfg()],
        objects=[
            {
                "type": "DatasetObject",
                "name": "breakfast_table",
                "category": "breakfast_table",
                "model": "kwmfdg",
                "bounding_box": [2, 1, 0.4],
                "position": [0.8, 0, 0.3],
                "orientation": [0, 0, 0.707, 0.707],
            },
            {
                "type": "DatasetObject",
                "name": "frail",
                "category": "frail",
                "model": "zmjovr",
                "scale": [2, 2, 2],
                "position": [0.6, -0.35, 0.5],
            },
            {
                "type": "DatasetObject",
                "name": "toy_figure1",
                "category": "toy_figure",
                "model": "issvzv",
                "scale": [0.75, 0.75, 0.75],
                "position": [0.6, 0, 0.5],
            },
        ],
    )


def _build_from_snapshot(snapshot_path):
    """Rebuild env config from a pipeline scene snapshot.

    The snapshot (saved by og.sim.save()) contains:
      - objects_info.init_info: class, category, model for each object
      - state.registry.object_registry: positions, orientations
    We reconstruct the object configs and apply saved positions, then
    override the robot controller to IK for teleop.
    """
    with open(snapshot_path, "r", encoding="utf-8") as f:
        snap = json.load(f)

    init_info = snap["objects_info"]["init_info"]
    obj_states = snap["state"]["registry"]["object_registry"]

    object_cfgs = []
    robot_name = None

    for obj_name, info in init_info.items():
        cls_name = info["class_name"]
        obj_args = info["args"]
        state = obj_states.get(obj_name, {})
        root = state.get("root_link", {})
        pos = root.get("pos", [0, 0, 0])
        ori = root.get("ori", [0, 0, 0, 1])

        if cls_name == "FrankaMounted":
            # Remember robot name + position; we'll build our own config
            robot_name = obj_name
            robot_pos = pos
            robot_ori = ori
            continue

        obj_cfg = {
            "type": "DatasetObject",
            "name": obj_name,
            "category": obj_args.get("category", obj_name),
            "model": obj_args.get("model", ""),
            "position": pos,
            "orientation": ori,
        }
        if obj_args.get("fixed_base"):
            obj_cfg["fixed_base"] = True
        if "bounding_box" in obj_args:
            obj_cfg["bounding_box"] = obj_args["bounding_box"]

        object_cfgs.append(obj_cfg)

    # Robot config with IK controller and saved position
    robot_cfg = _robot_cfg()
    if robot_name:
        robot_cfg["position"] = robot_pos
        robot_cfg["orientation"] = robot_ori

    n_objs = len(object_cfgs)
    print(f"[Teleop] Loaded snapshot: {snapshot_path}")
    print(f"[Teleop] {n_objs} objects, robot at {robot_pos[:2]}")

    return dict(
        scene={"type": "Scene"},
        robots=[robot_cfg],
        objects=object_cfgs,
        task={"type": "DummyTask"},
    )


def _build_task_scene_cfg(task_type, seed=0):
    """Generate a fresh task scene using the empty-scene pipeline helpers."""
    import numpy as np

    from omnigibson.task_generation.empty_scene_pipeline import (
        _build_clutter_objects,
        _build_stack_objects,
        _build_transfer_objects,
        _load_surface_catalog,
        _make_obj_cfg,
        _pick_surface,
    )

    rng = np.random.default_rng(seed)

    if task_type == "clutter":
        obj_cfgs, roles, selection = _build_clutter_objects(rng, "medium")
    elif task_type == "stack":
        obj_cfgs, roles, selection = _build_stack_objects(rng, "medium")
    elif task_type == "transfer":
        obj_cfgs, roles, selection = _build_transfer_objects(rng)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    surface_cat, surface_model, _ = _pick_surface(rng)
    catalog = _load_surface_catalog()
    surface_height = catalog.get(surface_cat, {}).get(
        surface_model, {}).get("height_m", 0.75)

    surface_cfg = _make_obj_cfg(
        name="support_surface", category=surface_cat,
        model=surface_model, position=[0.0, 0.0, surface_height / 2.0],
        fixed_base=True,
    )

    # Place objects on the table in a grid
    table_top_z = surface_height + 0.03
    for i, obj_cfg in enumerate(obj_cfgs):
        row, col = divmod(i, 3)
        obj_cfg["position"] = [-0.15 + col * 0.15, -0.15 + row * 0.15, table_top_z]

    robot_cfg = _robot_cfg()
    robot_cfg["position"] = [0.0, -0.65, 0.0]

    role_summary = {r: sum(1 for v in roles.values() if v == r) for r in set(roles.values())}
    print(f"[Teleop] Fresh {task_type} scene: {surface_cat}/{surface_model}, "
          f"objects={len(obj_cfgs)}, roles={role_summary}")

    return dict(
        scene={"type": "Scene"},
        robots=[robot_cfg],
        objects=[surface_cfg] + obj_cfgs,
        task={"type": "DummyTask"},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SO-101 → Franka Teleop Demo")
    parser.add_argument("--zmq-port", type=int, default=5557, help="ZMQ server port")
    parser.add_argument("--zmq-host", type=str, default="127.0.0.1", help="ZMQ server host")
    parser.add_argument("--pos-scale", type=float, default=5.0, help="Position scaling factor")
    parser.add_argument("--rot-scale", type=float, default=1.0, help="Rotation scaling factor")
    parser.add_argument("--steps", type=int, default=10000, help="Number of sim steps")
    parser.add_argument("--snapshot", type=str, default=None,
                        help="Path to a pipeline scene snapshot JSON (scene_ep*.json)")
    parser.add_argument("--task", type=str, default=None,
                        choices=["clutter", "stack", "transfer"],
                        help="Generate a fresh task scene")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --task mode")
    args = parser.parse_args()

    # Build scene config (priority: snapshot > task > simple)
    if args.snapshot:
        cfg = _build_from_snapshot(args.snapshot)
    elif args.task:
        cfg = _build_task_scene_cfg(args.task, seed=args.seed)
    else:
        cfg = _build_simple_scene_cfg()

    # Create environment
    env = og.Environment(configs=cfg)
    env.reset()

    # Set camera
    og.sim.viewer_camera.set_position_orientation(
        position=[-0.22, 0.99, 1.09],
        orientation=[-0.14, 0.47, 0.84, -0.23],
    )

    robot = env.robots[0]

    # Initialize SO-101 teleop agent
    teleop_cfg = SO101TeleopConfig(
        zmq_host=args.zmq_host,
        zmq_port=args.zmq_port,
        position_scale=args.pos_scale,
        rotation_scale=args.rot_scale,
    )
    agent = SO101TeleopAgent(config=teleop_cfg)

    label = args.snapshot or args.task or "simple"
    print("\n" + "=" * 50)
    print(f"SO-101 → Franka Teleop Ready  [{label}]")
    print("Move the SO-101 leader arm to control Franka")
    print("Ctrl+C to exit")
    print("=" * 50 + "\n")

    try:
        for step in range(args.steps):
            action = agent.get_action(robot)
            env.step(action)

            if step % 300 == 0 and step > 0:
                status = "connected" if agent.is_connected else "waiting for SO-101 data..."
                print(f"Step {step}/{args.steps} — {status}")

    except KeyboardInterrupt:
        print("\nStopping teleop...")
    finally:
        agent.stop()
        og.clear()


if __name__ == "__main__":
    main()
