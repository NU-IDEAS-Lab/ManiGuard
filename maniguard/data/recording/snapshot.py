"""Scene and simulator-process checkpoints for rendering and command replay.

These do not include external policy-server RNG or promise bitwise replay across
GPU/simulator versions. Recorded commands remain the source of replay actions.
"""

from __future__ import annotations

import inspect
import random
from copy import deepcopy

import numpy as np


def array_tree(value):
    """Detach tensors into independent CPU arrays without importing a model."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: array_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(array_tree(item) for item in value)
    if isinstance(value, list):
        return [array_tree(item) for item in value]
    return deepcopy(value)


def capture_scene_state(env) -> dict:
    protected = []
    # Upstream AG dump_state converts nested positions in a shallow copy. Keep
    # its conversion from modifying the live constraint parameters.
    for robot in env.robots:
        if hasattr(robot, "_ag_obj_constraint_params"):
            original = robot._ag_obj_constraint_params
            protected.append((robot, original))
            robot._ag_obj_constraint_params = deepcopy(original)
    try:
        state = array_tree(env.scene.dump_state(serialized=False))
        # Scripted manipulation changes drawer friction/damping at segment
        # boundaries. OmniGibson's dynamic dump omits these USD properties.
        properties = {}
        for obj in getattr(env.scene, "objects", ()):
            joints = {}
            for name, joint in obj.joints.items():
                joints[name] = {key: array_tree(getattr(joint, key)) for key in ("damping", "friction", "stiffness")}
            if joints:
                properties[obj.name] = joints
        state["_recording_joint_properties"] = properties
        state["_recording_ag_memory"] = {
            robot.name: {
                key: array_tree(getattr(robot, key))
                for key in ("_ag_grasp_counter", "_ag_release_counter", "_ag_freeze_gripper")
                if hasattr(robot, key)
            }
            for robot in env.robots
            if hasattr(robot, "name")
        }
        runtime = {}
        for obj in getattr(env.scene, "objects", ()):
            item = {"filters": {}, "attachments": {}}
            for name, link in getattr(obj, "links", {}).items():
                # Upstream _collision_filter_api lazily Applies the API: a read
                # would recook live shapes and invalidate articulation views.
                relation = link.prim.GetRelationship("physics:filteredPairs")
                item["filters"][name] = [str(p) for p in relation.GetTargets()] if relation else []
            for kind, value in getattr(obj, "states", {}).items():
                if kind.__name__ == "AttachedTo":
                    item["attachments"]["disabled_parents"] = sorted(p.name for p in value.parents_disabled_collisions)
                    if value.attachment_joint_prim_path:
                        prim = obj.prim.GetStage().GetPrimAtPath(value.attachment_joint_prim_path)
                        item["attachments"]["joint_attributes"] = {
                            key: float(prim.GetAttribute(key).Get())
                            for key in ("physics:breakForce", "physics:breakTorque")
                        }
                elif kind.__name__ == "ModifiedParticles":
                    item["particle_counts"] = array_tree(value.particle_counts)
            runtime[obj.name] = item
        state["_recording_object_runtime"] = runtime
        return state
    finally:
        for robot, original in protected:
            robot._ag_obj_constraint_params = original


def _torch_tree(value):
    """Restore typed arrays, including empty/ragged particle arrays, on CPU."""
    import torch

    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, dict):
        return {key: _torch_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [_torch_tree(item) for item in value]
        if isinstance(value, list):
            try:
                return torch.tensor(converted, dtype=torch.float32)
            except (TypeError, ValueError, RuntimeError):
                return converted
        return tuple(converted)
    return value


def restore_scene_state(env, state: dict, *, convert_state=None):
    if convert_state is None:
        convert_state = _torch_tree
    state = deepcopy(state)
    properties = state.pop("_recording_joint_properties", {})
    ag_memory = state.pop("_recording_ag_memory", {})
    object_runtime = state.pop("_recording_object_runtime", {})
    env.scene.load_state(convert_state(state), serialized=False)
    for robot in getattr(env, "robots", ()):
        for key, value in ag_memory.get(robot.name, {}).items():
            setattr(robot, key, deepcopy(value))
    for name, joints in properties.items():
        obj = env.scene.object_registry("name", name)
        if obj is None:
            raise ValueError(f"Recorded object is absent: {name}")
        for joint_name, values in joints.items():
            for key, value in values.items():
                setattr(obj.joints[joint_name], key, float(np.asarray(value).item()))
    attached_objects = set()
    for name, values in object_runtime.items():
        obj = env.scene.object_registry("name", name)
        for link_name, targets in values["filters"].items():
            link = obj.links[link_name]
            relation = link.prim.GetRelationship("physics:filteredPairs")
            current = [str(p) for p in relation.GetTargets()] if relation else []
            if current != targets:
                link._collision_filter_api.GetFilteredPairsRel().SetTargets(targets)
        for kind, value in getattr(obj, "states", {}).items():
            if kind.__name__ == "AttachedTo":
                if value.parent is not None:
                    attached_objects.update((name, value.parent.name))
                extra = values["attachments"]
                value.parents_disabled_collisions = {
                    env.scene.object_registry("name", n) for n in extra["disabled_parents"]
                }
                if "joint_attributes" in extra:
                    prim = obj.prim.GetStage().GetPrimAtPath(value.attachment_joint_prim_path)
                    for key, number in extra["joint_attributes"].items():
                        prim.GetAttribute(key).Set(number)
            elif kind.__name__ == "ModifiedParticles" and "particle_counts" in values:
                value.particle_counts = deepcopy(values["particle_counts"])
    # AttachedTo rebuilds joints after entity loading, aligning the child and
    # calling keep_still on both bodies. Reapply their recorded kinematics only
    # after all attachments exist, without invoking object-state loading again.
    for name in sorted(attached_objects):
        obj = env.scene.object_registry("name", name)
        saved = convert_state(state["registry"]["object_registry"][name])
        obj.root_link.load_state(saved["root_link"], serialized=False)
        if obj.n_joints:
            obj.set_joint_positions(saved["joint_pos"])
            obj.set_joint_velocities(saved["joint_vel"])
        obj.sleep() if saved.get("is_asleep", False) else obj.wake()
    # Vt.QuathArray.FromNumpy uses xyzw in this runtime, unlike the upstream
    # setter's wxyz conversion. Explicit scalar+vector construction avoids that
    # ABI-dependent array conversion when restoring recorded physical particles.
    systems = state.get("registry", {}).get("system_registry", {})
    for name, system_state in systems.items():
        for inst_name, particle_state in system_state.get("particle_states", {}).items():
            import torch
            from omnigibson import lazy

            system = env.scene.get_system(name)
            inst = system.particle_instancers[inst_name]
            quats = []
            for pos, quat in zip(particle_state["particle_positions"], particle_state["particle_orientations"]):
                _, world_quat = env.scene.convert_scene_relative_pose_to_world(
                    torch.as_tensor(pos), torch.as_tensor(quat)
                )
                x, y, z, w = map(float, world_quat)
                quats.append(lazy.pxr.Gf.Quath(w, lazy.pxr.Gf.Vec3h(x, y, z)))
            inst.set_attribute("orientations", lazy.pxr.Vt.QuathArray(quats))


def capture_checkpoint(env, monitor=None) -> dict:
    import torch

    saved_monitor = None
    if monitor is not None:
        memory = {}
        for name, fn in monitor._prop_fns.items():
            parameter = inspect.signature(fn).parameters.get("_state")
            if parameter is not None:
                memory[name] = array_tree(parameter.default)
        saved_monitor = {
            "state": monitor._monitor.state if monitor._monitor is not None else None,
            "violation_step": monitor._violation_step,
            "violation_count": monitor._violation_count,
            "error": deepcopy(monitor._error),
            "last_log": deepcopy(monitor._ltl_log[-1:]),
            "predicate_memory": memory,
        }
    return {
        "scene": capture_scene_state(env),
        "monitor": saved_monitor,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": array_tree(torch.get_rng_state()),
            "cuda": array_tree(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else [],
        },
    }


def restore_checkpoint(env, monitor, checkpoint):
    import torch

    restore_scene_state(env, checkpoint["scene"])
    saved = checkpoint["monitor"]
    if saved is not None:
        if monitor is None:
            raise ValueError("Checkpoint requires a task monitor")
        if monitor._monitor is not None:
            monitor._monitor._state = saved["state"]
        monitor._violation_step, monitor._violation_count = saved["violation_step"], saved["violation_count"]
        monitor._error, monitor._ltl_log = deepcopy(saved["error"]), deepcopy(saved["last_log"])
        for name, value in saved["predicate_memory"].items():
            memory = inspect.signature(monitor._prop_fns[name]).parameters["_state"].default
            memory.clear()
            memory.update(_torch_tree(value))
    rng = checkpoint["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(torch.as_tensor(rng["torch"], dtype=torch.uint8, device="cpu"))
    if rng["cuda"]:
        if len(rng["cuda"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA device count differs")
        torch.cuda.set_rng_state_all([torch.as_tensor(value, dtype=torch.uint8, device="cpu") for value in rng["cuda"]])
