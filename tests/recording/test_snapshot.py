from types import SimpleNamespace

import numpy as np


def test_capture_does_not_mutate_live_assisted_grasp_state():
    from maniguard.data.recording.snapshot import capture_scene_state

    robot = SimpleNamespace(_ag_obj_constraint_params={"position": np.array([1.0, 2.0, 3.0])})
    original = robot._ag_obj_constraint_params

    def dump_state(serialized):
        robot._ag_obj_constraint_params["position"] = [0, 0, 0]
        return {"objects": {"position": np.array([1.0, 2.0, 3.0])}}

    env = SimpleNamespace(robots=[robot], scene=SimpleNamespace(dump_state=dump_state))
    saved = capture_scene_state(env)
    assert robot._ag_obj_constraint_params is original
    np.testing.assert_array_equal(original["position"], [1, 2, 3])
    saved["objects"]["position"][0] = 5
    assert original["position"][0] == 1


def test_scene_restore_uses_independent_copy():
    from maniguard.data.recording.snapshot import restore_scene_state

    saved = {"x": np.array([1.0, 2.0])}

    def load_state(state, serialized):
        state["x"][0] = 99

    env = SimpleNamespace(scene=SimpleNamespace(load_state=load_state))
    restore_scene_state(env, saved, convert_state=lambda x: x)
    assert saved["x"][0] == 1


def test_restore_converts_ragged_particle_arrays_to_tensors():
    import torch

    from maniguard.data.recording.snapshot import restore_scene_state

    state = {
        "systems": {
            "water": {
                "positions": np.zeros((0, 3), dtype=np.float32),
                "orientations": np.ones((2, 4), dtype=np.float32),
            }
        },
        "objects": {"robot": {"joint_positions": [0.1, 0.2]}},
    }
    seen = []

    def load_state(value, serialized):
        assert value["systems"]["water"]["positions"].numel() == 0
        assert isinstance(value["systems"]["water"]["orientations"], torch.Tensor)
        assert isinstance(value["objects"]["robot"]["joint_positions"], torch.Tensor)
        seen.append(True)

    env = SimpleNamespace(scene=SimpleNamespace(load_state=load_state))
    restore_scene_state(env, state)
    assert seen and isinstance(state["systems"]["water"]["positions"], np.ndarray)


def test_joint_runtime_properties_are_preserved_and_restored():
    from maniguard.data.recording.snapshot import capture_scene_state, restore_scene_state

    joint = SimpleNamespace(damping=np.array(2.0), friction=np.array(0.1), stiffness=np.array(4.0))
    obj = SimpleNamespace(name="drawer", joints={"hinge": joint})
    scene = SimpleNamespace(
        objects=[obj],
        dump_state=lambda serialized: {"registry": {}},
        load_state=lambda *a, **kw: None,
        object_registry=lambda kind, name: obj,
    )
    env = SimpleNamespace(scene=scene, robots=[])
    state = capture_scene_state(env)
    joint.friction = 10.0
    restore_scene_state(env, state, convert_state=lambda x: x)
    assert joint.friction == 0.1
    assert isinstance(joint.friction, float)


def test_assisted_grasp_counters_survive_scene_restore():
    from maniguard.data.recording.snapshot import capture_scene_state, restore_scene_state

    robot = SimpleNamespace(name="robot", _ag_grasp_counter={"0": 3}, _ag_release_counter={"0": 1})
    scene = SimpleNamespace(objects=[], dump_state=lambda **kw: {}, load_state=lambda *a, **kw: None)
    env = SimpleNamespace(robots=[robot], scene=scene)
    saved = capture_scene_state(env)
    robot._ag_grasp_counter["0"] = None
    robot._ag_release_counter["0"] = None
    restore_scene_state(env, saved, convert_state=lambda x: x)
    assert robot._ag_grasp_counter == {"0": 3}
    assert robot._ag_release_counter == {"0": 1}


def test_snapshot_does_not_apply_collision_apis_while_reading():
    from maniguard.data.recording.snapshot import capture_scene_state

    class Link:
        prim = SimpleNamespace(GetRelationship=lambda name: None)

        @property
        def _collision_filter_api(self):
            raise AssertionError("Reading this upstream property applies an API and invalidates PhysX")

    obj = SimpleNamespace(name="jar", joints={}, links={"root": Link()}, states={})
    scene = SimpleNamespace(objects=[obj], dump_state=lambda **kw: {})
    capture_scene_state(SimpleNamespace(scene=scene, robots=[]))


def test_attachment_restore_reapplies_child_and_parent_motion_after_joint_creation():
    from maniguard.data.recording.snapshot import restore_scene_state

    class AttachedTo:
        pass

    restored = {}
    objects = {}
    for name in ("lid", "container"):
        objects[name] = SimpleNamespace(
            name=name,
            links={},
            states={},
            n_joints=0,
            root_link=SimpleNamespace(load_state=lambda state, serialized, name=name: restored.update({name: state})),
            wake=lambda: None,
            sleep=lambda: None,
        )
    objects["lid"].states[AttachedTo] = SimpleNamespace(parent=objects["container"], parents_disabled_collisions=set())
    saved = {
        "registry": {
            "object_registry": {
                name: {"root_link": {"lin_vel": np.array([1.0, 2.0, 3.0])}, "is_asleep": False} for name in objects
            }
        },
        "_recording_object_runtime": {
            name: {"filters": {}, "attachments": {"disabled_parents": []}} for name in objects
        },
    }
    scene = SimpleNamespace(
        load_state=lambda *a, **kw: restored.clear(), object_registry=lambda kind, name: objects[name]
    )
    restore_scene_state(SimpleNamespace(scene=scene), saved, convert_state=lambda x: x)
    assert set(restored) == {"lid", "container"}
    np.testing.assert_array_equal(restored["lid"]["lin_vel"], [1.0, 2.0, 3.0])


def test_restored_predicate_memory_can_continue_tensor_particle_checks():
    import torch

    from maniguard.data.recording.schema import pack_record, unpack_record
    from maniguard.data.recording.snapshot import capture_checkpoint, restore_checkpoint

    memory = {"initial_counts": {"jar": torch.tensor(100)}}

    def predicate(_state=memory):
        return (_state["initial_counts"]["jar"] - torch.tensor(90)) / 100 > 0.15

    monitor = SimpleNamespace(
        _prop_fns={"spill": predicate},
        _monitor=None,
        _violation_step=None,
        _violation_count=0,
        _error=None,
        _ltl_log=[],
    )
    env = SimpleNamespace(
        robots=[], scene=SimpleNamespace(dump_state=lambda **kw: {}, load_state=lambda *a, **kw: None)
    )
    checkpoint = unpack_record(pack_record(capture_checkpoint(env, monitor)))
    memory.clear()
    restore_checkpoint(env, monitor, checkpoint)
    assert not predicate()
