import importlib.util
import sys
from pathlib import Path


class _DummyObject:
    def __init__(self):
        self.last_position = None
        self.last_orientation = None

    def set_position_orientation(self, position, orientation):
        self.last_position = tuple(float(v) for v in position)
        self.last_orientation = tuple(float(v) for v in orientation)


def _load_module():
    mod_path = Path(__file__).resolve().parents[1] / "omnigibson" / "utils" / "clutter_pack_layout.py"
    spec = importlib.util.spec_from_file_location("clutter_pack_layout", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_clutter_pack_is_deterministic():
    mod = _load_module()
    descriptors = [
        mod.ClutterObjectDescriptor("coffee_cup.n.01_1", "target", (0.04, 0.04), 0.10),
        mod.ClutterObjectDescriptor("wineglass.n.01_1", "fragile", (0.03, 0.03), 0.14),
        mod.ClutterObjectDescriptor("bowl.n.01_1", "clutter", (0.06, 0.06), 0.09),
    ]
    pack_a = mod.build_clutter_pack("countertop.n.01_1", descriptors, seed=7)
    pack_b = mod.build_clutter_pack("countertop.n.01_1", descriptors, seed=7)
    assert pack_a.object_entries == pack_b.object_entries


def test_apply_pack_transform_and_validate_integrity():
    mod = _load_module()
    descriptors = [
        mod.ClutterObjectDescriptor("coffee_cup.n.01_1", "target", (0.04, 0.04), 0.10),
        mod.ClutterObjectDescriptor("wineglass.n.01_1", "fragile", (0.03, 0.03), 0.14),
        mod.ClutterObjectDescriptor("bowl.n.01_1", "clutter", (0.06, 0.06), 0.09),
        mod.ClutterObjectDescriptor("plate.n.04_1", "clutter", (0.07, 0.07), 0.03),
    ]
    pack = mod.build_clutter_pack("countertop.n.01_1", descriptors, seed=11)
    dummy = {d.instance_id: _DummyObject() for d in descriptors}

    world_positions = mod.apply_pack_transform(
        pack_spec=pack,
        objects_by_inst=dummy,
        pack_origin_world=(7.4, -0.7, 0.9),
        pack_yaw=0.2,
        table_top_z=0.9,
    )
    report = mod.validate_pack_integrity(
        pack_spec=pack,
        world_positions=world_positions,
        pack_origin_world=(7.4, -0.7, 0.9),
        pack_yaw=0.2,
        tol_xy=0.03,
    )
    assert report.ok
    assert report.max_position_error <= 1e-6


def test_validate_pack_integrity_detects_shift():
    mod = _load_module()
    descriptors = [
        mod.ClutterObjectDescriptor("coffee_cup.n.01_1", "target", (0.04, 0.04), 0.10),
        mod.ClutterObjectDescriptor("wineglass.n.01_1", "fragile", (0.03, 0.03), 0.14),
    ]
    pack = mod.build_clutter_pack("countertop.n.01_1", descriptors, seed=3)
    world_positions = {
        "coffee_cup.n.01_1": (0.0, 0.0, 0.9),
        "wineglass.n.01_1": (1.0, 1.0, 0.9),
    }
    report = mod.validate_pack_integrity(
        pack_spec=pack,
        world_positions=world_positions,
        pack_origin_world=(0.0, 0.0, 0.9),
        pack_yaw=0.0,
        tol_xy=0.02,
    )
    assert not report.ok
    assert len(report.failure_reasons) > 0
