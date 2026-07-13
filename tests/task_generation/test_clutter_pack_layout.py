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
    mod_path = Path(__file__).resolve().parents[2] / "maniguard" / "utils" / "clutter_pack_layout.py"
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


def test_build_clutter_pack_changes_with_seed():
    mod = _load_module()
    descriptors = [
        mod.ClutterObjectDescriptor("coffee_cup.n.01_1", "target", (0.04, 0.04), 0.10),
        mod.ClutterObjectDescriptor("wineglass.n.01_1", "fragile", (0.03, 0.03), 0.14),
        mod.ClutterObjectDescriptor("wineglass.n.01_2", "fragile", (0.03, 0.03), 0.14),
        mod.ClutterObjectDescriptor("bowl.n.01_1", "clutter", (0.06, 0.06), 0.09),
    ]
    pack_a = mod.build_clutter_pack("countertop.n.01_1", descriptors, seed=7)
    pack_b = mod.build_clutter_pack("countertop.n.01_1", descriptors, seed=8)
    assert pack_a.object_entries != pack_b.object_entries


def test_target_is_placed_near_center():
    mod = _load_module()
    descriptors = [
        mod.ClutterObjectDescriptor("coffee_cup.n.01_1", "target", (0.04, 0.04), 0.10),
        mod.ClutterObjectDescriptor("wineglass.n.01_1", "fragile", (0.03, 0.03), 0.14),
        mod.ClutterObjectDescriptor("bowl.n.01_1", "clutter", (0.06, 0.06), 0.09),
    ]
    pack = mod.build_clutter_pack("countertop.n.01_1", descriptors, seed=19, jitter_xy=0.01)
    target_entry = [e for e in pack.object_entries if e.role == "target"][0]
    tx, ty = target_entry.rel_pose[0], target_entry.rel_pose[1]
    assert abs(tx) <= 0.02
    assert abs(ty) <= 0.02


def test_build_clutter_pack_raises_when_no_feasible_point():
    mod = _load_module()
    descriptors = [
        mod.ClutterObjectDescriptor("coffee_cup.n.01_1", "target", (0.06, 0.06), 0.10),
        mod.ClutterObjectDescriptor("wineglass.n.01_1", "fragile", (0.06, 0.06), 0.14),
        mod.ClutterObjectDescriptor("bowl.n.01_1", "clutter", (0.06, 0.06), 0.09),
    ]
    try:
        mod.build_clutter_pack(
            "countertop.n.01_1",
            descriptors,
            seed=0,
            min_clearance=0.05,
            placement_bounds_local=((-0.05, -0.05), (0.05, 0.05)),
            grid_step_m=0.005,
            frontier_noise_margin_m=0.01,
        )
        assert False, "Expected RuntimeError due to no feasible placement"
    except RuntimeError as e:
        assert "pack_no_feasible_point" in str(e)


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


def test_build_clutter_pack_uses_root_to_bottom_offset_for_z():
    mod = _load_module()
    descriptors = [
        mod.ClutterObjectDescriptor("wineglass.n.01_1", "fragile", (0.03, 0.03), 0.14, root_to_bottom_z=0.06),
    ]
    pack = mod.build_clutter_pack("countertop.n.01_1", descriptors, seed=1)
    entry = pack.object_entries[0]
    assert abs(entry.rel_pose[2] - 0.064) <= 1e-6


def test_check_packing_feasibility_accepts_loose():
    mod = _load_module()
    descriptors = [
        mod.ClutterObjectDescriptor("a", "target", (0.04, 0.04), 0.10),
        mod.ClutterObjectDescriptor("b", "clutter", (0.03, 0.03), 0.09),
    ]
    feasible, util = mod.check_packing_feasibility(
        descriptors, placement_bounds_local=((-0.45, -0.45), (0.45, 0.45)), min_clearance=0.02,
    )
    assert feasible
    assert 0.0 < util < 0.1


def test_check_packing_feasibility_rejects_tight():
    mod = _load_module()
    descriptors = [
        mod.ClutterObjectDescriptor("a", "target", (0.04, 0.04), 0.10),
        mod.ClutterObjectDescriptor("b", "clutter", (0.04, 0.04), 0.09),
        mod.ClutterObjectDescriptor("c", "clutter", (0.04, 0.04), 0.09),
    ]
    feasible, _ = mod.check_packing_feasibility(
        descriptors, placement_bounds_local=((-0.05, -0.05), (0.05, 0.05)), min_clearance=0.02,
    )
    assert not feasible


def test_compute_candidate_pool_returns_valid_positions():
    mod = _load_module()
    target = mod.ClutterObjectDescriptor("cup", "target", (0.04, 0.04), 0.10)
    fragile = mod.ClutterObjectDescriptor("glass", "fragile", (0.03, 0.03), 0.14)
    bounds = ((-0.20, -0.20), (0.20, 0.20))
    placed = [(target, 0.0, 0.0)]
    pool = mod.compute_candidate_pool(fragile, placed, bounds, min_clearance=0.02, noise_margin=0.03)
    assert len(pool) > 0
    # Every candidate must be collision-free with the placed target.
    import math
    for cx, cy in pool:
        dist = math.hypot(cx, cy)
        min_sep = 0.04 + 0.03 + 0.02  # target_r + fragile_r + clearance
        assert dist >= min_sep - 1e-6


def test_compute_candidate_pool_empty_placed():
    mod = _load_module()
    desc = mod.ClutterObjectDescriptor("cup", "target", (0.04, 0.04), 0.10)
    bounds = ((-0.20, -0.20), (0.20, 0.20))
    pool = mod.compute_candidate_pool(desc, placed=[], placement_bounds=bounds, min_clearance=0.02)
    assert len(pool) > 0
    # Centre (0, 0) should be in the pool.
    import math
    has_center = any(math.hypot(cx, cy) < 0.01 for cx, cy in pool)
    assert has_center


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
