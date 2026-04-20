import importlib.util
import sys
from pathlib import Path


def _load_module():
    mod_path = Path(__file__).resolve().parents[1] / "sentinel" / "utils" / "kitchen_bar_workspace.py"
    spec = importlib.util.spec_from_file_location("kitchen_bar_workspace", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compute_kitchen_bar_zone_in_bar_and_outside_sink_keepout():
    mod = _load_module()
    bar = ((7.80, -1.05), (8.45, 1.30))
    sink = ((8.00, -0.10), (8.30, 0.45))
    zone = mod.compute_kitchen_bar_zone(bar, sink)

    # zone is fully inside bar
    assert zone.red_zone_bounds[0][0] >= zone.bar_bounds[0][0]
    assert zone.red_zone_bounds[0][1] >= zone.bar_bounds[0][1]
    assert zone.red_zone_bounds[1][0] <= zone.bar_bounds[1][0]
    assert zone.red_zone_bounds[1][1] <= zone.bar_bounds[1][1]

    # zone does not overlap keepout
    assert not mod.bounds_overlap(zone.red_zone_bounds, zone.sink_keepout_bounds, tol=1e-6)


def test_compute_kitchen_bar_zone_prefers_positive_long_axis_side():
    mod = _load_module()
    # Same geometry style as house_double_floor_lower: long axis on y.
    bar = ((7.80, -0.98), (8.42, 1.44))
    sink = ((7.85, -0.28), (8.37, 0.58))
    zone = mod.compute_kitchen_bar_zone(bar, sink)

    # For sink_left_v1 in this hardcoded pipeline, zone must sit on positive y side.
    assert zone.long_axis == "y"
    assert zone.red_zone_bounds[0][1] > zone.sink_keepout_bounds[1][1]


def test_compute_zone_capacity_reports_utilization():
    mod = _load_module()
    stats = mod.compute_zone_capacity(
        red_zone_bounds=((7.90, -0.90), (8.35, -0.35)),
        half_extents_xy=((0.04, 0.04), (0.03, 0.03), (0.06, 0.06)),
        per_object_padding=0.02,
    )
    assert stats.available_area > 0.0
    assert stats.required_area > 0.0
    assert 0.0 < stats.utilization < 1.0


def test_compute_kitchen_bar_zone_raises_if_keepout_blocks_zone():
    mod = _load_module()
    bar = ((0.0, 0.0), (0.6, 1.0))
    # Keepout blocks nearly all negative long-axis side after margins.
    sink = ((0.1, 0.0), (0.5, 0.8))

    try:
        mod.compute_kitchen_bar_zone(
            bar,
            sink,
            edge_margin_m=0.05,
            sink_keepout_margin_m=0.10,
            sink_side_clearance_m=0.05,
            min_zone_span_m=0.25,
        )
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "span" in str(e).lower() or "overlap" in str(e).lower()
