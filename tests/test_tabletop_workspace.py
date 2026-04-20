import importlib.util
import math
import sys
from pathlib import Path


def _load_module():
    mod_path = Path(__file__).resolve().parents[1] / "sentinel" / "utils" / "tabletop_workspace.py"
    spec = importlib.util.spec_from_file_location("tabletop_workspace", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compute_zone_capacity_reports_utilization():
    mod = _load_module()
    stats = mod.compute_zone_capacity(
        red_zone_bounds=((0.0, 0.0), (0.60, 0.50)),
        half_extents_xy=((0.04, 0.04), (0.03, 0.03), (0.06, 0.06)),
        per_object_padding=0.02,
    )
    assert stats.available_area > 0.0
    assert stats.required_area > 0.0
    assert 0.0 < stats.utilization < 1.0


def test_compute_tabletop_zone_selects_largest_remaining_region_around_resident_obstacle():
    mod = _load_module()
    surface = ((0.0, 0.0), (1.0, 2.0))
    laptop = ((0.05, 0.05), (0.35, 0.65))
    zone = mod.compute_tabletop_zone(
        surface_bounds_xy=surface,
        obstacle_bounds_seq=(laptop,),
        edge_margin_m=0.05,
        obstacle_keepout_margin_m=0.05,
        obstacle_side_clearance_m=0.02,
        min_zone_span_m=0.20,
    )

    assert zone.red_zone_bounds[0][1] >= 0.72
    assert zone.red_zone_bounds[1][1] <= 1.95
    assert zone.red_zone_bounds[0][0] >= 0.05
    assert zone.red_zone_bounds[1][0] <= 0.95
    assert len(zone.obstacle_keepout_bounds_seq) == 1


def test_bounds_overlap_respects_tolerance():
    mod = _load_module()
    a = ((0.0, 0.0), (1.0, 1.0))
    b = ((1.001, 0.0), (2.0, 1.0))
    assert not mod.bounds_overlap(a, b, tol=1e-4)
    assert mod.bounds_overlap(a, b, tol=0.01)


def test_normalize_bounds_sorts_corners():
    mod = _load_module()
    bounds = mod.normalize_bounds(((2.0, 3.0), (-1.0, 1.5)))
    assert bounds == ((-1.0, 1.5), (2.0, 3.0))


def test_compute_tabletop_zone_reports_capacity_stats():
    mod = _load_module()
    zone = mod.compute_tabletop_zone(
        surface_bounds_xy=((0.0, 0.0), (1.2, 0.8)),
        object_half_extents_xy=((0.05, 0.05), (0.04, 0.04)),
        edge_margin_m=0.05,
    )
    assert zone.capacity_stats is not None
    assert zone.capacity_stats.available_area > 0.0
    assert math.isfinite(zone.capacity_stats.utilization)
