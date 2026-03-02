import importlib.util
import sys
from pathlib import Path


def _load_module():
    mod_path = Path(__file__).resolve().parents[1] / "omnigibson" / "utils" / "franka_edge_align.py"
    spec = importlib.util.spec_from_file_location("franka_edge_align", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_select_best_table_edge_prefers_nearest():
    mod = _load_module()
    objects = (
        mod.EdgeAlignObject("target", "target", (-0.85, 0.10)),
        mod.EdgeAlignObject("fragile", "fragile", (-0.75, 0.04)),
    )
    edge = mod.select_best_table_edge(((-1.0, -0.5), (1.0, 0.5)), objects)
    assert edge == "x_min"


def test_compute_weighted_edge_anchor_biases_target():
    mod = _load_module()
    objects = (
        mod.EdgeAlignObject("target", "target", (0.0, 0.32)),
        mod.EdgeAlignObject("fragile", "fragile", (0.0, -0.25)),
        mod.EdgeAlignObject("clutter", "clutter", (0.0, -0.20)),
    )
    anchor = mod.compute_weighted_edge_anchor(
        edge_label="x_min",
        table_aabb_xy=((-1.0, -0.6), (1.0, 0.6)),
        pack_objects_world=objects,
        role_weights={"target": 3.0, "fragile": 2.0, "clutter": 1.0},
        robot_half_extent_xy=(0.24, 0.24),
        edge_margin_m=0.05,
    )
    assert anchor > 0.0


def test_place_franka_edge_aligned_scans_for_first_collision_free_candidate():
    mod = _load_module()
    objects = (
        mod.EdgeAlignObject("target", "target", (-0.85, 0.0)),
        mod.EdgeAlignObject("fragile", "fragile", (-0.75, 0.1)),
    )

    # Only rank 0 candidate collides.
    calls = {"count": 0}

    def checker(_pose):
        calls["count"] += 1
        return ("blocker",) if calls["count"] == 1 else ()

    result = mod.place_franka_edge_aligned(
        mod.EdgeAlignRequest(
            table_aabb_xy=((-1.0, -0.5), (1.0, 0.5)),
            pack_objects_world=objects,
            role_weights=mod.DEFAULT_ROLE_WEIGHTS,
            robot_half_extent_xy=(0.24, 0.24),
            edge_gap_m=0.03,
            edge_margin_m=0.05,
            scan_offsets_m=(0.0, 0.05, -0.05),
            collision_checker=checker,
        )
    )
    assert result.edge_label == "x_min"
    assert result.candidate_rank == 1
    assert result.collision_hits == ()
    assert abs(result.gap_actual - 0.03) < 1e-6


def test_place_franka_edge_aligned_is_deterministic():
    mod = _load_module()
    objects = (
        mod.EdgeAlignObject("target", "target", (0.2, -0.3)),
        mod.EdgeAlignObject("fragile", "fragile", (0.25, -0.1)),
        mod.EdgeAlignObject("clutter", "clutter", (0.1, -0.2)),
    )
    req = mod.EdgeAlignRequest(
        table_aabb_xy=((-1.0, -0.5), (1.0, 0.5)),
        pack_objects_world=objects,
        role_weights=mod.DEFAULT_ROLE_WEIGHTS,
        robot_half_extent_xy=(0.24, 0.24),
        edge_gap_m=0.03,
        edge_margin_m=0.05,
        scan_offsets_m=(0.0, 0.05, -0.05),
        collision_checker=lambda _pose: (),
    )
    res_a = mod.place_franka_edge_aligned(req)
    res_b = mod.place_franka_edge_aligned(req)
    assert res_a == res_b
