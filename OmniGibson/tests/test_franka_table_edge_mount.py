import importlib.util
import sys
from pathlib import Path


def _load_module():
    mod_path = Path(__file__).resolve().parents[1] / "omnigibson" / "utils" / "franka_table_edge_mount.py"
    spec = importlib.util.spec_from_file_location("franka_table_edge_mount", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_table_edge_mount_returns_feasible_edge_contact_pose():
    mod = _load_module()
    request = mod.TableEdgeMountRequest(
        target_position=(-0.95, 0.05, 0.9),
        table_aabb_xy=((-1.0, -0.5), (1.0, 0.5)),
        desired_gap=0.03,
        reachable_distance_range=(0.05, 1.20),
    )

    result = mod.plan_table_edge_mount(request)
    assert result.best_pose is not None
    best = result.ranked_candidates[0]
    assert best.feasible
    assert best.gap_ok
    assert best.edge_label in {"x_min", "x_max", "y_min", "y_max"}


def test_table_edge_mount_reports_collision_failure():
    mod = _load_module()
    request = mod.TableEdgeMountRequest(
        target_position=(0.0, 0.0, 0.9),
        table_aabb_xy=((-0.6, -0.4), (0.6, 0.4)),
        desired_gap=0.03,
        obstacles=(
            mod.TableEdgeMountObstacle(
                name="big_blocker",
                center_xy=(0.0, 0.0),
                radius=5.0,
            ),
        ),
    )

    result = mod.plan_table_edge_mount(request)
    assert result.best_pose is None
    assert result.failure_reason in {"all_candidates_in_collision", "no_feasible_candidates_after_scoring"}


def test_table_edge_mount_gap_constraint_is_respected():
    mod = _load_module()
    request = mod.TableEdgeMountRequest(
        target_position=(0.1, 0.25, 0.9),
        table_aabb_xy=((-0.8, -0.6), (0.8, 0.6)),
        desired_gap=0.04,
        gap_tolerance=0.01,
    )

    result = mod.plan_table_edge_mount(request)
    assert result.best_pose is not None
    best = result.ranked_candidates[0]
    assert best.gap_ok
    assert best.gap_error <= request.gap_tolerance


def test_table_edge_mount_is_deterministic():
    mod = _load_module()
    request = mod.TableEdgeMountRequest(
        target_position=(0.35, -0.1, 0.85),
        table_aabb_xy=((-0.7, -0.5), (0.7, 0.5)),
        desired_gap=0.03,
    )

    res_a = mod.plan_table_edge_mount(request)
    res_b = mod.plan_table_edge_mount(request)
    assert res_a.best_pose == res_b.best_pose
    assert res_a.ranked_candidates[:5] == res_b.ranked_candidates[:5]
