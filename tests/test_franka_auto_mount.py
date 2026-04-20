import importlib.util
import sys
from pathlib import Path


def _load_module():
    mod_path = Path(__file__).resolve().parents[1] / "sentinel" / "utils" / "franka_auto_mount.py"
    spec = importlib.util.spec_from_file_location("franka_auto_mount", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_auto_mount_is_deterministic_for_same_seed():
    mod = _load_module()
    request = mod.AutoMountRequest(
        scene="Rs_int",
        target_object="coffee_cup.n.01_1",
        target_position=(0.0, 0.0, 0.0),
        seed=13,
        max_candidates=32,
    )

    res_a = mod.plan_franka_auto_mount(request)
    res_b = mod.plan_franka_auto_mount(request)

    assert res_a.best_pose == res_b.best_pose
    assert res_a.ranked_candidates[:5] == res_b.ranked_candidates[:5]


def test_auto_mount_reports_failure_reason_when_all_colliding():
    mod = _load_module()
    request = mod.AutoMountRequest(
        scene="Rs_int",
        target_object="coffee_cup.n.01_1",
        target_position=(0.0, 0.0, 0.0),
        seed=0,
        max_candidates=24,
        obstacles=(
            mod.AutoMountObstacle(name="obs0", position_xy=(0.0, 0.0), radius=2.0),
        ),
    )

    result = mod.plan_franka_auto_mount(request)

    assert result.best_pose is None
    assert result.failure_reason in {"all_candidates_in_collision", "no_feasible_candidates_after_scoring"}
    assert result.debug_metrics["num_feasible"] == 0.0


def test_auto_mount_returns_ranked_candidates():
    mod = _load_module()
    request = mod.AutoMountRequest(
        scene="Rs_int",
        target_object="coffee_cup.n.01_1",
        target_position=(1.0, -1.0, 0.0),
        seed=5,
        max_candidates=20,
    )

    result = mod.plan_franka_auto_mount(request)

    assert len(result.ranked_candidates) == 20
    assert result.ranked_candidates[0].score >= result.ranked_candidates[-1].score
