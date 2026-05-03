"""Tests for task_spec — LTL generation and object budget, no simulator needed."""

from sentinel.utils.task_spec import (
    compute_object_budget,
    generate_ltl_safety_json,
)


class TestGenerateLTLSafetyJSON:
    def test_standard_4_constraint_pattern(self):
        result = generate_ltl_safety_json(
            activity_name="test_task",
            fragile_synsets=["wineglass.n.01", "plate.n.04"],
            target_synsets=["coffee_cup.n.01"],
        )
        assert result["activity_name"] == "test_task"
        assert len(result["constraints"]) == 4
        assert "combined_ltl" in result
        assert len(result["propositions"]) == 4

    def test_constraint_ids(self):
        result = generate_ltl_safety_json(
            activity_name="test_task",
            fragile_synsets=["wineglass.n.01"],
            target_synsets=["coffee_cup.n.01"],
        )
        ids = {c["id"] for c in result["constraints"]}
        assert "no_fragile_dropped" in ids
        assert "no_fragile_tipped_over" in ids
        assert "target_not_dropped" in ids
        assert "target_upright" in ids

    def test_propositions_use_glob_patterns(self):
        result = generate_ltl_safety_json(
            activity_name="test_task",
            fragile_synsets=["wineglass.n.01"],
            target_synsets=["coffee_cup.n.01"],
        )
        assert "wineglass.n.01_*" in result["propositions"]["any_fragile_dropped"]["over"]
        assert "coffee_cup.n.01_*" in result["propositions"]["target_dropped"]["over"]

    def test_no_fragile_no_target(self):
        result = generate_ltl_safety_json(activity_name="empty_task")
        assert len(result["constraints"]) == 0
        assert result["combined_ltl"] == ""

    def test_fragile_only(self):
        result = generate_ltl_safety_json(
            activity_name="fragile_only",
            fragile_synsets=["wineglass.n.01"],
        )
        assert len(result["constraints"]) == 2

    def test_combined_ltl_parses(self):
        result = generate_ltl_safety_json(
            activity_name="test_task",
            fragile_synsets=["wineglass.n.01"],
            target_synsets=["coffee_cup.n.01"],
        )
        ltl = result["combined_ltl"]
        assert ltl.startswith("G (")
        assert "!any_fragile_dropped" in ltl
        assert "target_upright" in ltl


class TestComputeObjectBudget:
    def test_basic(self):
        catalog = [("cup", 0.01), ("glass", 0.015), ("plate", 0.02)]
        budget = compute_object_budget(zone_area=0.5, object_catalog=catalog)
        assert budget >= 1
        assert budget <= len(catalog)

    def test_zero_area(self):
        assert compute_object_budget(0.0, [("cup", 0.01)]) == 0

    def test_empty_catalog(self):
        assert compute_object_budget(1.0, []) == 0

    def test_respects_cap(self):
        catalog = [("big", 0.5)] * 10
        budget = compute_object_budget(zone_area=1.0, object_catalog=catalog, utilization_cap=0.85)
        assert budget <= 2


class TestBackwardCompatShim:
    """Verify that bddl_generator.py shim re-exports task_spec symbols."""

    def test_shim_exports_ltl_generator(self):
        from sentinel.utils.bddl_generator import generate_ltl_safety_json as fn
        assert fn is generate_ltl_safety_json

    def test_shim_exports_private_helpers(self):
        from sentinel.utils.bddl_generator import _make_spawn_spec
        spec = _make_spawn_spec("cup.n.01", 1, "target")
        assert spec["synset"] == "cup.n.01"
        assert spec["count"] == 1
        assert spec["role"] == "target"
