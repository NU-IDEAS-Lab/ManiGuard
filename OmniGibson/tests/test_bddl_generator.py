"""Tests for bddl_generator — pure text generation, no simulator needed."""

import json
import os
import tempfile

import pytest

from omnigibson.utils.bddl_generator import (
    BDDLGenConfig,
    ObjectSpec,
    compute_object_budget,
    generate_bddl_problem,
    generate_ltl_safety_json,
    write_activity_files,
)


class TestGenerateBDDLProblem:
    def _make_config(self):
        """Default config uses grasped goal (no cabinet needed)."""
        return BDDLGenConfig(
            activity_name="test_task",
            support_synset="breakfast_table.n.01",
            support_room="living_room",
            goal_predicate="grasped",
            objects=[
                ObjectSpec(synset="coffee_cup.n.01", count=1, role="target"),
                ObjectSpec(synset="wineglass.n.01", count=3, role="fragile"),
                ObjectSpec(synset="plate.n.04", count=2, role="clutter"),
            ],
        )

    def _make_placement_config(self):
        """Placement goal config (inside cabinet) for backward compat."""
        return BDDLGenConfig(
            activity_name="test_task",
            support_synset="countertop.n.01",
            support_room="kitchen",
            goal_synset="cabinet.n.01",
            goal_room="kitchen",
            goal_predicate="inside",
            objects=[
                ObjectSpec(synset="coffee_cup.n.01", count=1, role="target"),
                ObjectSpec(synset="wineglass.n.01", count=3, role="fragile"),
                ObjectSpec(synset="plate.n.04", count=2, role="clutter"),
            ],
        )

    def test_produces_valid_pddl_structure(self):
        config = self._make_config()
        text = generate_bddl_problem(config)
        assert "(define (problem test_task-0)" in text
        assert "(:domain omnigibson)" in text
        assert "(:objects" in text
        assert "(:init" in text
        assert "(:goal" in text

    def test_objects_section(self):
        config = self._make_config()
        text = generate_bddl_problem(config)
        assert "coffee_cup.n.01_1 - coffee_cup.n.01" in text
        assert "wineglass.n.01_1 wineglass.n.01_2 wineglass.n.01_3 - wineglass.n.01" in text
        assert "plate.n.04_1 plate.n.04_2 - plate.n.04" in text

    def test_grasped_goal_includes_agent(self):
        config = self._make_config()
        text = generate_bddl_problem(config)
        assert "agent.n.01_1 - agent.n.01" in text
        assert "floor" not in text

    def test_grasped_goal_uses_target(self):
        config = self._make_config()
        text = generate_bddl_problem(config)
        assert "(grasped agent.n.01_1 coffee_cup.n.01_1)" in text

    def test_init_places_objects_on_support(self):
        config = self._make_config()
        text = generate_bddl_problem(config)
        assert "(ontop coffee_cup.n.01_1 breakfast_table.n.01_1)" in text
        assert "(ontop wineglass.n.01_1 breakfast_table.n.01_1)" in text

    def test_support_room(self):
        config = self._make_config()
        text = generate_bddl_problem(config)
        assert "(inroom breakfast_table.n.01_1 living_room)" in text

    def test_placement_goal_uses_inside(self):
        config = self._make_placement_config()
        text = generate_bddl_problem(config)
        assert "(inside coffee_cup.n.01_1 cabinet.n.01_1)" in text

    def test_placement_goal_rooms(self):
        config = self._make_placement_config()
        text = generate_bddl_problem(config)
        assert "(inroom countertop.n.01_1 kitchen)" in text
        assert "(inroom cabinet.n.01_1 kitchen)" in text

    def test_placement_goal_no_agent(self):
        config = self._make_placement_config()
        text = generate_bddl_problem(config)
        assert "agent.n.01" not in text


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


class TestWriteActivityFiles:
    def test_writes_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            activity_dir = os.path.join(tmpdir, "test_activity")
            bddl_text = "(define (problem test-0) (:domain omnigibson))"
            ltl_safety = {"activity_name": "test", "constraints": []}

            bddl_path, json_path = write_activity_files(activity_dir, bddl_text, ltl_safety)

            assert os.path.isfile(bddl_path)
            assert os.path.isfile(json_path)

            with open(bddl_path) as f:
                assert f.read() == bddl_text

            with open(json_path) as f:
                loaded = json.load(f)
                assert loaded["activity_name"] == "test"
