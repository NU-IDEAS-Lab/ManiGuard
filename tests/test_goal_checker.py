"""Unit tests for maniguard.eval.goal_checker.

Tests the parsing and evaluation logic without OmniGibson — predicates
are mocked so the test suite runs in any Python env (no GPU / Isaac Sim).
"""

import pytest
from unittest.mock import MagicMock, patch

from maniguard.eval.goal_checker import (
    GoalChecker,
    GoalRegionChecker,
    build_goal_checker,
    _collect_names,
    _eval_node,
)
from maniguard.utils.goal_region import GoalRegionSpec


# ---------------------------------------------------------------------------
# Helpers: mock OmniGibson objects with .states[Predicate].get_value(other)
# ---------------------------------------------------------------------------

def _make_mock_obj(name, state_values=None):
    """Create a mock OG object with controllable state returns."""
    obj = MagicMock()
    obj.name = name
    # state_values: {(predicate_class_name, other_obj_name): bool}
    state_values = state_values or {}

    def _make_state(pred_name):
        state = MagicMock()
        def get_value(other):
            return state_values.get((pred_name, other.name), False)
        state.get_value = get_value
        return state

    # Build states dict that maps state classes to mock state objects.
    # We intercept __getitem__ so obj.states[Inside] works.
    states = MagicMock()
    def states_getitem(cls):
        return _make_state(cls.__name__)
    states.__getitem__ = states_getitem
    obj.states = states
    return obj


# ---------------------------------------------------------------------------
# Tests: _collect_names
# ---------------------------------------------------------------------------

class TestCollectNames:
    def test_flat_list(self):
        conds = [
            {"predicate": "inside", "subject": "potato_1", "reference": "pot_2"},
            {"predicate": "grasping", "subject": "robot", "reference": "cup_3"},
        ]
        assert _collect_names(conds) == {"potato_1", "pot_2", "robot", "cup_3"}

    def test_compound_and(self):
        conds = {
            "op": "and",
            "terms": [
                {"predicate": "inside", "subject": "a", "reference": "b"},
                {"predicate": "ontop", "subject": "c", "reference": "d"},
            ]
        }
        assert _collect_names(conds) == {"a", "b", "c", "d"}

    def test_compound_not(self):
        conds = {
            "op": "not",
            "term": {"predicate": "touching", "subject": "x", "reference": "y"},
        }
        assert _collect_names(conds) == {"x", "y"}

    def test_nested(self):
        conds = {
            "op": "and",
            "terms": [
                {"predicate": "inside", "subject": "a", "reference": "b"},
                {"op": "not", "term": {"predicate": "touching", "subject": "c", "reference": "d"}},
            ]
        }
        assert _collect_names(conds) == {"a", "b", "c", "d"}

    def test_empty(self):
        assert _collect_names([]) == set()
        assert _collect_names({}) == set()


# ---------------------------------------------------------------------------
# Tests: _eval_node (with mocked predicate evaluation)
# ---------------------------------------------------------------------------

class TestEvalNode:
    def _mock_objects(self, names):
        return {n: _make_mock_obj(n) for n in names}

    @patch("maniguard.eval.goal_checker._eval_predicate")
    def test_flat_list_all_true(self, mock_pred):
        mock_pred.return_value = True
        objects = self._mock_objects(["a", "b", "c"])
        node = [
            {"predicate": "inside", "subject": "a", "reference": "b"},
            {"predicate": "ontop", "subject": "a", "reference": "c"},
        ]
        ok, detail = _eval_node(node, objects, None)
        assert ok is True
        assert all(detail.values())

    @patch("maniguard.eval.goal_checker._eval_predicate")
    def test_flat_list_one_false(self, mock_pred):
        mock_pred.side_effect = [True, False]
        objects = self._mock_objects(["a", "b", "c"])
        node = [
            {"predicate": "inside", "subject": "a", "reference": "b"},
            {"predicate": "ontop", "subject": "a", "reference": "c"},
        ]
        ok, detail = _eval_node(node, objects, None)
        assert ok is False

    @patch("maniguard.eval.goal_checker._eval_predicate")
    def test_compound_or(self, mock_pred):
        mock_pred.side_effect = [False, True]
        objects = self._mock_objects(["a", "b", "c"])
        node = {
            "op": "or",
            "terms": [
                {"predicate": "inside", "subject": "a", "reference": "b"},
                {"predicate": "inside", "subject": "a", "reference": "c"},
            ]
        }
        ok, detail = _eval_node(node, objects, None)
        assert ok is True

    @patch("maniguard.eval.goal_checker._eval_predicate")
    def test_compound_not_true(self, mock_pred):
        mock_pred.return_value = False  # predicate is false → NOT is true
        objects = self._mock_objects(["a", "b"])
        node = {
            "op": "not",
            "term": {"predicate": "touching", "subject": "a", "reference": "b"},
        }
        ok, detail = _eval_node(node, objects, None)
        assert ok is True

    @patch("maniguard.eval.goal_checker._eval_predicate")
    def test_compound_not_false(self, mock_pred):
        mock_pred.return_value = True  # predicate is true → NOT is false
        objects = self._mock_objects(["a", "b"])
        node = {
            "op": "not",
            "term": {"predicate": "touching", "subject": "a", "reference": "b"},
        }
        ok, detail = _eval_node(node, objects, None)
        assert ok is False

    @patch("maniguard.eval.goal_checker._eval_predicate")
    def test_nested_and_not(self, mock_pred):
        # (inside a b) AND NOT(touching c d)
        mock_pred.side_effect = [True, False]  # inside=True, touching=False
        objects = self._mock_objects(["a", "b", "c", "d"])
        node = {
            "op": "and",
            "terms": [
                {"predicate": "inside", "subject": "a", "reference": "b"},
                {"op": "not", "term": {"predicate": "touching", "subject": "c", "reference": "d"}},
            ]
        }
        ok, detail = _eval_node(node, objects, None)
        assert ok is True

    def test_missing_object(self):
        objects = {"a": _make_mock_obj("a")}  # "b" missing
        node = [{"predicate": "inside", "subject": "a", "reference": "b"}]
        ok, detail = _eval_node(node, objects, None)
        assert ok is False

    def test_empty_list(self):
        ok, detail = _eval_node([], {}, None)
        assert ok is False
        assert detail == {}


# ---------------------------------------------------------------------------
# Tests: build_goal_checker
# ---------------------------------------------------------------------------

class TestBuildGoalChecker:
    def test_with_conditions(self):
        scene_info = {
            "goal_conditions": [
                {"predicate": "inside", "subject": "potato_1", "reference": "pot_2"}
            ]
        }
        gc = build_goal_checker(scene_info)
        assert gc is not None
        assert gc.raw_conditions == scene_info["goal_conditions"]

    def test_empty_conditions(self):
        assert build_goal_checker({"goal_conditions": []}) is None
        assert build_goal_checker({}) is None

    def test_compound_conditions(self):
        scene_info = {
            "goal_conditions": {
                "op": "and",
                "terms": [
                    {"predicate": "inside", "subject": "a", "reference": "b"},
                    {"op": "not", "term": {"predicate": "touching", "subject": "c", "reference": "d"}},
                ]
            }
        }
        gc = build_goal_checker(scene_info)
        assert gc is not None

    def test_goal_region_takes_precedence(self):
        scene_info = {
            "goal_region": {
                "mode": "held_intersection",
                "shape": "sphere",
                "family": "table",
                "target_name": "cup_1",
                "support_name": "desk_1",
                "marker_name": "goal_region__cup_1",
                "center_world": [0.0, 0.0, 0.0],
                "radius_m": 0.1,
                "color_rgba": [0.1, 0.8, 0.2, 0.35],
                "target_width_m": 0.1,
                "anchor_local_xy": [0.0, 0.2],
                "pack_bbox_robot_local_xy": [[0.0, 0.0], [0.1, 0.1]],
                "support_bounds_robot_local_xy": [[-1.0, -1.0], [1.0, 1.0]],
                "clamped_to_support_bounds": False,
            },
            "goal_conditions": [{"predicate": "inside", "subject": "a", "reference": "b"}],
        }
        gc = build_goal_checker(scene_info)
        assert isinstance(gc, GoalRegionChecker)


# ---------------------------------------------------------------------------
# Tests: GoalChecker.check (integration with mock env)
# ---------------------------------------------------------------------------

class TestGoalCheckerIntegration:
    def _mock_env(self, obj_names, robot_name="robot"):
        env = MagicMock()
        robot = _make_mock_obj(robot_name)
        robot.contact_list = MagicMock(return_value=[])
        env.robots = [robot]

        objects = {robot_name: robot}
        for name in obj_names:
            objects[name] = _make_mock_obj(name)

        def object_registry(key, name):
            return objects.get(name)
        env.scene.object_registry = object_registry
        return env, objects

    @patch("maniguard.eval.goal_checker._eval_predicate")
    def test_check_success(self, mock_pred):
        mock_pred.return_value = True
        env, _ = self._mock_env(["potato_1", "pot_2"])
        gc = GoalChecker(raw_conditions=[
            {"predicate": "inside", "subject": "potato_1", "reference": "pot_2"}
        ])
        gc.resolve(env)
        ok, detail = gc.check(env)
        assert ok is True

    @patch("maniguard.eval.goal_checker._eval_predicate")
    def test_check_failure(self, mock_pred):
        mock_pred.return_value = False
        env, _ = self._mock_env(["potato_1", "pot_2"])
        gc = GoalChecker(raw_conditions=[
            {"predicate": "inside", "subject": "potato_1", "reference": "pot_2"}
        ])
        gc.resolve(env)
        ok, detail = gc.check(env)
        assert ok is False


class TestGoalRegionChecker:
    @patch("maniguard.eval.goal_checker.object_intersects_goal_region")
    @patch("maniguard.eval.goal_checker.robot_holds_target")
    def test_held_intersection_success(self, mock_holds, mock_intersects):
        mock_holds.return_value = True
        mock_intersects.return_value = True
        spec = GoalRegionSpec(
            mode="held_intersection",
            shape="sphere",
            family="table",
            target_name="cup_1",
            support_name="desk_1",
            marker_name="goal_region__cup_1",
            center_world=(0.0, 0.0, 0.0),
            radius_m=0.1,
            color_rgba=(0.1, 0.8, 0.2, 0.35),
            target_width_m=0.1,
            anchor_local_xy=(0.0, 0.2),
            pack_bbox_robot_local_xy=((0.0, 0.0), (0.1, 0.1)),
            support_bounds_robot_local_xy=((-1.0, -1.0), (1.0, 1.0)),
            clamped_to_support_bounds=False,
        )
        checker = GoalRegionChecker(raw_region=spec)
        env, objects = TestGoalCheckerIntegration()._mock_env(["cup_1", "goal_region__cup_1"])
        checker.resolve(env)
        ok, detail = checker.check(env)
        assert ok is True
        assert detail["held"] is True
        assert detail["intersects"] is True
