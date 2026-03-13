"""Tests for surface_discovery — pure geometry, no simulator needed."""

import pytest

from omnigibson.utils.surface_discovery import (
    SurfaceAnalysis,
    SurfaceCandidate,
    SurfaceObstacle,
    analyze_surface,
    detect_obstacles_on_surface,
    is_obstacle_like,
    is_table_like,
    rank_approach_edges,
    score_surface,
)


class TestIsTableLike:
    def test_known_categories(self):
        assert is_table_like("bar")
        assert is_table_like("coffee_table")
        assert is_table_like("dining_table")
        assert is_table_like("desk")
        assert is_table_like("counter")
        assert is_table_like("countertop")

    def test_non_table(self):
        assert not is_table_like("chair")
        assert not is_table_like("sofa")
        assert not is_table_like("lamp")

    def test_substring_match(self):
        assert is_table_like("kitchen_counter_large")
        assert is_table_like("folding_table")

    def test_case_insensitive(self):
        assert is_table_like("Bar")
        assert is_table_like("COUNTER")


class TestIsObstacleLike:
    def test_sink(self):
        assert is_obstacle_like("sink") == "sink"
        assert is_obstacle_like("drop_in_sink") == "sink"
        assert is_obstacle_like("kitchen_sink") == "sink"

    def test_stove(self):
        assert is_obstacle_like("stove") == "stove"
        assert is_obstacle_like("cooktop") == "stove"

    def test_non_obstacle(self):
        assert is_obstacle_like("cup") == ""
        assert is_obstacle_like("chair") == ""


class TestScoreSurface:
    def test_good_table(self):
        aabb = ((0.0, 0.0), (1.0, 0.8))
        score = score_surface(aabb, top_z=0.85, category="bar")
        assert score > 0.5

    def test_too_small(self):
        aabb = ((0.0, 0.0), (0.1, 0.1))
        score = score_surface(aabb, top_z=0.8, category="table")
        assert score == 0.0

    def test_too_high(self):
        aabb = ((0.0, 0.0), (1.0, 1.0))
        score = score_surface(aabb, top_z=2.0, category="table")
        assert score == 0.0

    def test_short_side_too_narrow(self):
        aabb = ((0.0, 0.0), (2.0, 0.1))
        score = score_surface(aabb, top_z=0.8, category="table")
        assert score == 0.0

    def test_category_bonus(self):
        aabb = ((0.0, 0.0), (1.0, 0.8))
        table_score = score_surface(aabb, top_z=0.85, category="bar")
        other_score = score_surface(aabb, top_z=0.85, category="shelf")
        assert table_score > other_score


class TestDetectObstacles:
    def test_finds_sink_on_surface(self):
        surface_aabb = ((0.0, 0.0), (2.0, 1.0))
        candidates = [
            {"name": "sink_1", "category": "drop_in_sink", "aabb_xy": ((0.2, 0.3), (0.6, 0.7)), "top_z": 0.85},
            {"name": "cup_1", "category": "cup", "aabb_xy": ((1.0, 0.5), (1.1, 0.6)), "top_z": 0.85},
        ]
        obstacles = detect_obstacles_on_surface(surface_aabb, 0.85, candidates)
        assert len(obstacles) == 1
        assert obstacles[0].name == "sink_1"
        assert obstacles[0].obstacle_type == "sink"

    def test_ignores_far_z(self):
        surface_aabb = ((0.0, 0.0), (2.0, 1.0))
        candidates = [
            {"name": "sink_1", "category": "sink", "aabb_xy": ((0.2, 0.3), (0.6, 0.7)), "top_z": 2.0},
        ]
        obstacles = detect_obstacles_on_surface(surface_aabb, 0.85, candidates)
        assert len(obstacles) == 0

    def test_ignores_non_overlapping(self):
        surface_aabb = ((0.0, 0.0), (1.0, 1.0))
        candidates = [
            {"name": "sink_1", "category": "sink", "aabb_xy": ((5.0, 5.0), (6.0, 6.0)), "top_z": 0.85},
        ]
        obstacles = detect_obstacles_on_surface(surface_aabb, 0.85, candidates)
        assert len(obstacles) == 0


class TestRankApproachEdges:
    def test_no_blockers_long_y(self):
        aabb = ((0.0, 0.0), (1.0, 2.0))
        edges = rank_approach_edges(aabb)
        assert edges[0] in ("x_min", "x_max")

    def test_no_blockers_long_x(self):
        aabb = ((0.0, 0.0), (2.0, 1.0))
        edges = rank_approach_edges(aabb)
        assert edges[0] in ("y_min", "y_max")

    def test_wall_blocks_one_edge(self):
        aabb = ((0.0, 0.0), (1.0, 1.0))
        walls = [((1.05, -1.0), (1.1, 2.0))]  # Wall right next to x_max
        edges = rank_approach_edges(aabb, wall_aabbs=walls)
        assert edges[-1] == "x_max"


class TestAnalyzeSurface:
    def test_basic(self):
        scene_objects = [
            {"name": "sink_1", "category": "sink", "aabb_xy": ((0.2, 0.3), (0.5, 0.6)), "top_z": 0.85},
        ]
        analysis = analyze_surface("bar_1", "bar", ((0.0, 0.0), (2.0, 1.0)), 0.85, scene_objects)
        assert isinstance(analysis, SurfaceAnalysis)
        assert analysis.surface.score > 0
        assert len(analysis.obstacles) == 1
        assert analysis.free_area > 0
        assert len(analysis.approach_edges) == 4
