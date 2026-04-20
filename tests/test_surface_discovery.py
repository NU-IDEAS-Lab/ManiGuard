"""Tests for surface_discovery — pure geometry, no simulator needed."""

import pytest

from sentinel.utils.surface_discovery import (
    SurfaceAnalysis,
    SurfaceCandidate,
    SurfaceObstacle,
    analyze_surface,
    check_edge_reachability,
    compute_robot_placement_box,
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


class TestComputeRobotPlacementBox:
    def test_x_min_edge(self):
        surface = ((0.0, 0.0), (2.0, 1.0))
        box = compute_robot_placement_box("x_min", surface, robot_footprint_xy=(0.4, 0.4), edge_gap_m=0.03)
        (bx0, by0), (bx1, by1) = box
        # Robot should be entirely to the left of the surface.
        assert bx1 < 0.0
        # Robot box should be robot-sized, not table-sized.
        assert abs((bx1 - bx0) - 0.4) < 1e-9
        assert abs((by1 - by0) - 0.4) < 1e-9

    def test_x_max_edge(self):
        surface = ((0.0, 0.0), (2.0, 1.0))
        box = compute_robot_placement_box("x_max", surface, robot_footprint_xy=(0.4, 0.4), edge_gap_m=0.03)
        (bx0, by0), (bx1, by1) = box
        # Robot should be entirely to the right of the surface.
        assert bx0 > 2.0
        assert abs((bx1 - bx0) - 0.4) < 1e-9

    def test_y_min_edge(self):
        surface = ((0.0, 0.0), (2.0, 1.0))
        box = compute_robot_placement_box("y_min", surface, robot_footprint_xy=(0.4, 0.4), edge_gap_m=0.03)
        (bx0, by0), (bx1, by1) = box
        assert by1 < 0.0
        assert abs((by1 - by0) - 0.4) < 1e-9

    def test_y_max_edge(self):
        surface = ((0.0, 0.0), (2.0, 1.0))
        box = compute_robot_placement_box("y_max", surface, robot_footprint_xy=(0.4, 0.4), edge_gap_m=0.03)
        (bx0, by0), (bx1, by1) = box
        assert by0 > 1.0
        assert abs((by1 - by0) - 0.4) < 1e-9

    def test_tangent_offset(self):
        surface = ((0.0, 0.0), (2.0, 1.0))
        box_center = compute_robot_placement_box("x_min", surface, robot_footprint_xy=(0.4, 0.4))
        box_shifted = compute_robot_placement_box("x_min", surface, robot_footprint_xy=(0.4, 0.4), tangent_offset=0.2)
        # Shifted box should be 0.2m higher in Y.
        center_cy = 0.5 * (box_center[0][1] + box_center[1][1])
        shifted_cy = 0.5 * (box_shifted[0][1] + box_shifted[1][1])
        assert abs(shifted_cy - center_cy - 0.2) < 1e-9


class TestCheckEdgeReachability:
    def test_open_table_all_edges_reachable(self):
        surface = ((5.0, 5.0), (7.0, 6.0))
        # No nearby objects.
        reachable = check_edge_reachability(surface, scene_object_aabbs=[])
        assert len(reachable) == 4

    def test_wall_blocks_one_edge(self):
        surface = ((0.0, 0.0), (2.0, 1.0))
        # Wall flush against x_min edge (blocks robot placement on that side).
        wall = ((-0.5, -1.0), (-0.01, 2.0))
        reachable = check_edge_reachability(surface, scene_object_aabbs=[wall])
        assert "x_min" not in reachable
        assert "x_max" in reachable

    def test_corner_table_two_walls(self):
        surface = ((0.0, 0.0), (1.5, 1.0))
        # Walls on x_min and y_min.
        walls = [
            ((-0.5, -1.0), (-0.01, 2.0)),  # blocks x_min
            ((-1.0, -0.5), (3.0, -0.01)),   # blocks y_min
        ]
        reachable = check_edge_reachability(surface, scene_object_aabbs=walls)
        assert "x_min" not in reachable
        assert "y_min" not in reachable
        assert "x_max" in reachable
        assert "y_max" in reachable

    def test_fully_enclosed_no_edges(self):
        surface = ((1.0, 1.0), (2.0, 2.0))
        # Walls on all four sides, very close.
        walls = [
            ((0.5, 0.0), (0.99, 3.0)),   # blocks x_min
            ((2.01, 0.0), (2.5, 3.0)),    # blocks x_max
            ((0.0, 0.5), (3.0, 0.99)),    # blocks y_min
            ((0.0, 2.01), (3.0, 2.5)),    # blocks y_max
        ]
        reachable = check_edge_reachability(surface, scene_object_aabbs=walls)
        assert len(reachable) == 0


class TestRankApproachEdgesWithReachability:
    def test_filters_to_reachable_only(self):
        aabb = ((0.0, 0.0), (2.0, 1.0))
        edges = rank_approach_edges(aabb, reachable_edges=["x_max", "y_max"])
        assert set(edges) == {"x_max", "y_max"}

    def test_empty_reachable_returns_empty(self):
        aabb = ((0.0, 0.0), (2.0, 1.0))
        edges = rank_approach_edges(aabb, reachable_edges=[])
        assert edges == []


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

    def test_unreachable_surface_scores_zero(self):
        # Table completely enclosed by walls.
        surface = ((1.0, 1.0), (2.0, 2.0))
        walls = [
            ((0.5, 0.0), (0.99, 3.0)),
            ((2.01, 0.0), (2.5, 3.0)),
            ((0.0, 0.5), (3.0, 0.99)),
            ((0.0, 2.01), (3.0, 2.5)),
        ]
        scene_objects = [
            {"name": "wall_1", "category": "wall", "aabb_xy": walls[0], "top_z": 2.5},
        ]
        analysis = analyze_surface(
            "table_1", "table", surface, 0.85, scene_objects,
            scene_object_aabbs=walls,
        )
        assert analysis.surface.score == 0.0
        assert len(analysis.approach_edges) == 0

    def test_reachable_surface_has_filtered_edges(self):
        surface = ((0.0, 0.0), (2.0, 1.0))
        # Wall blocks x_min only.
        wall = ((-0.5, -1.0), (-0.01, 2.0))
        analysis = analyze_surface(
            "bar_1", "bar", surface, 0.85, [],
            scene_object_aabbs=[wall],
        )
        assert analysis.surface.score > 0
        assert "x_min" not in analysis.approach_edges
        assert len(analysis.approach_edges) >= 1
