import math

import numpy as np

from omnigibson.task_generation.support_surface_profiles import (
    bounds_area,
    choose_profile_region,
    connected_components,
    ensure_support_surface_profiles_document,
    evaluate_region_reachability,
    get_support_surface_profile,
    largest_true_rectangle,
    load_support_surface_profiles,
    make_empty_support_surface_profiles_document,
    make_profile_status,
    mask_to_usable_regions,
    maximal_rectangle_cover,
    profile_generation_area_m2,
    profile_regions_to_world,
    region_bounds_xy,
    save_support_surface_profiles,
    select_dominant_top_plane_height,
    set_support_surface_profile,
)


class TestSupportSurfaceProfileDocument:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "profiles.json"
        doc = make_empty_support_surface_profiles_document()
        entry = {
            "category": "desk",
            "model": "demo",
            "top_plane_z_local": 0.75,
            "usable_regions": [],
            "candidate_for_generation": False,
            "review_status": "auto_pending",
        }
        doc = set_support_surface_profile(doc, "desk", "demo", entry)
        save_support_surface_profiles(doc, str(path))
        loaded = load_support_surface_profiles(str(path), use_cache=False)
        assert get_support_surface_profile(loaded, "desk", "demo")["model"] == "demo"

    def test_profile_status(self):
        assert make_profile_status(None) == "missing"
        assert make_profile_status({"review_status": "rejected"}) == "rejected"
        assert make_profile_status({"candidate_for_generation": False, "usable_regions": []}) == "non_candidate"
        assert make_profile_status({"candidate_for_generation": True, "usable_regions": [{"region_id": "r0"}]}) == "available"

    def test_ensure_document_merges_nested_defaults(self):
        doc = ensure_support_surface_profiles_document(
            {
                "version": "support_surface_profiles_v1",
                "frame_convention": {"type": "object_local_top_plane_xy"},
                "generator_defaults": {"region_shape": "rect"},
                "profiles": {},
            }
        )
        assert doc["generator_defaults"]["review_status"] == "auto_pending"
        assert doc["frame_convention"]["xy_units"] == "m"


class TestDominantTopPlane:
    def test_selects_highest_dense_band(self):
        heights = [0.72] * 30 + [0.74] * 3 + [0.40] * 40
        top_z, count = select_dominant_top_plane_height(heights, band_size_m=0.02, min_count_ratio=0.2)
        assert abs(top_z - 0.72) < 1e-6
        assert count == 30


class TestMaskGeometry:
    def test_connected_components(self):
        mask = np.array(
            [
                [1, 1, 0, 0],
                [1, 0, 0, 1],
                [0, 0, 0, 1],
            ],
            dtype=bool,
        )
        components = connected_components(mask)
        sizes = sorted(len(c) for c in components)
        assert sizes == [2, 3]

    def test_l_shape_rect_cover(self):
        mask = np.array(
            [
                [1, 1, 1, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=bool,
        )
        x_edges = [0.0, 0.1, 0.2, 0.3, 0.4]
        y_edges = [0.0, 0.1, 0.2, 0.3]
        regions = mask_to_usable_regions(mask, x_edges, y_edges, min_component_cells=1, min_region_area_m2=0.0)
        assert len(regions) == 2
        total_area = sum(region["area_m2"] for region in regions)
        assert abs(total_area - 0.05) < 1e-9
        for region in regions:
            assert region["shape"] == "rect"

    def test_largest_true_rectangle_finds_center_block(self):
        mask = np.array(
            [
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
            ],
            dtype=bool,
        )
        assert largest_true_rectangle(mask) == (0, 4, 1, 4)

    def test_maximal_rectangle_cover_splits_l_shape(self):
        mask = np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 1],
            ],
            dtype=bool,
        )
        cover = maximal_rectangle_cover(mask)
        assert cover == [(0, 3, 0, 2), (2, 3, 2, 4)]

    def test_bounds_area(self):
        assert abs(bounds_area(((0.0, 0.0), (0.5, 0.25))) - 0.125) < 1e-9


class TestReachability:
    def test_near_edge_region_reachable(self):
        reach = evaluate_region_reachability(
            ((0.20, 0.20), (0.50, 0.60)),
            ((0.0, 0.0), (1.2, 0.9)),
        )
        assert reach["reachable"] is True
        assert "x_min" in reach["edge_labels"] or "y_min" in reach["edge_labels"]

    def test_center_of_huge_table_not_reachable(self):
        reach = evaluate_region_reachability(
            ((1.6, 1.6), (2.4, 2.4)),
            ((0.0, 0.0), (4.0, 4.0)),
        )
        assert reach["reachable"] is False


class TestRegionSelection:
    def test_profile_generation_area_returns_largest_region(self):
        profile = {
            "usable_regions": [
                {
                    "region_id": "region_00",
                    "xy_min": [0.0, 0.0],
                    "xy_max": [0.5, 0.4],
                    "area_m2": 0.20,
                },
                {
                    "region_id": "region_01",
                    "xy_min": [0.0, 0.0],
                    "xy_max": [0.3, 0.4],
                    "area_m2": 0.12,
                },
            ]
        }
        assert abs(profile_generation_area_m2(profile) - 0.20) < 1e-9

    def test_profile_generation_area_ignores_too_narrow_regions(self):
        profile = {
            "usable_regions": [
                {
                    "region_id": "region_00",
                    "xy_min": [0.0, 0.0],
                    "xy_max": [0.04, 1.10],
                    "area_m2": 0.044,
                },
                {
                    "region_id": "region_01",
                    "xy_min": [0.0, 0.0],
                    "xy_max": [0.45, 0.35],
                    "area_m2": 0.1575,
                },
            ]
        }
        assert abs(profile_generation_area_m2(profile, min_span_xy_m=(0.28, 0.28)) - 0.1575) < 1e-9

    def test_choose_profile_region_uses_largest_sized_region(self):
        profile = {
            "usable_regions": [
                {
                    "region_id": "region_00",
                    "xy_min": [0.0, 0.0],
                    "xy_max": [0.4, 0.4],
                    "area_m2": 0.16,
                },
                {
                    "region_id": "region_01",
                    "xy_min": [0.5, 0.0],
                    "xy_max": [0.9, 0.3],
                    "area_m2": 0.12,
                },
                {
                    "region_id": "region_02",
                    "xy_min": [1.0, 0.0],
                    "xy_max": [1.7, 0.5],
                    "area_m2": 0.35,
                },
            ]
        }
        chosen = choose_profile_region(profile, required_area_m2=0.13)
        assert chosen["region_id"] == "region_02"

    def test_choose_profile_region_skips_narrow_strip(self):
        profile = {
            "usable_regions": [
                {
                    "region_id": "region_00",
                    "xy_min": [0.0, 0.0],
                    "xy_max": [0.04, 1.00],
                    "area_m2": 0.04,
                },
                {
                    "region_id": "region_01",
                    "xy_min": [0.10, 0.10],
                    "xy_max": [0.55, 0.55],
                    "area_m2": 0.2025,
                },
            ]
        }
        chosen = choose_profile_region(profile, min_span_xy_m=(0.28, 0.28))
        assert chosen["region_id"] == "region_01"

    def test_choose_profile_region_falls_back_to_any_region_when_needed(self):
        profile = {
            "usable_regions": [
                {
                    "region_id": "region_00",
                    "xy_min": [0.0, 0.0],
                    "xy_max": [0.3, 0.3],
                    "area_m2": 0.09,
                },
                {
                    "region_id": "region_01",
                    "xy_min": [0.4, 0.0],
                    "xy_max": [1.0, 0.4],
                    "area_m2": 0.24,
                },
            ]
        }
        chosen = choose_profile_region(profile, required_area_m2=0.20)
        assert chosen["region_id"] == "region_01"


class TestWorldTransforms:
    def test_regions_to_world_with_rotation(self):
        profile = {
            "top_plane_z_local": 0.75,
            "usable_regions": [
                {
                    "region_id": "region_00",
                    "shape": "rect",
                    "xy_min": [0.0, 0.0],
                    "xy_max": [0.4, 0.2],
                    "area_m2": 0.08,
                }
            ],
        }
        yaw = math.pi / 2.0
        quat = [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
        regions_world = profile_regions_to_world(
            profile,
            position_xyz_world=[1.0, 2.0, 0.0],
            orientation_xyzw_world=quat,
            scale_xyz=[1.0, 1.0, 1.0],
        )
        assert len(regions_world) == 1
        region = regions_world[0]
        (x0, y0), (x1, y1) = region["world_bounds_xy"]
        assert x0 < x1 and y0 < y1
        assert abs(region["area_m2"] - 0.08) < 1e-9
        assert abs(region["top_plane_z_world"] - 0.75) < 1e-9
        assert region["world_span_xy_m"] == [0.2, 0.4]
        assert region["reachable_edge_labels"] == []
        assert region_bounds_xy(region, world=True) == ((x0, y0), (x1, y1))

    def test_regions_to_world_respects_scale_for_span_and_area(self):
        profile = {
            "top_plane_z_local": 0.5,
            "usable_regions": [
                {
                    "region_id": "region_00",
                    "shape": "rect",
                    "xy_min": [0.0, 0.0],
                    "xy_max": [0.4, 0.2],
                    "area_m2": 0.08,
                }
            ],
        }
        regions_world = profile_regions_to_world(
            profile,
            position_xyz_world=[1.0, 2.0, 0.0],
            orientation_xyzw_world=[0.0, 0.0, 0.0, 1.0],
            scale_xyz=[2.0, 3.0, 1.0],
        )
        region = regions_world[0]
        assert region["world_span_xy_m"] == [0.8, 0.6]
        assert abs(region["area_m2"] - 0.48) < 1e-9

    def test_choose_profile_region_world_filter_uses_world_scaled_span(self):
        profile = {
            "usable_regions": [
                {
                    "region_id": "region_00",
                    "xy_min": [0.0, 0.0],
                    "xy_max": [0.4, 0.2],
                    "area_m2": 0.08,
                }
            ],
        }
        world_regions = profile_regions_to_world(
            {"top_plane_z_local": 0.0, "usable_regions": profile["usable_regions"]},
            position_xyz_world=[0.0, 0.0, 0.0],
            orientation_xyzw_world=[0.0, 0.0, 0.0, 1.0],
            scale_xyz=[2.0, 3.0, 1.0],
        )
        chosen = choose_profile_region(
            profile,
            world_regions=world_regions,
            required_area_m2=0.4,
            min_span_xy_m=(0.6, 0.7),
        )
        assert chosen is not None
        assert chosen["region_id"] == "region_00"
