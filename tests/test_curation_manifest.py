import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


def _load_module():
    mod_path = (
        Path(__file__).resolve().parents[1] / "sentinel" / "task_generation"
        / "curation"
        / "curation_manifest.py"
    )
    spec = importlib.util.spec_from_file_location("curation_manifest", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_merges_defaults_and_resolves_snapshot(tmp_path):
    mod = _load_module()
    run_dir = tmp_path / "benchmark_20260319_170914"
    scene_dir = run_dir / "office_cubicles_left"
    scene_dir.mkdir(parents=True)
    snapshot = scene_dir / "scene_ep2.json"
    snapshot.write_text("{}", encoding="utf-8")

    manifest_path = tmp_path / "curation_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "benchmark_run_dir": str(run_dir),
                "activity_prefix": "auto_clutter_on",
                "defaults": {
                    "canonical_episode": 1,
                    "repair_mode": "snapshot_first",
                },
                "scenes": {
                    "office_cubicles_left": {
                        "status": "repair",
                        "canonical_episode": 2,
                        "surface_name": "coffee_table_mtchfd_0",
                        "issue_tags": ["surface_too_large", "reachability"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = mod.load_curation_manifest(str(manifest_path))
    entry = manifest.get_scene_entry("office_cubicles_left")

    assert entry.status == "repair"
    assert entry.surface_name == "coffee_table_mtchfd_0"
    assert entry.repair_mode == "snapshot_first"
    assert entry.resolve_snapshot_path() == str(snapshot)


def test_manifest_resolves_repo_relative_benchmark_run_dir_from_nested_manifest(tmp_path):
    mod = _load_module()
    repo_root = tmp_path
    run_dir = repo_root / "outputs" / "benchmark_runs" / "benchmark_20260319_170914"
    scene_dir = run_dir / "gates_bedroom"
    scene_dir.mkdir(parents=True)
    snapshot = scene_dir / "scene_ep1.json"
    snapshot.write_text("{}", encoding="utf-8")

    manifest_dir = run_dir
    manifest_path = manifest_dir / "curation_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "benchmark_run_dir": "outputs/benchmark_runs/benchmark_20260319_170914",
                "scenes": {
                    "gates_bedroom": {
                        "status": "repair",
                        "repair_mode": "rerender_only",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    old_cwd = Path.cwd()
    try:
        import os

        os.chdir(repo_root)
        manifest = mod.load_curation_manifest(str(manifest_path))
    finally:
        os.chdir(old_cwd)

    entry = manifest.get_scene_entry("gates_bedroom")

    assert entry.benchmark_run_dir == str(run_dir.resolve())
    assert entry.resolve_snapshot_path() == str(snapshot.resolve())


def test_apply_scene_entry_to_args_sets_runtime_overrides(tmp_path):
    mod = _load_module()
    entry = mod.SceneCurationEntry(
        scene_model="grocery_store_asian",
        benchmark_run_dir=str(tmp_path),
        clutter_density="low",
        require_support_and_upright_after_pack=True,
        remove_other_object_categories=("breakfast_table",),
        pack_min_clearance=0.03,
        zone_edge_margin_m=0.05,
        obstacle_bounds_override_xy=((1.0, 2.0), (3.0, 4.0)),
        perimeter_clear_margin_m=0.8,
        mount_workspace_front_m=0.55,
        mount_workspace_side_m=0.35,
        mount_workspace_rear_m=0.95,
        pin_support_base=True,
        post_mount_settle_steps=6,
        video_viewer_only=True,
        video_candidate_mode="support_relative_v1",
        support_clear_mode="remove_all",
        perimeter_clear_mode="aggressive",
        video_candidate_views=(
            {
                "label": "diag_left_far",
                "eye": (1.0, 2.0, 3.0),
                "lookat": (4.0, 5.0, 6.0),
            },
        ),
        video_final_view="diag_left_far",
    )
    args = Namespace(
        clutter_density="medium",
        remove_other_object_categories=(),
        pack_min_clearance=None,
        zone_edge_margin_m=None,
        perimeter_clear_margin_m=None,
        mount_workspace_front_m=None,
        mount_workspace_side_m=None,
        mount_workspace_rear_m=None,
        pin_support_base=False,
        post_mount_settle_steps=None,
        video_viewer_only=False,
        video_candidate_mode=None,
        support_clear_mode=None,
        perimeter_clear_mode=None,
        video_candidate_views=(),
        video_final_view=None,
    )

    mod.apply_scene_entry_to_args(args, entry)

    assert args.clutter_density == "low"
    assert args.remove_other_object_categories == ("breakfast_table",)
    assert args.pack_min_clearance == 0.03
    assert args.zone_edge_margin_m == 0.05
    assert args.perimeter_clear_margin_m == 0.8
    assert args.mount_workspace_front_m == 0.55
    assert args.mount_workspace_side_m == 0.35
    assert args.mount_workspace_rear_m == 0.95
    assert args.pin_support_base is True
    assert args.post_mount_settle_steps == 6
    assert args.video_viewer_only is True
    assert args.video_candidate_mode == "support_relative_v1"
    assert args.support_clear_mode == "remove_all"
    assert args.perimeter_clear_mode == "aggressive"
    assert args.video_candidate_views[0]["label"] == "diag_left_far"
    assert args.video_final_view == "diag_left_far"


def test_manifest_parses_video_candidate_views(tmp_path):
    mod = _load_module()
    run_dir = tmp_path / "benchmark"
    run_dir.mkdir()
    manifest_path = tmp_path / "curation_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "benchmark_run_dir": str(run_dir),
                "scenes": {
                    "gates_bedroom": {
                        "status": "repair",
                        "video_candidate_views": [
                            {
                                "label": "diag_left_far",
                                "eye": [1, 2, 3],
                                "lookat": [4, 5, 6],
                            },
                            {
                                "label": "diag_right_mid",
                                "eye": [7, 8, 9],
                                "lookat": [10, 11, 12],
                            },
                        ],
                        "video_final_view": "diag_right_mid",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = mod.load_curation_manifest(str(manifest_path))
    entry = manifest.get_scene_entry("gates_bedroom")

    assert len(entry.video_candidate_views) == 2
    assert entry.video_candidate_views[0]["eye"] == (1.0, 2.0, 3.0)
    assert entry.video_candidate_views[1]["lookat"] == (10.0, 11.0, 12.0)
    assert entry.video_final_view == "diag_right_mid"


def test_manifest_parses_obstacle_bounds_override(tmp_path):
    mod = _load_module()
    run_dir = tmp_path / "benchmark"
    run_dir.mkdir()
    manifest_path = tmp_path / "curation_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "benchmark_run_dir": str(run_dir),
                "scenes": {
                    "grocery_store_asian": {
                        "status": "repair",
                        "obstacle_bounds_override_xy": [[-1.16, -3.84], [-0.82, -3.30]],
                        "require_support_and_upright_after_pack": True
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = mod.load_curation_manifest(str(manifest_path))
    entry = manifest.get_scene_entry("grocery_store_asian")

    assert entry.obstacle_bounds_override_xy == ((-1.16, -3.84), (-0.82, -3.3))
    assert entry.require_support_and_upright_after_pack is True


def test_manifest_applies_resident_surface_controls(tmp_path):
    mod = _load_module()
    entry = mod.SceneCurationEntry(
        scene_model="Rs_int",
        benchmark_run_dir=str(tmp_path),
        use_resident_surface_obstacles=True,
        require_resident_surface_stability=True,
    )
    args = Namespace()

    mod.apply_scene_entry_to_args(args, entry)

    assert args.use_resident_surface_obstacles is True
    assert args.require_resident_surface_stability is True


def test_manifest_applies_pack_clearance_search_controls(tmp_path):
    mod = _load_module()
    entry = mod.SceneCurationEntry(
        scene_model="Wainscott_0_int",
        benchmark_run_dir=str(tmp_path),
        pack_min_clearance=0.02,
        pack_clearance_floor_m=0.006,
        pack_clearance_step_m=0.002,
        pack_clearance_search_mode="expand_from_floor",
    )
    args = Namespace()

    mod.apply_scene_entry_to_args(args, entry)

    assert args.pack_min_clearance == 0.02
    assert args.pack_clearance_floor_m == 0.006
    assert args.pack_clearance_step_m == 0.002
    assert args.pack_clearance_search_mode == "expand_from_floor"


def test_manifest_parses_support_priority_controls(tmp_path):
    mod = _load_module()
    run_dir = tmp_path / "benchmark"
    run_dir.mkdir()
    manifest_path = tmp_path / "curation_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "benchmark_run_dir": str(run_dir),
                "scenes": {
                    "office_cubicles_right": {
                        "status": "repair",
                        "video_candidate_mode": "support_relative_v1",
                        "support_clear_mode": "remove_all",
                        "perimeter_clear_mode": "aggressive",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = mod.load_curation_manifest(str(manifest_path))
    entry = manifest.get_scene_entry("office_cubicles_right")

    assert entry.video_candidate_mode == "support_relative_v1"
    assert entry.support_clear_mode == "remove_all"
    assert entry.perimeter_clear_mode == "aggressive"


def test_manifest_applies_mount_controls(tmp_path):
    mod = _load_module()
    entry = mod.SceneCurationEntry(
        scene_model="grocery_store_half_stocked",
        benchmark_run_dir=str(tmp_path),
        mount_gap_m=0.28,
        mount_anchor_offset_m=-0.35,
        mount_base_pose_xyyaw=(1.18, 0.18, 3.14159),
    )
    args = Namespace()

    mod.apply_scene_entry_to_args(args, entry)

    assert args.mount_gap_m == 0.28
    assert args.mount_anchor_offset_m == -0.35
    assert args.mount_base_pose_xyyaw == (1.18, 0.18, 3.14159)
