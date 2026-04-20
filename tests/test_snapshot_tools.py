import importlib.util
import sys
from argparse import Namespace
from pathlib import Path


def _load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_expected_category_counts():
    mod_path = (
        Path(__file__).resolve().parents[1]
        / "sentinel"
        / "task_generation"
        / "curation"
        / "inspect_curated_snapshot.py"
    )
    mod = _load_module(mod_path)

    counts = mod.build_expected_category_counts(
        {
            "target_synset": "bowl.n.01",
            "fragile_picks": ["goblet.n.01", "goblet.n.01", "wineglass.n.01"],
            "clutter_picks": ["plate.n.04"],
        }
    )

    assert dict(counts) == {"bowl": 1, "goblet": 2, "wineglass": 1, "plate": 1}


def test_counter_missing_reports_only_shortfall():
    mod_path = (
        Path(__file__).resolve().parents[1]
        / "sentinel"
        / "task_generation"
        / "curation"
        / "inspect_curated_snapshot.py"
    )
    mod = _load_module(mod_path)

    missing = mod.counter_missing(
        mod.build_expected_category_counts(
            {
                "target_synset": "bowl.n.01",
                "fragile_picks": ["goblet.n.01", "goblet.n.01"],
                "clutter_picks": ["plate.n.04"],
            }
        ),
        {"bowl": 1, "goblet": 1, "plate": 4},
    )

    assert missing == {"goblet": 1}


def test_scene_category_counts_normalize_chalice_to_goblet(tmp_path):
    mod_path = (
        Path(__file__).resolve().parents[1]
        / "sentinel"
        / "task_generation"
        / "curation"
        / "inspect_curated_snapshot.py"
    )
    mod = _load_module(mod_path)

    snapshot = tmp_path / "scene_ep1.json"
    snapshot.write_text(
        """
        {
          "objects_info": {
            "init_info": {
              "chalice_1": {"args": {"category": "chalice"}},
              "teacup_1": {"args": {"category": "teacup"}}
            }
          }
        }
        """.strip()
    )

    counts = mod._scene_category_counts(str(snapshot))

    assert dict(counts) == {"goblet": 1, "teacup": 1}


def test_scene_category_counts_normalize_cocktail_glass_to_goblet(tmp_path):
    mod_path = (
        Path(__file__).resolve().parents[1]
        / "sentinel"
        / "task_generation"
        / "curation"
        / "inspect_curated_snapshot.py"
    )
    mod = _load_module(mod_path)

    snapshot = tmp_path / "scene_ep1.json"
    snapshot.write_text(
        """
        {
          "objects_info": {
            "init_info": {
              "cocktail_glass_1": {"args": {"category": "cocktail_glass"}},
              "mug_1": {"args": {"category": "mug"}}
            }
          }
        }
        """.strip()
    )

    counts = mod._scene_category_counts(str(snapshot))

    assert dict(counts) == {"goblet": 1, "mug": 1}


def test_resolve_effective_camera_config_prefers_cli_values():
    mod_path = (
        Path(__file__).resolve().parents[1]
        / "sentinel"
        / "task_generation"
        / "curation"
        / "replay_curated_scene.py"
    )
    mod = _load_module(mod_path)

    entry = Namespace(
        video_camera_eye=(1.0, 2.0, 3.0),
        video_camera_lookat=(4.0, 5.0, 6.0),
    )
    args = Namespace(
        video_camera_eye=[7.0, 8.0, 9.0],
        video_camera_lookat=None,
    )

    config = mod.resolve_effective_camera_config(entry, args)

    assert config == {
        "eye": (7.0, 8.0, 9.0),
        "lookat": (4.0, 5.0, 6.0),
    }


def test_build_video_view_specs_uses_camera_fallback_candidates():
    from sentinel.task_generation.pipeline_common import build_video_view_specs

    class _Obj:
        def __init__(self, pos):
            self._pos = pos
            self.aabb = ((pos[0] - 0.2, pos[1] - 0.2, max(0.0, pos[2] - 0.1)),
                         (pos[0] + 0.2, pos[1] + 0.2, pos[2] + 0.1))

        def get_position_orientation(self):
            return (self._pos, (0.0, 0.0, 0.0, 1.0))

    args = Namespace(
        video_candidate_views=(),
        video_final_view=None,
        _scene_curation=Namespace(issue_tags=("bad_camera_framing",)),
    )
    robot = _Obj((0.0, 0.0, 0.0))
    target = _Obj((0.6, 0.2, 0.8))

    views = build_video_view_specs(args, robot, target)

    assert [view["label"] for view in views] == [
        "opposite_side_front",
        "left_overview",
        "right_overview",
    ]
    assert views[0]["canonical"] is True
    assert all(len(view["eye"]) == 3 and len(view["lookat"]) == 3 for view in views)


def test_build_video_view_specs_support_relative_mode():
    from sentinel.task_generation.pipeline_common import build_video_view_specs

    class _Obj:
        def __init__(self, pos, aabb=None):
            self._pos = pos
            self.aabb = aabb

        def get_position_orientation(self):
            return (self._pos, (0.0, 0.0, 0.0, 1.0))

    args = Namespace(
        video_candidate_views=(),
        video_final_view=None,
        video_candidate_mode="support_relative_v1",
        _scene_curation=Namespace(issue_tags=()),
    )
    robot = _Obj((9.8, 3.4, 0.0))
    target = _Obj((9.9, 4.1, 0.86))
    support = _Obj((9.8, 4.1, 0.6), aabb=((9.2, 3.7, 0.1), (10.4, 4.5, 0.8)))
    active = {
        "cup": _Obj((9.7, 4.0, 0.86)),
        "glass": _Obj((10.0, 4.2, 0.86)),
    }

    views = build_video_view_specs(
        args,
        robot,
        target,
        support_obj=support,
        active_objects_by_inst=active,
    )

    assert [view["label"] for view in views] == [
        "opposite_side_front",
        "left_overview",
        "right_overview",
    ]
    assert views[0]["canonical"] is True
    assert all(view["eye"][2] > view["lookat"][2] for view in views)
