import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    mod_path = Path(__file__).resolve().parents[1] / "omnigibson" / "utils" / "clutter_recipe.py"
    spec = importlib.util.spec_from_file_location("clutter_recipe", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_recipe_roundtrip(tmp_path):
    mod = _load_module()
    recipe = mod.ClutterSceneRecipe(
        scene_model="Rs_int",
        task_name="retrieve_filled_cup_from_clutter_safely",
        clutter_level="low",
        seed=1,
        runtime_mode="cached",
        mount_result={"position": (0.0, 0.0, 0.0), "orientation": (0.0, 0.0, 0.0, 1.0)},
    )
    out = tmp_path / "recipe.json"
    mod.save_clutter_scene_recipe(recipe, str(out))

    loaded = mod.load_clutter_scene_recipe(str(out))
    assert loaded.scene_model == recipe.scene_model
    assert loaded.task_name == recipe.task_name
    assert loaded.runtime_mode == "cached"


def test_recipe_rejects_non_cached_mode(tmp_path):
    mod = _load_module()
    data = {
        "scene_model": "Rs_int",
        "task_name": "retrieve_filled_cup_from_clutter_safely",
        "clutter_level": "high",
        "seed": 2,
        "runtime_mode": "online",
    }
    p = tmp_path / "bad_recipe.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    try:
        mod.load_clutter_scene_recipe(str(p))
    except ValueError as e:
        assert "runtime_mode" in str(e)
    else:
        raise AssertionError("Expected ValueError for non-cached runtime_mode")
