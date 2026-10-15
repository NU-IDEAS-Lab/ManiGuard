"""The evaluator must configure physics from the same discovered scene set it runs."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from maniguard.data.datagen.primitives.scene import _needs_gpu_dynamics
from maniguard.eval.scene_discovery import discover_scenes

SOURCE = Path(__file__).resolve().parents[2] / "maniguard/eval/benchmark.py"


def physics_selection(cfg, root):
    tree = ast.parse(SOURCE.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    start = next(
        i
        for i, n in enumerate(main.body)
        if isinstance(n, ast.ImportFrom) and n.module == "maniguard.data.datagen.primitives.scene"
    )
    end = next(
        i
        for i, n in enumerate(main.body[start:], start)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "_init_omnigibson"
    )
    called = []
    ns = {
        "cfg": cfg,
        "resolved_root": root,
        "Path": Path,
        "discover_scenes": discover_scenes,
        "_init_omnigibson": lambda cfg, needs_gpu_dynamics: called.append(needs_gpu_dynamics),
    }
    # Execute only the local discovery/initialization seam, without starting Isaac Sim.
    exec(compile(ast.Module(body=main.body[start : end + 1], type_ignores=[]), str(SOURCE), "exec"), ns)  # noqa: S102
    return called[0]


@pytest.fixture
def bench(tmp_path):
    for scene, liquid in [("task_0000/base", False), ("task_0001/base", True), ("task_0001/env", True)]:
        path = tmp_path / scene
        path.mkdir(parents=True)
        (path / "scene_ep1.json").write_text(
            json.dumps({"objects_info": {"init_info": {"beaker_0": {"args": {"category": "beaker"}}}}})
        )
        (path / "diagnostics.jsonl").write_text(
            json.dumps(
                {
                    "pipeline": "liquid_transport" if liquid else "table",
                    "selection": {"target_synset": "beaker.n.02", **({"system_name": "water"} if liquid else {})},
                    "goal_region": {"target_name": "beaker_0"},
                    "prompt": "move the beaker",
                }
            )
        )
    return tmp_path


@pytest.mark.parametrize(
    "scenes,filter_,gpu",
    [
        (None, "", True),
        (["task_0001"], "", True),
        (["task_0001/base"], "", True),
        (["task_0000/base"], "", False),
        (None, "task_0000/*", False),
    ],
)
def test_physics_matches_selected_scenarios(bench, scenes, filter_, gpu):
    cfg = SimpleNamespace(scenes=scenes, scene_filter=filter_, max_scenes=None)
    assert physics_selection(cfg, bench) is gpu


def test_spill_spec_requires_gpu_even_without_selection_hint():
    assert _needs_gpu_dynamics({"ltl_safety": {"propositions": {"spill": {"check": "spill"}}}})
