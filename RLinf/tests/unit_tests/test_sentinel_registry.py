from __future__ import annotations

import json
from pathlib import Path

from rlinf.envs.sentinel.registry import (
    build_prompt,
    build_runtime_scene_info,
    build_scene_registry,
    slice_scene_registry_for_worker,
)


def test_build_prompt_formats_target_and_support():
    prompt = build_prompt("coffee_cup.n.01", "breakfast_table_skczfi_1")
    assert "coffee cup" in prompt
    assert "breakfast table skczfi 1" in prompt


def test_build_scene_registry_reads_snapshot_layout(tmp_path: Path):
    benchmark_root = tmp_path / "benchmark"
    activity_root = tmp_path / "activity_definitions"
    scene_dir = benchmark_root / "TestScene"
    scene_dir.mkdir(parents=True)
    (activity_root / "auto_clutter_on_TestScene").mkdir(parents=True)

    (scene_dir / "scene_ep1.json").write_text("{}", encoding="utf-8")
    (scene_dir / "diagnostics.jsonl").write_text(
        json.dumps(
            {
                "activity_name": "auto_clutter_on_TestScene",
                "surface": "table_0",
                "selection": {"target_synset": "goblet.n.01"},
                "active_object_summary": [
                    {"role": "target", "scene_object_name": "goblet_1"}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (activity_root / "auto_clutter_on_TestScene" / "problem0.bddl").write_text(
        "(define (problem auto_clutter_on_TestScene-0))",
        encoding="utf-8",
    )

    registry = build_scene_registry(benchmark_root, activity_root)
    assert len(registry) == 1
    spec = registry[0]
    assert spec.scene_name == "TestScene"
    assert spec.activity_name == "auto_clutter_on_TestScene"
    assert spec.target_synset == "goblet.n.01"
    assert spec.target_object_name == "goblet_1"
    assert spec.support_object_name == "table_0"


def test_slice_scene_registry_is_deterministic(tmp_path: Path):
    benchmark_root = tmp_path / "benchmark"
    activity_root = tmp_path / "activity_definitions"
    for idx in range(3):
        scene_name = f"Scene{idx}"
        scene_dir = benchmark_root / scene_name
        scene_dir.mkdir(parents=True)
        (activity_root / f"auto_clutter_on_{scene_name}").mkdir(parents=True)
        (scene_dir / "scene_ep1.json").write_text("{}", encoding="utf-8")
        (scene_dir / "diagnostics.jsonl").write_text(
            json.dumps(
                {
                    "activity_name": f"auto_clutter_on_{scene_name}",
                    "surface": f"table_{idx}",
                    "selection": {"target_synset": "bowl.n.01"},
                    "active_object_summary": [
                        {"role": "target", "scene_object_name": f"bowl_{idx}"}
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (
            activity_root / f"auto_clutter_on_{scene_name}" / "problem0.bddl"
        ).write_text("(define (problem test-0))", encoding="utf-8")

    registry = build_scene_registry(benchmark_root, activity_root)
    first = slice_scene_registry_for_worker(registry, num_envs=2, seed_offset=1)
    second = slice_scene_registry_for_worker(registry, num_envs=2, seed_offset=1)
    assert [item.scene_name for item in first] == [item.scene_name for item in second]


def test_build_runtime_scene_info_injects_cached_task_metadata():
    scene_info = {
        "metadata": {},
        "objects_info": {
            "init_info": {
                "chalice_355": {"args": {"category": "chalice"}},
                "plate_350": {"args": {"category": "plate"}},
                "coffee_table_dnsjnv_0": {"args": {"category": "coffee_table"}},
            }
        },
    }
    diagnostics = {
        "surface": "coffee_table_dnsjnv_0",
        "active_object_summary": [
            {"inst_id": "goblet.n.01_1", "scene_object_name": "chalice_355"},
            {"inst_id": "plate.n.04_1", "scene_object_name": "plate_350"},
        ],
    }
    problem_text = """
    (define (problem local-test)
        (:objects
            goblet.n.01_1 - goblet.n.01
            plate.n.04_1 - plate.n.04
            breakfast_table.n.01_1 - breakfast_table.n.01
            agent.n.01_1 - agent.n.01
        )
        (:init
            (ontop goblet.n.01_1 breakfast_table.n.01_1)
            (ontop plate.n.04_1 breakfast_table.n.01_1)
        )
        (:goal (and (grasped agent.n.01_1 goblet.n.01_1)))
    )
    """

    runtime_scene_info = build_runtime_scene_info(scene_info, diagnostics, problem_text)
    inst_to_name = runtime_scene_info["metadata"]["task"]["inst_to_name"]

    assert inst_to_name["goblet.n.01_1"] == "chalice_355"
    assert inst_to_name["plate.n.04_1"] == "plate_350"
    assert inst_to_name["breakfast_table.n.01_1"] == "coffee_table_dnsjnv_0"
    assert inst_to_name["agent.n.01_1"] == "agent_0"
