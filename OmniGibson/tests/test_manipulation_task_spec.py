import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    mod_path = Path(__file__).resolve().parents[1] / "omnigibson" / "utils" / "manipulation_task_spec.py"
    spec = importlib.util.spec_from_file_location("manipulation_task_spec", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_spec_parses_retrieve_filled_cup_task():
    mod = _load_module()
    spec = mod.build_manipulation_task_spec("retrieve_filled_cup_from_clutter_safely")

    assert spec.task_name == "retrieve_filled_cup_from_clutter_safely"
    assert len(spec.target_ids) > 0
    assert "cabinet.n.01_1" in spec.support_ids
    assert "wineglass.n.01_1" in spec.fragile_ids
    assert "is_fragile_broken" in spec.safety_status_rules


def test_build_spec_parses_clear_lane_task():
    mod = _load_module()
    spec = mod.build_manipulation_task_spec("clear_clutter_lane_for_target_transfer")

    assert spec.task_name == "clear_clutter_lane_for_target_transfer"
    assert len(spec.target_ids) > 0
    assert "cabinet.n.01_1" in spec.support_ids
    assert "sink.n.01_1" in spec.support_ids
    assert any(pred.name == "inside" for pred in spec.goal_predicates if not pred.negated)


def test_build_spec_rejects_unsupported_goal_predicate():
    mod = _load_module()
    predefined_problem = """(define (problem synthetic-0)
    (:domain omnigibson)
    (:objects
        apple.n.01_1 - apple.n.01
        table.n.02_1 - table.n.02
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )
    (:init
        (ontop apple.n.01_1 table.n.02_1)
        (inroom table.n.02_1 kitchen)
        (inroom floor.n.01_1 kitchen)
        (ontop agent.n.01_1 floor.n.01_1)
    )
    (:goal
        (and
            (ontop apple.n.01_1 table.n.02_1)
            (sliced apple.n.01_1 apple.n.01_1)
        )
    )
)"""

    with pytest.raises(mod.ManipulationTaskSpecValidationError, match="unsupported goal predicates"):
        mod.build_manipulation_task_spec("synthetic_task", predefined_problem=predefined_problem)
