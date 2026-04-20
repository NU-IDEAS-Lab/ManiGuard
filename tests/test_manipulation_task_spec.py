import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    mod_path = Path(__file__).resolve().parents[1] / "sentinel" / "utils" / "manipulation_task_spec.py"
    spec = importlib.util.spec_from_file_location("manipulation_task_spec", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_grasped_goal_infers_target_from_second_arg():
    mod = _load_module()
    predefined_problem = """(define (problem synthetic-grasp-0)
    (:domain omnigibson)
    (:objects
        coffee_cup.n.01_1 - coffee_cup.n.01
        table.n.02_1 - table.n.02
        agent.n.01_1 - agent.n.01
    )
    (:init
        (ontop coffee_cup.n.01_1 table.n.02_1)
        (inroom table.n.02_1 kitchen)
    )
    (:goal
        (and
            (grasped agent.n.01_1 coffee_cup.n.01_1)
        )
    )
)"""
    spec = mod.build_manipulation_task_spec("synthetic_grasp", predefined_problem=predefined_problem)
    assert "coffee_cup.n.01_1" in spec.target_ids
    assert "agent.n.01_1" not in spec.target_ids
    assert "table.n.02_1" in spec.support_ids


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
