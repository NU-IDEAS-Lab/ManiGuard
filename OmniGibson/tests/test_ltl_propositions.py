import os
import sys

import numpy as np
import pytest
import yaml

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
OG_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if OG_ROOT not in sys.path:
    sys.path.insert(0, OG_ROOT)

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.bddl_utils import SUPPORTED_PREDICATES


LOG_PATH = os.environ.get(
    "LTL_TEST_LOG",
    os.path.join(TESTS_DIR, "ltl_test_output.log"),
)


def _log_line(line: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(line + "\n")


def _ensure_logo_asset():
    logo_path = os.path.join(OG_ROOT, "docs", "assets", "OmniGibson_logo.png")
    os.makedirs(os.path.dirname(logo_path), exist_ok=True)
    if not os.path.exists(logo_path):
        with open(logo_path, "wb") as logo_file:
            logo_file.write(b"")


def _load_behavior_demo_config():
    config_filename = os.path.join(og.example_config_path, "r1pro_behavior.yaml")
    with open(config_filename, "r") as config_file:
        cfg = yaml.load(config_file, Loader=yaml.FullLoader)
    cfg["task"]["online_object_sampling"] = False
    cfg["task"]["use_presampled_robot_pose"] = True
    cfg["env"]["external_sensors"] = []
    print(f"[ltl-test] loaded config: {config_filename}")
    return cfg


def _make_env():
    if og.sim:
        og.sim.stop()
    with gm.unlocked():
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = False
        gm.HEADLESS = True
    print("[ltl-test] initializing environment")
    cfg = _load_behavior_demo_config()
    _ensure_logo_asset()
    return og.Environment(configs=cfg)


# def test_ltl_propositions_generation_behavior_env_demo():
#     env = None
#     try:
#         env = _make_env()
#         env.reset()

#         if not hasattr(env.task, "get_ltl_propositions"):
#             raise RuntimeError(
#                 "BehaviorTask missing get_ltl_propositions; ensure tests use the local OmniGibson checkout."
#             )
#         prop_set = env.task.get_ltl_propositions()
#         _log_line(f"\n=== Atomic propositions ({len(prop_set)}) ===")
#         for prop in prop_set:
#             _log_line(
#                 f"{prop.name} | category={prop.category} | desc={prop.description} | args={prop.args}"
#             )
#         grounded = env.task.proposition_generator.get_grounded_goal_options()
#         if grounded:
#             _log_line("\n=== Grounded goal options (quantifiers expanded) ===")
#             for option_idx, option in enumerate(grounded):
#                 _log_line(f"goal_option[{option_idx}] (conjunction of atoms):")
#                 for prop_name, negated in option:
#                     prefix = "not " if negated else ""
#                     _log_line(f"  {prefix}{prop_name}")
#         print(f"[ltl-test] propositions total: {len(prop_set)}")
#         print(f"[ltl-test] categories: {sorted(prop_set.categories.keys())}")
#         assert len(prop_set) > 0

#         categories = set(prop_set.categories.keys())
#         assert "unary_state" in categories
#         assert "binary_relation" in categories
#     finally:
#         if env is not None:
#             og.clear()


# def test_ltl_label_consistency_behavior_env_demo():
#     env = None
#     try:
#         env = _make_env()
#         env.reset()

#         if not hasattr(env.task, "get_ltl_propositions"):
#             raise RuntimeError(
#                 "BehaviorTask missing get_ltl_propositions; ensure tests use the local OmniGibson checkout."
#             )
#         prop_set = env.task.get_ltl_propositions()
#         label_array = prop_set.get_label()
#         label_dict = prop_set.get_label_dict()
#         _log_line("\n=== Initial labels ===")
#         for prop in prop_set:
#             _log_line(f"{prop.name}: {label_dict[prop.name]}")
#         print(f"[ltl-test] label shape: {label_array.shape}, true={int(np.sum(label_array))}")

#         assert label_array.shape == (len(prop_set),)
#         assert len(label_dict) == len(prop_set)

#         for idx, prop in enumerate(prop_set):
#             assert bool(label_array[idx]) == label_dict[prop.name]
#             assert label_dict[prop.name] == prop.evaluate()

#         for _ in range(3):
#             action = env.robots[0].action_space.sample()
#             env.step(action * 0.1)

#             step_array = prop_set.get_label()
#             step_dict = prop_set.get_label_dict()
#             _log_line(f"\n=== Step labels (step={_}) ===")
#             for prop in prop_set:
#                 _log_line(f"{prop.name}: {step_dict[prop.name]}")
#             print(f"[ltl-test] step true={int(np.sum(step_array))}")

#             assert step_array.shape == (len(prop_set),)
#             assert len(step_dict) == len(prop_set)
#             for idx, prop in enumerate(prop_set):
#                 assert bool(step_array[idx]) == step_dict[prop.name]
#                 assert step_dict[prop.name] == prop.evaluate()
#     finally:
#         if env is not None:
#             og.clear()


def test_ltl_safety_constraints_loaded_behavior_env_demo():
    env = None
    try:
        env = _make_env()
        env.reset()

        constraints = getattr(env.task, "safety_constraints", None)
        assert constraints is not None
        assert "task" in constraints
        assert "scene" in constraints

        all_constraints = []
        all_constraints.extend(constraints.get("task", []))
        all_constraints.extend(constraints.get("scene", []))
        assert len(all_constraints) > 0

        validation = getattr(env.task, "safety_validation", None)
        assert validation is not None
        combined_ltl = validation.get("combined_ltl", "")

        print(f"[ltl-test] safety constraints loaded: {len(all_constraints)}")
        print(f"[ltl-test] combined LTL: {combined_ltl}")
        _log_line("\n=== Safety constraints ===")
        for constraint in all_constraints:
            ltl_str = constraint.get("ltl", "")
            _log_line(f"{constraint.get('id')}: {ltl_str}")
            if ltl_str:
                assert ltl_str in combined_ltl

        if len(all_constraints) > 1:
            assert " & " in combined_ltl
    finally:
        if env is not None:
            og.clear()


# def test_ltl_labels_match_bddl_predicates_behavior_env_demo():
#     env = None
#     try:
#         env = _make_env()
#         env.reset()

#         if not hasattr(env.task, "get_ltl_propositions"):
#             raise RuntimeError(
#                 "BehaviorTask missing get_ltl_propositions; ensure tests use the local OmniGibson checkout."
#             )
#         prop_set = env.task.get_ltl_propositions()
#         label_dict = prop_set.get_label_dict()

#         for prop in prop_set:
#             pred_name = prop.args[1]
#             pred_cls = SUPPORTED_PREDICATES.get(pred_name)
#             assert pred_cls is not None
#             if len(prop.args) == 2:
#                 pred = pred_cls(
#                     env.task.object_scope,
#                     env.task.backend,
#                     [prop.args[0]],
#                     env.task.activity_conditions.parsed_objects,
#                     generate_ground_options=False,
#                 )
#             else:
#                 pred = pred_cls(
#                     env.task.object_scope,
#                     env.task.backend,
#                     [prop.args[0], prop.args[2]],
#                     env.task.activity_conditions.parsed_objects,
#                     generate_ground_options=False,
#                 )

#             try:
#                 print(f"[ltl-test] evaluating proposition '{prop.name}' against predicate '{pred_name}'")
#                 print(f"[ltl-test] proposition results: {label_dict[prop.name]} against predicate results: {pred.evaluate()}")
#                 pred_value = bool(pred.evaluate())
#             except (KeyError):
#                 continue
#             assert label_dict[prop.name] == pred_value
#     finally:
#         if env is not None:
#             og.clear()
