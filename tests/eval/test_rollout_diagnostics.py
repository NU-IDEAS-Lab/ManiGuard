"""Execute the evaluator's rollout code with CPU stand-ins for GPU/network calls."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SOURCE = Path(os.environ.get(
    "MANIGUARD_TEST_BENCHMARK_SOURCE",
    Path(__file__).resolve().parents[2] / "maniguard/eval/benchmark.py",
))


def _rollout_code():
    tree = ast.parse(SOURCE.read_text())
    # Compile real initialization, rollout and result construction without
    # bootstrapping Isaac Sim or loading a scene.
    for block in ast.walk(tree):
        if not isinstance(block, ast.For):
            continue
        body = []
        for node in block.body:
            if isinstance(node, ast.Try) and not any(isinstance(n, ast.While) for n in node.body):
                body.extend(node.body)
            else:
                body.append(node)
        starts = [i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "step_idx" for t in n.targets)]
        if not starts:
            continue
        start = starts[0]
        end_init = next(i for i in range(start, len(body)) if isinstance(body[i], ast.ImportFrom))
        loop = next(i for i in range(end_init, len(body)) if isinstance(body[i], ast.Try)
                    and any(isinstance(n, ast.While) for n in body[i].body))
        end = next(i for i in range(loop, len(body)) if isinstance(body[i], ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "result" for t in body[i].targets))
        helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == "_record_action_finiteness"]
        return compile(ast.Module(body=helpers + body[start:end_init] + body[loop:end + 1],
                                  type_ignores=[]), str(SOURCE), "exec")
    raise AssertionError("Rollout block not found")


def run_rollout(stage=None, *, error=None, actions=None, binarize=False, convert=False,
                nonfinite_obs=False, success=False, monitor_error=False,
                conversion_value=np.inf, bounds=None, conversion_dim=8):
    calls = {"actions": [], "monitor_steps": [], "queries": 0}
    actions = np.zeros((2, 7 if convert else 8), dtype=np.float32) if actions is None else actions.copy()

    def fail(where):
        if stage == where:
            raise error or AttributeError("'NoneType' object has no attribute 'view'")

    tree = ast.parse(SOURCE.read_text())
    query_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "query_policy")
    query_ns = {"np": np}
    exec(compile(ast.Module(body=[query_fn], type_ignores=[]), str(SOURCE), "exec"), query_ns)

    def query(*_):
        calls["queries"] += 1
        if calls["queries"] > 4:
            raise RuntimeError("test stopped a non-progressing policy loop")
        fail("query_policy")
        return query_ns["query_policy"](SimpleNamespace(act=lambda _: actions), {}, "random",
                                        SimpleNamespace(action_dim=7 if convert else 8))

    def step(action):
        calls["actions"].append(action.value.copy())
        fail("env_step")
        return None, 0.25, False, False, {}

    def observe(*_):
        fail("extract_obs")
        state = np.zeros(8)
        if nonfinite_obs:
            state[0] = np.nan
        return {"states": state}

    def check(*_):
        fail("goal_check")
        return success, {}

    def ik(*_):
        fail("action_conversion")
        return np.full(conversion_dim, conversion_value, dtype=np.float32)

    def monitor_step(index):
        if monitor_error:
            raise error or ValueError("predicate unavailable")
        calls["monitor_steps"].append(index)

    class Tensor:
        def __init__(self, value):
            self.value = value

        def unsqueeze(self, _):
            return self

    monitor = SimpleNamespace(step=monitor_step, summary=lambda: {"formula": "G p"},
                              violated=False, violation_step=None, violation_count=0)
    ns = dict(
        np=np, os=SimpleNamespace(environ={}), torch=SimpleNamespace(from_numpy=Tensor),
        cfg=SimpleNamespace(max_steps=2, execute_horizon=2, gripper_binarize=binarize,
                            ik_eef_to_joint=convert, save_video=False, success_hold_steps=1,
                            tau_move=0.05, tau_reach=0.1, seed=0),
        env=SimpleNamespace(step=step), robot=None, policy=None, client_type="stub",
        query_policy=query, extract_obs=observe, eef_delta_to_joint_action=ik,
        scene_info={"prompt": "test", "name": "task_0000/base", "target_name": "jar",
                    "target_rooms": []}, episode_seed=1,
        action_space=SimpleNamespace(shape=(8,), low=np.full(8, -np.inf if bounds is None else bounds[0]),
                                         high=np.full(8, np.inf if bounds is None else bounds[1])),
        obs={"states": np.zeros(8)}, ever_contacted=False, first_contact_step=None,
        _task_objs=[], _target_obj=None, target2spawn_max_dist=0.0,
        eef2target_min_dist=float("inf"), ever_grasped=False, grasp_steps=0,
        monitor=monitor, goal_checker=SimpleNamespace(check=check), metrics="both", run_success=True,
    )
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(_rollout_code(), ns)
    # Diagnostic values must survive strict JSON serialization, including bad actions.
    result = json.loads(json.dumps(ns["result"], allow_nan=False))
    return result, calls, output.getvalue()


@pytest.mark.parametrize("stage,steps,returned", [
    ("query_policy", 0, 0), ("action_conversion", 0, 0), ("env_step", 0, 0),
    ("extract_obs", 0, 1), ("goal_check", 1, 1),
])
def test_view_exception_is_a_runtime_error_not_a_completed_rollout(stage, steps, returned):
    result, _, output = run_rollout(stage, convert=stage == "action_conversion")
    assert (result["status"], result["success"], result["nan_terminated"], result["steps"]) == (
        "crashed", None, False, steps)
    assert "rollout_diagnostics" in result
    diag = result["rollout_diagnostics"]
    assert diag["phase"] == stage
    assert diag["env_steps_returned"] == returned
    assert diag["termination_reason"] == "exception"
    assert diag["exception"]["type"] == "AttributeError"
    assert "NoneType" in diag["exception"]["message"]
    assert "Traceback" in diag["exception"]["traceback"] and "Traceback" in output
    assert "OOD failure cascade" not in output


def test_oom_error_is_retained_without_automatic_causal_reclassification():
    result, _, _ = run_rollout("env_step", error=RuntimeError("CUDA out of memory"))
    assert result["status"] == "crashed" and not result["nan_terminated"]
    assert "rollout_diagnostics" in result
    diag = result["rollout_diagnostics"]
    assert diag["exception"]["message"] == "CUDA out of memory"
    assert diag["termination_reason"] == "exception"


def test_remote_view_error_keeps_server_exception_type_and_policy_phase():
    message = "policy server: 'str' object has no attribute 'view'"
    result, _, output = run_rollout("query_policy", error=RuntimeError(message))
    assert result["status"] == "crashed" and not result["nan_terminated"]
    diag = result["rollout_diagnostics"]
    assert diag["exception"]["type"] == "RuntimeError"
    assert diag["exception"]["message"] == message
    assert diag["phase"] == "query_policy" and diag["action_attempts"] == 0
    assert diag["last_action_finite"] == {"raw": None, "controller": None, "clipped": None}
    assert "Traceback" in output


def test_conversion_failure_does_not_claim_an_uncomputed_command_is_finite():
    result, _, _ = run_rollout("action_conversion", convert=True)
    diag = result["rollout_diagnostics"]
    assert diag["action_attempts"] == 1 and diag["observations_returned"] == 0
    assert diag["last_action_finite"] == {"raw": True, "controller": None, "clipped": None}


def test_nonfinite_observation_records_boundary_and_preserves_monitor_prefix():
    result, calls, _ = run_rollout(nonfinite_obs=True)
    assert result["status"] == "numerical_failed" and result["nan_terminated"]
    assert calls["monitor_steps"] == [] and result["steps"] == 1
    assert "rollout_diagnostics" in result
    diag = result["rollout_diagnostics"]
    assert diag["termination_reason"] == "nonfinite_observation"
    assert diag["env_steps_returned"] == diag["observations_returned"] == 1
    assert diag["last_finite_observation_step"] == 0
    assert diag["exception"] is None


@pytest.mark.parametrize("index,value,binarize", [(0, np.nan, False), (0, np.inf, False),
                                                (7, np.nan, True)])
def test_bad_raw_action_is_recorded_before_binarization(index, value, binarize):
    actions = np.zeros((2, 8), dtype=np.float32)
    actions[0, index] = value
    result, calls, _ = run_rollout(actions=actions, binarize=binarize)
    assert result["steps"] == 0 and result["nan_terminated"]
    assert result["status"] == "numerical_failed"
    assert calls["actions"] == []
    assert "rollout_diagnostics" in result
    diag = result["rollout_diagnostics"]
    assert diag["first_nonfinite_action"] == {"boundary": "raw", "action_attempt": 1,
                                               "indices": [index]}
    assert diag["last_action_finite"] == {"raw": False, "controller": None, "clipped": None}
    assert diag["termination_reason"] == "nonfinite_action"


def test_nonfinite_conversion_is_distinct_from_policy_output():
    result, _, _ = run_rollout(convert=True)
    assert "rollout_diagnostics" in result
    diag = result["rollout_diagnostics"]
    assert diag["first_nonfinite_action"]["boundary"] == "controller"
    assert diag["last_action_finite"] == {"raw": True, "controller": False, "clipped": None}
    assert result["status"] == "numerical_failed" and result["steps"] == 0


@pytest.mark.parametrize("success,steps", [(True, 1), (False, 2)])
def test_regular_rollout_keeps_success_horizon_and_monitor_calls(success, steps):
    result, calls, _ = run_rollout(success=success)
    assert (result["status"], result["success"], result["steps"]) == ("completed", success, steps)
    assert calls["monitor_steps"] == list(range(1, steps + 1))
    assert "rollout_diagnostics" in result
    diag = result["rollout_diagnostics"]
    assert diag["exception"] is diag["termination_reason"] is diag["first_nonfinite_action"] is None
    assert diag["env_steps_returned"] == diag["observations_returned"] == steps
    assert diag["last_finite_observation_step"] == steps


@pytest.mark.parametrize("error", [ValueError("predicate unavailable"),
                                  AttributeError("'NoneType' object has no attribute 'view'")])
def test_monitor_failure_stops_rollout_and_leaves_scores_unavailable(error):
    result, calls, output = run_rollout(monitor_error=True, error=error, success=True)
    assert result["status"] == "monitor_failed"
    assert len(calls["actions"]) == 1
    assert result["steps"] == 1
    assert result["success"] is None
    assert result["ltl_violated"] is None
    assert result["counted_violation"] is None
    assert result["safety_evaluated"] is False
    assert result["outcome"] == "monitor_failed"
    assert not result["nan_terminated"]
    diag = result["rollout_diagnostics"]
    assert diag["phase"] == "safety_monitor"
    assert diag["termination_reason"] == "safety_monitor_error"
    assert diag["exception"]["type"] == type(error).__name__
    assert str(error) in diag["exception"]["traceback"]
    assert "Traceback" in output


@pytest.mark.parametrize("monitor_failed", [False, True])
def test_saved_summary_cannot_hide_monitor_failures(tmp_path, monitor_failed):
    rows = [{"status": "completed", "success": True, "ltl_monitored": True,
             "ltl_violated": False, "ever_contacted": True, "counted_violation": False}]
    if monitor_failed:
        rows.append({"status": "monitor_failed", "success": None,
                     "ltl_violated": None, "counted_violation": None})
    results = tmp_path / "results.jsonl"
    results.write_text("\n".join(json.dumps(r) for r in rows))
    tree = ast.parse(SOURCE.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    start = next(i for i,n in enumerate(main.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id == "_rows" for t in n.targets))
    code = compile(ast.Module(body=main.body[start:], type_ignores=[]), str(SOURCE), "exec")
    exits = []
    exec(code, {"json": json, "results_path": results, "output_dir": tmp_path,
                "metrics": "both", "run_success": True,
                "sys": SimpleNamespace(stdout=io.StringIO()),
                "os": SimpleNamespace(_exit=exits.append)})
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["n_monitor_failed"] == int(monitor_failed)
    assert summary["success_rate"] == (None if monitor_failed else 1.0)
    assert summary["contact_gated_violation_rate"] == (None if monitor_failed else 0.0)
    assert exits == [1 if monitor_failed else 0]


@pytest.mark.parametrize("kwargs", [
    {"stage": "query_policy"},
    {"stage": "env_step", "error": RuntimeError("CUDA out of memory")},
    {"nonfinite_obs": True},
    {"actions": np.full((2, 8), np.inf)},
])
def test_incomplete_rollout_never_has_final_success_or_safety_verdict(kwargs):
    result, _, _ = run_rollout(**kwargs)
    assert result["success"] is result["ltl_violated"] is result["counted_violation"] is None
    assert result["safety_evaluated"] is False
    assert result["outcome"] == result["status"]


@pytest.mark.parametrize("convert", [False, True])
def test_infinite_command_is_rejected_before_clipping_can_hide_it(convert):
    actions = np.zeros((2, 7), dtype=np.float32) if convert else np.full((2, 8), np.inf)
    result, calls, _ = run_rollout(actions=actions, convert=convert, bounds=(-1., 1.))
    assert result["status"] == "numerical_failed" and calls["actions"] == []
    assert result["rollout_diagnostics"]["first_nonfinite_action"]["boundary"] == (
        "controller" if convert else "raw")


def test_invalid_clipped_command_never_reaches_physics():
    result, calls, _ = run_rollout(bounds=(0., np.nan))
    assert result["status"] == "numerical_failed" and calls["actions"] == []
    assert result["rollout_diagnostics"]["first_nonfinite_action"]["boundary"] == "clipped"


def test_valid_conversion_and_clipping_still_run_normally():
    result, calls, _ = run_rollout(convert=True, conversion_value=2., bounds=(-1., 1.))
    assert result["status"] == "completed" and result["steps"] == 2
    assert all(np.array_equal(a, np.ones(8)) for a in calls["actions"])


def test_bad_second_action_preserves_valid_prefix_without_a_final_verdict():
    actions = np.zeros((2, 8), dtype=np.float32)
    actions[1, 0] = np.nan
    result, calls, _ = run_rollout(actions=actions)
    assert len(calls["actions"]) == 1 and calls["monitor_steps"] == [1]
    assert result["steps"] == result["nan_terminated_step"] == 1
    assert result["status"] == "numerical_failed" and result["success"] is None
    diag = result["rollout_diagnostics"]
    assert diag["env_steps_returned"] == diag["observations_returned"] == 1
    assert diag["first_nonfinite_action"]["action_attempt"] == 2
