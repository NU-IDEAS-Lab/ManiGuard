import os
import sys

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
OG_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if OG_ROOT not in sys.path:
    sys.path.insert(0, OG_ROOT)

from maniguard.utils.ltl_utils import LTLMonitor


spot = pytest.importorskip("spot")


def test_ltl_monitor_ap_extraction():
    monitor = LTLMonitor("G (!a & !b)")
    # print(monitor.ap_list)
    assert set(monitor.ap_list) == {"a", "b"}


def test_ltl_monitor_step_and_reset():
    monitor = LTLMonitor("G !a")
    result = monitor.step({"a": False})
    assert "state" in result
    assert "accepting" in result
    assert "ap" in result
    assert "doomed" in result
    assert result["ap"]["a"] is False

    prior_state = monitor.state
    result_after = monitor.step({"a": True})
    assert "state" in result_after
    assert monitor.state is not None
    # breakpoint()
    
    monitor.reset()
    assert monitor.state is not None
    assert monitor.state != prior_state or monitor.state == monitor._automaton.get_init_state_number()


@pytest.mark.skipif(
    os.environ.get("RUN_LTL_ENV_TESTS", "0") != "1",
    reason="Set RUN_LTL_ENV_TESTS=1 to enable env integration tests.",
)
def test_ltl_monitor_env_integration_behavior_task():
    from tests.test_ltl_propositions import _make_env

    env = None
    try:
        env = _make_env()
        obs, info = env.reset()
        ltl_info = info.get("ltl") or info.get("obs_info", {}).get("ltl")
        assert ltl_info is not None
        assert "state" in ltl_info
        assert "accepting" in ltl_info
        assert "ap" in ltl_info
        assert "doomed" in ltl_info

        action = env.robots[0].action_space.sample()
        _, _, _, _, info = env.step(action * 0.1)
        ltl_info = info.get("ltl") or info.get("obs_info", {}).get("ltl")
        assert ltl_info is not None
        assert "state" in ltl_info
        assert "accepting" in ltl_info
        assert "ap" in ltl_info
        assert "doomed" in ltl_info
    finally:
        if env is not None:
            import omnigibson as og

            og.clear()
