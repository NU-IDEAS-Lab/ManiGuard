import os
import sys

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
OG_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if OG_ROOT not in sys.path:
    sys.path.insert(0, OG_ROOT)

from omnigibson.utils.ltl_utils import LTLMonitor


# spot = pytest.importorskip("spot")


def test_ltl_monitor_ap_extraction():
    monitor = LTLMonitor("G (!a & !b)")
    print(monitor.ap_list)
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

    monitor.reset()
    assert monitor.state is not None
    assert monitor.state != prior_state or monitor.state == monitor._automaton.get_init_state_number()
