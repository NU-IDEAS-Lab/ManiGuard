from copy import deepcopy


def test_monitor_export_does_not_evaluate_or_advance_and_returns_copy():
    from maniguard.utils.safety_monitor import TaskLTLMonitor

    monitor = object.__new__(TaskLTLMonitor)
    monitor._error = None
    monitor._ltl_log = [{"step": 3, "ap": {"upright": True}, "state": 2, "doomed": False, "accepting": True}]
    monitor._label_dict = lambda: (_ for _ in ()).throw(AssertionError("must not evaluate"))
    expected = deepcopy(monitor._ltl_log[0])
    record = monitor.recording_state(3)
    assert record["observation"] == expected and record["valid"]
    record["observation"]["ap"]["upright"] = False
    assert monitor._ltl_log[0] == expected
    assert not monitor.recording_state(4)["valid"]
    monitor._error = {"step": 4, "message": "failed"}
    assert not monitor.recording_state(3)["valid"]
