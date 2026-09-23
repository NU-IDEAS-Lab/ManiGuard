"""Goal and LTL acceptance checks for demonstration collection.

Use eval.goal_checker.build_goal_checker for the task goal and
utils.safety_monitor.TaskLTLMonitor for the supplied LTL specification. Reset
before each attempt and advance through the executor's step callback.
The engine combines the goal result with the family-specific acceptance check.
"""
from __future__ import annotations


class SafetyGate:
    """Per-demo success + LTL gate. Built once per task (Spot init is not cheap); call
    ``reset()`` before each variant run, ``step()`` after each executed env step, then check
    ``violated`` (void the demo) and ``success()`` (goal reached).

    Success delegates to the eval success checker (``GoalRegionChecker`` for goal_region
    families like clutter; ``GoalChecker`` for goal_conditions families like cabinet =
    ``inside & closed``), so a collected demo's success is by construction identical to how the
    policy is later judged at eval."""

    def __init__(self, env, *, success_checker, ltl_safety: dict | None = None,
                 scene_model: str | None = None, surface_name: str | None = None):
        self.env = env
        self._success = success_checker
        self._step_idx = 0
        self.monitor = None
        ltl_safety = ltl_safety or {}
        if ltl_safety:
            from maniguard.utils.safety_monitor import TaskLTLMonitor, build_active_objects_for_ltl
            active = build_active_objects_for_ltl(env, ltl_safety, surface_name)
            self.monitor = TaskLTLMonitor(
                env, ltl_safety=ltl_safety, scene_model=scene_model,
                active_objects_by_inst=active)

    def reset(self) -> None:
        self._step_idx = 0
        if self.monitor is not None:
            self.monitor.reset()
            self.monitor.step(0)        # seed the initial labels

    def step(self) -> None:
        """Advance the LTL monitor by one executed env step."""
        self._step_idx += 1
        if self.monitor is not None:
            self.monitor.step(self._step_idx)

    @property
    def violated(self) -> bool:
        return bool(self.monitor is not None and self.monitor.violated)

    @property
    def violation_step(self):
        return None if self.monitor is None else self.monitor.violation_step

    @property
    def ltl_enabled(self) -> bool:
        return self.monitor is not None

    def success(self) -> bool:
        """eval-consistent goal check (the engine ANDs this with the family's success_extra)."""
        ok, _ = self._success.check(self.env)
        return bool(ok)


def build_gate(env, diagnostics: dict, *, surface_name: str | None = None) -> SafetyGate:
    """Build the shared goal checker and LTL monitor from task diagnostics. Pass scene_model=None because task safety specifications are supplied directly for the reconstructed empty Scene."""
    from maniguard.eval.goal_checker import build_goal_checker

    checker = build_goal_checker(diagnostics)
    if checker is None:
        raise ValueError("task has neither goal_region nor goal_conditions for a success check")
    return SafetyGate(env, success_checker=checker,
                      ltl_safety=diagnostics.get("ltl_safety") or {},
                      scene_model=None, surface_name=surface_name)
