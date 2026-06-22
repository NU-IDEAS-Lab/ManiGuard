"""SafetyGate — the real-time success + LTL-safety verifier the engine runs for the WHOLE
execution (doc §0.1 revised).

A datagen demo is kept ONLY if it ENDS in success AND was SAFE at every executed step; any
LTL violation instantly voids it. We NEVER collect "success but not safe" data.

Both checks reuse the SAME shared modules as teleop, the bench-finalize step, and the eval
runner — so "success" and "safe" mean exactly the same thing everywhere they are judged:

  * **success**: ``eval.goal_checker.build_goal_checker`` (``GoalRegionChecker`` for goal_region
    families like clutter = held + in-sphere; ``GoalChecker`` for goal_conditions families like
    cabinet = ``inside & closed``).
  * **LTL**: ``utils.safety_monitor.TaskLTLMonitor`` + ``build_active_objects_for_ltl`` (the glob
    -> scene-object resolver), stepped every executed env step; the gate exposes ``violated``.
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
    """Build a SafetyGate from a task's diagnostics row. Success comes from the eval checker
    (goal_region OR goal_conditions, auto-selected); LTL from ``ltl_safety``. ``scene_model=None``
    (6fam-base tasks are empty Scenes; matches the eval runner)."""
    from maniguard.eval.goal_checker import build_goal_checker

    checker = build_goal_checker(diagnostics)
    if checker is None:
        raise ValueError("task has neither goal_region nor goal_conditions for a success check")
    return SafetyGate(env, success_checker=checker,
                      ltl_safety=diagnostics.get("ltl_safety") or {},
                      scene_model=None, surface_name=surface_name)
