"""SafetyGate — the real-time success + LTL-safety verifier the engine runs for the WHOLE
execution (doc §0.1 revised).

A datagen demo is kept ONLY if it ENDS in success AND was SAFE at every executed step; any
LTL violation instantly voids it. We NEVER collect "success but not safe" data.

  * **success** (eval-consistent, matches ``eval.goal_checker.GoalRegionChecker``): the
    target is AG-held AND its AABB intersects the goal sphere.
  * **LTL**: ``utils.safety_monitor.TaskLTLMonitor`` stepped every executed env step; the
    gate exposes ``violated``.

The diagnostics' LTL patterns reference objects by glob (``teacup_*``, ``desk.n.01_*``,
``agent.*``), so we rebuild ``{inst_id: obj}`` against the loaded scene exactly like the eval
runner does (replicated clean here — datagen stays self-contained, no eval import).
"""
from __future__ import annotations

import fnmatch
from typing import Any

from maniguard.utils.goal_region import (
    GoalRegionSpec, object_intersects_goal_region, robot_holds_target,
)

_OBJECT_TAXONOMY = None


def _category_synset_lemma(category: str) -> str:
    """OmniGibson category -> its BDDL synset lemma (``roasting_pan`` -> ``roaster``).
    LTL patterns name objects by synset lemma, not always the OG category. ``""`` if N/A."""
    global _OBJECT_TAXONOMY
    if not category:
        return ""
    try:
        if _OBJECT_TAXONOMY is None:
            from bddl.object_taxonomy import ObjectTaxonomy
            _OBJECT_TAXONOMY = ObjectTaxonomy()
        syn = _OBJECT_TAXONOMY.get_synset_from_category(category)
        return syn.split(".n.")[0] if syn else ""
    except Exception:  # noqa: BLE001
        return ""


def build_active_objects(env, ltl_safety: dict, surface_name: str | None) -> dict:
    """Reconstruct ``{inst_id: obj}`` so the diagnostics LTL glob patterns resolve to loaded
    objects (``agent.*`` -> robot; ``<cat>_*`` / ``<synset>.n.*_*`` -> category/synset/role
    matches; unresolved synset -> the support surface backstop). Replicated from the eval
    runner so monitoring behaves identically."""
    patterns = set()
    for pdef in ((ltl_safety or {}).get("propositions") or {}).values():
        for key in ("over", "relative_to"):
            v = pdef.get(key)
            if isinstance(v, list):
                patterns.update(v)
            elif isinstance(v, str):
                patterns.add(v)

    robot = env.robots[0] if env.robots else None
    objs = list(env.scene.objects)
    cat2lemma = {}
    for o in objs:
        c = getattr(o, "category", "")
        if c and c not in cat2lemma:
            cat2lemma[c] = _category_synset_lemma(c)
    surface_obj = (
        env.scene.object_registry("name", surface_name) if surface_name else None
    )

    active: dict[str, Any] = {}
    for pat in patterns:
        prefix = pat[:-2] if pat.endswith("_*") else pat
        if prefix.startswith("agent"):
            if robot is not None:
                active[f"{prefix}_0"] = robot
            continue
        base = prefix.split(".n.")[0]
        matched = [
            o for o in objs
            if getattr(o, "category", "") == base
            or cat2lemma.get(getattr(o, "category", "")) == base
        ]
        matched += [
            o for o in objs
            if o not in matched and fnmatch.fnmatch(getattr(o, "name", ""), pat)
        ]
        if not matched and ".n." in prefix:
            role_matched = [o for o in objs if getattr(o, "name", "").startswith(base + "_")]
            if role_matched:
                matched = role_matched
            elif surface_obj is not None:
                print(f"  [LTL] pattern {pat!r} unresolved by category/synset; "
                      f"using diagnostics surface {surface_name!r}", flush=True)
                matched = [surface_obj]
        for i, obj in enumerate(matched):
            active[f"{prefix}_{i}"] = obj
    return active


class SafetyGate:
    """Per-demo success + LTL gate. Built once per task (Spot init is not cheap); call
    ``reset()`` before each variant run, ``step()`` after each executed env step, then check
    ``violated`` (void the demo) and ``held_in_goal`` (success)."""

    def __init__(self, env, *, target, goal_spec: GoalRegionSpec,
                 ltl_safety: dict | None = None, scene_model: str | None = None,
                 surface_name: str | None = None):
        self.env = env
        self.target = target
        self.goal_spec = goal_spec
        self._step_idx = 0
        self.monitor = None
        ltl_safety = ltl_safety or {}
        if ltl_safety:
            from maniguard.utils.safety_monitor import TaskLTLMonitor
            active = build_active_objects(env, ltl_safety, surface_name)
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

    def held_in_goal(self) -> bool:
        """eval-consistent success: target AG-held AND its AABB intersects the goal sphere."""
        return bool(robot_holds_target(self.env, self.target)
                    and object_intersects_goal_region(self.target, self.goal_spec))

    # success alias (the engine ANDs this with the family's success_extra)
    def success(self) -> bool:
        return self.held_in_goal()


def build_gate(env, diagnostics: dict, target, *, surface_name: str | None = None) -> SafetyGate:
    """Build a SafetyGate from a task's diagnostics row (``goal_region`` + ``ltl_safety``).
    ``scene_model=None`` (6fam-base tasks are empty Scenes; matches the eval runner)."""
    goal_spec = GoalRegionSpec.from_json(diagnostics["goal_region"])
    return SafetyGate(env, target=target, goal_spec=goal_spec,
                      ltl_safety=diagnostics.get("ltl_safety") or {},
                      scene_model=None, surface_name=surface_name)
