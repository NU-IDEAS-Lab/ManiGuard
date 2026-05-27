"""Generic LTL safety monitor for BEHAVIOR manipulation tasks.

This module provides a data-driven LTL monitoring system that reads safety
constraints from ``ltl_safety.json`` files (per-activity and per-scene) and
evaluates them every simulation step.  New tasks only require a JSON file --
no Python code per task.

All proposition evaluation is delegated to OmniGibson object states (including
the ``Upright`` state).  The monitor itself contains no physics or geometry
logic.

Main classes
------------
TaskLTLMonitor
    Drop-in replacement for task-specific monitors (e.g. ``KitchenBarLTLMonitor``).
    Loads safety JSON, builds proposition evaluators, wraps the Spot-based
    ``LTLMonitor`` from :mod:`omnigibson.utils.ltl_utils`.

ObjectResolver
    Resolves synset glob patterns (``"wineglass.n.01_*"``) and special tokens
    (``"floor"``) to wrapped OmniGibson scene objects, filtered by an optional
    *active-objects* set so that culled / parked objects are excluded.

SafetyPropositionEvaluator
    Builds per-proposition ``eval_fn`` closures from JSON definitions using
    only registered OmniGibson object states.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Set

import omnigibson as og
from omnigibson.object_states import (
    Covered,
    Dropped,
    Filled,
    Inside,
    NextTo,
    OnFire,
    OnTop,
    Open,
    Touching,
    ToggledOn,
    Upright,
)
from maniguard.utils.ltl_utils import (
    LTLMonitor,
    get_spot_runtime_status,
    spot,
    spot_runtime_available,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  State registry — maps JSON state names to OmniGibson state classes.
#  Every state used in ltl_safety.json must appear here.
# ---------------------------------------------------------------------------

_OG_UNARY_STATES: Dict[str, Any] = {
    "on_fire": OnFire,
    "toggled_on": ToggledOn,
    "open": Open,
    "upright": Upright,
    "dropped": Dropped,
}

_OG_BINARY_STATES: Dict[str, Any] = {
    "touching": Touching,
    "ontop": OnTop,
    "inside": Inside,
    "nextto": NextTo,
    "covered": Covered,
    "filled": Filled,
}

_ALL_OG_STATES: Dict[str, Any] = {**_OG_UNARY_STATES, **_OG_BINARY_STATES}

# Map of state class → list of (param_name_in_json, attribute_name_on_state).
# Used to apply per-proposition params from the JSON onto the state instance.
_STATE_CONFIGURABLE_PARAMS: Dict[Any, list] = {
    Upright: [("max_tilt_deg", "max_tilt_deg")],
    Dropped: [("floor_z", "floor_z"), ("z_margin", "z_margin")],
}


# ---------------------------------------------------------------------------
#  ObjectResolver
# ---------------------------------------------------------------------------

class ObjectResolver:
    """Resolves synset glob patterns to wrapped OmniGibson scene objects.

    Parameters
    ----------
    env : og.Environment
        The OmniGibson environment (must have ``env.task.object_scope``).
    active_objects_by_inst : dict, optional
        If provided, only instance IDs present in this dict are returned.
        This filters out culled / parked objects whose poses may be invalid.
    """

    def __init__(self, env, active_objects_by_inst: Optional[Dict] = None):
        self._env = env
        self._active_objs: Dict[str, Any] = (
            dict(active_objects_by_inst) if active_objects_by_inst else {}
        )
        self._active_insts: Optional[Set[str]] = (
            set(active_objects_by_inst.keys()) if active_objects_by_inst else None
        )

    def resolve_patterns(self, patterns) -> Dict[str, Any]:
        """Return ``{inst_id: wrapped_obj}`` for all patterns."""
        if isinstance(patterns, str):
            patterns = [patterns]
        merged: Dict[str, Any] = {}
        for pat in patterns:
            merged.update(self._resolve_one(pat))
        return merged

    def _resolve_one(self, pattern: str) -> Dict[str, Any]:
        scope = getattr(self._env.task, "object_scope", {}) or {}
        result: Dict[str, Any] = {}
        for inst, ent in scope.items():
            if not fnmatch.fnmatch(inst, pattern):
                continue
            if self._active_insts is not None and inst not in self._active_insts:
                continue
            if ent is None or not getattr(ent, "exists", False):
                continue
            wrapped = getattr(ent, "wrapped_obj", None)
            if wrapped is not None:
                result[inst] = wrapped
        if result:
            return result

        # Fallback for non-BDDL tasks (e.g. DummyTask used by every
        # empty-Scene pipeline): the task carries no ``object_scope``,
        # so the BDDL loop above is a no-op. Match patterns directly
        # against active_objects_by_inst — those are real DatasetObjects
        # the pipeline already resolved by name.
        for inst, obj in self._active_objs.items():
            if obj is None:
                continue
            if fnmatch.fnmatch(inst, pattern):
                result[inst] = obj
        return result


# ---------------------------------------------------------------------------
#  SafetyPropositionEvaluator
# ---------------------------------------------------------------------------

class SafetyPropositionEvaluator:
    """Builds ``eval_fn`` closures from JSON proposition definitions.

    Each eval_fn takes no arguments and returns ``bool``.
    All evaluation is delegated to OmniGibson object states.
    """

    def __init__(self, resolver: ObjectResolver):
        self._resolver = resolver

    def build(self, prop_name: str, prop_def: dict) -> Callable[[], bool]:
        # Custom evaluator types (not backed by a single OG state class).
        check = prop_def.get("check", "")
        if check == "spill":
            return self._build_spill(prop_def)
        if check == "overhead_forbidden":
            return self._build_overhead_forbidden(prop_def)
        if check == "inverted":
            return self._build_inverted(prop_def)
        if check == "particles_on_surface":
            return self._build_particles_on_surface(prop_def)

        state = prop_def.get("state", "")
        state_lower = state.lower()

        if state_lower in _OG_UNARY_STATES:
            return self._build_unary(prop_def, _OG_UNARY_STATES[state_lower])
        if state_lower in _OG_BINARY_STATES:
            return self._build_binary(prop_def, _OG_BINARY_STATES[state_lower])

        raise ValueError(
            f"Unknown state '{state}' in proposition '{prop_name}'. "
            f"Known states: {sorted(_ALL_OG_STATES)}."
        )

    def _build_unary(self, prop_def, state_cls) -> Callable[[], bool]:
        subjects = self._resolver.resolve_patterns(prop_def.get("over", []))
        check = prop_def.get("check", "any")
        params = prop_def.get("params", {})

        print(
            f"[LTL] unary {state_cls.__name__} over "
            f"{prop_def.get('over', [])} -> {len(subjects)} subject(s): "
            f"{sorted(subjects.keys())}",
            flush=True,
        )

        # Apply configurable params from JSON onto each object's state instance.
        self._apply_state_params(subjects, state_cls, params)

        def eval_fn(_subj=subjects, _cls=state_cls, _chk=check):
            results = []
            for name, obj in _subj.items():
                try:
                    results.append(bool(obj.states[_cls].get_value()))
                except Exception as exc:
                    log.warning("unary %s check failed for %s: %s", _cls.__name__, name, exc)
                    results.append(False)
            if not results:
                return False if _chk == "any" else True
            return any(results) if _chk == "any" else all(results)

        return eval_fn

    def _build_binary(self, prop_def, state_cls) -> Callable[[], bool]:
        subjects = self._resolver.resolve_patterns(prop_def.get("over", []))
        check = prop_def.get("check", "any")

        relative_to = prop_def.get("relative_to")
        rel_objs = self._resolver.resolve_patterns(relative_to) if relative_to else {}

        def eval_fn(_subj=subjects, _rel=rel_objs, _cls=state_cls, _chk=check):
            results = []
            for s_name, s_obj in _subj.items():
                for r_name, r_obj in _rel.items():
                    if s_obj is None or r_obj is None:
                        continue
                    try:
                        results.append(bool(s_obj.states[_cls].get_value(r_obj)))
                    except Exception as exc:
                        log.warning(
                            "binary %s check failed for (%s, %s): %s",
                            _cls.__name__, s_name, r_name, exc,
                        )
                        results.append(False)
            if not results:
                return False if _chk == "any" else True
            return any(results) if _chk == "any" else all(results)

        return eval_fn

    def _build_spill(self, prop_def: dict) -> Callable[[], bool]:
        """Build an evaluator that detects liquid spill from a container.

        Tracks the initial particle count inside each container on first
        evaluation. Returns True (spilled) when the fraction of particles
        lost exceeds ``spill_threshold``.
        """
        from omnigibson.object_states import ContainedParticles

        subjects = self._resolver.resolve_patterns(prop_def.get("over", []))
        system_name = prop_def.get("system_name", "water")
        params = prop_def.get("params", {})
        threshold = float(params.get("spill_threshold", 0.15))
        env = self._resolver._env

        # Mutable state captured by the closure.
        state = {"initial_counts": None}

        def eval_fn(
            _subj=subjects, _sys_name=system_name, _env=env,
            _threshold=threshold, _state=state, _cp=ContainedParticles,
        ):
            # Fail loudly if the particle system is missing — a silent False
            # here would mask env setup bugs as "no safety violation".
            system = _env.scene.get_system(_sys_name)

            # Read current particle counts per container.
            current = {}
            for inst, obj in _subj.items():
                try:
                    data = obj.states[_cp].get_value(system)
                    current[inst] = data.n_in_volume
                except Exception as exc:
                    log.warning("particle count read failed for %s: %s", inst, exc)
                    current[inst] = 0

            # Record baseline on first call.
            if _state["initial_counts"] is None:
                _state["initial_counts"] = dict(current)
                return False  # No spill possible on first evaluation.

            # Check if any container has lost more than threshold.
            for inst, initial in _state["initial_counts"].items():
                if initial <= 0:
                    continue
                now = current.get(inst, 0)
                loss = (initial - now) / initial
                if loss > _threshold:
                    return True
            return False

        return eval_fn

    def _build_overhead_forbidden(self, prop_def: dict) -> Callable[[], bool]:
        """Build an evaluator that detects when a carried object passes over forbidden zones.

        Returns True (violation) when the carried object's xy position overlaps
        any forbidden zone's xy AABB footprint and its z is above the zone.

        JSON definition::

            {
                "check": "overhead_forbidden",
                "carried": ["sponge.n.01_*"],
                "zones": ["hardback.n.01_*", "laptop.n.01_*"],
                "params": {"margin_m": 0.02}
            }
        """
        carried_objs = self._resolver.resolve_patterns(prop_def.get("carried", []))
        zone_objs = self._resolver.resolve_patterns(prop_def.get("zones", []))
        margin = float(prop_def.get("params", {}).get("margin_m", 0.02))

        def eval_fn(_carried=carried_objs, _zones=zone_objs, _margin=margin):
            for c_name, c_obj in _carried.items():
                try:
                    c_pos = c_obj.get_position_orientation()[0]
                    cx, cy, cz = float(c_pos[0]), float(c_pos[1]), float(c_pos[2])
                except Exception as exc:
                    log.warning("carried object %s pose lookup failed: %s", c_name, exc)
                    continue
                for z_name, z_obj in _zones.items():
                    try:
                        z_min, z_max = z_obj.aabb
                        zx0 = float(z_min[0]) - _margin
                        zy0 = float(z_min[1]) - _margin
                        zx1 = float(z_max[0]) + _margin
                        zy1 = float(z_max[1]) + _margin
                        z_top = float(z_max[2])
                    except Exception as exc:
                        log.warning("zone %s aabb lookup failed: %s", z_name, exc)
                        continue
                    if zx0 <= cx <= zx1 and zy0 <= cy <= zy1 and cz > z_top:
                        return True
            return False

        return eval_fn

    def _build_inverted(self, prop_def: dict) -> Callable[[], bool]:
        """Check if an object is flipped upside down (tilt > threshold from vertical).

        JSON definition::

            {"check": "inverted", "over": ["mug.n.04_*"],
             "params": {"min_tilt_deg": 120.0}}
        """
        import math

        subjects = self._resolver.resolve_patterns(prop_def.get("over", []))
        min_tilt = float(prop_def.get("params", {}).get("min_tilt_deg", 120.0))

        def eval_fn(_subj=subjects, _min=min_tilt):
            for name, obj in _subj.items():
                try:
                    quat = obj.get_position_orientation()[1]
                    x, y, z, w = [float(v) for v in quat[:4]]
                    zz = 1.0 - 2.0 * (x * x + y * y)
                    zz = max(-1.0, min(1.0, zz))
                    tilt_deg = math.degrees(math.acos(zz))
                    if tilt_deg >= _min:
                        return True
                except Exception as exc:
                    log.warning("inverted check failed for %s: %s", name, exc)
                    continue
            return False

        return eval_fn

    def _build_particles_on_surface(self, prop_def: dict) -> Callable[[], bool]:
        """Check if physical particles exist near a surface (e.g. water on table).

        JSON definition::

            {"check": "particles_on_surface", "surface": ["breakfast_table.n.01_*"],
             "params": {"system_name": "water", "z_margin": 0.05}}
        """
        surface_objs = self._resolver.resolve_patterns(prop_def.get("surface", []))
        system_name = prop_def.get("params", {}).get("system_name", "water")
        z_margin = float(prop_def.get("params", {}).get("z_margin", 0.05))
        env = self._resolver._env

        def eval_fn(_surfaces=surface_objs, _sys=system_name, _env=env, _zm=z_margin):
            # Fail loudly if the particle system is missing — a silent False
            # here would mask env setup bugs as "no particles on surface".
            system = _env.scene.get_system(_sys)
            if system.n_particles == 0:
                return False
            for name, s_obj in _surfaces.items():
                try:
                    from omnigibson.object_states.contact_particles import ContactParticles
                    n = len(s_obj.states[ContactParticles].get_value(system))
                    if n > 0:
                        return True
                except Exception as exc:
                    log.warning("contact-particles check failed for %s: %s", name, exc)
                    continue
            return False

        return eval_fn

    @staticmethod
    def _apply_state_params(subjects: Dict[str, Any], state_cls, params: dict):
        """Apply JSON params (e.g. ``max_tilt_deg``) to the state instances."""
        param_map = _STATE_CONFIGURABLE_PARAMS.get(state_cls)
        if not param_map or not params:
            return
        for obj in subjects.values():
            state_inst = obj.states.get(state_cls)
            if state_inst is None:
                continue
            for json_key, attr_name in param_map:
                if json_key in params:
                    setattr(state_inst, attr_name, float(params[json_key]))


# ---------------------------------------------------------------------------
#  JSON loading
# ---------------------------------------------------------------------------

def _load_scene_safety(scene_model: str) -> dict:
    """Load ``ltl_safety.json`` from the scene assets directory.

    Scene-level safety lives with the asset (next to the scene's USD,
    not in BDDL), so this filesystem path stays — it's not a backward
    compat shim. Task-level safety, on the other hand, is now passed
    inline through ``TaskLTLMonitor(ltl_safety=...)``.
    """
    try:
        from omnigibson.utils.asset_utils import get_scene_path
    except ImportError:
        return {}
    path = os.path.join(get_scene_path(scene_model), "safety", "ltl_safety.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _merge_safety_data(task_data: dict, scene_data: dict) -> dict:
    """Merge task-level and scene-level safety definitions.

    * Constraint lists are concatenated.
    * Proposition dicts are merged (task-level first, scene-level overrides).
    * ``combined_ltl`` formulas are AND-joined.
    * Scene constraints without a ``combined_ltl`` get one auto-built from
      individual ``constraints[*].ltl`` entries.
    """
    merged: dict = {
        "constraints": (
            list(task_data.get("constraints", []))
            + list(scene_data.get("constraints", []))
        ),
        "propositions": {
            **task_data.get("propositions", {}),
            **scene_data.get("propositions", {}),
        },
    }

    task_ltl = task_data.get("combined_ltl", "").strip()
    scene_ltl = scene_data.get("combined_ltl", "").strip()

    if not scene_ltl and scene_data.get("constraints"):
        parts = [c["ltl"] for c in scene_data["constraints"] if "ltl" in c]
        if parts:
            scene_ltl = " & ".join(f"({p})" for p in parts)

    if task_ltl and scene_ltl:
        merged["combined_ltl"] = f"({task_ltl}) & ({scene_ltl})"
    else:
        merged["combined_ltl"] = task_ltl or scene_ltl

    return merged


def _auto_generate_scene_propositions(
    merged: dict,
    resolver: ObjectResolver,
) -> dict:
    """Generate evaluators for scene-level APs that lack proposition defs.

    Scene-level ``ltl_safety.json`` files often embed instance-state names
    directly in the LTL formula (e.g. ``agent.n.01_1_on_fire``) without a
    ``propositions`` block.  This function parses those AP names and creates
    definitions so the evaluator can handle them.
    """
    if not merged.get("combined_ltl"):
        return merged
    if not spot_runtime_available(require_buddy=False):
        status = get_spot_runtime_status(require_buddy=False)
        log.warning("[TaskLTLMonitor] Spot runtime invalid for AP parsing: %s", status["error"])
        return merged

    formula = spot.formula(merged["combined_ltl"])
    ap_names = sorted(str(ap) for ap in spot.atomic_prop_collect(formula))
    props = dict(merged.get("propositions", {}))

    for ap in ap_names:
        if ap in props:
            continue
        generated = _try_parse_instance_state_ap(ap, resolver)
        if generated is not None:
            props[ap] = generated

    merged["propositions"] = props
    return merged


def _try_parse_instance_state_ap(
    ap_name: str,
    resolver: ObjectResolver,
) -> Optional[dict]:
    """Try to parse ``<instance_id>_<state_name>`` and build a proposition def.

    For example, ``"agent.n.01_1_on_fire"`` -> instance ``agent.n.01_1``,
    state ``on_fire``.  We try matching known state names from longest to
    shortest suffix.
    """
    known_states = sorted(_ALL_OG_STATES.keys(), key=len, reverse=True)
    for state_name in known_states:
        suffix = f"_{state_name}"
        if ap_name.endswith(suffix):
            instance_id = ap_name[: -len(suffix)]
            objs = resolver.resolve_patterns([instance_id])
            if not objs:
                continue
            if state_name in _OG_UNARY_STATES:
                return {
                    "state": state_name,
                    "over": [instance_id],
                    "check": "any",
                }
            elif state_name in _OG_BINARY_STATES:
                # Binary states need a relative_to, which we can't infer.
                continue
    return None


# ---------------------------------------------------------------------------
#  TaskLTLMonitor
# ---------------------------------------------------------------------------

class TaskLTLMonitor:
    """Generic LTL safety monitor for any BEHAVIOR manipulation task.

    Usage in a runner::

        monitor = TaskLTLMonitor(
            env, ltl_safety=activity_ltl_safety,
            scene_model=scene_model,
            active_objects_by_inst=active_objects,
        )
        monitor.reset()
        initial = monitor.step(0)

        for step_idx in range(1, max_steps + 1):
            sim_step(...)
            info = monitor.step(step_idx)
            if monitor.violated:
                break

        summary = monitor.summary()

    Parameters
    ----------
    env : og.Environment
    ltl_safety : dict
        Task-level safety spec (constraints + propositions + combined LTL
        formula) produced by ``maniguard.utils.task_spec``'s activity
        generators and embedded in each task's ``diagnostics.jsonl``.
        Required — pass ``{}`` to disable task-level monitoring.
    activity_name : str, optional
        Logged for diagnostics; no filesystem lookup happens.
    scene_model : str or None
        Scene model identifier. When set, scene-level safety constraints
        are still loaded from ``scenes/<scene_model>/safety/ltl_safety.json``
        (asset-side, not BDDL-side). ``None`` skips them.
    active_objects_by_inst : dict, optional
        ``{inst_id: obj}`` for objects actually placed in the scene. If
        given, only these objects are monitored — culled objects are
        ignored, and patterns resolve against this dict when the env's
        BDDL task has no ``object_scope`` (e.g. DummyTask).
    """

    def __init__(
        self,
        env,
        *,
        ltl_safety: dict,
        activity_name: str = "",
        scene_model: Optional[str] = None,
        active_objects_by_inst: Optional[Dict] = None,
    ):
        self._env = env
        self._resolver = ObjectResolver(env, active_objects_by_inst)
        self._evaluator = SafetyPropositionEvaluator(self._resolver)
        self._violation_count = 0
        self._violation_step: Optional[int] = None
        self._ltl_log: List[dict] = []

        task_data = dict(ltl_safety) if ltl_safety else {}
        scene_data = _load_scene_safety(scene_model) if scene_model else {}
        merged = _merge_safety_data(task_data, scene_data)
        merged = _auto_generate_scene_propositions(merged, self._resolver)

        self._formula_str: str = merged.get("combined_ltl", "")
        self._constraints: list = merged.get("constraints", [])

        # Build eval functions for each declared proposition.
        self._prop_fns: Dict[str, Callable[[], bool]] = {}
        for prop_name, prop_def in merged.get("propositions", {}).items():
            try:
                self._prop_fns[prop_name] = self._evaluator.build(prop_name, prop_def)
            except Exception as exc:
                log.warning("[TaskLTLMonitor] Skipping proposition '%s': %s", prop_name, exc)

        print(f"[LTL] Propositions: {sorted(self._prop_fns.keys())}")

        # Initialise the Spot automaton.
        if spot_runtime_available(require_buddy=True) and self._formula_str:
            try:
                self._monitor = LTLMonitor(self._formula_str)
                self._monitor.reset()
                print(f"[LTL] Monitor initialised: {self._formula_str}")
                print(f"[LTL]   APs: {self._monitor.ap_list}")
            except Exception as exc:
                log.warning("[TaskLTLMonitor] LTLMonitor init failed: %s", exc)
                self._monitor = None
        else:
            self._monitor = None
            if not spot_runtime_available(require_buddy=True):
                status = get_spot_runtime_status(require_buddy=True)
                print(f"[LTL] WARNING: Spot runtime invalid — monitoring disabled. {status['error']}")
            elif not self._formula_str:
                print("[LTL] WARNING: No LTL formula found — monitoring disabled.")

    # -- per-step interface -------------------------------------------------

    def _label_dict(self) -> Dict[str, bool]:
        return {name: fn() for name, fn in self._prop_fns.items()}

    def step(self, step_idx: int) -> dict:
        """Advance the monitor by one simulation step."""
        labels = self._label_dict()

        if self._monitor is None:
            info = {"state": None, "accepting": False, "doomed": False, "ap": labels}
        else:
            info = self._monitor.step(labels)

        doomed = bool(info.get("doomed", False))
        if doomed and self._violation_step is None:
            self._violation_count += 1
            self._violation_step = step_idx
            og.log.warning("LTL safety violation at step %d: %s", step_idx, labels)
            print(f"[LTL] VIOLATION at step {step_idx}: {labels}")

        entry = {
            "step": step_idx,
            "ap": {k: bool(v) for k, v in labels.items()},
            "state": info.get("state"),
            "accepting": bool(info.get("accepting", False)),
            "doomed": doomed,
        }
        self._ltl_log.append(entry)
        return entry

    def reset(self):
        """Reset the monitor for a new episode."""
        if self._monitor is not None:
            self._monitor.reset()
        self._violation_step = None
        self._ltl_log.clear()

    # -- accessors ----------------------------------------------------------

    @property
    def violated(self) -> bool:
        return self._violation_step is not None

    @property
    def violation_step(self) -> Optional[int]:
        return self._violation_step

    @property
    def violation_count(self) -> int:
        return self._violation_count

    def summary(self) -> dict:
        """Return a serialisable summary for JSONL logging."""
        return {
            "formula": self._formula_str,
            "constraints": self._constraints,
            "violated": self.violated,
            "violation_step": self._violation_step,
            "violation_count": self._violation_count,
            "total_steps_monitored": len(self._ltl_log),
            "log": self._ltl_log,
        }
