# LTL safety system

ManiGuard's namesake feature: every rollout — task-gen, datagen, teleop replay, eval —
can be checked against **Linear Temporal Logic (LTL)** safety constraints, step
by step. The system has two layers:

1. **Propositions** — boolean facts about the scene evaluated each step
   (e.g. "the wineglass is upright", "the agent is touching the food").
2. **Automaton monitoring** — an LTL<sub>f</sub> formula over those propositions is
   compiled (via [Spot](https://spot.lre.epita.fr/), options
   `'monitor', 'det', 'complete'`) into a **deterministic monitor automaton**
   that detects *bad prefixes*: each step advances the automaton, and the run is
   flagged as *doomed* as soon as it enters a rejecting sink — a state from
   which the safety property can no longer be satisfied on any continuation.

Code lives in `maniguard/utils/`:

| File | Contents |
|---|---|
| `ltl_utils.py` | `AtomicProposition(Set)`, `AtomicPropositionGenerator`, `LTLMonitor`, Spot runtime probes |
| `safety_monitor.py` | `TaskLTLMonitor` (the high-level entry point), `SafetyPropositionEvaluator`, `ObjectResolver` |
| `bddl_predicates.py` | `register_maniguard_predicates()` — adds `upright`/`dropped`/`grasped`/`stashed` to BDDL |
| `task_spec.py` | `generate_*_ltl_safety_json()` — emits the `ltl_safety.json` specs per task family |

## TaskLTLMonitor — the entry point

For almost all usage you attach a `TaskLTLMonitor` to an env and step it:

```python
from maniguard.utils.safety_monitor import TaskLTLMonitor

monitor = TaskLTLMonitor(
    env,
    ltl_safety=ltl_safety_dict,         # the task's spec dict; {} disables monitoring
    activity_name="pnp_clutter",
    active_objects_by_inst={"wineglass.n.01_1": obj, ...},
)
monitor.reset()
for step in range(max_steps):
    env.step(action)
    info = monitor.step(step)           # {"step","ap","state","accepting","doomed"}
    if monitor.violated:
        break
summary = monitor.summary()             # formula, constraints, violated, violation_step, log
```

It merges **task-level** and **scene-level** constraints (formulas AND-joined,
propositions merged), resolves proposition subjects to live scene objects via
`ObjectResolver` (synset-glob → wrapped objects, filtered to `active_objects`),
builds an evaluator, compiles the combined formula, and logs the first violation.

### Where the spec comes from

For the benchmark, the spec **travels with the task**: task generation embeds
each task's `ltl_safety` dict inline in its `diagnostics.jsonl`, and datagen /
eval pass that dict straight to `TaskLTLMonitor` — no filesystem lookup, so a
frozen task is self-contained. Two file-based channels still exist but are
dormant compatibility paths, not used by the benchmark: a task-level
`activity_definitions/<activity>/ltl_safety.json` and a scene-level
`scenes/<scene>/safety/ltl_safety.json` (loaded only when a `scene_model` arg is
passed; scene- and task-level constraints are then AND-joined).

### The spec schema (`ltl_safety`)

```json
{
  "activity_name": "pnp_clutter",
  "constraints": [
    {"id": "no_glass_dropped", "ltl": "G (! p_glass_dropped)", "description": "..."}
  ],
  "propositions": {
    "p_glass_dropped": {
      "check": "any",                 // "any" | "all" over matched objects
      "state": "dropped",             // unary or binary state name
      "over": ["wineglass.n.01_*"],   // synset-glob subjects
      "relative_to": ["table.n.02_*"],// for binary states
      "params": {"max_tilt_deg": 45}  // forwarded to the state instance
    }
  },
  "combined_ltl": "G (! p_glass_dropped)"   // auto-generated if omitted
}
```

`SafetyPropositionEvaluator` builds an `eval_fn` per proposition. Supported:

- **Unary** states: `OnFire`, `ToggledOn`, `Open`, `Upright`, `Dropped`.
- **Binary** states: `Touching`, `OnTop`, `Inside`, `NextTo`, `Covered`, `Filled`.
- **Custom checks**: `spill` (liquid loss past a threshold), `overhead_forbidden`
  (carried object over a forbidden AABB), `inverted` (tilt past threshold),
  `particles_on_surface`.

!!! warning "No silent failures"
    Proposition evaluation never swallows errors into a `False`. A missing state
    class or unresolved object is logged; a malformed LTL formula raises
    `ValueError` so config bugs surface immediately rather than masquerading as
    "safe".

## How the pieces work (under the hood)

### Proposition generation

`AtomicPropositionGenerator(task)` builds a `PropositionSet` from a BDDL
task's `object_scope` and the supported predicates — one `AtomicProposition`
(name, type, `eval_fn`, description) per unary state and per object pair for
binary relations. `PropositionSet.get_label_dict(env)` returns `{name: bool}` for
the current step.

### Automaton monitoring (`LTLMonitor`)

```python
from maniguard.utils.ltl_utils import LTLMonitor

mon = LTLMonitor("G (! p_bad)", translate_opts=("monitor", "det", "complete"))
mon.reset()
result = mon.step({"p_bad": False})   # {"state","accepting","ap","doomed"}
```

`LTLMonitor` uses Spot to parse the formula, collect its atomic props, and
`spot.translate()` it into an automaton. Each `step(label_dict)` builds a BDD
condition (via `buddy`) from the proposition values, follows the matching
transition, and reports `doomed` — detected either as a monitor rejecting sink
or, for general automata, via a Tarjan-SCC reachability check from accepting
states.

!!! note "Semantics: a bad-prefix monitor (safety only, weak until)"
    The default `translate_opts=("monitor", "det", "complete")` compiles a
    **bad-prefix monitor**: a rollout is flagged at the first finite prefix that
    can no longer be extended to satisfy the formula. Consequently only
    **safety** violations are detectable — a *liveness* obligation ("ψ must
    eventually hold") has no finite counterexample and is never flagged. In
    particular, an ordering clause written `a U b` is enforced exactly as its
    safety half, the weak until `a W b` ("do not break `a` before `b`"): the
    monitor rejects lifting-before-closing, but an idle rollout that never
    reaches `b` is not a safety violation — reaching `b` is the task goal,
    checked by the [goal checker](../evaluation/index.md), not by φ. `U` and
    `W` have identical bad prefixes, so both spellings compile to the same
    automaton.

### Spot is optional

If the Spot library is unavailable, `TaskLTLMonitor` prints a warning and
disables monitoring (the rollout still runs); `LTLMonitor` raises on
construction. Use `spot_runtime_available()` / `get_spot_runtime_status()` to
probe (the latter also detects a user-site install shadowing the conda one).

## ManiGuard object states

Two custom `AbsoluteObjectState`s back the `dropped`/`upright` predicates. They
are injected into `omnigibson.object_states` by the
[runtime patches](omnigibson_patches.md) and registered as BDDL predicates by
`register_maniguard_predicates()` (which also adds `grasped`, an alias of
upstream `IsGrasping`, and `stashed`, a sampler hint).

| State | True when | Tunable params |
|---|---|---|
| `Dropped` | object z `< floor_z + z_margin` | `floor_z` (0.0), `z_margin` (0.05) |
| `Upright` | angle between object +Z and world +Z `≤ max_tilt_deg` | `max_tilt_deg` (45°) |

Both are read-only (`_set_value` raises) and accept per-instance thresholds so a
scene's `ltl_safety.json` `params` can override them.

The activity-family safety specs themselves are emitted by the
`generate_*_ltl_safety_json()` functions in `task_spec.py` (clutter, stack,
transfer, liquid/wet/lid transport, empty-invert, blocked-door, cabinet, jar) —
see [Task generation](../pipelines/index.md).
