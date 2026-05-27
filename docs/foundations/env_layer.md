# Environment layer

`maniguard/envs/` does **not** define a live environment class — the active RL
stack uses `maniguard.rl.tasks.pick_and_lift.PickAndLiftTask` directly. What
remains are three building blocks that every downstream stage (task-gen replay,
eval, RL reset) shares:

| Module | Role |
|---|---|
| `registry.py` | Discover frozen scene snapshots and parse them into typed specs |
| `frozen_task_runtime.py` | Build an OmniGibson env config from a snapshot + controller presets |
| `perturbation_runtime.py` | Apply visual/structural perturbations to a loaded scene |

## Scene registry (`registry.py`)

A *benchmark root* is a directory of per-scene subdirs, each containing a frozen
`scene_ep1.json` snapshot and a `diagnostics.jsonl`. `build_scene_registry()`
walks it and returns a list of immutable `ManiGuardSceneSpec` records:

```python
from maniguard.envs.registry import build_scene_registry

registry = build_scene_registry(
    benchmark_root="outputs/benchmarks/pnp_clutter",
    activity_root="behavior-1k/bddl3/bddl/activity_definitions",
    scene_names=["Benevolence_1_int"],   # optional filter
    max_scenes=None,
)
spec = registry[0]   # ManiGuardSceneSpec(scene_name, scene_file, diagnostics_file,
                     #   activity_name, problem_file, target_synset,
                     #   target_object_name, support_object_name/_label, prompt)
```

Each spec is built by cross-referencing three sources: the scene snapshot
(object init info), the first line of `diagnostics.jsonl` (activity name,
selection, `active_object_summary` roles, surface), and the dataset-local
`problem0.bddl`. Helpers in the same module:

- `build_runtime_scene_info()` / `build_runtime_task_metadata()` — attach an
  `inst_to_name` map (BDDL instance id → scene object name) to the snapshot's
  task metadata so a `DummyTask` env can resolve LTL propositions.
- `extract_scene_robot_setup()` / `strip_scene_robots_from_scene_info()` — pull
  the saved robot pose/joints out of a snapshot and remove scene robots so the
  runtime can re-insert a canonical robot.
- `slice_scene_registry_for_worker(registry, num_envs, seed_offset)` —
  deterministically shard scenes across parallel envs/workers.

## Frozen-snapshot env builder (`frozen_task_runtime.py`)

`build_env_config()` turns a snapshot into an OmniGibson env config dict, with
scene robots stripped and a canonical robot re-inserted under locked conventions:

```python
from maniguard.envs.frozen_task_runtime import build_env_config, FrozenTaskRuntimeSession

cfg = build_env_config(
    scene_info, diagnostics,
    controller_preset="joint_position",   # see table below
    grasping_mode="assisted",             # "sticky" for lid / thin-object pipelines
    camera_names=["cam_opposite", "cam_left", "cam_right"],
    action_frequency=20, rendering_frequency=20, physics_frequency=120,
)

with FrozenTaskRuntimeSession(headless=True) as sess:   # boots OmniGibson, stops sim on exit
    env = sess.og.Environment(configs=cfg)
    ...
```

### Controller presets

The arm/gripper controller pair is selected by a **single `controller_preset`
arg** — never reach into the controller config dict directly. Two conventions
are locked across pipelines: `action_normalize=False` (raw radians/meters) and
`grasping_mode="assisted"` (override to `"sticky"` for lids/thin objects).

| Preset | Arm controller | Used by |
|---|---|---|
| `joint_position` | `JointController`, absolute position | teleop replay, validation, default |
| `joint_position_impedance` | `JointController` + impedances, `input_limits=None` | cuRobo Phase-A replays (accurate tracking, no clip) |
| `osc` | `OperationalSpaceController`, raw 6-D pose-delta | pnp Phase-B replay, VLA policies emitting EEF deltas (OpenPI pi0.5) |
| `ik` | `InverseKinematicsController`, binary gripper | live teleop (GELLO / SO-101) |

Other runtime helpers: `FrozenTaskRuntimeSession` (context manager that boots
OmniGibson headless and stops the sim on exit), `ReviewVideoRecorder` +
`position_diagnostics_cameras()` (taskgen-style multi-camera review MP4s),
`step_idle()`, `compute_floor_z()`, `save_scene_snapshot()`, and
`resolve_runtime_python()` (locates a torch-capable interpreter; override with
`SENTINEL_RUNTIME_PYTHON`).

## Runtime perturbations (`perturbation_runtime.py`)

When a scene's task metadata carries a `perturbation` spec (written by the
[perturbation generator](../one_machine_pro6000_eval.md)),
`apply_runtime_perturbations(env)` materializes it on the loaded scene:

- **Visual overrides** — per-object diffuse color / texture swaps.
- **Local reconstruct** instructions, dispatched by `type`:
  `restack_chain`, `liquid_refill_target`, `place_lid_on_container`,
  `restage_on_support`.

Each returns a structured result (`applied`, `reason`, counts) so the caller can
log exactly what was and wasn't materialized — no silent failures.
