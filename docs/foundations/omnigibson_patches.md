# OmniGibson patches & configs

ManiGuard consumes OmniGibson as an **unmodified upstream dependency**. The
behaviors it needs that aren't in upstream are installed at runtime by
`maniguard/_omnigibson_patches.py`, applied automatically the first time you
`import maniguard`:

```python
import maniguard   # calls maniguard._omnigibson_patches.apply()
```

`apply()` is idempotent. Set `SENTINEL_SKIP_OMNIGIBSON_PATCH=1` to skip it
entirely (e.g. lightweight pure-Python consumers that never touch the
simulator). If OmniGibson isn't importable, the eager patches are silently
skipped so non-sim tooling still works.

## Two kinds of patch

**1. Import-time injection.** A `MetaPathFinder` wraps the loader for
`omnigibson.object_states` so the instant that subpackage finishes importing,
ManiGuard injects `Dropped`, `Upright` (from
[`maniguard.object_states`](ltl_safety.md#maniguard-object-states)), and
`Grasped` (an alias of upstream `IsGrasping`). This must happen *during* `import
omnigibson` because downstream modules reference those names at module load.

**2. Eager class/function patches**, applied after `import omnigibson` completes:

| Patch | What it does | Why |
|---|---|---|
| `_extend_factory_lists` | Adds `Dropped`/`Upright` to the default state set | so any object can carry them |
| `_patch_grasp_goal` | `GraspGoal` gains a `hold_steps` counter | require a grasp to be held N steps |
| `_patch_grasp_reward` | `GraspReward` falls back when robot lacks `torso_lift_link` | Franka has no torso link |
| `_patch_sampling_utils` | tensor dtype/device-safe `draw_debug_markers` | avoid CPU/GPU tensor mismatches |
| `_patch_franka_longfinger` | long-finger Franka USD/URDF/cuRobo assets + denser AG ray points | accurate assisted-grasp with long fingers |
| `_patch_create_joint_skip_render` | skips `og.sim.render()` in `create_joint` when all poses are supplied | fixes an AG-path segfault (render invalidates articulation handles inside a physics callback) |
| `_patch_attachable_for_f_link_objects` | auto-adds `attachable` to objects with an F meta-link | lid↔container coupling needs both sides attachable |
| `_patch_attached_to_disable_collision` | filters collisions between attached child/parent | stops light containers flying apart on attach |
| `apply_ag_throttle_from_env` | throttles assisted-grasp to every Nth substep | cuts redundant per-substep AG raycasts in long rollouts |
| `_register_bddl_predicates` | registers `upright`/`dropped`/`grasped`/`stashed` BDDL predicates | expose ManiGuard states to BDDL |

## Environment variables

Runtime toggles for the patch layer (all read at `import maniguard`):

| Variable | Effect |
|---|---|
| `SENTINEL_SKIP_OMNIGIBSON_PATCH=1` | skip all patches (no sim needed) |
| `SENTINEL_SKIP_LONGFINGER=1` | skip the long-finger Franka asset patch |
| `SENTINEL_AG_SUBSTEP_INTERVAL=N` | fire assisted-grasp once per N physics substeps (set N ≈ physics-steps-per-action-step to fire once per env step) |

These variables are read at runtime by the patch module; nothing else needs to
be configured to enable the patches.
