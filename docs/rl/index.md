# Reinforcement learning

ManiGuard trains grasp / pick-and-lift policies with **Stable-Baselines3 PPO**
on OmniGibson. The RL stack is self-contained SB3 — there is no external
distributed-RL dependency.

## Module layout (`maniguard/rl/`)

| Subpackage | Role |
|---|---|
| `algorithms/` | PPO training entry (`ppo.py`) + checkpoint eval (`eval.py`) |
| `envs/` | OG scene builder (`grasp_reset_scene.py`) + SB3 `VecEnv` adapters (`wrappers.py`: `build_vec_env`, `ManiGuardSB3VectorEnvironment`) |
| `tasks/` | `PickAndLiftTask` — grasp-hold detection + goal-region success termination |
| `training/` | shared trainer loop (`run_training`) + callbacks (checkpoint, W&B, viewer video, periodic eval) |
| `cli/` | shared argparse builders (`add_env_args`, `add_training_args`, …) |
| `grasps/` | GraspGen + cuRobo grasp-dataset lifecycle → `.pt` for `GraspDatasetResetter` |

## End to end

```
python -m maniguard.rl.algorithms.ppo --diagnostics-file <task>/diagnostics.jsonl --num-envs 4 ...
        │  cli/common.py: parse + validate args; set OG macros
        ▼
envs/wrappers.build_vec_env
   ├─ envs/grasp_reset_scene.build_config   (empty scene + Franka + target + PickAndLiftTask)
   └─ grasps/reset.GraspDatasetResetter      (samples grasps_<cat>_<model>.pt at each reset)
        ▼
SB3 PPO(MultiInputPolicy)  ──►  training.run_training (callbacks)  ──►  ckpts/ppo_final.zip
```

`GraspDatasetResetter` starts each episode from a *valid grasp* sampled from the
per-object `.pt` dataset, so PPO learns the lift/transport rather than re-solving
the grasp every reset. When no grasp dataset exists for the target, grasp reset
is skipped and training still runs.

## Entry points

| Command | Purpose |
|---|---|
| `python -m maniguard.rl.algorithms.ppo …` | train (see [PPO grasp training](../rl_training.md) for flags + scaling) |
| `python -m maniguard.rl.algorithms.eval …` | roll out a checkpoint and dump `metrics.json` |

## Grasp reset datasets (`rl/grasps/`)

The `grasps/` subpackage builds the per-object grasp dataset the resetter
consumes: `render_grasps.py` drives GraspGen (ZMQ) → cuRobo motion plan →
physics validation, writing
`outputs/grasp_datasets/<cat>_<model>/grasps_<cat>_<model>.pt`. See the
[GraspGen / cuRobo grasp pipeline](../graspgen_pipeline.md) for the full
install + run recipe.

## See also

- [PPO grasp training](../rl_training.md) — run recipe, key args, RTX-4090 scaling.
- [GraspGen / cuRobo grasp pipeline](../graspgen_pipeline.md) — building the reset dataset.
- [Environment layer](../foundations/env_layer.md) — controller presets used by the env.
