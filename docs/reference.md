# Reference

A one-page map of the package: where each piece lives, its main entry point, and
the section that documents it.

## Module map

| Package | Role | Docs | Main entry point |
|---|---|---|---|
| `task_generation/` | generate frozen task scenes | [Task Generation](pipelines/index.md) | `python -m maniguard.task_generation.<name>_pipeline` · `…run_benchmark` |
| `data/teleop/` | SO-101 / GELLO human teleop | [Data Collection](data_collection/index.md) | `python -m maniguard.data.teleop.{so101,gello}_franka_teleop` |
| `data/datagen/` | scripted sim demo collection → LeRobot | [Sim datagen](datagen/pipeline.md) | `python -m maniguard.data.datagen.{driver,sweep,to_lerobot}` |
| `data/playback.py` | render teleop HDF5 → SFT obs | [Data Collection](data_collection/index.md) | `python -m maniguard.data.playback` |
| `data/lerobot/` | teleop HDF5 → LeRobot export + norm stats | [SFT](sft/index.md) | `python -m maniguard.data.lerobot.{multitask_lerobot_export,norm_stats}` |
| `data/real_teleop/` | real npz → LeRobot (DROID joint) | [SFT (real)](sft/dataset_and_config.md) | `python -m maniguard.data.real_teleop.real_teleop_to_droid` |
| `data/scene/` | benchmark-set prep | [Evaluation](evaluation/index.md) | `python -m maniguard.data.scene.{benchmark_repair,trim_scene_to_room,rewrite_scene_robot}` |
| `eval/` | policy benchmark loop | [Evaluation](evaluation/index.md) | `python -m maniguard.eval.benchmark --config <yaml>` |
| `serve/` | websocket policy server | [Evaluation](evaluation/index.md) | `python -m maniguard.serve.openpi_native` |
| `utils/` | LTL safety, `task_spec`, geometry | [LTL safety](foundations/ltl_safety.md) | *(library)* |
| `envs/` | scene registry + frozen-snapshot runtime | [Environment layer](foundations/env_layer.md) | *(library)* |
| `object_states/` | `Dropped`, `Upright` | [LTL safety](foundations/ltl_safety.md) | *(library)* |
| `_omnigibson_patches.py` | runtime OmniGibson patches | [OmniGibson patches](foundations/omnigibson_patches.md) | applied on `import maniguard` |

## Environment variables

These keep the `SENTINEL_` prefix for backward compatibility.

| Variable | Effect |
|---|---|
| `SENTINEL_SKIP_OMNIGIBSON_PATCH=1` | skip all runtime patches (no simulator needed) |
| `SENTINEL_SKIP_LONGFINGER=1` | skip the long-finger Franka asset patch |
| `SENTINEL_AG_SUBSTEP_INTERVAL=N` | fire assisted-grasp once per N physics substeps |
| `SENTINEL_RUNTIME_PYTHON` | torch-capable interpreter for frozen-task replay |
| `OMNIGIBSON_HEADLESS=1` · `VK_ICD_FILENAMES` | headless GPU rendering |
| `OMNIGIBSON_DATA_PATH` | override the BEHAVIOR dataset root |
| `CUDA_VISIBLE_DEVICES` | GPU selection |

## Running tests

```bash
conda activate behavior
pytest tests/ -v        # maniguard-side LTL + task-gen + eval unit tests
```

## See also

- [Architecture overview](architecture/overview.md) — the lifecycle and repo layout.
- [Add a custom pipeline](pipelines/custom_pipeline.md) — extending task generation.
- [Controller, data, action & eval](sft/end_to_end.md) — keeping training and eval consistent.
