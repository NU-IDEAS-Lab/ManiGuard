# Task Generation Pipeline — Dev Log

## 2026-04-09

### Empty-scene pipeline refactor
- `7c494752` batch mode (`--batch-size`), refactored `run_sim` into `_run_episode_inner`, BDDL generation now receives resolved object selection for consistency

## 2026-04-08

### Object-first scene selection architecture
- `53053df5` partial room load + stashed object placement — 6x speedup on large scenes (office_large: 576s → 95s), fixes sampler crash on crowded tables
- `5574c571` stashed init for stack and transfer BDDL — all 4 pipelines now use stashed
- `d8335c74` object-first scene selection architecture — `build_scene_surface_catalog`, `estimate_object_set_footprint`, `select_objects` abstract method, auto-scene when no `--scene-model`
- `8f81ecaf` implement `select_objects` for all 5 pipelines — clutter, stack, transfer, liquid, wet transport
- `c23439fd` benchmark auto-scene — `run_benchmark` defaults to auto-select mode with `--num-trials`, `--scenes` for explicit scene list

### Lid-before-transport pipeline (temporal Until constraint)
- `0705b1d5` lid-container pair registry (20 verified attachment pairs from asset metadata) + food/liquid activity generators + LTL Until constraint
- `167be100` LidTransportPipeline (food) + LidLiquidTransportPipeline (teapot/kettle with water), `--lid-mode food|liquid`

### Empty-before-invert pipeline (temporal Until + particles on surface)
- `pending` EmptyInvertPipeline — liquid-filled container on table, must empty before inverting, table must stay dry
- New evaluators: `inverted` (tilt > 120°), `particles_on_surface` (water ContactParticles on table)
- LTL: `(!container_inverted) U (!container_filled)` + `G(!water_on_table)`

### Wet transport pipeline (overhead forbidden)
- `3594851c` `overhead_forbidden` evaluator in `SafetyPropositionEvaluator` — first distance-based safety check
- `bb6c92aa` wet transport pipeline — liquid container + water-sensitive zone objects (books, laptops, keyboards)
- `b83300fa` taxonomy update with distance-based + temporal order action items
- Fixed `tablet.n.01` → `tablet.n.05` synset, added `monitor.n.04` to WATER_SENSITIVE_POOL
- Changed from wet sponge (`Covered(water)` doesn't stick to rigid bodies) to filled container (`Filled(water)` is stable)

### Known issues
- **Liquid pipeline + auto-scene: many scenes segfault with GPU dynamics.** Isaac Sim crashes on certain scenes when `USE_GPU_DYNAMICS=True` (bookcase emitter orthogonal transform assertion, and other scene-specific crashes). Need to build a whitelist of GPU-dynamics-compatible scenes or add retry logic with scene fallback.
- **Transfer: food slides off source.** Bread on frying_pan consistently fails OnTop gate check. Need better food-source compatibility matching or physics stabilization.

## 2026-04-07

### Pipeline simplification
- `e196347f` simplify scene clearing — merged 3 clear functions into `clear_support_area`, fixed `compute_floor_z` (was always 0.0)
- `255e8356` unconditional support pinning, removed `stabilize_support_object`
- Deduplicated aabb computation (compute once after pin + clear)

### Stack pipeline: 3 task variants
- `30d757f6` expanded footprint catalog with 27 new categories
- `79fbcd82` split into StackSamePipeline / StackFlatPipeline / StackReceptaclePipeline
- Per-role model pinning (same-synset target and stack get different models in flat/receptacle)
- Fixed `identify_objects` bug where same-synset objects all went to target
- `c2a2564f` task taxonomy doc

### Liquid transport refactor
- `40f26930` LiquidTransportPipeline inherits ClutterPipeline, expanded LIQUID_CONTAINER_POOL to 20 synsets (141 models)
- `2082be39` fixed Tensor JSON serialization in diagnostics
- `e728ccf0` taxonomy update

### Food transfer improvements
- `d6b0c109` place source/dest on table (was relying on sampler which placed them elsewhere)
- `4fe73e04` expanded pools: 19 food, 9 source, 20 dest (~3,344 combos)
- `44ce0630` taxonomy update
