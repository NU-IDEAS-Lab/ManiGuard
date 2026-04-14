# SENTINEL-Lite — Dev Log

## 2026-04-14

### SO-101 leader arm → Franka teleop pipeline
- `0bc5529d` SO-101 → Franka teleop with HDF5 recording + playback: loads pipeline scene snapshots, swaps FrankaMounted → FrankaPanda (raised 0.5m), IK controller, 3 external cameras matching pipeline views.
- Gripper mapping rewritten to binary-native -1/+1 (was 0/1 with edge-case scaling through `command_input_limits=(0,1)` → binary's `target[0] >= 0` check sat on the boundary).
- Hotkeys: C = checkpoint, R = rollback, S = toggle success, Q = clean exit. Q is load-bearing because carb's SIGINT handler bypasses Python try/finally — Ctrl+C left HDF5 as a 96-byte truncated header.
- `so101_franka_playback.py` replays recorded trajectories. Default is fast-scrub (`include_contacts=False`, `include_robot_control=False`, `n_render_iterations=1`) — ~30x faster than the default physics-on path. `--with-physics` for strict re-simulation, `--record` to dump observations via `DataPlaybackWrapper`.
- `empty_scene_pipeline` standardised its per-episode snapshot on `og.sim.save()` so the JSON format matches BasePipeline's and the teleop entrypoint can consume either.
- `93baf896` gitignore local IDE config and the SO-101 hardware repo clone.

### Unified camera setup across task-gen / teleop / finetuning / eval
- `0a32cc76` `omnigibson/utils/camera_setup.py` introduces canonical constants (`EXTERNAL_CAMERA_NAMES = cam_opposite / cam_left / cam_right`, `CAMERA_RESOLUTION = 256`, `POLICY_EXTERNAL_CAMERAS_DEFAULT = ["cam_opposite"]`) and helpers `build_external_camera_configs`, `compose_main_image`, `normalize_policy_cameras`. Task-generation + teleop adopt the helper; both stay at Kit's default ~1280×720 since MP4s are for human review only and nothing downstream reads those pixels as policy input.
- `a087952e` RLinf renames `external_sensor0` → `cam_opposite` in both the yaml and code (`_apply_policy_camera`, `_extract_obs`, video-frame capture). `_extract_obs` now reads `sentinel_cfg.policy_external_cameras` and composes `main_images` via `compose_main_image` (single → unchanged; multi → concat along W). Policy I/O contract unchanged. Existing LIBERO SFT ckpt stays compatible — same camera pose, new obs-dict key.
- `06a97512` `tools/evaluate_benchmark.py` gets `--policy-external-cameras` and `--camera-resolution` CLI; `extract_obs` routes through the same `compose_main_image` so train/eval see identical tensor layout. Eval applies the `image_height/width` setter + `env.load_observation_space()` workaround because Kit viewport init silently overrides `sensor_kwargs` (StanfordVL/OmniGibson#266, #1875).
- Not resolving: V1 profile's `external_camera_resolution=(240, 416)` kept as-is (no ckpt pressure yet, but haven't confirmed a retrain target).

## 2026-04-13

### placeable_surfaces_v1.json: filter not-ready models from B1K knowledgebase
- `build_placeable_surfaces.py` now drops any `(category, model)` flagged as not-ready in `bddl3/bddl/generated_data/complaints.json` (objects with unprocessed QA complaints). The knowledgebase's `ready` property considers only unprocessed complaints (`bddl3/bddl/knowledge_base/processing.py:484`).
- Counts: 192 → 145 models, 218 → 165 surfaces, 14 → 13 categories (`lab_table` dropped entirely — all 3 models had unprocessed complaints).
- 47 skipped models are listed verbatim in the JSON's `skipped_not_ready` field for auditability.
- Skipped by category: desk (19), coffee_table (10), console_table (5), lab_table (3), bar (2), commercial_kitchen_table (2), checkout_counter / conference_table / countertop / nightstand / pedestal_table / reception_desk (1 each).
- Dominant unprocessed complaint types: `nth-metalink` (30), `material` (13), `appearance` (7) — all cosmetic / metadata issues, none structural (no `collision`/`scale`/`joint`/`category`).
- Override: pass `--no-require-ready` to keep not-ready entries (for debugging).

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
