# SENTINEL-Lite — Dev Log

## 2026-04-19 — `refactor/omnigibson` branch: BEHAVIOR-1K + RLinf → submodules, sentinel/ is now the only owned tree

Large restructure landed on `refactor/omnigibson` over 12 commits. End state: the repo is ~99K tracked LOC (177 files) + two submodules (behavior-1k @ v3.7.2, RLinf @ 5714022) — down from ~3.58M LOC / 19K files of mixed-origin sources sitting at the root.

**Sentinel extraction**:
- `OmniGibson/omnigibson/{task_generation,utils}/*` and our object states (`dropped`, `upright`) moved out of the OmniGibson tree into `sentinel/{task_generation,utils,object_states}/`. Import paths were rewritten in every consumer; `sentinel/task_generation/__init__.py` placeholder was replaced with the real tree.
- `sentinel/utils/ltl_utils.py` was confirmed 100% Sentinel code (not in StanfordVL/BEHAVIOR-1K at any version) and moved out. Likewise for `test_ltl_monitor.py` / `test_ltl_propositions.py`, which now live at top-level `tests/`.
- Four Franka-cached yamls (`franka_mounted_behavior_cached*.yaml`) moved to `sentinel/configs/` with a `config_path()` helper; upstream demo scripts (`grasp_task_demo.py`, `grasp_policy_demo.py`, etc.) moved to `sentinel/examples/`.

**Runtime patches on vanilla OmniGibson**: `sentinel/_omnigibson_patches.py` is called from `sentinel/__init__.py`. It does three things:
1. Installs a `sys.meta_path` post-load hook on `omnigibson.object_states` that injects `Dropped`, `Upright`, and a `Grasped` alias for `IsGrasping` the moment the subpackage finishes importing — so downstream modules (e.g. `bddl_utils.SUPPORTED_PREDICATES`) that reference `object_states.Grasped` at module-load time see them.
2. Eager monkey-patches on already-loaded modules: `factory._DEFAULT_STATE_SET` gets `Dropped` + `Upright`; `GraspGoal` learns a `hold_steps` counter; `GraspReward` falls back to `root_link` when the robot has no `torso_lift_link`; `sampling_utils.draw_debug_markers` becomes tensor-dtype/device safe.
3. Registers four custom BDDL predicates (`upright`, `dropped`, `grasped`, `stashed`) on `omnigibson.utils.bddl_utils.SUPPORTED_PREDICATES` via `sentinel.utils.bddl_predicates`.

`sentinel/tasks/sentinel_grasp_task.py` is a `Registerable` subclass of `GraspTask` that fixes three upstream quirks SENTINEL relies on: commented-out `GraspGoal` registration (revived, now with `hold_steps`), missing `trunk_control_idx` on mounted Franka, and list-vs-tensor reset-pose dicts. Task configs now reference `type: SentinelGraspTask`. Set `SENTINEL_SKIP_OMNIGIBSON_PATCH=1` to opt out of everything.

**BDDL generator change**: `sentinel/utils/bddl_generator.py` now always emits `agent.n.01_1` + `floor.n.01_1` in `:objects` and `(ontop agent.n.01_1 floor.n.01_1)` + `(inroom floor.n.01_1 <support_room>)` in `:init`. Upstream BDDLSampler rejects tasks that leave the agent without a kinematic init in non-empty scenes; sentinel's pipelines override the pose via `place_franka_edge_aligned` after sampling, so the floor placement only exists to satisfy upstream's validity check.

**OmniGibson tree restored to pristine v3.7.2** (zero tracked drift vs `StanfordVL/BEHAVIOR-1K@v3.7.2`): `object_states/{__init__,factory,robot_related_states}.py`, `tasks/{behavior_task,grasp_task}.py`, `termination_conditions/grasp_goal.py`, `reward_functions/grasp_reward.py`, `utils/{bddl_utils,sampling_utils}.py`, `envs/env_base.py`, plus a small handful of root-level files and demo scripts. Dead code dropped along the way: the entire LTL integration inside `behavior_task.py` (we had our own), `inroom_object_name_whitelist` plumbing (only curation used it, and curation was gone), agent-pose-stash-and-restore logic (pipelines always place the robot explicitly), and the `env_base.py` `update_ltl_monitor` hook.

**Submodule conversion**:
- Seven BEHAVIOR-1K upstream directories (`OmniGibson/`, `bddl3/`, `joylo/`, `docs/`, `asset_pipeline/`, `knowledgebase/`, `eval-jobqueue/`) and root files (`setup.sh`, `setup.ps1`, `mkdocs.yml`, `ruff.toml`) collapsed into a single `behavior-1k/` submodule pinned at the v3.7.2 tag (commit 88454bd04).
- `RLinf/` replaced with a submodule pinned at 5714022 (`feat: support custom model registration (#920)`).
- `.gitmodules` now lists both.

**Other cleanups**:
- Franka-cached-config chain (`franka_mounted_mvp_runner_*.py` + their yamls + the kitchen-bar activity it depended on) deleted; none of it was exercised anymore.
- 11 hand-drafted safety BDDLs in `bddl3/bddl/activity_definitions/` (`clean_surface_near_food`, …) deleted — `retrieve_filled_cup_from_clutter_safely` was the only referenced one, and its only consumers (the dead cached config + a broken test) were also removed.
- `scene_generation/` (untracked leftover) removed from disk.
- `checkpoints/`, `Isaac-GR00T/`, `GR00T-N1.6-DROID/`, `RLinf-Gr00t-SFT-Stack-cube/`, `RLinf-pi05-SFT-Stack-cube/`, `RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT/` — six weight directories (~42 GB on disk, all gitignored) moved into `vla_models/`. Tracked code that hardcoded the old paths (sentinel/openpi/configs.py, sentinel/serve/gr00t_server.py, the SFT goblet yaml, prepare_sft_data.sh, etc.) was updated.
- Root `setup.sh` and `my_first_env.py` deleted — the authoritative `setup.sh` now lives inside `behavior-1k/`; sentinel install is lightweight enough that a wrapper was just drift.
- README rewritten for the new layout; test expectations updated for the agent+floor BDDL shape; defunct `tests/test_ltl_propositions.py` dropped.

**Test status**: `pytest tests/` reports 135 passed, 3 pre-existing failures (same three that failed on `dev` before this branch), 1 skipped. Full-sim smoke tests on `clutter_scene_pipeline` (non-empty scene) and `empty_scene_pipeline --setup stack` both complete with `Gate: pass=True, Episode done, violated=False`.

**Known follow-up**: sentinel is not yet pip-installable (no `pyproject.toml`); Isaac-GR00T and friends still live in `vla_models/` as user-downloaded weight dirs rather than submodules. Neither blocks merge.

## 2026-04-16

### VectorEnvironment validation
- Confirmed `og.VectorEnvironment` works with benchmark scene snapshots on RTX 4090: 2 and 3 parallel envs, partial room loading via `load_room_instances`, DummyTask, headless and GUI modes all pass.
- `scripts/test_vec_env.py` — standalone test that discovers scenes from an HF-style dataset directory, builds OG configs from `scene_ep1.json` + `diagnostics.jsonl`, and runs N envs with random actions. Includes viewer camera auto-positioning to frame all envs.
- Viewer camera teleoperation (`sim.enable_viewer_camera_teleoperation()`) confirmed working for visual inspection of multi-env runs.

### Action item: switch SentinelEnv to VectorEnvironment for RL training
- Currently `SentinelEnv` creates N independent `og.Environment` instances and steps them sequentially in a Python loop. `og.VectorEnvironment` shares a single `og.sim.step()` across all envs — should give a significant speedup for same-scene multi-seed training.
- Plan: add a `use_vec_env: true` option to `sentinel_cfg`. When enabled and all envs share the same scene config, SentinelEnv delegates to `og.VectorEnvironment` internally instead of managing individual env instances. The RLinf interface (`step`/`chunk_step`/`reset`) stays unchanged.
- Prerequisite: `VectorEnvironment` currently takes a single config dict. For heterogeneous scenes (different trials), either extend it to accept a list of configs or restrict vec-env mode to same-scene runs.
- Connects to: `configs/rl/sentinel_clutter_ppo_openpi_pi05.yaml`, `sentinel/envs/sentinel_env.py`, the HF-downloaded taskgen dataset at `sentinel-lite-taskgen-staging/`.

## 2026-04-15 (continued)

### Decouple teleop from OmniGibson
- Moved the three Sentinel-specific teleop sources out of `OmniGibson/` into `sentinel/teleop/`:
  - `OmniGibson/omnigibson/teleop/so101_teleop.py` → `sentinel/teleop/so101_teleop.py`
  - `OmniGibson/omnigibson/examples/teleoperation/so101_franka_teleop.py` → `sentinel/teleop/so101_franka_teleop.py`
  - `OmniGibson/omnigibson/examples/teleoperation/so101_franka_playback.py` → `sentinel/teleop/so101_franka_playback.py`
- Dropped the now-empty `OmniGibson/omnigibson/teleop/` package (it was ours from day one; upstream never had it).
- Rewrote internal imports: `omnigibson.teleop.so101_teleop` → `sentinel.teleop.so101_teleop`. Updated `teleop_bridge/README.md` launch example to `python -m sentinel.teleop.so101_franka_teleop`.
- Populated `sentinel/teleop/__init__.py` with the module docstring that pointed to this follow-up.
- OmniGibson subtree is now one step closer to pure-upstream; remaining Sentinel-specific code there is only `omnigibson/task_generation/*` and a few utility helpers (`camera_setup.py`, LTL hooks). Those stay for now because they're deeply integrated into the pipeline and OmniGibson utility surface.

## 2026-04-15

### Teleop → SFT data pipeline (Pi0.5 / OpenPI)
- `tools/playback_teleop_to_hdf5.py` Stage 1: DataPlaybackWrapper replay with real physics; extracts cam_opposite RGB, wrist RGB, and 8D state (`eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)`). State layout mirrors IsaacLab-Stack-Cube's `_wrap_obs` so Pi0.5 ckpts trained on that dataset transfer cleanly as SFT init.
- `tools/hdf5_to_lerobot.py` Stage 2: flat HDF5 → LeRobot v2.1 parquet + AV1 MP4 via `LeRobotDataset.create()`. Feature keys (`image`, `wrist_image`, `state`, `actions`) line up 1:1 with RLinf's `OmniGibsonDataConfig.RepackTransform`.
- `tools/compute_norm_stats.py` Stage 3a: mean/std/q01/q99 per feature, written to `{ckpt}/assets/{asset_id}/norm_stats.json` for openpi to pick up.
- `tools/prepare_sft_data.sh` wraps the three stages idempotently (skip if outputs exist) and prints the final SFT launch command.
- `RLinf/rlinf/models/embodiment/openpi/dataconfig/__init__.py` +1: `TrainConfig(name="pi05_sentinel_goblet", ...)` — `OmniGibsonDataConfig(repo_id="sentinel/goblet_pick_place", prompt_from_task=True, extra_delta_transform=False)` with 7D EEF delta actions.
- `RLinf/examples/sft/config/sentinel_goblet_sft_openpi.yaml`: new launch config pointing at the stack-cube ckpt for init + local LeRobot dataset. LoRA + bs=1 + no value head to fit on 31 GB host RAM (RAM OOM on full-param SFT).

### Stage 1 camera setup bug
- First version of `playback_teleop_to_hdf5.py` didn't position `cam_opposite` after `DataPlaybackWrapper` built the env — sensor stayed at the default pose → every frame was a near-uniform gray image (std ≈ 4.4 on uint8). Caught by eyeballing the exported MP4 after the batch had run 11/21 trajectories; killed and restarted.
- Fix: `_setup_cameras_from_scene(env)` calls `task_generation.utils.video.{build_video_view_specs, setup_cameras}` with the scene's `support_surface` + any non-robot object as `target_obj`, reproducing the opposite-side overview teleop saw. Post-fix std jumped to ~30.
- Isaac Sim's `~/.cache/ov/_cache.lock` was retained after SIGKILL of the previous Kit process, causing the next run to hang silently at `app ready`. Manually removed.

### Decouple Sentinel from RLinf + reorganize into `sentinel/` package
- Bumped vendored RLinf to upstream `5714022` (`feat: support custom model registration`). Old locally-patched copy preserved at `RLinf_local/` for one cycle, then deleted to reclaim 17 GB. RLinf/ subtree is now byte-equivalent to upstream.
- Extracted every Sentinel-specific piece that previously lived inside `RLinf/rlinf/envs/sentinel/`, `RLinf/rlinf/models/embodiment/openpi/{dataconfig,policies}/omnigibson_*.py`, and `RLinf/examples/embodiment/config/sentinel_*.yaml` into an out-of-tree `sentinel_ext/` package. RLinf is no longer edited; patches applied via `sentinel_ext/rlinf_patches.py` at import time: (1) extends `rlinf.envs.SupportedEnvType` Enum with `SENTINEL` member via `_member_map_` / `_value2member_map_` direct mutation (EnumMeta.__setattr__ forbids setattr on registered names, so we skip that step — `__getattr__` falls back to the map), (2) wraps `rlinf.envs.get_env_cls` to dispatch sentinel→`SentinelEnv`, (3) wraps `rlinf.config.validate_embodied_cfg` to inject the `franka_mounted_sentinel` omnigibson_cfg.
- Ray workers don't inherit the launcher's imports. `sentinel_ext/_autoimport/sitecustomize.py` + prepending that dir to `PYTHONPATH` makes every Python interpreter under the training PATH auto-import sentinel_ext on startup, catching both the main process and all Ray actors.
- Renamed the whole package `sentinel_ext/` → `sentinel/` (more canonical now that it's our main package, not just an RLinf ext). Reorganised: `tools/*.py` split by purpose — SFT data prep → `sentinel/data/`, evaluation → `sentinel/eval/`, inference servers → `sentinel/serve/`. Shell launchers → `scripts/`. Empty `sentinel/task_generation/` and `sentinel/teleop/` placeholders reserved for the future OmniGibson decouple. `sentinel/__init__.py` is now side-effect-free; RLinf/openpi-coupled registrations are explicit imports in `sitecustomize.py` and `sentinel/launchers.py`, so `python -m sentinel.eval.benchmark` works in the `behavior` conda env (no `openpi` needed).

### End-to-end eval (websocket policy + OmniGibson) verified on `RLinf-pi05-SFT-Stack-cube` 0-shot
- Policy server (`sentinel/serve/pi05_franka.py`) and eval client (`sentinel/eval/benchmark.py`) on same box, GPU shared (RTX 4080 Laptop 12 GB). Ran 30-step, 600-step, 1500-step, and 3000-step rollouts on `clutter_goblet_00`; all completed without crashing the server.
- Four independent fixes were needed to make eval actually work end-to-end:
  1. **Pipeline Scene vs InteractiveTraversableScene**: benchmark.py assumed every snapshot was an `InteractiveTraversableScene` with a named `scene_model`. Pipeline-generated snapshots use a bare `Scene` with everything baked in via `scene_file`. Branching on `init_info.class_name` now handles both.
  2. **Double-robot instantiation**: for Scene-type snapshots, the robot is inside `scene_file`; passing an extra `robot_cfg` causes a conflict. Skip the explicit robots block when `class_name == "Scene"`.
  3. **Obs schema gaps**: upstream RLinf's `openpi_action_model.obs_processor` now unconditionally reads `env_obs["extra_view_images"]`. Added `extra_view_images: None` to `extract_obs`. Also upgraded `states` to the 8D layout (`eef_pos+axisangle+gripper_qpos(2)`) so the schema matches both the training-side playback output and IsaacLab stack-cube's `_wrap_obs`.
  4. **Eval cameras stuck at default pose**: the previous bug from `playback.py` Stage 1 was never back-ported to eval — `cam_opposite` stayed at the world origin, so every rendered frame was near-uniform gray (std ≈ 4). Added `_setup_eval_cameras(env)` that calls `task_generation.utils.video.{build_video_view_specs, setup_cameras}` post-reset, mirroring the teleop + playback rigs. Post-fix frame std ≈ 31.
- `sentinel/data/rewrite_scene_robot.py` — reusable utility that clones a benchmark scene directory and rewrites the snapshot to swap `FrankaMounted`→`FrankaPanda`, raise the base by 0.5 m, install an `InverseKinematicsController`, and stub out the saved OSC controller goals. This is the same transform `so101_franka_teleop._build_from_snapshot` does on the fly, but persisted so eval/SFT consume a benchmark that already matches the teleop training distribution. Produced `clutter_goblet_00_frankapanda/` as an example.

### Follow-ups (not done today)
- **D435 sim preset**: OmniGibson's `VisionSensor` exposes `focal_length`/`horizontal_aperture` but no built-in D435 profile; RLinf has `CAMERA_INTRINSICS` only for A1/R1Pro (not D435, not Franka), and `envs/realworld/camera.py` uses `pyrealsense2` on real hardware only. Plan: add `preset="d435"` to `camera_setup.build_external_camera_configs` — `focal_length ≈ 15.17 mm` matches D435's 69.4° horizontal FOV at the default 20.995 mm aperture. Open questions before implementing: which cameras (wrist / cam_opposite / both), resolution (256² vs 192×256 for D435's native 4:3), and whether to re-render the existing 21 teleop trajectories (~1.5 h).
- **0-shot eval success signal**: `success=True` at step 590 on the pre-camera-fix run was almost certainly a heuristic false positive — the policy was seeing gray frames, so actions were ≈ random, and the robot happened to brush the goblet hard enough to trigger `check_grasp`. Post-fix runs give `False` at 600 steps, which is the more honest baseline. Better: tighten the success check to require held-during-lift or `OnTop(goblet, plate)` rather than a single contact.
- **OmniGibson decouple**: same pattern as RLinf — move `OmniGibson/omnigibson/{task_generation,teleop}/*` and our OmniGibson patches out into `sentinel/task_generation/` + `sentinel/teleop/` + `sentinel/omnigibson_patches/`, replace `OmniGibson/` with an upstream clone. Placeholders already exist.
- **SFT actually running**: full-parameter Pi0.5 SFT OOM'd the 31 GB host RAM after FSDP worker init. Pipeline is verified end-to-end up to model-weight load; actually training likely needs a larger-RAM box or LoRA at a smaller rank than 32.

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
