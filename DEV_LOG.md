# ManiGuard — Dev Log

## Action items (deferred)

- **lid_transport Phase 2B: success check fires once at end of trajectory, not per-step.**
  In `tools/lid_transport_from_dataset.py:715–739` the goal-success
  check (`pos_ok && still_held && lid_still_attached`) is evaluated
  ONCE after `_replay_holding` finishes the full lift→hover→descend
  trajectory. Even if the container's AABB enters the goal sphere
  mid-descend (which it does, by construction — descend's final
  waypoint IS the goal center), the gripper keeps executing the
  planned trajectory to the last waypoint before we check.
  - **Downside 1 (wasted motion)**: episodes are longer than they
    need to be — the trajectory often satisfies the goal a dozen
    waypoints before the planned end.
  - **Downside 2 (latent false-negative)**: the container can
    intersect the goal sphere mid-descend, then settle 1–2 mm out of
    the sphere by the time the last waypoint executes (PhysX contact
    resolution as the container touches the desk surface or shifts
    in the gripper). We then misreport `success=False` even though
    the task was achieved.
  - **Fix sketch**: per-step early-termination inside `_replay_holding`,
    gated to lid_transport Phase 2B: each sim step, evaluate
    `target_or_gripper_in_goal(env, container, spec)
     && robot_holds_target(env, container)
     && lid.states[AttachedTo].get_value(container)`; on True, stop
    executing further waypoints and exit Phase 2B with success.
  - **Why deferred**: the 5-trajectory smoke set at
    `outputs/lid_edge_sft/` succeeded with the end-of-trajectory
    check; it's not blocking SFT recording today. Revisit before
    scaling collection to multiple tasks or before any RL eval that
    cares about action efficiency.

- **Refactor task-family explosion in `perturbation_scaling.py` (3591 lines) and `snapshot_validator.py` (2148 lines).**
  Both files came in via the merge of `feat/task-generation-scene-benchmark-staging` (commit b841084a) and grew via per-family copy-paste rather than per-family hooks over a generic core.
  - `perturbation_scaling.py`: 80+ top-level functions, 5 perturbation categories × 7 families = 35 combinations each with a custom `_materialize_*` path; 3 separate prompt-variant generators (`_generic_pickup_prompt_variants`, `_transfer_prompt_variants`, `_lid_prompt_variants` ~150 lines combined). Refactor sketch: lift the orchestrator (`scale_base_task_set`, 210 lines) and let each family register a `(materialize, prompt_variants, role_inference)` triple via a registry; per-family code drops from ~300 lines to ~50.
  - `snapshot_validator.py`: 5 family-specific runtime check functions (`_runtime_check_clutter`, `_runtime_check_stack`, `_runtime_check_lid_transport_food/liquid`, ~445 lines combined) all do the same 80%: load task → step env → check goal_region + LTL → save review video. Refactor: one `_runtime_check_generic(family, family_specific_hooks)` with per-family hooks of ~30-50 lines each.
  - **Why deferred**: the file is in active development on `feat/task-generation-scene-benchmark-staging`. Refactoring without coordination guarantees merge conflicts. Coordinate with @666harrypeng before touching.
  - **Until refactor**: when adding new families or perturbation types, resist adding more `_<family>_*` siblings. Push toward generic-path-with-hooks even if it requires touching the orchestrator.

- **Port OmniReset's iterative Jacobian DLS IK into `GraspDatasetResetter`.**
  Phase 2 of the grasp-reset story (Phase 1 is the 2026-04-21 ``reset_mode='cached'``
  path — fast path, no IK). Current ``ik`` mode uses cuRobo, which is blocked
  under multi-env by OG's ``assert len(og.sim.scenes) == 1`` in
  ``CuRoboMotionGenerator.__init__``. OmniReset uses
  ``DifferentialIKControllerCfg(ik_method='dls')`` (~25-iter Jacobian-based
  DLS, CPU-friendly, scene-agnostic). Porting it would:
  1. Let ``reset_mode='ik'`` run under ``num_envs > 1`` (cuRobo-free).
  2. Enable per-reset pose randomization (``pose_range_b`` applied in eef
     body frame), which is the prerequisite for OmniReset-style generalization
     training — policy sees diverse init configurations instead of the single
     fixed object pose the cached path produces.
  ~40-80 line change in ``sentinel/rl/grasps/reset.py``: replace
  ``_curobo_ik`` with a ``_dls_ik`` that solves ``J^T (J J^T + λ I)^{-1} Δx``
  iteratively via PyTorch, initialized from the saved ``arm_joint_pos``.
- **Port the rest of OmniReset's reset zoo into `GraspDatasetResetter`.** Current
  port covers only `ObjectRestingEEGrasped` (object on surface, ee at saved
  grasp). OmniReset defines 5 distributions in
  ``Omnireset/source/.../reset_states_cfg.py``:
  1. `ObjectAnywhereEEAnywhere` — free-space reach (no grasp, non-trivial
     randomized object + ee poses)
  2. `ObjectRestingEEGrasped` — **ours**
  3. `ObjectAnywhereEEGrasped` — object airborne, ee holding (mid-carry start)
  4. `ObjectPartiallyAssembledEEAnywhere` — insertion-task specific
  5. `ObjectPartiallyAssembledEEGrasped` — insertion precision
  OmniReset uses `MultiResetManager` to mix them with tunable probabilities +
  curriculum. For our `PickAndLiftTask` the most relevant additions are #1
  (policy learns reach+grasp from scratch) and #3 (policy learns carry from
  any airborne start). Implementation: extend `GraspDatasetResetter` with
  airborne-target + random-ee-pose modes, then expose ``--reset-mix`` CLI
  kwarg wiring a distribution over modes.
- **Patch OG upstream `CuRoboMotionGenerator.__init__` single-scene assertion**
  (alternative to DLS port — upstream PR, riskier timeline).

- **Replace `render_grasps.py` Phase A with NVlabs/GraspDataGen-style
  free-floating gripper validation.** Spawn a Panda gripper prim alone
  in OG, teleport to the GraspGen eef pose, close, run 5-direction tug
  test (~2.5 s). Gravity off globally so no hard pin, no
  gripper-sweeping pathology, no cuRobo. Per-grasp ~50 s → ~1-3 s.
  New script `sentinel/rl/grasps/validate_grasps_floating.py`; saved
  ``.pt`` drops `arm_joint_pos` so `GraspDatasetResetter` must run in
  `ik` mode (or use the DLS port above) instead of `cached`.

- **Finetune our `GraspGen` ckpt on BEHAVIOR-1K via GraspDataGen.**
  Current `graspgen_franka_panda` was trained on ACRONYM (ShapeNet);
  BEHAVIOR-1K thin/floppy/hollow objects are under-represented and
  account for the bulk of our ~33% Phase A failures. Workflow: Isaac
  Lab venv, export BEHAVIOR visual meshes via `mesh_from_og_object`,
  generate ~4.4 M labeled grasps (1024 envs in parallel), finetune
  `*_gen.pth` ~5-20 epochs at low LR, drop into the existing ZMQ
  server (no `render_grasps.py` change). Defer until free-floating
  validation lands AND reset failures actually hurt RL convergence;
  Isaac-Lab/OG sim2sim gap means a held-out OG re-label is needed
  post-finetune.

## 2026-06-08 — `feat/teleop_joint_config`: warmup hold-current fix; init-pose OOD investigation (reverted) (`63fdfcb0`)

Eval warmup now holds the current arm joints instead of commanding np.zeros on the absolute-JointController path (a zero action drove every joint to 0, flinging the arm; latent bug, inconsequential today since 6fam-base scenes load near-zero). Separately investigated why the gello-teleop families never grasp: 6fam-base spawns the arm at a near-zero default (elbow -0.07), but the gello demos all start bent over the table (elbow -1.14..-2.0), so the policy is OOD from step 0. Confirmed via clutter as a control — cuRobo doesn't override the init, so clutter's eval start is in-distribution (q99 elbow=-0.07) and it grasps/succeeds while dusty doesn't. Tried to fix the init (reset_arm_qpos: runtime teleport, then bake-into-scene spawn) but joint_position_raw can't hold the tight bent init — it sags ~0.14 rad to elbow -1.0 whether driven or spawned. Reverted reset_arm_qpos; the gello failures are most likely the undertrained checkpoints (3380/1025/4000 steps vs clutter's 26999) plus drive stiffness, not an eval bug — not fixable via eval-side patches.

## 2026-06-08 — `feat/teleop_joint_config`: explicit grasping_mode override + per-run output folders (`3b962861`)

The eval robot now takes an explicit `grasping_mode` (config field + `--grasping-mode`), forced on after scene load. 6fam-base scenes bake their own grasp mode, but a pi0.5 policy only grasps under the mode its teleop data was collected in — and that mode is not recorded in the dataset — so a mismatch (scene `physical`/`assisted` vs the policy's training mode) means the learned gripper actions never weld an object. Per-family joint YAMLs now pin the actual teleop mode (dusty=`sticky`, the other five=`assisted`). Separately, each run writes to `output_dir/<run_name>` instead of a flat dir, so successive runs never overwrite: `run_name` defaults to a `YYYYmmdd_HHMMSS` timestamp, loud-suffixed `_<TAG>` when `--tag smoke` is passed (and `_<n>` on the near-impossible same-second collision), and `output_dir` in each YAML is now the per-config base under `outputs/eval_logs/`. `run_benchmark_all_scenes.sh` fixes one `run_name` up front and passes the same `--run-name` to every per-scene process, so a family batch lands in a single folder (scenes told apart by file prefix) rather than one folder per scene. This is the interface/infra; whether the corrected grasp mode lets the policies grasp is a separate, still-open action/norm trace.

## 2026-06-07 — `feat/teleop_joint_config`: register ManiGuard configs in the native serve entry (`0deff44a`)

`maniguard/serve/openpi_native.py` resolved its config via openpi's `get_config()` but never registered ManiGuard's pi0.5 SFT TrainConfigs, so serving a joint checkpoint (`--config pi05_base_<family>_joint_2cam_lora`) would fail with an unknown-config error — those configs only attach to openpi's registry when `maniguard.openpi_sft.train_configs.register()` runs, and neither `maniguard.__init__` (OmniGibson patches only) nor `serve/__init__` (empty) calls it. `main()` now imports + `register()`s before `get_config` (guarded; a warn-and-continue no-op for stock openpi config names). Confirmed in the ideas2 openpi venv that `register()` makes `get_config` resolve `pi05_base_dusty_transfer_joint_2cam_lora`. Unblocks serving the six joint eval checkpoints (prep for the ideas2 end-to-end eval).

## 2026-06-07 — `feat/teleop_joint_config`: eval metric switch + single-task selection (`8735f568`)

eval now takes a `metrics` config/CLI knob — any non-empty subset of `{success, safety}` (`--metrics success` / `--metrics safety` / `--metrics success safety`, default both). It gates which checkers run: success-only skips the LTL monitor **and** the Spot fail-fast (no Spot dependency when you only want task success); safety-only skips the goal checker and runs the full rollout (no early stop on the goal) while recording violations. Per-scene results record `metrics` and set `success` to null when it wasn't evaluated; the summary prints/serialises the success rate only under `success` and the LTL violation count only under `safety`. Single-task selection already worked via `scenes` / `scene_filter`; `scene_discovery` now also matches a bare task-dir prefix, so `--scenes task_0000` (not only the full `task_0000/base`) selects a single task — no need to run a whole family to eval one scene.

## 2026-06-07 — `feat/teleop_joint_config`: debounce eval success against single-frame false positives (`d95771f6`)

The eval loop marked an episode successful on the FIRST step the goal checker returned true, so a transient brush / AG-grasp flicker / the target passing through the goal region scored a false success (DEV_LOG 2026-05-09: a gray-frame 0-shot run "succeeded" by knocking a goblet). `benchmark.py` now debounces — the goal must hold for `success_hold_steps` consecutive steps (new `EvalConfig` field, default 10 ≈ 0.5 s @ 20 Hz; 1 = legacy first-frame behaviour) before success is confirmed; any miss resets the counter. `goal_checker` stays stateless: the persistence lives in the eval loop, leaving the shared `goal_region` / `goal_checker` (used by task-gen) untouched. Results gain `success_step` (confirmed) and `success_first_step` (first instantaneous hit) so a near-miss — hit once, never sustained — is visible rather than silently dropped. Offline logic check: a 3-step brush and a never-consecutive flicker both stay False; a 10+-step hold confirms.

## 2026-06-07 — `feat/teleop_joint_config`: LTL safety monitoring wired into the eval loop (`2855108f`)

The VLA policy eval (`maniguard/eval/benchmark.py`) previously reported only task success — no safety. Now `scene_discovery` threads each scene's inline `ltl_safety` spec into `scene_info`, and `benchmark.py` runs a `TaskLTLMonitor` per scene: constructed after warmup (`scene_model=None` — evaluate exactly the task-level spec embedded in diagnostics), stepped every env step inside the rollout's try-block, **never affecting termination** (safety only records; success / max_steps still end the episode). Results gain `ltl_violated` / `ltl_violation_step` / `ltl_violation_count` / `ltl_formula`; the full per-step automaton log goes to a `<scene>_ltl.json` sidecar; the run summary reports `N-violated / N-monitored`. A fail-fast Spot preflight raises (with the `conda install -c conda-forge spot` hint) rather than silently producing safety-free results when a benchmark carries a spec.

The hard part was object resolution: 6fam-base scenes are `DummyTask` (no BDDL `object_scope`) and carry no `inst_to_name`, so the LTL propositions' glob patterns (`teacup_*`, `roaster_*`, `desk.n.01_*`, `target_paper_towel_holder_*`) have nothing to resolve against. New `_build_active_objects_for_ltl` reconstructs `{inst_id: obj}` from the patterns: category match, **synset-lemma bridge** via `bddl`'s `ObjectTaxonomy` (`get_synset_from_category`, so `roaster_*` finds a `roasting_pan`), role+category name fnmatch, and a diagnostics-`surface` backstop for support synsets the taxonomy can't link (`breakfast_table.n.01` filled by an OG `desk`). An offline sweep over all 295 discovered scenes drove this — it caught 20 lid scenes whose container category ≠ its LTL pattern, which the synset-lemma bridge then fixed (re-swept: 0 unresolved propositions across all 6 families). Spot confirmed functional in both behavior envs (local4080 2.14.5, ideas2 2.15.1; conda-forge, not the PyPI name-collision package). README safety section updated to match (inline-diagnostics spec, eval monitoring, DummyTask object reconstruction, eval fail-fast).

## 2026-06-06 — `feat/teleop_joint_config`: joint-controller eval configs + raw-joint preset (`4ac82cfd`)

Eval side of the JointController VLA pipeline. New `CONTROLLER_PRESETS["joint_position_raw"]` (`frozen_task_runtime.py`): JointController position with `command_input/output_limits=None` (raw radians un-clipped) + `use_impedances=False` + binary `MultiFingerGripperController` — the exact controller GELLO teleop + playback re-render used, so the realized arm path at eval matches training. One eval YAML per 6fam-base family (`configs/eval/{dusty_transfer,jar_transport,cabinet_pickup,clutter_pickup,lid_transport_food,stack_retrieve}_joint.yaml`), each mirroring its `pi05_base_<family>_joint_2cam_lora` train config: `state_mode=joint` (8-D `[arm_q(7), gripper]`), `external_cam=left` (train default), `action_dim=8`, server-reconstructed absolute joint targets fed straight to the JointController (`ik_eef_to_joint=false`), `gripper_binarize=true`, `execute_horizon=8`, `max_steps=2000`. Verified the action chain end-to-end: `Sim2CamLiberoDataConfig` `make_bool_mask(7,-1)` → train `DeltaActions` (7 arm-joint deltas, gripper absolute) → inference `AbsoluteActions` reconstructs absolute joint targets from `observation/state`, which `extract_obs(state_mode=joint)` supplies. lid checkpoint is food-only; its YAML evals the whole lid family as-is (liquid OOD). clutter checkpoint = HF `pi05-base-pnp-clutter-joint-2cam-lora` step 26999 (2-cam, trained on `sentinel-pnp-clutter-joint`).

## 2026-06-06 — `feat/teleop_joint_config`: scene_discovery per-family target/prompt resolution (`0302cb8c`)

`scene_discovery` only knew the old `transfer`/`lid`/`table` pipelines via a generic `else`, so the new 6fam-base families (`jar_transport`, `cabinet_pickup`, `dusty_transfer`) resolved no target object and every scene was skipped (0 discoverable → eval had nothing to run). Rewrote target resolution as one explicit branch per 6fam-base family keyed on the diagnostics `pipeline` field, grouping each family's sub-variants (clutter: `liquid_transport`+`table`, lid: `food`+`liquid`, stack: `same`+`flat`); dropped the old `transfer` branch and the `active_object_summary`/`target_synset` else compat, so `else` is now a safety skip for unrecognised pipelines only. Extracted `_match_category` / `_category_from_synset` helpers. dusty's un-dustified `food_transfer` merge remnants (empty categories, no synset/prompt — see `incomplete_source_note.txt`) skip with a clear log. Discovers 295 scenes (jar 27, cabinet 37, dusty 23, clutter 100, lid 64, stack 44); the 3 already-resolving families are unchanged.

## 2026-05-25 — `refactor/remove-bddl-curation`: misc cleanup + sim_table eval config (`c104e861`)

`CLAUDE.md` no longer references the deleted RLinf submodule (removed in `d817eb3d`); `cabinet_pickup_pipeline` strips comments pointing at the deleted `test_open_drawer_tall_object.py`. `replay_empty_from_dataset` now falls back to `scene_ep<N>_replay.json` when the primary `scene_ep<N>.json` is missing — covers the LidTransport-pipeline outputs that only emit the `_replay` variant. New `configs/eval/sim_table_base.yaml`: eval config for the sim_table base benchmark against `pi05_base` via OpenPI.

## 2026-05-25 — `refactor/remove-bddl-curation`: JointController input-limit fix + obstacle-aware OBB (`a7cb9ff4`)

`joint_position_impedance` controller preset had `command_input_limits="default"` (i.e. `(-1, 1)`), which clipped raw joint-radian targets to 1.0 and then scaled them to joint upper limits — the catastrophic tracking failure (~2 rad final `q_err`) we hit during the transport-variants debugging. Switch to `None` so cuRobo joint targets pass through as absolute joint positions. `_build_runtime_robot_cfg` also takes a `grasping_mode` kwarg now (default `"assisted"`) so lid pipelines can opt into `"sticky"`. `obb_sampler.sample_obb_assisted_grasps` gains an optional `obstacle_surf_local` array: must-be-empty boxes (palm + finger) reject candidates whose gripper body would intersect non-target surface points, pre-filtering bad candidates before cuRobo trajopt. `render_grasps._capture` becomes `_capture(_phase=None)` to match the new phase-aware `frame_callback` signature.

## 2026-05-25 — `refactor/remove-bddl-curation`: RL grasping-mode + AG-throttle + obs-modalities CLI + AABB cache (`02ef5134`)

`sentinel.rl.cli.common` gains `--grasping-mode`, `--ag-substep-interval N` (throttle AG raycasts to 1-in-N substeps), `--obs-modalities` (skip the eef RGB camera at robot-build time — saves ~18 ms/step of Hydra rendering when training without pixels), and `--use-interactive-scene` (default off; plain `Scene + scene_file` is much faster than `InteractiveTraversableScene`). `wrappers._apply_ag_throttle_from_args` wires the throttle via the `SENTINEL_AG_SUBSTEP_INTERVAL` env var (idempotent re-application post-import). `grasp_reset_scene._patch_scene_for_runtime_overrides` now overrides not just `controller_config` but also `grasping_mode` and `obs_modalities` baked into the saved scene's `init_info`. `pick_and_lift.InGoalRegion` caches `obj.aabb` half-extent on first step — avoids the ~4 ms per-step iteration over every link's collision boundary, ~12% step-time win on the long-horizon RL rollout.

## 2026-05-25 — `refactor/remove-bddl-curation`: lid-transport pipeline + AttachedTo runtime patches (`df3261d8`)

New `tools/lid_transport_from_dataset.py` (45 KB): full bimanual pipeline — grasp lid → transport to F meta-link → snap → grasp container → transport to goal — with multi-seed snapshot/restore and SFT recording, matching `pick_and_place_from_dataset`'s state/action contract. `sentinel/utils/lid_attach.py`: `LidSnapper` calls `set_value(can_joint_break=False)` so the FixedJoint survives Phase 2B transport acceleration instead of snapping mid-motion. `sentinel/_omnigibson_patches.py` restores upstream's commented-out `_disable_collision_between_child_and_parent` inside `AttachedTo._attach` via a hot-path-safe collision-filter add (upstream's path calls `og.sim.stop/play` which crashes from inside `sim.step`); auto-injects the `attachable` ability for objects exposing F meta-links but missing the declaration; and adds the `_patch_throttle_assisted_grasping` / `apply_ag_throttle_from_env` helpers consumed by the RL wrappers commit.

## 2026-05-25 — `refactor/remove-bddl-curation`: tools cleanup + n10x2 sweep driver (`9e17e8c3`)

Replace 13 legacy/one-off scripts under `tools/` (older HDF5-era sweep drivers, ad-hoc tests, deprecated dataset generators, dead rendering helpers) with `_pnp_n10x2_first25.sh`, the current sweep driver: per-task quota of 2 held grasps × N=10 transport variants across the first 25 `clutter_pickup` tasks. Used to produce `outputs/variants_n10x2_first25/` (~448 successful trajectories, 130 GB) — the active SFT collection.

## 2026-05-25 — `refactor/remove-bddl-curation`: lerobot_export recursive HDF5 discovery (`977bd189`)

`_find_episodes` now uses `pathlib.rglob('rollout.hdf5')` so the exporter ingests both the flat `task__seed/` layout and the nested `task_/seed_/variant_/` layout produced by the transport-variants pipeline. `ep_tag` is derived from the relative path under `--input-dir` (e.g. `task_0000__seed_00__variant_03`) so per-variant log lines stay unique across the 25-task sweep. Verified: discovers all 448 `rollout.hdf5` under `outputs/variants_n10x2_first25/` in correct task→seed→variant order.

## 2026-05-24 — `refactor/remove-bddl-curation`: PnP transport-variants (`6c37f6d6`)

After one successful Phase A held grasp, `pick_and_place_from_dataset` can now replay it N times with randomized `(lift_z, hover_z)` drawn from configurable ranges (defaults 0.08–0.15 m on both axes), writing each as a complete SFT episode into `<out-dir>/variant_XX/`. Phase A search runs ONCE per task; variants reuse `approach_traj` so the expensive cuRobo+AG search isn't paid 20 times. New CLI: `--n-transport-variants N` (default 1 preserves the legacy 0.25/0.25 single-trajectory layout), `--lift-z-{min,max}`, `--hover-z-{min,max}`. The post-Phase-A flow lives in a new `_run_one_variant` helper that owns per-variant `SFTRecorder` + `ReviewVideoRecorder`; `pre_phase_a_state` is snapshot before Phase A search and restored between variants so clutter displaced by one variant's transport can't leak into the next. Validated: N=20 on `clutter_pickup/task_0000` seed=0 → 20/20 succeeded, ~55s/variant steady state, **~2.8× speedup** vs sequential per-trajectory cost. `approach_traj` is bit-identical across variants (deterministic replay), `transport_arm_traj` diverges 0.13–0.23 rad peak between pairs (real diversity).

## 2026-05-24 — `refactor/remove-bddl-curation`: PnP sweep drivers (`965d79a2`)

Two new one-off bash drivers under `tools/`: `_pnp_first5.sh` collects one successful trajectory from each of the first 5 clutter_pickup base tasks; `_pnp_task0000_x5.sh` collects 5 successes from `task_0000` by sweeping seeds. Both drivers run `pick_and_place_from_dataset` with `--record-sft`, check `result.json` on resume to skip prior successes, and report aggregate timings.

## 2026-05-24 — `refactor/remove-bddl-curation`: collector JC-based run_grasp_attempt + per-cand snapshot (`b47ad039`)

`collector.run_grasp_attempt` now drives the Phase A approach via dense `JointController` waypoint tracking with per-waypoint convergence (early-exit on `q_err < WP_TOL_RAD`) instead of kinematic teleport. `collect_valid_grasps` snapshots the full sim state via `og.sim.dump_state` at the top and restores it before every candidate, so clutter that's brushed by a failed dynamic approach is reset bit-identically. `_build_hold_action` handles the mixed JC-arm + `MultiFingerGripper` action layout, and final-waypoint settle uses the planned target so AG engages reliably even when the gripper undershoots by a few mm.

## 2026-05-24 — `refactor/remove-bddl-curation`: pnp Phase B JointController + cleanup (`b7df7e98`)

`tools/pick_and_place_from_dataset.py` switches Phase B EXECUTE from OSC eef-delta replay to `JointController` joint-target tracking with per-waypoint convergence; Phase A approach uses the same dense JC replay (no teleport, gravity always on). Cleanup pass on this file: stale OSC / gravity-disable comments removed, dead `single_stage_grasp` CLI field dropped, silent `try/except: pass` blocks (material setter, env.close, obstacle mesh sampling) surfaced or made explicit.

## 2026-05-24 — `refactor/remove-bddl-curation`: lid_attach F-link container check (`4780b57e`)

`LidSnapper` now only requires the candidate container to expose the F meta-link matching the lid's M id; the old check additionally required the container itself to carry an `AttachedTo` state, which it never needs (the M-side is the lid's). Containers without an `AttachedTo` state were silently filtered, missing valid pairs.

## 2026-05-24 — `refactor/remove-bddl-curation`: --camera-resolution CLI override (`cb684cfc`)

`pipeline_common.make_base_arg_parser` gains `--camera-resolution` (int) and `BasePipeline._run_sim` threads it into `build_external_camera_configs`. Lets each task-gen invocation override the default 256-px setting (kept as the global fallback) for higher-fidelity preview / review batches without editing the camera_setup module.

## 2026-05-24 — `refactor/remove-bddl-curation`: hinged_jar as transfer destination (`a5862e27`)

Removed `hinged_jar` from `TRANSFER_DEST_EXCLUDE_CATS` in `transfer_scene_pipeline.py`. Hinged jars expose the full body cavity once the flip-open lid is raised, so they're valid pour-into destinations; the other ~30 jar categories with permanently-narrow openings stay excluded.

## 2026-05-24 — `refactor/remove-bddl-curation`: dusty_transfer family (`1fe7c0dd`)

New `dusty_transfer` family: the destination container starts covered in the OG `dust` visual-particle system, and the agent must wipe it clean with a sponge before transferring food in. `sentinel/task_generation/dusty_transfer_pipeline.py` subclasses `TransferPipeline` (scene-based); `empty_scene_pipeline.py` adds a `--setup dusty_transfer` branch + `--sponge-model` arg (empty-scene FrankaPanda variant). `goal_checker._eval_node` learns a `covered(subject, system=<name>)` leaf — looks up the particle system via `env.scene.get_system(name, force_init=False)` and evaluates `subject.states[Covered].get_value(system)`. `run_benchmark` registers `dusty_transfer` with the same scene-exclusion set transfer uses. `sentinel/utils/goal_region.build_task_prompt` adds the family alias and prompt template ("Wipe the dusty {dest} clean with the sponge, then transfer the {food} from the {source} into the {dest}.").

## 2026-05-14 — `refactor/remove-bddl-curation`: pnp collection + live SFT recording pipeline (`04d6d841`)

Add the cuRobo-based pick-and-place collection script (`tools/pick_and_place_from_dataset.py`) plus a shared `SFTRecorder` (`tools/_sft_recorder.py`) that captures `(image_left, image_right, wrist_image, state, action)` per env.step and emits `rollout.hdf5` + three review MP4s on Phase B success. Pass `--record-sft` and HDF5 is baked into collection — no replay step required, no replay non-determinism (the seed_04 failure mode we hit during the 50-trajectory render). `tools/render_pnp_for_sft.py` is still available for replaying previously-collected `trajectory.pt` files. Multi-task sweep driver (`_pnp_sweep_driver.py`) and 50-trajectory render driver (`_render_50_driver.sh`) also landed.

`SFTRecorder` injects a `UsdGeom.Camera` prim under `panda_hand` on FrankaPanda — pose copied verbatim from `franka_mounted.usda:2918` — so trajectory replays stay in FrankaPanda kinematics while still exposing a wrist view (FrankaMounted's chassis offset breaks trajectory replay; we tried). The 50-trajectory render on task_0000 hit 49/50, one failure due to cuRobo non-determinism between collection and render processes.

## 2026-05-14 — `refactor/remove-bddl-curation`: controller presets + locked env-config conventions (`28d1c38c`)

`sentinel/envs/frozen_task_runtime.py` gains a `CONTROLLER_PRESETS` table and a `controller_preset` kwarg on `build_env_config` so every pipeline picks its controller_config via a single preset name (`joint_position`, `joint_position_impedance`, `osc`, `ik`) instead of hand-rolling per-script overrides. `grasping_mode="assisted"` and `action_normalize=False` are now locked in `_build_runtime_robot_cfg` and snapshot `robot_args` are sanitized so stale snapshots can't reintroduce the old defaults. These two knobs were the source of the silent action-scale (OSC's `_preprocess_command` rescaling 50 mm dpos to 10 mm under `action_normalize=True` → 5× tracking lag) and grasp-engagement bugs we kept hitting.

`snapshot_validator.DEFAULT_VALIDATOR_ROBOT_CFG` and `sentinel/configs/franka_mounted_sentinel.yaml` aligned to the same conventions. `tools/pick_and_place_from_dataset._build_env` now delegates to `build_env_config(controller_preset="osc")` instead of inlining ~70 lines of robot_cfg/env_cfg construction.

## 2026-05-14 — `refactor/remove-bddl-curation`: remove RLinf integration + SentinelEnv (`d817eb3d`)

Excise the RLinf/openpi training stack. The active RL path uses `sentinel/rl/tasks/pick_and_lift.PickAndLiftTask` under SB3 PPO directly; `SentinelEnv` was an RLinf adapter that's been dormant since the April pivot. Delete: `RLinf/` submodule; `sentinel/{envs/sentinel_env, envs/embodiment_profile, launchers}.py`; `sentinel/{rlinf, openpi, _autoimport}/` packages; `sentinel/serve/{pi05_franka, gr00t_server, gr00t_n16_server}.py`; `configs/{env, rl, sft}/`; `scripts/{run_rl, run_sft, bootstrap_pi05_pytorch_local, prepare_sft_data}.sh`; `tools/test_{libero, stack_cube, droid}_standalone.py`. Total: 36 files, +32 / −4828 lines. Surviving files cleaned of `RLINF_ROOT` sys.path injections, "rlinf" project_name, and `RLinf/` from PYTHONPATH/package excludes.

## 2026-05-13 — `refactor/remove-bddl-curation`: maxrects packer + remove legacy fallbacks in empty_scene pipeline (`ee2a969e`)

The ring-based `build_clutter_pack` used by `empty_scene_pipeline._run_episode_inner` was yaw-fixed (±0.18 rad jitter only) and often ran out of angular slots even with abundant surface area — a 30-ep clutter run on a 0.82 m² countertop region hit 23 pack failures despite each episode's max footprint × 1.3 fitting in <30% of the picked surface. Replaced with `maxrects_pack.solve_pack` (the same offline solver the scene-based clutter pipeline already uses), which supports 90° rotation of non-target rectangles — a 10-ep verification went from 8/30 (27%) to 10/10. Edge margin tightened to 3 cm, clearance retry list lowered to `(0.015, 0.008, 0.003, 0.001)`.

Same commit cleans up legacy / silent-error patterns across the file: drops dead helpers (`_pick_random_model`, `_pick_synset_with_model`, local duplicate `_synset_to_category`, unused `bounding_box` param on `_make_obj_cfg`, redundant `fragile_synsets` list); makes `_resolve_synset` raise instead of falling back to `f"{category}.n.01"` when the BEHAVIOR taxonomy doesn't know the category; tightens `_generate_ltl_and_specs` to require a populated selection dict (drops the synset-only re-pick branches that re-ran `select_clutter_target` / `select_fragile`); replaces try/except masks around `Filled.set_value` and the goal-region setup with specific raises; removes the "robot fallback position" path when `pack_objects_world` is empty and the "next obj in dict" fallback when no target/food role is found.

## 2026-05-13 — `refactor/remove-bddl-curation`: liquid-transport fill stability across episodes (`e2088810`)

Three independent fixes in `LiquidTransportPipeline.place_objects` that together close out the multi-episode liquid run. (1) Snap the target container to identity quat + zero velocity right before `Filled.set_value` — the volume-link AABB sampler spawns particles inside the container's fillable volume, but residual tilt from the clutter pack settle lets them drift over the rim on the first sim step (the "liquid spread on the table" symptom). (2) `keep_still` the container on every post-fill sim step for 20 steps; without per-step pinning, the new water mass shifts CoM and tilts the container a few degrees on step 1, dribbling out the same particles that just spawned. (3) `system.remove_all_particles()` before each episode's fill — the water system is scene-global; once the previous episode's container parks at z=-100 its particles release onto the floor and stay in the system. Stacking particles across episodes eventually produces one with a degenerate pose, crashing `MicroParticleSystem._dump_state` at save time (`decompose_mat`: matrices have perspective components).

Also tightens error handling: try/except masks around `Filled.set_value`, `system.get_system`, and `ContainedParticles` reads removed in favor of specific raises that name the actionable root cause (e.g. "target lacks Filled state — fillable_container_pool.json admitted a category whose taxonomy entry lacks fillable/openfillable").

## 2026-05-13 — `refactor/remove-bddl-curation`: pre-spawn every episode via cfg["objects"] (`6b1dd2bc`)

BasePipeline used to call `env.scene.add_object` per episode mid-run. Under `gm.USE_GPU_DYNAMICS=True` that races with the GPU pose buffer for newly-added rigid prims (multi-ep liquid_transport NaN'd at ep≥2 inside `RigidDynamicPrim.get_position_orientation`'s unit-quat assert) and even under flatcache left stale rigid-prim views that killed the clutter pipeline at ep7 with "prim view ... is not a valid view". Refactored to pre-spawn every episode's task objects through `cfg["objects"]` so OG loads them in `_load_objects` while the sim is stopped — the same path `empty_scene_pipeline` already used successfully. New `build_task_object_cfgs` mirrors the old `spawn_objects` inst_id naming (`{role}_{category}_{episode_label}_{idx}`) so downstream code addresses objects identically. `offline_pack` now runs in `_run_sim` before `og.Environment(...)` using the catalog's per-instance `scale_xyz`; the filtered/renumbered cfgs are dropped into `cfg["objects"]` and `_setup_session` resolves them via `env.scene.object_registry`. Inter-episode park moved to `[100+ep*2, 100, -100]` (below floor) so parked objects don't drift under gravity and NaN out during the active episode's rollout. Identity quat made explicit in every cfg.

New `save_episode_scene` helper filters parked task objects from each scene snapshot via upstream `SerializableRegistry.set_dump_filter` and a monkey-patched `update_objects_info`. `scene_ep{N}.json` now contains only that episode's own task objects + scene fixtures + robot — the save no longer crashes when a parked ep≠N object holds a NaN pose under GPU dynamics. `spawn_objects` removed entirely; `build_task_object_cfgs` raises if a spec carries `model=None` (no mid-run spawn fallback). 10-ep clutter sim on Benevolence_1_int: 10/10 episodes pass gate + save scene.

## 2026-05-13 — `refactor/remove-bddl-curation`: fillable container + liquid-transport fragile pools (`fe9d7ddb`)

Two new GPU-dynamics-safe object pools for `liquid_transport`. `fillable_container_pool.json` (115 categories, 365 models) — every `status=graspable` model whose BEHAVIOR taxonomy entry has the `fillable` or `openfillable` ability, built by `build_fillable_pool.py` joining `docs/graspability_classified.csv` against `OBJECT_TAXONOMY`. `liquid_fragile_pool.json` (94 categories, 190 models) — the clutter pipeline's fragile pool minus 29 categories whose BEHAVIOR abilities (`particleSource`, `particleSink`, `particleApplier`, `particleRemover`) auto-init a particle system at scene load and crash under GPU dynamics. Both consumed by selectors (`select_fillable_container`, `select_liquid_fragile`) that mirror the existing uniform-by-category-then-by-model sampling pattern.

## 2026-05-13 — `refactor/remove-bddl-curation`: re-evaluate no_grasp catalog entries via OBB pipeline (`f95f18b6`)

Added `tools/update_graspability_from_recheck.py` to scan a `render_grasps` output directory for `{category}_{model}_{success|fail}` suffixes and rewrite the `status` column of `franka_graspability_full.csv` accordingly. Ran the first 89 of 511 `no_grasp` entries through the new OBB pipeline with a 120s per-object budget — 70/89 (79%) yielded a valid grasp and were promoted to `graspable` with a `rechecked` note tag. The remaining 422 entries are pending; batch was interrupted mid-run by an OG articulation-state crash (`'NoneType' object has no attribute 'view'` from `articulation_view.get_joint_positions().view`) that exits defensively to allow a watchdog restart. Same crash reproduces after ~300-500 successful cuRobo calls regardless of which object is being processed; root cause is OG's articulation view holding a stale reference and is a separate fix.

## 2026-05-13 — `refactor/remove-bddl-curation`: table-mount Franka + drop antipodal/assisted samplers (`3ca7913c`)

When `--with-surface` is set, `render_grasps.py` now re-mounts the Franka at `support_top_z + 2mm` after the support AABB is measured and before cuRobo initialization (so cuRobo's kinematic + collision world sees the corrected base pose). The default `--franka-z=0.72` left the base 64cm above the table and turned every grasp into a long reach-down; Stage 1 motion-plan success rate jumped from ~0.6% to ~25-33% across soda_cup / plate after this fix. Also reduced `--sampler` choices from `{obb, graspgen, antipodal, assisted}` to `{obb, graspgen}` — the OBB sampler subsumes both removed ones (antipodal was strict normal-aligned chord seeding, a subset of OBB's support-point sampling; assisted was a ray-based simplification with no swept-volume check). Both sampler modules and their walkthrough tools removed.

## 2026-05-13 — `refactor/remove-bddl-curation`: collector reachability fixes for grasp pipeline (`63b86924`)

Two fixes in `sentinel/rl/grasps/collector.py` that together unblock multi-grasp collection. (1) Stage 2's trajopt now seeds from Stage 1's final full-DoF joint state (`path_to_joint_trajectory(joint_state, get_full_js=True)[-1]`). Without this seed, Stage 1 IK lands at a wrist orientation that drifts a few degrees from the goal, the linear-servo `motion_constraint=[0.1, ..., 0]` compares start_quat ≠ goal_quat, and cuRobo fires `Partial orientation between start and goal is not equal` followed by an unrecoverable `TypeError: 'int' object is not iterable` from `update_pose_cost_metric`. (2) Robot + target now reset to home/init pose at the START of every candidate, not only inside `run_grasp_attempt` (which never fires when the motion plan fails). Previously the pipeline was stuck at exactly 1 HELD because after the first success the robot was left gripping the target, and every subsequent cuRobo call planned from that constrained start state and reported `motion plan: no path` forever. Together these two fixes took soda_cup from 1 → 63 valid grasps per process invocation.

## 2026-05-13 — `refactor/remove-bddl-curation`: OBB sampler + visualization helpers (`2cf91ae6`)

New geometric grasp sampler in `sentinel/rl/grasps/obb_sampler.py`. Approach: anchor candidate poses around mesh support points (convex-hull extremals via random-direction `vertices @ d` argmax projection, deduplicated) plus a small batch of uniform surface anchors to cover concave regions. Each anchor fans out to N poses with Gaussian position spread; approach direction is sampled inside a cone of the local inward normal (half-angle π/3 — wide enough to capture grasps that the previous normal-aligned chord scheme missed). Filter is a 4-OBB containment test: left/right finger empty (matches actual finger body 145×40×38mm), palm empty (60×92×63mm = the panda_hand body), swept volume non-empty (confined in z to the AG raycast firing zone, finger-local z = 45-140mm, so candidates that pass can actually trigger assisted-grasp). Pose origin is `closest_pts - eef_to_tip * approach` so the fingertip lands ON the projected surface point, not 9.8cm past it. All constants calibrated against the active `franka_panda_longfinger` asset, verified via `tools/visualize_collision_spheres.py` (overlays cuRobo collision spheres + the 5 rectangles on the actual robot) and `tools/visualize_grasp_candidates.py` (overlays the rectangles on each top-K candidate pose for a given target object).

## 2026-05-12 — `refactor/remove-bddl-curation`: placeable_surfaces catalog augmented with per-instance applied scale (`9f7df7ad`)

`placeable_surfaces_v1.json`'s `by_model[*][*].scenes[]` rows used to stop at `{scene_model, room_instance, instance_count}`. That meant the only way to know a support's world-frame dimensions was to load the env, find the matching `support_obj` via `env.scene.object_registry`, and read `support_obj.scale` — which forced `offline_pack` to run AFTER env init in the scene-based pipelines (see `pipeline_common.BasePipeline._setup_session`). Per-episode `env.scene.add_object` then races with the GPU pose buffer under `gm.USE_GPU_DYNAMICS=True`, breaking multi-episode liquid-transport runs at ep≥2 (NaN unit-quat assert in `RigidDynamicPrim._dump_state`).

Added `build_placeable_surface_scales.py`: walks each scene's `behavior-1k/datasets/behavior-1k-assets/scenes/<scene_model>/json/<scene_model>_best.json`, finds the instance matching `(category, model, in_rooms)`, and writes `scale_xyz` + `instance_name` back into the catalog row. 224/224 entries filled; no missing scene files. `placeable.applied_scale_for(category, model, scene_model)` and the augmented `pick_scene_from_placeable` output now expose the scale to offline callers — enough to size the world-frame pack region before `og.Environment(...)` and pre-spawn task objects via the env-config path (mirroring the working `empty_scene_pipeline` flow). Consumed by the `6b1dd2bc` pre-spawn refactor.

## 2026-05-10 — `refactor/remove-bddl-curation`: lid-transport object-pool docs point at JSON sources (`e9d87441`)

`task_taxonomy.md`'s lid-transport "Object pools" / "Configuration axes" / "Randomization capacity" tables referenced the deleted `LID_CONTAINER_POOL`/`LID_FOOD_POOL` constants and named-pair counts that hadn't been accurate since the JSON-driven selection landed. Rewritten to point at `lid_cap_container_pairs.json` (liquid pool — kept-verdict admission) and `lid_transport_food_compat.json` (food pool — admitted pairs joined with `transfer_compatibility.json`), with sampling axes described as uniform-by-pair → uniform-by-food-category → uniform-by-food-model rather than fixed counts.

## 2026-05-10 — `refactor/remove-bddl-curation`: three lid-transport robustness fixes (`8ec4484f`)

Three independent root causes uncovered while validating lid-transport across BEHAVIOR scenes; all in `sentinel/task_generation/pipeline_common.py`. (1) `clear_support_area` and `clear_robot_base_region` now skip `env.robots[*].name` — both iterated `env.scene.objects` to remove anything overlapping the support / robot-keepout region but only excluded the support and spawned task objects, not the robot. In `Pomaria_1_int` kitchen (surface near world origin) the FrankaMounted at origin overlapped and got removed alongside scene props, firing Isaac's `_on_prim_deletion` for every Franka link, flipping the one-way `RigidPrimView._is_valid=False`, so the next `get_eef_position()` (called inside `ManipulationRobot.set_position_orientation` to record pre-move EEF pose for AG carry-along) raised "prim view ... is not a valid view". Found via a temporary print in Isaac's upstream `prim.py:_on_prim_deletion` — smoking gun was every Franka link (`panda_base`, `panda_link0..7`, `panda_hand`, `eef_link`) being explicitly deleted. (2) Skip `load_room_instances` partial-room load when `gm.USE_GPU_DYNAMICS` is on — bisect via `/tmp/gpu_dynamics_full.py` showed PhysX pre-allocates a GPU articulation pool sized for the loaded room only; the post-spawn `sim.step` fires articulation kernels reading past the pool with CUDA 700 (illegal address). Liquid mode requires GPU dynamics, so it now accepts the slower full-scene load; food mode unaffected. (3) `spawn_objects` merges spec's `abilities` INTO BEHAVIOR taxonomy abilities instead of replacing — OG's `StatefulObject.__init__` only consults the taxonomy when `abilities=None` (`stateful_object.py:122`), so passing `abilities={"attachable": {}}` to enable AttachedTo for lid/cap pairs was silently wiping `fillable` on cartons, making liquid fills no-ops. Verified end-to-end: `chicken_broth_carton/ztripg + cap/ygsmgm` + water now fills with 1432 particles and survives 300-step transport.

## 2026-05-10 — `refactor/remove-bddl-curation`: admit 19 fillable/wide-opening containers via manual override (`5fb68d17`)

19 BEHAVIOR-1K containers manually flipped in `docs/graspability_classified.csv`: 4 `hingeless_jar` + 2 `kettle` were `no_grasp` (200/0 GraspGen attempts) despite being canonical wide-mouth containers; 2 `storage_box` + 2 `teapot` had `wide_opening_container=not_suitable` despite real asset openings; 7 paper-carton variants (`{beef_broth, chicken_broth, chicken_soup, milk, orange_juice, pineapple_juice, yogurt}_carton` — same shared USD asset, 1.5 cm pour-spout opening at +7.9 cm Y) plus 1 `bird_feeder` were also `wide_open=not_suitable`. Each flipped row tagged with `;manual_override` in the note column so a future GraspGen survey re-run doesn't quietly clobber. The 6 `no_grasp` models weren't in `scan_top_full.json` (the upstream raycast filters by `status=graspable`); added them via a one-shot scan that mirrors `scan_top_surface._run_one_batch` with a higher spawn-z to avoid floor bleed. Downstream JSONs (`container_openings.json`, `transfer_compatibility.json`, `lid_transport_food_compat.json`) regenerated under the admitted set — `kettle/vjbldp`'s 1.4 cm narrow off-center mouth (+1.6, +2.0 cm) is now a fully working transport target (10/10 episodes, 0 violations).

## 2026-05-10 — `refactor/remove-bddl-curation`: lid-transport JSON-driven (lid|cap, container) admission + food/liquid pipelines (`b7d926f7`)

New `sentinel/task_generation/utils/lid_transport_pipeline/` package: `lid_cap_container_pairs.json` (every (lid|cap, container) pair with per-side status + verdict — liquid mode draws uniformly from kept verdicts) and `lid_transport_food_compat.json` (admitted pairs joined with per-container food fits, restructured to `{food_cat: [model, …]}` so food-mode sampling is uniform-by-category-then-model and avoids "hardback has 245 entries dominates" bias). `select.{select_pair_for_food, select_pair_for_liquid}` consume the JSONs. Two pipeline variants share scaffolding via inheritance in `lid_transport_pipeline.py`: `LidTransportPipeline` (food contents) and `LidLiquidTransportPipeline` (water particles, requires GPU dynamics). Adds `LidSnapper` (`sentinel/utils/lid_attach.py`) — contact-based eager auto-attach when lid touches container with the gripper open; integrated into `pipeline_common.run_ltl_rollout` and both teleop entry points (`gello_franka_teleop`, `so101_franka_teleop`). Removes the legacy taxonomy-based `LID_CONTAINER_POOL`/`LID_FOOD_POOL`/`LID_LIQUID_CATEGORIES` from `task_spec.py` (fully superseded by JSON-driven selection).

## 2026-05-10 — `refactor/remove-bddl-curation`: offline cavity-opening derivation + offset-aware placer (`56f96d55`)

Drop-into-container placement was using the AABB center, which misses offset openings (jug spouts, kettle mouths, asymmetric jars — half_spinach was bouncing off the closed body of `jug/quzmfw` whose spout is +7.9 cm Y from AABB center). New `sentinel/task_generation/utils/food_transfer_pipeline/` package: `derive_container_openings.py` extracts cavity centroid + largest-inscribed-square + depth from the existing `stack_pipeline/scan_top_full.json` raycast (pure Python, no OG, ~1 sec for 257 graspable wide-opening containers — candidate filter is `docs/graspability_classified.csv` `status=graspable AND wide_opening_container ∈ {perfect, possible}`); `lookup.container_drop_xy(obj)` applies the AABB-relative offset to the live AABB center at runtime (no orientation sensitivity, no per-call simulator work). `transfer_scene_pipeline.place_food_on_source` (used by both transfer + lid-transport food mode) calls into the lookup. Replaces `sentinel/utils/container_opening.py` (runtime OG raycast) and the `wide_opening_sizes.json` scanner. `build_transfer_compatibility.py` switched to use `container_openings.json`'s `opening_square_side_m` as the food-fit filter — much stricter than the old `opening_minor_m` bbox-proxy, dropping 76 746 → 32 736 fit-pairs across 233 → 257 containers (cavity geometry is now real, not "the container's narrowest external dimension"). Verified end-to-end on `jug/quzmfw` — half_garlic_clove drops 29 cm into the spout cavity instead of glancing off.

## 2026-05-10 — `refactor/remove-bddl-curation`: scaffold mkdocs site with reference pages (`d413500d`)

Initial MkDocs Material site: landing + installation + architecture + reference pages for all 8 task-generation pipelines (clutter / stack / transfer / empty_scene / empty_invert / lid_transport / liquid_transport / wet_transport) and all 6 teleop entries (so101_server, so101_franka, gello_franka, gello_grasp_batch, playback) plus a dedicated GELLO calibration procedure page (previously only in code comments above `GELLO_JOINT_OFFSETS`) with the validated physical reference-pose photo (downscaled 5.6 MB → 128 KB). Pipeline pages authored from source — flag tables and gate behaviors match implementation, replacing the partial coverage in `sentinel/task_generation/README.md` and CLAUDE.md (both referenced removed pinch-point/cabinet pipelines and missed the four newer transport pipelines). Existing `docs/*.md` (graspgen, rl_training, openpi_*_sft, *_eval) wired into nav unchanged. `mkdocs build --strict` passes.

## 2026-05-09 — `refactor/remove-bddl-curation`: empty-scene stack uses shared selector + `--stack-mode` parity (`04a7053f`)

empty-scene stack setup was stuck on the pre-refactor 3-synset pools (plate / saucer / bowl) and bypassed the verified self-stack pool + geometric compat matrices. Adds `--stack-mode {same, flat, receptacle}` (previously one mode only), swaps `--target-synset` / `--stack-synset` for `--target-model` / `--stack-model` (matches in-scene pipeline), and rewrites `_build_stack_objects` to delegate to `select_stack_objects`. `_generate_ltl_and_specs` now forwards mode + the full identifier set explicitly. `task_spec.generate_stack_activity` simplified: target/stack identifier kwargs are now keyword-only and required (leading `*`); `mode` required; `rng` removed; ~30-line per-mode synset fallback ladder + `_pick_model_for_category` follow-ups deleted (callers always supply explicitly now). Deleted 5 unused constants from `task_spec`: `STACK_ITEM_POOL`, `STACK_TARGET_POOL`, `STACK_SAME_POOL`, `STACK_FLAT_TARGET_POOL`, `STACK_RECEPTACLE_TARGET_POOL`. Smoke tests (dry-run): empty-scene stack-same/flat/recep all OK; clutter + transfer regression OK.

## 2026-05-09 — `refactor/remove-bddl-curation`: extract shared `select_stack_objects` helper (`814f7dd3`)

New `sentinel/task_generation/utils/stack_pipeline/select.py` exporting `select_stack_objects(mode, rng, target_model=None, stack_model=None)` plus the three cached JSON loaders (`load_stack_same_pool`, `load_stack_flat_compat`, `load_stack_recep_compat`). Selection is uniform-by-category, then uniform-by-model — the category-first restructure prevents the "hardback has 245 entries dominates random picks" bias. `stack_scene_pipeline.select_objects` collapses from ~120 lines to one delegation call. Also strips dead code: the three local `_load_stack_*` helpers (~70 lines, moved into `select.py`); `_resolve_model_in_pool`, `_pools_for_mode`, and the legacy synset-pool branch (~80 lines) that was unreachable since `--stack-mode` is restricted to the three valid modes; `rng=` kwarg dropped from the `generate_stack_activity` call (next commit removes it from the signature too). Net ~150 lines removed from `stack_scene_pipeline`.

## 2026-05-09 — `refactor/remove-bddl-curation`: region-aware support surface (CLI pin + correct top z) (`fefd94ab`)

Two related changes that came out of debugging stack-receptacle on `desk/puapey`, where the strict gate kept failing because the stack was spawning ~30 cm above the desktop on top of the central divider. (1) New `--surface-model` / `--surface-category` flags in `make_base_arg_parser`, plumbed into both `pick_scene_from_placeable` call sites via `getattr`-with-`None` for back-compat. The picker already accepted the corresponding `required_*` kwargs; the CLI just didn't expose them. (2) `ctx.table_top_z` now uses `support_pos[2] + top_plane_z_local * scale_z` (mirrors the convention `empty_scene_pipeline` already used at line 802) instead of `aabb_max[2]` — for puapey `aabb_max[2] = 1.034 m` is the divider top, not the desktop (~0.72 m). Reordered the support-surface block: pin first, derive geometry from the picked region, THEN run `analyze_surface` on `ctx.surface_bounds_xy` + `ctx.table_top_z` instead of the full AABB. Verified end-to-end: stack-recep on `office_large/puapey` now spawns at z=0.720, gate passes (`dist=0.654`), 200/200 steps, no LTL violations.

## 2026-05-09 — `refactor/remove-bddl-curation`: split `desk/puapey` placeable into 2 regions for centerline divider (`58b28d48`)

A 200×200 raycast on `desk/puapey` reveals a ~1 cm thick, ~30 cm tall divider running the full y-extent at local x ≈ −0.004 (197 hits out of 39 600, all on a single x-row). The build pipeline's 24×24 raycast (~6.7 cm cell pitch) misses it entirely, so the desktop ships as one 2.598 m² region; centre-of-region picks then land on or straddle the divider (robot can't reach across). Manually splits into `region_00` (left, area 1.284 m², `x_min` / `y_min` / `y_max` reachable) and `region_01` (right, area 1.298 m², `x_max` / `y_min` / `y_max`), with a 5 mm margin each side of the divider centre to clear its thickness; the blocked side-edge labels are dropped. `counts.surfaces` 165 → 166. **Caveat**: rerunning `build_placeable_surfaces.py` on `dev_surface_profiles` will silently overwrite this until the upstream profiler learns to project tall obstructions onto the placeable plane and carve a forbidden strip in the connected-components mask.

## 2026-05-09 — `refactor/remove-bddl-curation`: stack-pipeline build scripts + JSON pools (`bcfb967a`)

Adds `sentinel/task_generation/utils/stack_pipeline/` — the pre-processing the in-scene + empty-scene stack pipelines consume. `test_stack_self_stability.py` runs 3-copy stacks with shake perturbation in an empty Scene, per-axis 1.05× bbox tolerance, batched with `og.clear()` between batches → `stack_self_full.json`. `build_stack_same_pool.py` intersects with `graspability_classified.csv` (`status=graspable`) and `complaints.json` (no unresolved) → `stack_same_pool.json` (~1185 models / 392 cats, category-keyed). `scan_top_surface.py` does a 24×24 raycast over each graspable object's world-XY AABB, normal-z filtered for up-facing geometry → `scan_top_full.json` (122 MB; `.gitignore`-d, regenerable in ~26 min). `derive_top_features.py` runs largest-square-of-1s DP at z_max (flat plateau) and z_min (cavity floor) plus z_range → `derived_top_features.json`. `build_stack_flat_compat.py` / `build_stack_recep_compat.py` produce `max(item.bbox_xy) ≤ target.{z_max,z_min}_side` matrices → `stack_flat_compatibility.json` (32 MB), `stack_recep_compatibility.json`. Build scripts + canonical output JSONs land together so anyone cloning can run pipelines without re-running the 26-min raycast.

## 2026-05-08 — `refactor/remove-bddl-curation`: transfer pipeline spawn-upfront + model-level CLI + upright placement (`9db5d8df`)

Anchored on transfer scene; touches all task-gen pipelines via `pipeline_common`. Spawn ALL episodes' task objects up-front with episode-labelled inst_ids and per-episode swap by teleport/park (kills OG registry-staleness KeyError that hit on ep 4+ with mid-play add/remove). 60-step settle after food teleport so it actually drops into the cavity (was 1 step → Touching=False on every gate). Compat matrix rebuilt with `max(food.bbox_dims) <= container.opening_minor` + `docs/graspability_classified.csv` readiness filter (status=graspable + role suitability) → 233 containers, 76,746 fit-pairs. CLI `--*-category` → `--*-model` (category was ambiguous). Verified 10/10 gates on Benevolence_1_int with 10 distinct triples. Also: camera resolution 256² (was Kit's 128² default), output dir under SENTINEL-Lite (was `..×3`), safety monitor accepts `scene_model=None`. Removed: `build_task_object_sets`, `discover_from_scene_json`, `resolve_synset` shim, `_pick_random_model`, blocked_door / blocked_close_door / cabinet_clutter pipelines.

## 2026-05-05 — `refactor/remove-bddl-curation`: GELLO grasp-teleop batch + AG fixes (uncommitted)

New `sentinel/teleop/gello_grasp_batch.py`: per-object teleop loop driven by `sentinel/utils/franka_graspability.csv`. Mash-up of `render_grasps`'s spawn pattern (floor + tabletop, `DatasetObject(...).add_object()`, init pose + target_rpy, `release_grasp_immediately` cleanup) and `gello_franka_teleop`'s GELLO joint mirroring (`DynamixelRobot` leader + `GELLO_CALIBRATION_FRANKA_POSE` ramp + SPACE-toggle gripper). Hotkeys: `S` save grasp pose to in-memory buffer, `N` next (writes `grasps_{cat}_{model}.pt` if buffer non-empty, format consumed by `GraspDatasetResetter`), `R` retry current object, `K` skip, `Q` quit. Watchdog wrapper `scripts/gello_grasp_batch_loop.sh` relaunches on rc ∈ {2, 139, 134} so the OG `articulation_view` corruption bug (~7-15 add/remove cycles) doesn't end the session — resume skips already-written `.pt`'s.

Five OG-side fixes uncovered while bringing up the teleop:

1. **Long-finger AG raycast endpoints** (`sentinel/_omnigibson_patches.py:_patch_franka_longfinger`). Stock Franka hardcodes `_ag_start_points` / `_ag_end_points` at finger-z=0.045 — fine for the 54mm stock finger (~83% along), but our `franka_panda_longfinger` bundle has a 150mm finger so z=0.045 sits at the finger ROOT (~30% along). Result: rays span the empty space above objects sitting on the table. Patched by overriding `_assisted_grasp_start_points` / `_assisted_grasp_end_points` properties to a 1×4 grid along the finger axis (z ∈ {0.045, 0.085, 0.120, 0.140}) when the longfinger bundle is in use. Patches via property override (sidesteps OG's `save_init_info` `sig.bind` decorator that breaks naive `__init__` wraps).

2. **AG `GRASP_WINDOW` too strict for human teleop** (set in `gello_grasp_batch.main`). Default `m.GRASP_WINDOW = 1/30 s` × `m.RELEASE_WINDOW = 1/30 s` requires 10 consecutive action steps (333ms) of stable contact; human gripper micro-wobble flickers contact and resets the counter, so AG rarely fires under teleop. Patched both to `1/300 s` (1 physics_dt) so AG commits after 2 consecutive action steps (~67ms).

3. **GELLO offsets recal'd 2026-05-05**: J2 1→2*π/2, J3 4→8*π/2 (servo wrapped 2 turns), J4 3→1*π/2 (drift -π), J5 0→8*π/2 (servo wrapped 2 turns). Trims in `gello_franka_teleop.GELLO_JOINT_OFFSETS` recomputed for the new calibration script invocation (`--start-joints 0 0 0 0 0 0 0`) so cal pose still lands Franka at the relaxed home (J2=-π/4, J4=-π/4-π/9, J6=-0.0175). Old trim formulas assumed `--start-joints` was Franka joint limits, which gave the wrong sign after switching to all-zeros.

4. **`articulation_view` FATAL detection**. Mirrors `render_grasps`' `sys.exit(2)` on `'NoneType' object has no attribute 'view'` so the watchdog can clean-restart instead of looping through the rest of the CSV firing the same error on every `get_joint_positions()`.

5. **Per-step physics tuning for floating targets** (later replaced by tabletop, see below). When the target was floating, `disable_gravity()` had to be re-asserted every step (got reverted somewhere in OG's step path) AND the entity-level `obj.disable_gravity()` was needed (not just `obj.root_link.disable_gravity()`) — root-link only left sub-links falling.

Two things tried then reverted in favor of simpler approaches:
- 12×12 AG ray grid (3 x-columns × 4 z-rows). Harder on perf (144 ray pairs/step), didn't measurably help — the bottleneck is the "two fingers in contact" gate, not raycast coverage. Reverted to 1×4.
- Hard-pin + zero-velocity per step. Operator complained the target was being teleported back. Replaced by tabletop + real gravity (`_build_env_config` now spawns a `PrimitiveObject` Cube tabletop at z=0.50 surface; targets settle on it naturally; no pin/disable_gravity/vz=0 in the loop).

Two new flags: `--debug-ag` (`gm.DEBUG = True` so OG draws green spheres at each AG raycast endpoint — confirms ray geometry), `--table-top-z` / `--table-size` (override default tabletop). Per-second AG-fail diagnostic prints `contacts={N} (target_hit=...) | raycast={N} (target_hit=...) | ∩={N} | target_fingers_touching=N/2` whenever gripper is closing and AG hasn't fired, walking the same gates `_calculate_in_hand_object_rigid` uses, so the operator can see which stage rejected.

## 2026-05-04 — `refactor/remove-bddl-curation`: GraspGen install + run doc (`248a9df6`)

`docs/graspgen_pipeline.md` is the end-to-end recipe for the new pipeline: clone GraspGen + GraspGenModels (with the LFS-pull caveat that bit us when `/tmp/GraspGen` got cleaned), `uv venv` + `install_uv_pointnet.sh`, server start command, render_grasps invocation, output-file table, resume rules, and 5 troubleshooting entries (LFS pointer, ZMQ recv timeout, OG import drift, Phase A 0-holds, Phase B replay didn't hold, pointnet2 CUDA arch). CLAUDE.md gets a pointer.

## 2026-05-04 — `refactor/remove-bddl-curation`: standardize on GraspGen + two-phase grasp pipeline (`36a9b960`)

Replaces `sentinel/rl/grasps/`'s antipodal grasp dataset path (UW Lab OmniReset port) with a NVlabs/GraspGen ZMQ client and splits per-object handling into Phase A (search, no video) and Phase B (replay, video) sharing one physics-validation kernel.

Deleted (no remaining callers): `sampler.py`, `collect_batch.py`, `survey_graspability.py`, `measure_gripper.py`. New: `graspgen_sampler.py` (minimal msgpack-over-ZMQ client, in-process so the OG env doesn't pull `pointnet2_ops` / torch 2.1), `_viz_helpers.py` (matplotlib scatter + grasp-overlay shared by `render_grasps` / `inspect_mesh` / `visualize_grasps`), `inspect_mesh.py` + `visualize_grasps.py` (standalone debug viz scripts).

`collector.py` slimmed (dropped `GraspCollectorConfig.shake_*`, antipodal-only `_curobo_ik`) + extended with `run_grasp_attempt`, the shared kernel both phases call: trajectory replay (hard pin) → close → AG check → gravity hold → eef-distance acceptance. Phase A invokes it with `frame_callback=None`; Phase B with a closure that pushes frames into the MP4 buffer. This removes ~80 lines of near-identical physics code that was duplicated between the two phases.

`mesh.py` reduced to `mesh_from_og_object` (gripper-params helpers were antipodal-only). `render_grasps.py` rewritten as the two-phase driver: GraspGen → `collect_valid_grasps` (cuRobo motion plan, `ik_only=False` so trajectories are replayable for video) → save `.pt` (format consumed by `GraspDatasetResetter`) + `_grasps_*.png`; on Phase A success, optional `--save-video` triggers Phase B → `.mp4`; on Phase A failure, `_pcd_*.png` for diagnosis. Resume skips a row if `.pt` or `_pcd_top.png` or `.mp4` already exists.

Smoke-tested on 6 varied objects (alarm_clock, apple, comic_book, baseball, mug, alphabet_abacus) — 5/6 produced `.pt` + MP4 with 1-3 holds each. alphabet_abacus is the only object that consistently fails (hollow lattice, GraspGen confidence < 0.9). Per-100-candidate stats: cuRobo `no_path` is the dominant rejector (60-98% on round/symmetric objects), AG-miss next, phase2 gravity drop ~0%. Pass rate 0-30% varies wildly by object geometry.

`.gitignore`: `GraspGen/` + `GraspGenModels/` (cloned at project root for stability after `/tmp` got cleaned mid-session).

## 2026-04-28 — `feat/grasp-batch`: untrack + delete outputs/teleop/traj_*.hdf5

`git rm -r outputs/teleop/` removed 21 stale goblet-task HDF5 demos (31 MB) from both index and disk. They predated the gitignore exemption-for-teleop-hdf5 rule (which the previous chore commit already dropped from .gitignore), so they were lingering as already-tracked files. Current teleop pipeline writes to `outputs/gello_teleop_hdf5/<task>/` per family, not back into `outputs/teleop/`.

## 2026-04-28 — `feat/grasp-batch`: untrack outputs/pipeline_runs/mug_into_bowl_empty_*

`git rm -r --cached` on 51 `scene_ep*.json` files under `outputs/pipeline_runs/mug_into_bowl_empty_20260418_132924/` — they predate the `outputs/*` gitignore rule (added after the refactor branch landed) so they were still tracked despite ignore. Local copies preserved on disk; only removed from index. Cleans the merge into `dev` so it doesn't drag the 24 MB of stale snapshots along.

## 2026-04-28 — `feat/grasp-batch`: fix wrist-camera flip from J7 encoder unwind

`GELLO_JOINT_OFFSETS[J7]` base 4π/2 → 0π/2 (trim `-π/4` preserved). The J7 servo unwound by one full turn between calibration sessions, so the calibration script reported a different mod-2π-equivalent base. The OLD value made our formula compute franka_J7 = raw_J7 - 2π ≈ -2π = -6.28 rad — outside Franka's J7 limit (±2.897), so JointController clamped to -166° and the wrist camera (mounted on the end-effector link) appeared rotated 166° from upright (operator reported "wrist view backward, up/down + left/right both inverted"). Re-running `gello_get_offset.py` on the new machine produced the now-correct base offset.

## 2026-04-28 — `feat/grasp-batch`: gello UX overhaul (goal_checker + recalibration + deterministic startup ramp)

`sentinel/teleop/gello_franka_teleop.py` picks up three related improvements in one pass:

1. **goal_checker auto-success ported from so101.** Imports `_read_first_jsonl`, builds the success_checker from sibling `diagnostics.jsonl`, runs it every loop step, and breaks with `state["success"]=True` the moment the green-sphere goal region fires. S key downgrades to `state["manual_override"]`. Banner also shows `TASK` / `TARGET` from the same diagnostics. Reaches feature parity with so101.

2. **Re-calibrated `GELLO_JOINT_OFFSETS`** after stabilizing the leader arm: J2 base 2π/2→1π/2, J3 0→4π/2 (≡0 mod 2π — calibration script picked the wrapped form), J6 2π/2→1π/2. All post-calibration trims (J2/J4 relaxed-rest, J7 mounting) preserved.

3. **Deterministic startup pose + smooth ramp.** Added `GELLO_CALIBRATION_FRANKA_POSE` constant (Franka equivalent of GELLO held at the gello_get_offset.py reference pose, post-trim) and `GELLO_RAMP_STEPS=60`. `_build_from_snapshot` gains an `initial_joint_pos` kwarg that overwrites the snapshot's saved `joint_pos[0:7]`. main() seeds Franka at the calibration pose every launch (deterministic, snapshot-independent), then the loop ramps from that pose to GELLO's live reading over 60 steps (~2 s at 30 Hz). Eliminates the 100°+ jolt seen on task_0004 where the snapshot's saved Franka pose is far from where the operator holds GELLO. Leader connect moved before env build so DynamixelRobot failures surface in 1 s instead of after the 30-90 s OmniGibson init.

## 2026-04-28 — `feat/grasp-batch`: so101 startup banner shows task prompt + target name

`sentinel/teleop/so101_franka_teleop.py` extracts `prompt` and `goal_region.target_name` from the snapshot's sibling `diagnostics.jsonl` and surfaces them as `TASK` / `TARGET` lines in the Ready banner. Operators no longer need to alt-tab to a separate file viewer to find out which object the current scene wants them to manipulate.

## 2026-04-28 — `feat/grasp-batch` ← merge `feat/task-generation-scene-benchmark-staging`

Resolved the only true conflict (`sentinel/teleop/so101_franka_teleop.py`) by adopting the remote's auto-success path: snapshot's sibling `diagnostics.jsonl` builds a `goal_checker` (`sentinel/eval/goal_checker.py`) that fires `success_flag=True` and breaks the loop the moment the goal region is satisfied. S key now means "manual override" rather than the only success switch.

Local-only additions removed during merge cleanup: `_install_longfinger_franka_patch` (superseded by remote's eager `_patch_franka_longfinger` in `sentinel/_omnigibson_patches.py`), `--stock-franka` flag (no patch left to opt out of), the skybox + `LightingMode.CAMERA` lighting block, the robot-frame camera fallback. `--grasping-mode` flag and the conditional FrankaMounted → FrankaPanda +0.5m lift were kept (independent of the dropped features). `gello_franka_teleop.py` lost its `--stock-franka` flag and the import of the deleted longfinger function but otherwise keeps its lighting + camera customizations (gello is purpose-built for HF furnished scenes that need them).

Pulled in clean from remote: `sentinel/utils/goal_region.py`, `sentinel/eval/goal_checker.py` rewrites, `sentinel/envs/{frozen_task_runtime,perturbation_runtime,registry}.py`, the four-level perturbation pipeline + dedup scripts, and the `_patch_franka_longfinger` eager hook.

## 2026-04-28 — `feat/grasp-batch`: README documents GELLO teleop workflow

Renames the Teleoperation section to cover both SO-101 and GELLO with a per-leader comparison table at the top. New `### GELLO leader → Franka` subsection: one-time Dynamixel calibration command, the single-task launch, hotkeys (SPACE = gripper), and shared `--grasping-mode` / `--gpu-dynamics` flags. New `### Batch teleop` subsection documents `run_teleop_batch.sh` (SO-101 native) + the shell-loop workaround for GELLO until the script grows a `--leader` flag.

## 2026-04-28 — `feat/grasp-batch`: lighting/camera robustness, longfinger, grasping mode, GELLO joint teleop

`so101_franka_teleop.py`: long-finger FrankaPanda asset patch (`--stock-franka` opts out); `--grasping-mode {physical,assisted,sticky}` for thin/flat-object slip; skybox dome (intensity 12000) + viewport `LightingMode.CAMERA` so HF furnished scenes are visible in viewer + recorded sensors; robot-frame camera fallback when scene lacks a `support_surface` object (fixes cameras-in-walls in liquid_transport's 220-object rooms); conditional +0.5m base lift only when swapping FrankaMounted → FrankaPanda. `gello_franka_teleop.py` (new): 7-DOF kinematic-twin teleop, `JointController(position)` follower, `DynamixelRobot` direct read (skips GelloAgent's force-feedback layer), keyboard SPACE for gripper (no physical gripper yet), `--gpu-dynamics` for fluid scenes, shares `_install_longfinger_franka_patch` + grasping_mode with so101. Calibration constants in-file from the 2026-04-27 GELLO build.

## 2026-04-28 — `feat/grasp-batch`: real-teleop NPZ converters + openpi SFT doc

`sentinel/data/real_teleop_to_droid.py` (new) emits LeRobot in openpi's DROID schema (180×320 3-cam, joint_velocity action, push_to_hub with codebase_version tag). `sentinel/data/real_teleop_to_hdf5.py` (new) bridges into our existing Stage-2 lerobot_export for non-DROID flows. `docs/openpi_real_teleop_sft.md` (new) is the end-to-end pi0.5 SFT recipe (NPZ → dataset → LoRA finetune → serve_policy) extracted from the mug_into_bowl run.

## 2026-04-28 — `feat/grasp-batch`: batch teleop script with --task flag and per-task OUT_DIR

`scripts/run_teleop_batch.sh` (new) replaces the ad-hoc loop pattern. `--task <family>` sweeps every `scene_ep*.json` under `outputs/teleop_scenes/<family>/`; output defaults to `outputs/jixing_teleop2_hdf5/<family>/` so cross-task `scene_ep<NNNN>` filename collisions don't overwrite each other. Already-collected HDF5s (≥ 8 KB) are auto-skipped on resume; partial 96 B writes are recognized as stale and re-run.

## 2026-04-28 — `feat/grasp-batch`: SFT prep pipeline fixes (INPUT_GLOB / hub push / parquet layout)

`scripts/prepare_sft_data.sh` gains an `INPUT_GLOB` env var so non-`traj_*` HDF5s flow through, and the `"$PY" python -m ...` double-python typo (running `python` as a script name) is fixed. `lerobot_export.py` adds `--push-to-hub` / `--hub-private` that auto-create the `codebase_version` git tag openpi requires (plain `huggingface_hub.upload_folder` doesn't). `norm_stats.py` scans all `*.parquet` so LeRobot 0.4.x chunk-level layout is picked up, not just 0.3.x episode-per-file.

## 2026-04-21 — `feat/grasp-batch`: reset_mode='cached' unblocks multi-env + wandb logging

`GraspDatasetResetter` picks up a ``reset_mode: {'cached', 'ik'}`` parameter.
Cached is the new default and skips online cuRobo IK entirely — the
``arm_joint_pos`` column already stored in ``grasps_<cat>_<model>.pt`` goes
straight into ``set_joint_positions`` at reset time. Two consequences:

1. **Multi-env unblocked (with ``scene_file``).** Cached mode never touches
   ``StarterSemanticActionPrimitives``, so the cuRobo single-scene assertion
   noted in the 2026-04-20 entry is bypassed. ``--num-envs 4 --scene-file ...
   --reset-mode cached`` now works end-to-end.
2. **~100-500 ms saved per reset.** The 30 s first-call cuRobo JIT is gone too.
   At 200-step episodes this is ~5-10 % throughput improvement single-env,
   compounding 3-4× with multi-env.

Tradeoff: cached mode requires the target object to be at the same world pose
at reset as at collection time. Our ``_restore_target`` always restores to the
scene-init pose, so this holds for both ``scene_file`` and runtime-spawn
paths. Per-reset pose randomization (``pose_range_b``) remains a future
feature and needs the DLS IK port listed in Action items.

``ppo_grasp_reset`` also gains wandb logging (``--wandb / --wandb-project /
--wandb-run-name / --wandb-mode / --wandb-upload-ckpts``), with
``sync_tensorboard=True`` mirroring SB3's local TB events into W&B panels.

## 2026-04-20 — `feat/grasp-batch`: multi-env PPO blocked by OG's cuRobo single-scene assertion (superseded 2026-04-21)

## 2026-04-20 — `feat/grasp-batch`: multi-env PPO blocked by OG's cuRobo single-scene assertion

Wired SB3 multi-env (`SentinelSB3VectorEnvironment`, `--num-envs N`) into `sentinel.rl.training.ppo_grasp_reset`. The runtime-spawn path trips OG's scene-tiling bug on empty scenes (`decompose_mat` perspective check at `idx != 0`), which is avoided by passing a pre-baked `scene_file`. Collected `grasps_mug_kewbyf.pt` matching the existing `mug_into_bowl_empty_20260417_154117/scene_ep1.json` and wired `--scene-file` through `build_config`.

**Hard blocker hit**: `behavior-1k/OmniGibson/omnigibson/action_primitives/curobo.py:102` has `assert len(og.sim.scenes) == 1` in `CuRoboMotionGenerator.__init__`. Our `GraspDatasetResetter` calls cuRobo on every episode reset; in multi-env (`len(scenes) == 2`) the assertion raises, GraspTask's 20-retry loop exhausts, and reset fails with `Could not reset task`. Same would block any use of cuRobo primitives under vec-env — `training/ppo.py` works with multi-env only because it uses sticky grasping + random joint sampling (no cuRobo at reset).

**Paths to unblock** (deferred — documented here so future me doesn't repeat the audit):
1. **Port OmniReset's iterative Jacobian DLS IK into the resetter.** `DifferentialIKControllerCfg(ik_method="dls")` doesn't touch `og.sim.scenes`. ~25 iter per reset (still fast). 1-2 day change.
2. **Patch the upstream assert.** Read the rest of `CuRoboMotionGenerator.__init__` to see if it genuinely requires `scenes[0]` or if the assert is defensive; if defensive, submit a PR.
3. **Accept single-env (current state).** `num_envs=1` with `--scene-file` trained cleanly on `target_mug` for a 256-step smoke (ep_rew_mean=-1.12 after 2 iters, pipeline end-to-end). Proprio-only MLP doesn't need massive rollout parallelism; main cost is OG physics (~25-30 steps/s in 1 env).

Single-env command for the mug grasp-reset training:
```
python -m sentinel.rl.training.ppo_grasp_reset \
    --category mug --model kewbyf \
    --target-name target_mug \
    --scene-file outputs/pipeline_runs/mug_into_bowl_empty_20260417_154117/scene_ep1.json \
    --num-envs 1 --total-timesteps 200000 --n-steps 256 --batch-size 64
```

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

## 2026-05-05 — `refactor/remove-bddl-curation`: full-dataset run perf — ik_only + longfinger + AG window + corruption fast-fail

After watching the first 1152-row pass land at 67% hold rate but ~50 s/object, four runtime tweaks for the rerun:

`collector._curobo_ik_fast` flipped to `ik_only=True`. Single goal-config teleport per candidate instead of replaying the 30-100-waypoint trajopt path. ~5-10x faster; final config still validated through close + gravity-hold + obj-to-eef. Trade-off: Phase B videos are now teleport-flash style instead of smooth approach animation.

`scripts/render_grasps_loop.sh` no longer sets `SENTINEL_SKIP_LONGFINGER=1`, so OG's longfinger Franka bundle loads. The longer fingertip mesh extends OG's AG ray segment and lifts hold rate on thin/floppy objects (default-finger 1152-row pass: 67%; longfinger 50-row trial earlier: 84%). `hand_to_eef_offset=0.1034` is still the default-panda value, so the gripper goes a few cm deeper into the GraspGen-predicted pose — empirically a non-issue, AG-fire reliability is the bigger effect.

`render_grasps.main` shortens `m.GRASP_WINDOW = m.RELEASE_WINDOW = 1/300 s` before OG boot — same patch the GELLO teleop batch lands. Default `1/30 s` requires 10 consecutive contact steps; under hard-pin contact jitter that flickers off frequently, hurting both Phase A AG-engage and Phase B replay drops.

`render_grasps.main`'s exception handler `sys.exit(2)`'s when an inner exception contains `'NoneType' object has no attribute 'view'`. After many target add/remove cycles OG's `articulation_view` invalidates, after which every `robot.get_joint_positions()` raises that error and the rest of the boot insta-fails. Fast-fail lets `scripts/render_grasps_loop.sh` restart immediately instead of burning ~30 s on every remaining pending row insta-failing it.

Wiped `outputs/grasp_datasets/graspgen_full/` and restarted from row 0. Mean per-object dropped to ~20 s on the first sample; ETA now ~5-15 hours for the 3405 pending rows (was ~36 hours with motion-plan + default fingers).

## 2026-05-04 — `refactor/remove-bddl-curation`: GraspGen install + run doc (`248a9df6`)

`docs/graspgen_pipeline.md` is the end-to-end recipe for the new pipeline: clone GraspGen + GraspGenModels (with the LFS-pull caveat that bit us when `/tmp/GraspGen` got cleaned), `uv venv` + `install_uv_pointnet.sh`, server start command, render_grasps invocation, output-file table, resume rules, and 5 troubleshooting entries (LFS pointer, ZMQ recv timeout, OG import drift, Phase A 0-holds, Phase B replay didn't hold, pointnet2 CUDA arch). CLAUDE.md gets a pointer.

## 2026-05-04 — `refactor/remove-bddl-curation`: standardize on GraspGen + two-phase grasp pipeline (`36a9b960`)

Replaces `sentinel/rl/grasps/`'s antipodal grasp dataset path (UW Lab OmniReset port) with a NVlabs/GraspGen ZMQ client and splits per-object handling into Phase A (search, no video) and Phase B (replay, video) sharing one physics-validation kernel.

Deleted (no remaining callers): `sampler.py`, `collect_batch.py`, `survey_graspability.py`, `measure_gripper.py`. New: `graspgen_sampler.py` (minimal msgpack-over-ZMQ client, in-process so the OG env doesn't pull `pointnet2_ops` / torch 2.1), `_viz_helpers.py` (matplotlib scatter + grasp-overlay shared by `render_grasps` / `inspect_mesh` / `visualize_grasps`), `inspect_mesh.py` + `visualize_grasps.py` (standalone debug viz scripts).

`collector.py` slimmed (dropped `GraspCollectorConfig.shake_*`, antipodal-only `_curobo_ik`) + extended with `run_grasp_attempt`, the shared kernel both phases call: trajectory replay (hard pin) → close → AG check → gravity hold → eef-distance acceptance. Phase A invokes it with `frame_callback=None`; Phase B with a closure that pushes frames into the MP4 buffer. This removes ~80 lines of near-identical physics code that was duplicated between the two phases.

`mesh.py` reduced to `mesh_from_og_object` (gripper-params helpers were antipodal-only). `render_grasps.py` rewritten as the two-phase driver: GraspGen → `collect_valid_grasps` (cuRobo motion plan, `ik_only=False` so trajectories are replayable for video) → save `.pt` (format consumed by `GraspDatasetResetter`) + `_grasps_*.png`; on Phase A success, optional `--save-video` triggers Phase B → `.mp4`; on Phase A failure, `_pcd_*.png` for diagnosis. Resume skips a row if `.pt` or `_pcd_top.png` or `.mp4` already exists.

Smoke-tested on 6 varied objects (alarm_clock, apple, comic_book, baseball, mug, alphabet_abacus) — 5/6 produced `.pt` + MP4 with 1-3 holds each. alphabet_abacus is the only object that consistently fails (hollow lattice, GraspGen confidence < 0.9). Per-100-candidate stats: cuRobo `no_path` is the dominant rejector (60-98% on round/symmetric objects), AG-miss next, phase2 gravity drop ~0%. Pass rate 0-30% varies wildly by object geometry.

`.gitignore`: `GraspGen/` + `GraspGenModels/` (cloned at project root for stability after `/tmp` got cleaned mid-session).

## 2026-04-28 — `feat/grasp-batch`: untrack + delete outputs/teleop/traj_*.hdf5

`git rm -r outputs/teleop/` removed 21 stale goblet-task HDF5 demos (31 MB) from both index and disk. They predated the gitignore exemption-for-teleop-hdf5 rule (which the previous chore commit already dropped from .gitignore), so they were lingering as already-tracked files. Current teleop pipeline writes to `outputs/gello_teleop_hdf5/<task>/` per family, not back into `outputs/teleop/`.

## 2026-04-28 — `feat/grasp-batch`: untrack outputs/pipeline_runs/mug_into_bowl_empty_*

`git rm -r --cached` on 51 `scene_ep*.json` files under `outputs/pipeline_runs/mug_into_bowl_empty_20260418_132924/` — they predate the `outputs/*` gitignore rule (added after the refactor branch landed) so they were still tracked despite ignore. Local copies preserved on disk; only removed from index. Cleans the merge into `dev` so it doesn't drag the 24 MB of stale snapshots along.

## 2026-04-28 — `feat/grasp-batch`: fix wrist-camera flip from J7 encoder unwind

`GELLO_JOINT_OFFSETS[J7]` base 4π/2 → 0π/2 (trim `-π/4` preserved). The J7 servo unwound by one full turn between calibration sessions, so the calibration script reported a different mod-2π-equivalent base. The OLD value made our formula compute franka_J7 = raw_J7 - 2π ≈ -2π = -6.28 rad — outside Franka's J7 limit (±2.897), so JointController clamped to -166° and the wrist camera (mounted on the end-effector link) appeared rotated 166° from upright (operator reported "wrist view backward, up/down + left/right both inverted"). Re-running `gello_get_offset.py` on the new machine produced the now-correct base offset.

## 2026-04-28 — `feat/grasp-batch`: gello UX overhaul (goal_checker + recalibration + deterministic startup ramp)

`sentinel/teleop/gello_franka_teleop.py` picks up three related improvements in one pass:

1. **goal_checker auto-success ported from so101.** Imports `_read_first_jsonl`, builds the success_checker from sibling `diagnostics.jsonl`, runs it every loop step, and breaks with `state["success"]=True` the moment the green-sphere goal region fires. S key downgrades to `state["manual_override"]`. Banner also shows `TASK` / `TARGET` from the same diagnostics. Reaches feature parity with so101.

2. **Re-calibrated `GELLO_JOINT_OFFSETS`** after stabilizing the leader arm: J2 base 2π/2→1π/2, J3 0→4π/2 (≡0 mod 2π — calibration script picked the wrapped form), J6 2π/2→1π/2. All post-calibration trims (J2/J4 relaxed-rest, J7 mounting) preserved.

3. **Deterministic startup pose + smooth ramp.** Added `GELLO_CALIBRATION_FRANKA_POSE` constant (Franka equivalent of GELLO held at the gello_get_offset.py reference pose, post-trim) and `GELLO_RAMP_STEPS=60`. `_build_from_snapshot` gains an `initial_joint_pos` kwarg that overwrites the snapshot's saved `joint_pos[0:7]`. main() seeds Franka at the calibration pose every launch (deterministic, snapshot-independent), then the loop ramps from that pose to GELLO's live reading over 60 steps (~2 s at 30 Hz). Eliminates the 100°+ jolt seen on task_0004 where the snapshot's saved Franka pose is far from where the operator holds GELLO. Leader connect moved before env build so DynamixelRobot failures surface in 1 s instead of after the 30-90 s OmniGibson init.

## 2026-04-28 — `feat/grasp-batch`: so101 startup banner shows task prompt + target name

`sentinel/teleop/so101_franka_teleop.py` extracts `prompt` and `goal_region.target_name` from the snapshot's sibling `diagnostics.jsonl` and surfaces them as `TASK` / `TARGET` lines in the Ready banner. Operators no longer need to alt-tab to a separate file viewer to find out which object the current scene wants them to manipulate.

## 2026-04-28 — `feat/grasp-batch` ← merge `feat/task-generation-scene-benchmark-staging`

Resolved the only true conflict (`sentinel/teleop/so101_franka_teleop.py`) by adopting the remote's auto-success path: snapshot's sibling `diagnostics.jsonl` builds a `goal_checker` (`sentinel/eval/goal_checker.py`) that fires `success_flag=True` and breaks the loop the moment the goal region is satisfied. S key now means "manual override" rather than the only success switch.

Local-only additions removed during merge cleanup: `_install_longfinger_franka_patch` (superseded by remote's eager `_patch_franka_longfinger` in `sentinel/_omnigibson_patches.py`), `--stock-franka` flag (no patch left to opt out of), the skybox + `LightingMode.CAMERA` lighting block, the robot-frame camera fallback. `--grasping-mode` flag and the conditional FrankaMounted → FrankaPanda +0.5m lift were kept (independent of the dropped features). `gello_franka_teleop.py` lost its `--stock-franka` flag and the import of the deleted longfinger function but otherwise keeps its lighting + camera customizations (gello is purpose-built for HF furnished scenes that need them).

Pulled in clean from remote: `sentinel/utils/goal_region.py`, `sentinel/eval/goal_checker.py` rewrites, `sentinel/envs/{frozen_task_runtime,perturbation_runtime,registry}.py`, the four-level perturbation pipeline + dedup scripts, and the `_patch_franka_longfinger` eager hook.

## 2026-04-28 — `feat/grasp-batch`: README documents GELLO teleop workflow

Renames the Teleoperation section to cover both SO-101 and GELLO with a per-leader comparison table at the top. New `### GELLO leader → Franka` subsection: one-time Dynamixel calibration command, the single-task launch, hotkeys (SPACE = gripper), and shared `--grasping-mode` / `--gpu-dynamics` flags. New `### Batch teleop` subsection documents `run_teleop_batch.sh` (SO-101 native) + the shell-loop workaround for GELLO until the script grows a `--leader` flag.

## 2026-04-28 — `feat/grasp-batch`: lighting/camera robustness, longfinger, grasping mode, GELLO joint teleop

`so101_franka_teleop.py`: long-finger FrankaPanda asset patch (`--stock-franka` opts out); `--grasping-mode {physical,assisted,sticky}` for thin/flat-object slip; skybox dome (intensity 12000) + viewport `LightingMode.CAMERA` so HF furnished scenes are visible in viewer + recorded sensors; robot-frame camera fallback when scene lacks a `support_surface` object (fixes cameras-in-walls in liquid_transport's 220-object rooms); conditional +0.5m base lift only when swapping FrankaMounted → FrankaPanda. `gello_franka_teleop.py` (new): 7-DOF kinematic-twin teleop, `JointController(position)` follower, `DynamixelRobot` direct read (skips GelloAgent's force-feedback layer), keyboard SPACE for gripper (no physical gripper yet), `--gpu-dynamics` for fluid scenes, shares `_install_longfinger_franka_patch` + grasping_mode with so101. Calibration constants in-file from the 2026-04-27 GELLO build.

## 2026-04-28 — `feat/grasp-batch`: real-teleop NPZ converters + openpi SFT doc

`sentinel/data/real_teleop_to_droid.py` (new) emits LeRobot in openpi's DROID schema (180×320 3-cam, joint_velocity action, push_to_hub with codebase_version tag). `sentinel/data/real_teleop_to_hdf5.py` (new) bridges into our existing Stage-2 lerobot_export for non-DROID flows. `docs/openpi_real_teleop_sft.md` (new) is the end-to-end pi0.5 SFT recipe (NPZ → dataset → LoRA finetune → serve_policy) extracted from the mug_into_bowl run.

## 2026-04-28 — `feat/grasp-batch`: batch teleop script with --task flag and per-task OUT_DIR

`scripts/run_teleop_batch.sh` (new) replaces the ad-hoc loop pattern. `--task <family>` sweeps every `scene_ep*.json` under `outputs/teleop_scenes/<family>/`; output defaults to `outputs/jixing_teleop2_hdf5/<family>/` so cross-task `scene_ep<NNNN>` filename collisions don't overwrite each other. Already-collected HDF5s (≥ 8 KB) are auto-skipped on resume; partial 96 B writes are recognized as stale and re-run.

## 2026-04-28 — `feat/grasp-batch`: SFT prep pipeline fixes (INPUT_GLOB / hub push / parquet layout)

`scripts/prepare_sft_data.sh` gains an `INPUT_GLOB` env var so non-`traj_*` HDF5s flow through, and the `"$PY" python -m ...` double-python typo (running `python` as a script name) is fixed. `lerobot_export.py` adds `--push-to-hub` / `--hub-private` that auto-create the `codebase_version` git tag openpi requires (plain `huggingface_hub.upload_folder` doesn't). `norm_stats.py` scans all `*.parquet` so LeRobot 0.4.x chunk-level layout is picked up, not just 0.3.x episode-per-file.

## 2026-04-21 — `feat/grasp-batch`: reset_mode='cached' unblocks multi-env + wandb logging

`GraspDatasetResetter` picks up a ``reset_mode: {'cached', 'ik'}`` parameter.
Cached is the new default and skips online cuRobo IK entirely — the
``arm_joint_pos`` column already stored in ``grasps_<cat>_<model>.pt`` goes
straight into ``set_joint_positions`` at reset time. Two consequences:

1. **Multi-env unblocked (with ``scene_file``).** Cached mode never touches
   ``StarterSemanticActionPrimitives``, so the cuRobo single-scene assertion
   noted in the 2026-04-20 entry is bypassed. ``--num-envs 4 --scene-file ...
   --reset-mode cached`` now works end-to-end.
2. **~100-500 ms saved per reset.** The 30 s first-call cuRobo JIT is gone too.
   At 200-step episodes this is ~5-10 % throughput improvement single-env,
   compounding 3-4× with multi-env.

Tradeoff: cached mode requires the target object to be at the same world pose
at reset as at collection time. Our ``_restore_target`` always restores to the
scene-init pose, so this holds for both ``scene_file`` and runtime-spawn
paths. Per-reset pose randomization (``pose_range_b``) remains a future
feature and needs the DLS IK port listed in Action items.

``ppo_grasp_reset`` also gains wandb logging (``--wandb / --wandb-project /
--wandb-run-name / --wandb-mode / --wandb-upload-ckpts``), with
``sync_tensorboard=True`` mirroring SB3's local TB events into W&B panels.

## 2026-04-20 — `feat/grasp-batch`: multi-env PPO blocked by OG's cuRobo single-scene assertion (superseded 2026-04-21)

## 2026-04-20 — `feat/grasp-batch`: multi-env PPO blocked by OG's cuRobo single-scene assertion

Wired SB3 multi-env (`SentinelSB3VectorEnvironment`, `--num-envs N`) into `sentinel.rl.training.ppo_grasp_reset`. The runtime-spawn path trips OG's scene-tiling bug on empty scenes (`decompose_mat` perspective check at `idx != 0`), which is avoided by passing a pre-baked `scene_file`. Collected `grasps_mug_kewbyf.pt` matching the existing `mug_into_bowl_empty_20260417_154117/scene_ep1.json` and wired `--scene-file` through `build_config`.

**Hard blocker hit**: `behavior-1k/OmniGibson/omnigibson/action_primitives/curobo.py:102` has `assert len(og.sim.scenes) == 1` in `CuRoboMotionGenerator.__init__`. Our `GraspDatasetResetter` calls cuRobo on every episode reset; in multi-env (`len(scenes) == 2`) the assertion raises, GraspTask's 20-retry loop exhausts, and reset fails with `Could not reset task`. Same would block any use of cuRobo primitives under vec-env — `training/ppo.py` works with multi-env only because it uses sticky grasping + random joint sampling (no cuRobo at reset).

**Paths to unblock** (deferred — documented here so future me doesn't repeat the audit):
1. **Port OmniReset's iterative Jacobian DLS IK into the resetter.** `DifferentialIKControllerCfg(ik_method="dls")` doesn't touch `og.sim.scenes`. ~25 iter per reset (still fast). 1-2 day change.
2. **Patch the upstream assert.** Read the rest of `CuRoboMotionGenerator.__init__` to see if it genuinely requires `scenes[0]` or if the assert is defensive; if defensive, submit a PR.
3. **Accept single-env (current state).** `num_envs=1` with `--scene-file` trained cleanly on `target_mug` for a 256-step smoke (ep_rew_mean=-1.12 after 2 iters, pipeline end-to-end). Proprio-only MLP doesn't need massive rollout parallelism; main cost is OG physics (~25-30 steps/s in 1 env).

Single-env command for the mug grasp-reset training:
```
python -m sentinel.rl.training.ppo_grasp_reset \
    --category mug --model kewbyf \
    --target-name target_mug \
    --scene-file outputs/pipeline_runs/mug_into_bowl_empty_20260417_154117/scene_ep1.json \
    --num-envs 1 --total-timesteps 200000 --n-steps 256 --batch-size 64
```

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

### LeRobot live-write + LTL integration (2026-05-25)
- `14b2cd58` `sentinel/data/lerobot_writer.py` — direct LeRobot v2.1 writer with no-PNG/no-encode passthrough patches; lerobot_export refactored to use shared utilities; legacy render_pnp_for_sft deleted
- `24fd2eb4` `pick_and_place_from_dataset` — `--lerobot-*` live-write CLI flags, auto-attach TaskLTLMonitor with cat→synset object resolution, Phase A cache reuse across variants (~6-10s saved/variant), per-stage prints in `_replay_holding`, early-exit-after-hover when target is inside goal (skips descend + final settle to avoid the cuRobo IK-flip bug)
- `a44a3db7` `tools/lerobot_from_mp4s.py` — one-off converter using hardlinks to migrate legacy fat-HDF5 + sibling-MP4 collections to LeRobot v2.1 (~25 min for 448 variants vs 9 h re-encoding); also strips image arrays from source HDF5 in place (294 MB → 340 KB per variant)
- Bonus: `datasets` downgraded to <3.0 in `behavior` env (lerobot 0.1.0 incompat with datasets 4.x Column return type); lerobot installed --no-deps so OmniGibson torch stack stays intact

### SFT-prior grasp shortcut + sweep driver
- `ac0458ed` `--phase-a-grasp-from-dataset` loads grasp pose from prior successful HDF5 (eef pose at gripper-transition step → base→world→target-local), uses it as a single OBB candidate; Phase A wall drops ~75s → ~22s with falls-back to OBB on miss
- `d5912b33` `tools/_pnp_sft_prior_n10.sh` sweep driver — 47 tasks × 10 variants each, appends to outputs/lerobot_pnp_sft_prior_n10/; one prior per task (lowest seed); follow-up will add per-prior loop for 5× more diversity

### Rebrand: SENTINEL-Lite → ManiGuard (2026-05-27)
- Renamed Python package `sentinel/` → `maniguard/`, all imports, `Sentinel*` identifiers, and branding; pyproject distribution `sentinel-lite` → `maniguard`; `.gitignore`/`_version` paths repointed. Preserved external contracts: `SENTINEL_*` env vars, asset id `sentinel_goblet_pick_place`, HF dataset ids, wandb `sentinel-grasp-reset`, and the `NU-IDEAS-Lab/SENTINEL-Lite` GitHub slug. Verified: `compileall` OK, guarded `import maniguard` OK, 125 tests collect in behavior env. Pre-existing WIP (data/eval refactor) left uncommitted.

### Docs: ManiGuard module walkthrough (2026-05-27)
- mkdocs site reorganized into lifecycle nav (Foundations + 5 stages). New Foundations pages (env layer, LTL safety, OmniGibson patches & configs); architecture overview rewritten. Task-gen pages refreshed, cabinet/jar/dusty pages added, offline-pool carousel + 6fam-base example renders, and an "Add a custom pipeline" guide. Carousel/triptych CSS+JS under docs/stylesheets + docs/javascripts.

### Refactor: data-collection consolidation (2026-05-27)
- Moved maniguard/teleop/ → data/teleop/ and tools/ cuRobo demo-collection scripts (+_sft_recorder) → data/curobo/; restructured data/ into lerobot/, real_teleop/, scene/ subpackages. Rewrote all imports + path-depth hacks; benchmark.py now imports maniguard.data.curobo._sft_recorder directly. compileall OK, 125 tests collect, strict docs build clean. (Carries pre-existing data/eval WIP.)

### Docs: SFT / Eval / RL sections + RLinf purge + Home gallery (2026-05-27)
- Added SFT controller/action/eval end-to-end guide, Evaluation overview, RL overview, and a Reference page; removed CLI/Outputs blocks site-wide; deleted the redundant Teleop-overview page (content folded into Data Collection) and documented the interactive ghost-gripper cuRobo collection. Purged all RLinf references from the docs site, README, and CLAUDE.md (RLinf is no longer used) and fixed stale module refs (ManiGuardEnv, maniguard_env.py, tasks/rlinf/openpi subpackages, serve module names). Rebuilt Home around the lifecycle nav with rendered Material icons and a 10×10 gallery of 100 6fam-base thumbnails (left/right overview; cabinet = right only).

### Docs hero montage + README polish + Pages workflow (2026-05-27)
- Combined the 100 6fam-base thumbnails into a single 10×10 montage (docs/index_gallery/montage.jpg); Home and README now use it as a hero image (dropped the 100 individual files + the .mg-gallery grid). Polished the README header (centered montage, tagline, nav links, Documentation link). Added site_url + a GitHub Actions workflow (.github/workflows/docs.yml) that builds --strict and gh-deploys the MkDocs site to GitHub Pages on push to main.

### Safety: jar_closed hinge-angle fix (2026-06-08)
- `f8dae056` `maniguard/utils/safety_monitor.py` — hinged_jar ships abilities={} so OmniGibson never attaches the Open state (KeyError every step) and `_build_unary` ignored `negated`; jar_closed was stuck False, firing a false-positive close_before_lift on any lift. Added `_is_open_via_joints` (joint-angle open-check, OmniGibson 5%-of-range convention) as the Open fallback + per-element `negated`; scope = jar_closed only (sole user of state:open/negated). Validated on the teleop'd jar scenes: task_0005 (lid closes) jar_closed flips True @ step 417 (0->1584 True), no false violation; task_0001 (lid never closes) stays False, real jar_upright tip-over still fires; 3972->0 "Open check failed" warnings.

### Safety: lid_on_container resolution fix (2026-06-09)
- `ec1b74f4` `maniguard/eval/benchmark.py` — `_build_active_objects_for_ltl` resolved LTL patterns by category / synset-lemma / name-fnmatch, else fell back to the support surface. lid_transport's `lid_on_container` is generated over synset `lid.n.02_*`, but the lid spawns under a ROLE name (`lid_cap_ep1_1`, category `cap`) -> no match -> fell back to the countertop -> `OnTop(countertop, container)` = always False, so any container lift fired a false `lid_before_lift` violation (and a capping policy would be falsely penalised). Added a role-prefix name match (`name.startswith(base+"_")`) for unresolved synset patterns before the surface backstop -> `lid.n.02_*` now resolves to `lid_cap_ep1_1`. Validated on task_0017: "unresolved" warning gone, `lid_on_container` evaluated every step over the real lid (stays False since lid/1025 skips capping -> the violation is now genuine).

### Eval: EVAL_USE_GPU_DYNAMICS for liquid scenes (2026-06-09)
- `3d1eedcb` `maniguard/eval/benchmark.py` — liquid/particle scenes (clutter `liquid_transport`) initialize a water particle system that needs `gm.USE_GPU_DYNAMICS=True`; without it the benchmark hard-crashes ("Failed to initialize water system" -> segfault). Added an env-gated `EVAL_USE_GPU_DYNAMICS` flag, set in `_init_omnigibson` before physx init (default off = CPU dynamics unchanged). Verified: clutter liquid 18/18 ran with `EVAL_USE_GPU_DYNAMICS=1`, no water-init crash.

### Eval: summary aggregate + OOD NaN-cascade handling (2026-06-10)
- `e8618ccc` `maniguard/eval/benchmark.py` — (1) summary.json recomputed from the accumulated results.jsonl (completed-only) so a batch run gets a real aggregate, not the last process's single-task stats. (2) The OOD failure cascade (a flailing policy drives the arm -> PhysX NaN -> the GPU articulation view is invalidated -> `get_joint_positions()` returns None inside `extract_obs`) is now caught and recorded as a clean failure (`status=completed, success=False, nan_terminated/_step`) instead of a crashed/excluded row. A proactive `obs["states"]` finiteness guard is the first layer but does NOT fire in practice — the joint state stays valid until it abruptly goes None — so the except-path reclassification is what catches it. Root cause: assisted-grasping raycasts emit a NaN `unitDir` once the gripper pose degenerates (a known upstream OmniGibson articulation bug, handled at the eval layer, not patched in OG). Caveat: `nan_terminated` rows have LTL monitored only up to the termination step. Also routed stack eval to the `*/env` scene variant (base/ has a spawn-failure bug). Validated: clutter 56/56 tasks completed, 0 crashes (6 nan_terminated); success ID 8/29, OOD by-scene 1/19, by-object 0/8.

### Eval: engagement ladder + contact-gated safety metric (2026-06-10)
- `9816f90d` `maniguard/eval/benchmark.py` + `eval_config.py` — new per-rollout engagement signals so a do-nothing rollout no longer counts as vacuously "safe". Whole-arm contact (`ContactBodies` ∩ `robot.links` over task objects = target+distractors, not the table/marker) gates safety: a rollout is safety-evaluated only if it contacted something, and a violation counts only at/after `first_contact_step`. Plus `ever_grasped`/`grasp_steps` (reuses the goal checker's `held`), `target2spawn_max_dist` (peak target offset from spawn), `eef2target_min_dist` (closest eef approach), and an outcome ladder `idle→reached→manipulated→success` (`tau_move`/`tau_reach` config knobs, 5cm/12cm). New fields land in `results.jsonl` + `summary.json` ALONGSIDE the existing success/violation metrics (kept unchanged). Design lives in `eval_engagement_metric_spec.md` (local, not committed). Pilot (10 OOD liquid → `outputs/eval_logs/_pilot_engagement_metric/`): cleanly split 3 vacuous-safe `idle` (never contacted) from 7 engaged, turning a diluted 70% violation rate into a contact-gated 100%.

### Tooling: eval_family.sh — full-family eval template (2026-06-11)
- `f18a42f9` `scripts/eval_family.sh` — one-command full-family eval: classifies every task of a 6fam-base family into the ID/OOD taxonomy + per-task GPU-dynamics, runs each in its own process into the standard `outputs/eval_logs/<leaf>/{ID,OOD/<tag>}/` layout, engagement metric built in, FORCE guard against clobbering finalized dirs. `clutter_pickup` classifier implemented inline; other families add a branch when their taxonomy is designed. Used for the clutter engagement full re-run (`clutter_pickup_joint_engagement/`, separate from finalized).

### Tooling: eval_family.sh — dusty taxonomy + robust parsing (2026-06-11)
- `d22bf319` `scripts/eval_family.sh` — added the dusty_transfer classifier branch (ID = the 3 teleop'd (food,source,dest) triples, food always potato; else OOD/by-object), switched diagnostics parsing to raw_decode (dusty's are pretty-printed multi-line), and a generic skip for unusable tasks (empty prompt / scene=None; dusty has 25 broken-source tasks, only 23 usable). One script now drives clutter OR dusty by the family arg. Validated: dusty 23/23 completed, 0 crashes.

### Tooling: eval_family.sh — lid taxonomy + ID_REPEAT (2026-06-12)
- `03b65d2b` `scripts/eval_family.sh` — lid_transport classifier branch (ID = teleop'd containers, only milk_carton in 6fam-base; food+novel -> OOD/by-object_novel; liquid -> OOD/by-target_liquid). New `ID_REPEAT=N` env (repeat each ID task N times for a stable rate when ID is tiny — lid ID=1, run 20x). gpu gate widened to `"liquid" in pipe`. Validated: lid 51/51 completed, 0 crashes. Result: lid 1/51 success; Probe-C (open-loop replay) ~3% err -> fit the data, closed-loop drift / data-quantity bottleneck (same as dusty).

### Tooling: eval_family.sh — jar taxonomy + empty-table scene guard (2026-06-12)
- `7fd45e1b` `scripts/eval_family.sh` — jar_transport classifier branch: the manipulated object is always `hinged_jar` and the motion is identical (close lid -> carry); only the food content inside varies, so ID = the 2 SFT-trained contents {jar_of_cumin, can_of_bay_leaves} (run `ID_REPEAT=7`), else OOD/by-content_novel. Also relaxed the task-skip guard to drop only on a missing prompt — jar tasks are empty-table (`scene_model=None`), which `benchmark.py` loads via the empty-`Scene` path from the saved scene_file; the old `not scene` check would have skipped all 27 (dusty's None tasks are prompt-less anyway, so they still skip; no past result changes). Validated: jar 45/45 completed, 0 crashes, 0 nan. Result: **jar 14/45 success (31%) — the BEST family**, ID 33% ≈ OOD 29% (content is a soft visual/language axis). Probe-C ~1-3% open-loop err AND it succeeds closed-loop = fully mastered. This **reshapes the cross-family conclusion**: dusty (0/23) and lid (1/51) fail because their manipulated OBJECT varies and 30 demos can't cover it — NOT from demo count; jar has the same 30 demos but a fixed operand and works. The bottleneck is SFT operand-distribution coverage, not the number of demos.

### Eval: stack_retrieve — 0/44, worst family (closed-loop collapse) (2026-06-12)
- `446383d6` `scripts/eval_family.sh` (stack+cabinet taxonomy + env-subdir) drove the stack run on the `task_*/env` variant (base/ has a spawn-failure bug). 3-bucket taxonomy ID/flat 23 + ID/same 11 {bowl,chili} + OOD/by-object_novel 10; real household scenes; grasp=assisted; cam=left. Result: **stack 0/44 success — the worst family, near-totally inert**: 39/44 (89%) never touched anything, and of the 5 that engaged, 4 toppled the stack (violation). Probe-C (open-loop replay of flat ep0 + bowl ep53) ~1–2% err → the policy fit its data; this is pure closed-loop collapse. The paradox: stack has the MOST SFT data (60 eps) yet the WORST result, because its eval distribution is the widest (diverse targets + real household scenes + the hardest skill, precise retrieval without toppling). 44/44 completed, 0 crashes. This + jar (best, 31%) bracket the cross-family thesis: open-loop fit is ~1–3% everywhere; the closed-loop outcome is set by how well SFT covers the eval's operand+scene+skill distribution, not by demo count.

### Eval: cabinet_pickup — sharpest ID/OOD split; all 6 families done (2026-06-12)
- `446383d6` `scripts/eval_family.sh` drove cabinet (ID_REPEAT=20: paper_towel_holder ×20 + 36 novel-target OOD; empty-table scene_model=None -> empty-Scene path; external_cam=right unique to cabinet; grasp=assisted; success = target in drawer + drawer closed, no held requirement). Result: **the sharpest split of any family — ID 12/20 = 60% success (best in-domain of all six), 55% safe-success; OOD 0/36, with 66% of engaged rollouts toppling something.** Same task + scene + skill, only the placed object changes -> cleanest operand-coverage proof. Probe-C ~1–2% (fit data; ID works closed-loop). 56/56 completed, 0 crashes. **All 6 families now evaluated.** FINAL cross-family ranking by success: cabinet-ID 60% > jar 31% > clutter-ID 28% >> lid 2% ~ dusty 0% ~ stack 0%. open-loop fit is ~1–3% in every family (never the bottleneck); the closed-loop outcome is decided by SFT coverage of the eval's operand (+scene+skill) distribution.

### Tooling: openloop_replay_probe.py — open-loop fit diagnostic (2026-06-12)
- `32ebc7b7` `tools/openloop_replay_probe.py` — feeds an SFT episode's recorded observations back to a running openpi policy server and compares the predicted action against the recorded ground-truth, normalized by each action dim's std. Separates "did the checkpoint fit its training data?" (open-loop) from "does it work closed-loop?" (what the OmniGibson eval measures); near-zero error ⇒ any eval failure is closed-loop drift/collapse, not undertraining. Parametrized over dataset + `--episode` + `--external-cam {left,right}` (cabinet serves the right overview, others left) + `--samples` + `--host/--port`. The standard companion diagnostic referenced in every family's eval README — empirically ~1–3% normalized error in all 6 families (incl. the 0-success ones).

### ManiGuard-Bench rebuild · Step 0 Part A: promote cam_left_shoulder (2026-06-12)
- `8ffa58af` `maniguard/utils/camera_setup.py` + `task_generation/utils/video.py` — first code of the 6fam-base → ManiGuard-Bench rebuild (spec = the Obsidian design doc; output goes to a NEW `outputs/lerobot_datasets/maniguard-bench/`, 6fam-base stays read-only source). Made `cam_left_shoulder` a first-class shared camera: added to `EXTERNAL_CAMERA_NAMES` (→4), centralized the blend in `left_shoulder_eye(opp,left)` (0.55*left+0.45*opp XY at left height), and both camera modes (support-relative + robot-frame) now emit the 4th view explicitly. Previously left_shoulder was only a jar/cabinet copy-paste, so the other 4 families rendered 3 views. Bench standardizes on robot-frame cameras (orientation rigorously verified sound across all 6); the render step will re-stamp diagnostics['cameras'] with the live-computed poses. Policy input unchanged (still cam_opposite only).

### ManiGuard-Bench rebuild · Step 0 Parts B–D: shared render step + robot-frame 4-cam + canonical init pose (2026-06-13)
- `e838a050` `maniguard/data/bench_builder/{__init__,render}.py` (new) + `maniguard/utils/{camera_setup,robot_pose}.py` + `task_generation/utils/video.py` — completes Step 0 (the bench render foundation). `render_task(scene_file, diagnostics, out_dir)` is the single shared render step every base/perturbation task uses: loads a snapshot (mirrors `eval.benchmark.build_og_config`; empty-`Scene` vs `InteractiveTraversableScene` branch off the header class + `scene_model`), puts the robot at the canonical init pose, places the 4 robot-frame cameras, RE-STAMPS `diagnostics['cameras']` with the live poses, and writes 4 static ~2s showcase MP4s (no physics step). `compute_robot_frame_views(env)` centralizes the robot-frame placement (forward = base +X projected to ground), shared by the render step and `setup_external_cameras_robot_frame` (DRY).
- **Render polish from end-to-end spot-check (Parts C/D)**: (1) opposite cam moved across the table — `opp_eye = rp + forward*(workspace_off+back_off)` so depth far→near is robot→pack→cam (was sitting behind the robot); (2) `left_shoulder_eye` decoupled from the opp/left blend (the opp move dragged the blend off the shoulder) → now computed directly from the robot frame (behind-left); (3) `robot_pose.BENCH_INIT_QPOS` = the OmniGibson FrankaPanda default `[0,-1.3,0,-2.87,0,2.0,0.75]` — J6=2.0 points the wrist cam down at the workspace, J7=0.75 keeps the two-finger gripper plane orthogonal to the arm; (4) static showcase (no `og.sim.step`, 60 frames @ 30fps).
- **Stability verified** on jar base (empty `Scene`), clutter base (empty `Scene`, different family — robot at world `[8.13,2.33]`, cameras still correctly robot-framed), and clutter env (`InteractiveTraversableScene` + office_large full room): both `_build_og_config` branches render 4 good externals + the onboard wrist; robust to per-snapshot robot naming (`agent_0` vs `robot_qipvba` — uses `robots[0]`, no hardcoded name). Wrist angle is family/scene-independent (pose + fixed eef mount). Note: `render_task` renders only the 4 externals; the wrist is a policy-obs stream (forced to 256 at collection/eval) and isn't part of the review videos.

### ManiGuard-Bench rebuild · P1.0: base finalize toolchain (2026-06-13)
- `2611cb5e` `maniguard/data/bench_builder/{finalize_base,validate_base,run_finalize_base}.py` (new) + `render.py` + `maniguard/utils/robot_pose.py` — finalizes a read-only 6fam-base task into a ManiGuard-Bench base task (NEW `maniguard-bench/` folder; source never touched, leftovers just aren't selected). `render_views(env, ...)` is the single shared render step: 4 robot-frame cameras + **idle-step** recording (`og.sim.step()` per frame while the arm is held at pose A by the stiff Isaac drive, so the clip shows physical stability), returning arm_drift/obj_disp — this supersedes Step 0's frozen render. `finalize_base_task` strips the source robot and bakes ONE uniform canonical robot (FrankaPanda+longfinger + `BENCH_CONTROLLER_PRESET=joint_position_raw` + `BENCH_GRASPING_MODE=assisted`, pulled from the shared `CONTROLLER_PRESETS`), enforces `base_z = support_top + ROBOT_MOUNT_OFFSET` (0.02, keeping xy/yaw), bakes pose A, saves the clean snapshot, renders, self-checks, and writes a `bench` provenance block. `validate_base_task` is an exhaustive OFFLINE 12-check QC (robot/pose/mount/config/objects/goal-marker/cameras/LTL-Tier-A/videos/diagnostics) read from the output artifacts. `run_finalize_base.py` batches it one fresh subprocess per task (success = output presence, segfault-tolerant), `--jobs` parallel workers, driver-side offline validate, full per-task rows aggregated into `base_manifest.jsonl`, `--skip-existing` to resume.
- **Config model clarified (design doc §2)**: the snapshot bakes two layers — ① hardware/physical invariants (FrankaPanda+longfinger, mount, pose A) the downstream reads but never changes, and ② software knobs (controller/grasping) the bench DECLARES a uniform default for but which eval/teleop override per-run (inert in the snapshot). This replaced an earlier `include_robots` (inherit-the-source-config) approach — that was an eval-compat compromise; the bench now owns a uniform canonical declaration, controller dict single-sourced in `CONTROLLER_PRESETS`.
- **Verified** on jar (no-op: mount/pose already conform, config normalized) + clutter (re-mounted +3.4 cm, pose baked): both save the identical canonical robot config (`JointController` raw `command_input_limits=None` + `assisted` + `action_normalize=False`), all 12 QC checks pass, `--jobs 2` runs two OmniGibson workers without CUDA OOM, `--skip-existing` resumes in seconds.

### ManiGuard-Bench rebuild · owned diagnostics schema (2026-06-13)
- `28af736a` `bench_builder/{finalize_base,render,validate_base}.py` — the bench now PRODUCES a clean owned diagnostics record instead of copying the source blob. `finalize_base` builds `out_diag` explicitly: task-IDENTITY fields carried by allowlist (prompt / ltl_safety spec / selection / goal_region / family task-def / activity_name / ...), physical/runtime fields recomputed fresh in-sim (`cameras`, uniform `surface_info` from the resolved surface AABB, `gate_pass` / `ltl_violated` / `steps_executed` over the bench's OWN idle-step, slim `ltl_summary` with the 301-step log + duplicate constraints dropped, `bench` provenance), derived `clutter_info`/`lid_info` for the two families whose source pipeline never wrote one, and a `bench.dropped_unexpected_src_fields` guardrail. Dropped: the source's `ltl_summary` (48 KB collection log) / `snapshots` / `videos` / `robot_base` (stale) and its runtime `gate_pass`/`ltl_violated`/`steps_executed`/`surface_info` (all RE-computed). Each task's diagnostics shrinks ~54 KB → ~5 KB.
- **Key principle (design doc §2 + §8)**: `gate_pass`/`ltl_violated`/`steps_executed` are RUNTIME results (the "spawn → run a few sim steps → is it stable / does LTL hold" check, generation-time analog), NOT static task data — the bench recomputes them over its own idle-step (LTL monitor stepped inside `render_views`), never copies the source's. `render_views` gained an optional `ltl_monitor` + returns `steps_executed`. `validate_base` checks the owned-schema fields + asserts `gate_pass==True` / `ltl_violated==False`; `no_fallen` now excludes the support surface + goal marker (a tall bar's origin sits ~0.6 m below its top — fixed the jar task_0026 false positive). Two downstream-refine notes logged (§9): the LTL active-object resolver is duplicated in bench_builder + benchmark (unify into shared infra), and lid's LTL hardcodes `breakfast_table.n.01` (resolves via surface fallback on other tables).

### ManiGuard-Bench rebuild · P1.1 jar + drop_list/prune_reindex (2026-06-13)
- `d965ff8f` `bench_builder/{prune_reindex.py(new),run_finalize_base.py}` — full-run fail handling. `run_finalize_base` writes `drop_list.json` next to `base_manifest.jsonl` only when a batch has fails (a CANDIDATE list for human review, not auto-drop — tool bugs get fixed + re-validated, only genuinely bad tasks get pruned; not persisted since the deterministic checks re-flag bad tasks on any re-run). `prune_reindex.py` deletes the user-decided dropped task folders and renames survivors to a gap-free `task_0000..` sequence (bench keeps its own contiguous index, decoupled from the read-only source; only the folder name + `_finalize_row`/manifest `task` carry the index — `activity_name`'s `trial_N` moves with the task, snapshot/videos carry none), rebuilds the manifest, writes `_index_map.json` (bench→source) for traceability, deletes `drop_list.json`; `--dry-run` prints the plan.
- **jar finalized (the first P1.1 family)**: 27 source tasks → finalize-all → 26 ok + 1 fail. The fail, jar **task_0004**, is a genuine bad task (light container + heavy lid + over-full contents → skewed COG tips over once physics runs; the fresh idle-step LTL check flagged `ltl_violated` at step 1 — the source's generation-time jitter rollout had missed it). Pruned + reindexed → **26 contiguous tasks, all ok** in `maniguard-bench/jar_transport/`.

### ManiGuard-Bench rebuild · P1.1 cabinet finalized (2026-06-13)
- **cabinet finalized (P1.1 family #2)**: 37 source tasks → finalize-all → 34 ok + 3 fail. Two fails (**task_0026 / task_0028**) were a `no_fallen` tool false-positive, not bad tasks: goal-region-less families (cabinet) name the surface entity `support_surface`, which the validator's old surface-name / `gr_support` / `marker_name` exclusion never matched, so the surface's own origin (a desk centre sits ~0.33 m below its top plane) tripped the "fallen task object" check while every other check passed and the objects barely moved (`obj_disp` ~0.002 m). Fixed `validate_base.no_fallen` to also resolve the real surface entity by **category+model** (double-match guards against excluding a same-category manipuland) → re-validate → ok.
- The one genuine fail, **task_0006**, is a real bad task: `target_bag_of_rice` rests at `d_xy=1.5 m` from the robot base (outside the 0.20–1.10 reach band), displaces 0.48 m over the idle-step settle, and violates LTL. Additionally the user dropped **task_0022** on semantic grounds (`target=pot_plant` — foliage makes grasp + drawer-insertion too tricky for a clean base task). `prune_reindex --drop 6,22` → **35 contiguous tasks, all ok** in `maniguard-bench/cabinet_pickup/`, with `_index_map.json` recording the bench→source remap.
- `bench_builder/review_snapshots.py` (new) — visual-QC helper: walks a family's existing `task_*` folders, grabs the LAST frame of each task's opposite (wide front) review video, and writes `<family>/snapshots/<task_id>.png` (a `snapshots/` folder sibling to the task folders, one-to-one by task id). Lets the reviewer flip through stills instead of scrubbing every video. Drop-agnostic — processes whatever tasks exist when run. `python -m maniguard.data.bench_builder.review_snapshots --family <fam>`.

### ManiGuard-Bench rebuild · P1.1 clutter finalized — liquid GPU-dynamics fix + actual object inventory (2026-06-13)
- **clutter finalized (P1.1 family #3)**: 56 source tasks = **29 dry `auto_clutter` + 27 liquid `auto_liquid_transport`** (a "carry the filled decanter/water-glass without spilling" activity bundled into the clutter folder). The first full run crashed all 27 liquid tasks (each produced only `scene_ep1.json`, no diagnostics/videos). Root cause: **the bench finalize never set `gm.USE_GPU_DYNAMICS`**, so it ran on the default CPU pipeline — fluid particle systems only simulate under the PhysX GPU pipeline, and the CPU pipeline NaN-segfaults on the first physics step. NOT cumulative / scene / semantic: the same scene succeeds dry and crashes liquid, and an isolated single-task re-run also crashes; the dry/liquid split correlates 100% with the ok/fail split (`liquid&FAIL=27, dry&ok=29, cross-terms=0`).
- **Fix** (`feat(bench): GPU-dynamics gating for fluid tasks + actual object inventory`): `finalize_base._needs_gpu_dynamics(diag)` gates on `selection.system_name` (the liquid clutter's `water`); fluid tasks set `gm.USE_GPU_DYNAMICS=True` + `gm.ENABLE_FLATCACHE=False` BEFORE the env is built (cannot toggle mid-session), dry tasks are left on the default CPU pipeline (subprocess-per-task isolates this). Mirrors the source `liquid_transport_pipeline`. `bench.gpu_dynamics` is recorded in provenance so downstream eval/collection knows to enable it too (§9-6). All 27 liquid tasks then finalize cleanly.
- **Actual object inventory** (same commit): the source pipeline drops objects it cannot place at generation time, so `spawn_specs` over-counts the real layout (13/56 tasks under-placed by 1-2). `clutter_info.n_clutter_objects` and the validator's `object_count` now use the ACTUAL count from the finalized snapshot, not `spawn_specs`: `object_count` became a PRESERVATION check (`len(non_robot) == bench.n_src_objects`, catches finalize-time loss, never false-warns on source drops); a new `spawn_shortfall` FAIL fires when actual `< 0.6 ×` designed (severe under-placement surfaces in the manifest as a drop candidate instead of needing per-task video review — none in clutter, min frac 0.71); and the manifest row gains `object_spawn_num` ("`designed -> placed`", e.g. `"7 -> 5"`; empty when fully placed) so under-placed tasks are scannable in `base_manifest.jsonl`. Provenance adds `n_src_objects / n_task_objects / n_task_intended`.
- Full 56-task re-run → 55 ok + 1 fail. The fail, **task_0053** (`auto_liquid_transport`), is a genuine bad task (`wineglass_138` topples below the surface at init → `init_ltl_doomed` + `ltl_violated`), deterministically reproduced across the fresh overwrite re-run. `prune_reindex --drop 53` → **55 contiguous tasks, all ok** in `maniguard-bench/clutter_pickup/`, `_index_map.json` recorded, snapshots regenerated (55/55), user-reviewed all ok.

### ManiGuard-Bench rebuild · P1.1 lid finalized — LTL normalization + offline surgery (2026-06-14)
- **lid finalized (P1.1 family #4)**: 32 source tasks = 19 dry (rigid food) + 13 liquid (`system_name=water`, kettle/jug/bottle — the GPU-dynamics gate from the clutter commit handles them, no code change). The `lid_before_lift = (container_on_support) U (lid_on_container)` constraint is a `U` (until), not the `G` of the other families — verified the bench idle-step handles it correctly: at init the lid isn't placed (B false) but the container is on support (A true) and accepting is still reachable → NOT doomed → `ltl_violated=False` (the source recorded the same for all 32; the lid being un-placed is a manipulation goal, not a spawn invariant).
- **Two hardcoded-synset LTL bugs fixed** via new `finalize_base._patch_lid_ltl` (applied BEFORE the monitor; both rewrites resolve to the SAME object so the monitor outcome / `ltl_violated` / gate are unchanged — only the carried spec text + the validate verdict change): (1) `lid_on_container.over` hardcodes `lid.n.02_*`, but 8 tasks use a `cap.n.02` (bottle/carton screw-cap) → resolved to 0 objects, the `check: all` AP went vacuously TRUE and `lid_before_lift` was silently unenforced (validate Tier-A correctly FAILED them) → rewrite to the spawned lid's synset; (2) `container_on_support.relative_to` hardcodes `breakfast_table.n.01_*` on 14 tasks regardless of the real surface → resolved only via the surface fallback (the §9-5 warn) → rewrite to the actual surface's category prefix `<category>_*` (the form the correctly-generated tasks already use), derived from the `surface` object name. §9-5 is now resolved at the source.
- **Drop-list**: 2 cap tasks (**0011 milk_carton, 0022 chicken_broth_carton**) dropped on the user's call; the other 6 cap tasks patched. `prune_reindex --drop 11,22` → 30 tasks.
- **Offline surgery (no sim re-run)**: two of the three cleanups are pure output-metadata and were applied by editing the existing diagnostics + re-validating, NOT a full re-run (verified `_DROP` feeds only `unexpected`, and the §9-5 relative_to rewrite resolves to the same surface so all in-sim products are byte-identical): (a) `_DROP` += 9 redundant lid top-level source fields (`container/food/item_category|model`, `n_objects_active|requested`, `system_name` — all dups of `selection`/`lid_info`/fresh provenance), clearing the `dropped_unexpected_src_fields` finalize warn on all 32; (b) the §9-5 `relative_to` rewrite. Only the 6 cap tasks needed a real sim re-run (content change). **Result: 30 contiguous tasks, ALL ok** in `maniguard-bench/lid_transport/`, snapshots 30/30, user-reviewed.
- **Gotcha logged**: never run `run_finalize_base --skip-existing` AFTER a `prune_reindex` — the driver aligns source↔output by index, so a reindexed output makes it re-finalize the (now-renamed) tail and re-create dropped tasks. Hit this once (spurious task_0030/0031 = dups of 0028/0029), deleted them and rebuilt the manifest by iterating the output dirs. **Prune must be the LAST step of a family.**

### ManiGuard-Bench rebuild · P1.1 stack finalized — env-recovery for blank-spawn tasks (2026-06-14)
- **stack finalized (P1.1 family #5)**: 45 source tasks (pull a flat/bottom object out from under a stack of 3; modes `stack_flat` 23 + `stack_same` 22; no liquid; LTL = 4 `G` safety props). **17/45 had a BLANK core scene** — the `base/scene_ep1_replay.json` (stripped snapshot the bench reads) contained only the surface + goal marker, zero task objects (objects never appeared in the source's own video — the user had hit this in teleop/eval). The source metadata LIES: it recorded `gate_pass=True / ontop_valid=True / ltl_violated=False` for all 45 including the 17 blanks. **Lesson: never trust source spawn metadata — verify the snapshot's task objects directly** (scan the registry for objects on the surface).
- **Root cause = a STRIP bug, not a spawn failure**: the objects WERE spawned and neatly stacked — they live in `env/scene_ep1.json` (full 200+ object room) for all 17, properly layered on the surface. The generation-time strip step that produced `base/scene_ep1_replay.json` lost them for these 17. So they are RECOVERABLE.
- **Recovery (reused tools, no code change)**: `replay_empty_from_dataset --scene-subdir env` re-strips the full-room `env/scene_ep1.json` (keeps surface + the spawn-spec task objects, drops the room, re-spawns the marker, replays 60 steps) into a staging dir, then `run_finalize_base --src-root <staging> --src-subdir replay_empty` finalizes from there. Verified end-to-end on task_0015 (blank → recovered router + 3 stacked cigarette packs). Staged all 44 env-having tasks (each → surface+4+marker); **task_0044 has NO env** so it finalized from its own (good, non-blank) `base/scene_ep1_replay.json`.
- Full 45 finalize → 42 ok + 3 fail. The 3 fails (**task_0022/0035/0037**) are genuinely unstable stacks (the top 2 stack objects topple below the surface during the idle-step → `no_fallen` + `ltl_violated` + `init_doomed`) — the fresh safety check working. User dropped the 3 fails + 14 more on manip-feasibility grounds (stacks too flat / objects too large to be a realistic retrieve). `prune_reindex --drop` 17 → **28 contiguous tasks, all ok** in `maniguard-bench/stack_retrieve/`, snapshots 28/28, user-reviewed.
- **task_0044 notes** (force-added to match a teleop task): its 6fam-base `base/` video shows a room and is MISLEADING — a stale generation-time render; the saved `base/scene_ep1_replay.json` is a clean empty `Scene` (breakfast_table + 4 bowls + marker, 0 room objects), structurally identical to the other stripped bases, so it finalizes to a clean empty scene like the rest. It has NO `env/` variant, so the Phase-2d env perturbation cannot be built for it in our finalize bench style (skip task_0044 there).

### ManiGuard-Bench rebuild · P1.1 dusty finalized — Phase 1 COMPLETE (2026-06-14)
- **dusty finalized (P1.1 family #6, the last)**: "wipe the dusty destination clean with the sponge, then transfer the food into it". 48 source tasks split cleanly by SCENE CLASS (verified directly, not from metadata): **0000-0022 (23) = `InteractiveTraversableScene` full-room, valid** (surface + 4 objects on it + dust); **0023-0047 (25) = `Scene` empty but BROKEN** (the `surface` string isn't in the scene and the task objects sit at floor height — no full-room source to recover from, so genuinely dropped). Used the 23 full-room tasks → `replay_empty` (room strip, like stack).
- **`replay_empty` dusty support** (`feat`): (1) **dust restore** — replay_empty rebuilds only the rigid `object_registry`, dropping the `system_registry`, so the dust would vanish. Reload the source dust particle group onto the rebuilt dest via the system's own `load_state` with the group's `particle_attached_obj_uuid` repointed to the new dest — this preserves the generation-time bottom-concentrated dust filter EXACTLY (NOT a fresh `Covered.set_value`, which would re-cover the whole object). Dust is a visual "Covered" particle (no GPU dynamics needed; the dusty pipeline never set it). (2) **sponge kept** — the dusty sponge is named `dust_sponge_0` (role-prefix, no `_ep`), which the old `_is_task_object_name` dropped; relaxed it to accept the category as a contiguous token run. (3) **sponge placement** — snap the sponge to the source/dest XY midpoint (the spec §4d ideal); 4/23 source spawns (0002/0013/0017/0021) flung it out of reach (`d_robot` 1.0-1.6 m) → moved to the midpoint (`d_robot` 0.49 m), so all 23 kept (none dropped). (4) **pretty-printed diagnostics** — dusty's `diagnostics.jsonl` is a single multi-line JSON object, not one-per-line JSONL.
- `validate_base`: `goal_region` is present-as-`None` for dusty (not absent) → guard `(diag.get("goal_region") or {})`.
- Full 23 finalize → **23 ok, all with dust + sponge preserved**, snapshots 23/23, user-reviewed. The 4 sponge-fixed tasks verified reachable in the bench output.
- **🎉 Phase 1 (base finalization) COMPLETE — all 6 families**: jar 26 · cabinet 35 · clutter 55 · lid 30 · stack 28 · dusty 23 = **197 contiguous, all-ok base tasks** in `maniguard-bench/`. Each: canonical FrankaPanda + pose A + uniform mount + 4 review cameras + idle-step + owned diagnostics schema, every task user-reviewed.

### ManiGuard-Bench rebuild · dusty rounded to 200 — dust filter module + layout fix + dest/food swap surgery (2026-06-14)
- **Goal**: round the 197 base total to a clean **200** by adding 3 more dusty tasks → dusty 23 → 26 (`dusty_transfer/`, 0000-0025 contiguous, all ok, snapshots 26/26). Grand total **jar 26 · cabinet 35 · clutter 55 · lid 30 · stack 28 · dusty 26 = 200**.
- **dust-bottom refine module** (`390b4f9b`, `task_generation/dust_bottom_filter.py`): promoted the one-off `_filter_dust_bottom.py` (untracked in SENTINEL-Lite — the tool that gave the 23 dusty bases their bottom-concentrated dust) into a proper module. Keeps only `z <= z_min + 20mm` dust, drops wall/rim, leaves flat containers (<30mm z-spread) untouched; offline `scene_ep1.json` edit + CLI. Reverified on all 48 6fam-base sources (already flat-filtered) and exercised live on freshly-generated deep containers (e.g. a bucket 20→14 particles, 249mm→6mm spread).
- **left/right layout fix** (`2a65ee07`, `transfer_scene_pipeline.place_objects` + `dusty_transfer_pipeline.make_edge_objects`): the first plan was to regenerate 3 fresh dusty tasks via `DustyTransferPipeline`. Found the generator placed source/dest nose-to-tail in front of the arm on Y-longer surface zones (a 90°-rotated layout vs the bases). Real cause: `place_objects` hardcoded the pair along world-X, while `select_best_table_edge` mounts the robot on the zone's SHORT edge facing the long axis (chosen from the ZONE aspect ratio, not the object pack — an earlier "drop the sponge from the edge-align pack" guess was only a partial red herring). Fix = lay source/dest along the zone's LONGER axis (X-long zones byte-for-byte unchanged, only Y-long corrected) + drop the parked +Y-edge sponge from the edge-align pack. Verified 6/6 left/right on a clean batch across X-long and Y-long zones (measured robot-frame lateral vs forward separation). **These fixes are NOT on the path of the final tasks** — the generator's object-selection + idle-step stability variance still made each fresh task a coin-flip on container quality/feasibility (narrow beaker/can tipping, big black buckets hiding dust), so it was abandoned for hand-picked additions; the fixes harden the generator for future runs.
- **dest+food swap surgery** (one-off, deliberately NOT committed — throwaway `/tmp/dusty_swap_surgery.py`): built the 3 final tasks deterministically from the already-reviewed bases instead. Picked 3 small, similar-footprint dests — bowl (0011) · saucepan (0005) · mixing_bowl (0019), using dust-XY-spread as a size proxy (0.08-0.11 m), avoiding the big stockpot/casserole and the flat plate/tray to prevent post-swap source collision — and **cyclically swapped each dest container + its bottom-filtered dust as one unit** (1→2, 2→3, 3→1), plus swapped the potato for a distinct gen-drawn food (cherry / half_blackberry / garlic_clove). Pure JSON edits: `objects_info.init_info` + `object_registry` (drop host's dest/food, add donor's dest at the host slot + new food on the host source) + the `system_registry.dust` group + diagnostics (`spawn_specs`, templated `prompt`, `goal_conditions` name refs, LTL `over` `<food>_*` pattern), then re-ran the proven `replay_empty` + `run_finalize_base` flow (rebuilds the dest by category, repoints the dust uuid to it, settles, re-renders 4 cams, validates). **KEY gotcha**: the saved dust `positions` are LOCAL offsets relative to the dest's `base_link` (centroid ~0, bottom-concentrated), NOT world coords — they must NOT be translated when transplanted; replay_empty re-attaches them to whatever dest it repoints to. All 3 → status ok (`obj_disp` 0.0, validate 15/15); dust (12/18/18) + food presence/position structurally re-confirmed in the finalized `scene_ep1.json` (don't trust the render). Appended as `task_0023-0025`, manifest + snapshots regenerated → 26/26.

### ManiGuard-Bench rebuild · Phase 2a target (appearance) perturbation COMPLETE (2026-06-15)
- **Goal**: the first OOD level. Each of the 200 base tasks gets a `target/` variant whose family-defined target object is forced to a **vivid, instantly-distinct color**, everything else identical to `base`. **200/200 target variants, all ok** (clutter 55 · cabinet 35 · lid 30 · stack 28 · jar 26 · dusty 26).
- **Uniform task-instance interface** (`8f44f927` `bench_builder/perturbation.py` + `75ef5dd1` `eval/benchmark.py`): every bench dir — `base` or any perturbation level — is just a *task instance* loaded the SAME way: build env (scene cfg is already adaptive: empty `Scene` vs room) → reset → **`apply_perturbation(env, diag)`**, the single post-load hook, data-driven by a uniform `diagnostics["perturbation"]` block (`kind` = base/target/language/location/env). For `target` it re-applies the recolor; for the others (changes baked into scene/diag) it's a no-op. The sim/eval NEVER branches on base-vs-perturbation — the user just points at the instance dir they want. Wired into `benchmark.py` (loads the instance's `diagnostics.jsonl`, applies post-reset).
- **Recolor mechanism — the journey to a UNIVERSAL recolor**: `diffuse_tint` alone is **multiplicative** (`final = tint × albedo`) → it visibly recolors light/mid objects but **cannot brighten a dark albedo** (a near-black tray × any tint stays near-black); `diffuse_color_constant` / texture-clear also failed to override textured MDL assets. The robust fix is the engine's OWN texture-change formula (used for Frozen/Cooked/Burnt): **`final_albedo = diffuse_tint × (orig_albedo + albedo_add)`**. Set `albedo_add = 1 − luminance(orig)` to lift a dark/textured albedo up to ~1 (washing out the original), THEN tint to the vivid color → ANY object becomes that color (verified on a near-black tray → vivid cyan). The additive lift is the key the plain tint lacked. `albedo_add` IS a settable MaterialPrim input (`material_prim.py`).
- **Palette + selection**: 6 vivid high-saturation candidates spread around the hue wheel (red/orange/yellow/green/cyan/magenta — no dark/muted entries), **cycled by task index** (`palette[task_index % 6]`) so a family's variants rotate through distinct colors, + a far-guard that skips a candidate sitting near the original. Deterministic. The recolor spec stored in diag = `{object, category, diffuse_tint, albedo_add, orig_color}` (NOTE: neither tint nor albedo_add serializes into `scene_ep1.json`, so the variant's snapshot is a byte-copy of `base` and the recolor is re-applied at every load from the spec — that's exactly what `apply_perturbation` does).
- **Generator** (`perturb_target.py`, subprocess-per-task like `run_finalize_base`): load base → resolve target → pick color + albedo_add → apply → `render_views` the recolored env (4 mp4s) → copy base `scene_ep1.json` + write the recolor spec → `target_manifest.jsonl` (+ reuse `validate_base_task` for structural QC). GPU-dynamics gate mirrored for fluid tasks.
- **lid two-provenance fix** (`81fb1488`): the full batch first failed 16/30 lid tasks with "no target category" — lid carries TWO role conventions for the capped container (`container` in older tasks 0000-0010, `target` in newer 0011+). `TARGET_ROLE` now lists candidate role(s) per family and `resolve_target_category` takes the first that resolves → all 30 build. **Per-family recolor target**: clutter/jar/cabinet/stack = `target`, lid = `container`|`target`, dusty = `source`.
- **stack-same bottom-target fix** (`3b5bcc7f`): user review caught the recolor hitting the TOP of the stack on stack-retrieve "same" tasks (0013-0020, 0027). Cause: a "same" stack is identical objects, so the recolor target (the BOTTOM, retrieved) and the stack share a category → `find_object_by_category` returned the first match (top). New **`resolve_target_object(env, diag, family)`**: resolve the category, then when ≥2 objects share it, disambiguate to the actual manipuland — the goal's `grasping robot <ref>` object (the bottom, e.g. `bowl_45`), else the **lowest-z** object (stack bottom). Unique-category cases (flat stacks, all other families) unchanged. All 15 stack-same target variants regenerated → recolored == bottom (verified by z). **Gotcha for later levels**: any per-task object resolution must handle same-category duplicates (stack-same), not just category match.
- **Phase 2a doc finalization** (`4a119334`): bench `README.md` rewritten from "base only" to the full base + perturbation contract (directory layout, 4-axis perturbation table, uniform task-instance load interface, `target` recolor mechanism + JSON spec example); spec doc §4b gains a same-category-dedup note + Phase 2a stack-same checkbox; `pick_tint` docstring corrected to the current 6-color palette (was stale blue/brown). README/spec are gitignored/Obsidian-external; the docstring is the only tracked change.

### ManiGuard-Bench rebuild · Phase 2b language (prompt rephrase) perturbation COMPLETE (2026-06-15)
- **Goal**: the lightest OOD level. Each base task gets a `language/` variant where ONLY the instruction text changes — scene/objects/poses/goal/physics byte-identical to `base`. **200/200 language variants, all ok** (clutter 55 · cabinet 35 · lid 30 · stack 28 · jar 26 · dusty 26).
- **Why there is no sim**: eval reads the task prompt straight from `diagnostics["prompt"]` (`scene_discovery.py:111`), and the uniform load hook `apply_perturbation` is already a no-op for `kind == "language"`. So rewriting that one string IS the complete change — zero simulation, zero runtime change. `language/scene_ep1.json` + the 4 `rollout_*.mp4` are byte-copies of `base/` (`filecmp` verified 0 diffs); only `diagnostics.jsonl` differs (rewritten `prompt` + a `perturbation{kind:language, language:{base_prompt, rephrased}}` block).
- **Rephrase = deterministic, family-aware ordered phrase substitution** on the finalized base prompt STRING (new module `bench_builder/perturb_language.py`, pure Python, no subprocess / no OmniGibson). NOT the legacy synset reconstruction: the bench base prompts have stable per-family shapes, so `FAMILY_REPHRASE_RULES` holds a short ordered list of `(find → replace)` phrase rules per family that rewrite verbs/connectives while object names, sides, and "or anything else" clauses pass through untouched → semantics preserved by construction, exactly one rephrase per base. E.g. `Pick up the X`→`Lift the X`, `move it into the green goal sphere`→`place it in the green goal sphere`, `Place the lid on`→`Put the lid on`, `Close the lid of … carry the closed jar into`→`Shut the lid of … move the closed jar into`, `Place X inside the open drawer … and close the drawer. Do not knock over Y`→`Put X into the open drawer … and shut the drawer. Do not tip over Y`, `Wipe the dusty X clean with the sponge … transfer`→`Clean the dusty X using the sponge … move`.
- **Banned-word guard is RELATIVE to base**: the legacy global banned list (`knock/into/without/careful/…`) conflicts with the base prompts themselves — cabinet base legitimately says "knock over", every family says "into the goal sphere"; these are canonical task words present in ID too, not leaks. So the guard flags only a hint word the rephrase ADDS that base didn't have (`banned(rephrase) − banned(base)` must be empty). Plus: rephrase ≠ base, no object word dropped (`into` whitelisted — it's an intentionally-swapped connective, not an object), non-empty.
- **Review artifact**: per-family `language_prompts.md` listing every `base → language` pair with the **changed words bolded** (word-level `difflib` diff via `mark_diff`) so the reviewer's eye lands on the change; **no snapshot** since pixels don't change. Driver flags: `--family/--tasks/--skip-existing/--dry-run` (dry-run previews rephrases without writing; `--skip-existing` rebuilds only manifest+md from existing variants, no file re-copy). Full cross-check across all 200: 0 identical-to-base, 0 rule-missed (every base shape matched its rules), 0 structure issues, prompt==rephrase everywhere, scene byte-identical to base. User reviewed all 6 `language_prompts.md`.

### ManiGuard-Bench rebuild · Phase 2c location (object position) perturbation COMPLETE (2026-06-16)
- **Goal**: the position OOD level (`ae94b5c5`). Each base task gets a `location/` variant where the task objects are shifted in-plane on the support surface while the **robot base + init pose A stay fixed** — the mis-alignment between the moved objects and the stationary arm is the OOD signal. **200/200 location variants, all ok**. Positions DO serialize into `scene_ep1.json`, so the moved+settled scene IS the snapshot and `apply_perturbation` is a no-op for `kind=="location"`. Ran in two tmux sessions (`local_1`/`local_2`, 3 families each, jobs 1 → 2 concurrent workers on one GPU, ~50 min).
- **`location_geom.py`** (pure geometry, unit-tested offline): per-family move units (spec §4d — clutter=each object independent, cabinet=target+obstacle independent along the drawer axis, jar/lid/stack=whole pack rigid, dusty=(source+food)+(dest)) + deterministic displacement (random-plane / along-slide-dir / xy-independent) + table-edge clamp (footprint within `surface_info.bounds_xy` − margin) + mode-aware clamp-retry + perpendicular fallback rule. `surface_info.bounds_xy` exists per family already (no must-write needed).
- **`perturb_location.py`** (subprocess-per-task worker + driver, mirrors `perturb_target`). Three hard-won fixes from a deep debug (all root-caused, not patched):
  - **teleport-impulse heal + settle-to-equilibrium**: `set_position_orientation` teleports a body and breaks the delicate resting contact of a tall/narrow object (a bottle), so a free settle from the raw teleport is CHAOTIC — it can tip fully over from the impulse even though the placed pose is a valid rest (PROVEN: an UNMOVED object set to its own pose also tips; a long-horizon run showed the held-then-freed bottle frozen at its rest over 250 steps). Fix: hold the moved objects still (zeroing velocity each step) for the first `SETTLE_HOLD=20` of `SETTLE_STEPS=80` to heal the contact, then free 60 steps so they converge to their true equilibrium BEFORE the snapshot is saved (an under-settled save had captured a 17° transient whose real rest was 9°). This is the cabinet_0026 "infeasible" task — it was never infeasible, just tipped by the teleport artifact.
  - **pose-A re-assert before save, no step in between**: the raw joint controller sags ~0.03 rad over the settle (> POSE_TOL 0.01), so re-set pose A and `env.scene.save()` immediately (a stepped save drifts and fails validate's `pose_A`). This is the stack_0022 "fail".
  - **FRESH gate + LTL recomputed on the MOVED scene**: a location variant is an independent task at the SAME level as base, so it must pass the SAME in-sim checks computed on the NEW positions (reachability, init not LTL-doomed, LTL not violated over settle, nothing fell), not inherit base's stale verdicts. (target/language inherit safely — they don't move objects; location must recompute. The user's question surfaced this gap.) Mirrors `finalize_base`'s gate+monitor sequence.
- **Retry**: two-phase — primary (spec direction/magnitude, `--max-attempts 5`) then a PERPENDICULAR fallback (cabinet's cross-slide axis at 0.5-0.8× bbox; a fresh random sweep for omnidirectional families) when primary fully fails. All failures reported with per-attempt reasons (gate/reach/ltl/fallen). cabinet_0007 (target at reach limit, slide axis blocked) recovered via fallback.
- **Surgical constraints for marginal tasks** (opt-in CLI, default off; the gate's [0.20,1.10] reach band is too permissive): `--reach-window LO,HI` (target must land in a comfortable annulus — fixed jar_0011/0015/0023 + lid_0006 that landed too far / on the arm base), cabinet keep-out (a moved object may not NEWLY overlap the cabinet AABB — the target's pre-existing drawer overlap is exempt; fixed cab_0021 behind + cab_0026), and `--force-move DX,DY` (deterministic rigid shift in a user-dictated safe direction, reach band NOT enforced since the user vetted the spot — hard-tuned cab_0007 image-right [0.13,0] out from under the drawer + cab_0028 depth-toward-arm [0,0.13] out of the drawer). 7 marginal tasks reviewed + fixed task-by-task with visual verification.
- **Audit before commit** (offline, per-task per-level): 200×4 levels (base/target/language/location) all complete + correct kind; 24 manifests complete + all ok (800 rows); target/language `scene_ep1.json` byte-identical to base, location differs; language prompts all differ; target recolor present + stack-same recolor==bottom; 8 surgical-fix tasks correct provenance. **One issue found + fixed**: stack `target_manifest` (and cabinet/jar/lid `location_manifest`) had been truncated by `--tasks` re-runs (driver writes the manifest in "w" mode for only the tasks it ran) → all rebuilt complete from the on-disk task dirs. Per-family `snapshots/` regenerated in-place as base|location opposite-view concat (200, 1:1 with tasks). **Bench now finalized for base + target + language + location; only Phase 2d `env` remains.**

### ManiGuard-Bench rebuild · Phase 2d env (surrounding-room) perturbation COMPLETE (2026-06-17)
- **Goal**: the last + heaviest OOD level (`7aea14b4`). Each base task — a clean empty `Scene` (table + objects + edge-mounted arm + marker) — is rigidly injected into a real BEHAVIOR room anchored on its support table, so the policy sees the IDENTICAL manipulation geometry against a totally different visual background (walls, furniture, clutter). **200/200 env variants, all ok** (clutter 55 · cabinet 35 · lid 30 · stack 28 · jar 26 · dusty 26; 14 on a substitute table). `apply_perturbation` is a no-op for `kind=="env"` — the room is baked into the merged `scene_ep1.json`.
- **Mechanism = offline scene-JSON MERGE** (`env_geom.py` + `perturb_env.py`, subprocess-per-task — OmniGibson can't rebuild the env in-process, external cameras go stale on a 2nd build): merged scene_file = the room's `<scene>_best.json` + base objects transformed by `T = T_room_table · T_base_table⁻¹` (drop the base table; the room holds the same model), top-level `init_info.class_name=InteractiveTraversableScene` → `build_env_config` instantiates room + injected objects in one load → mount arm on the room table → Stage-1 declutter → settle → fresh gate+LTL+fall/sink/arm verdict on the FINAL saved scene → render 4 cams. `data/table_scene_db.json` (offline `build_table_scene_db`) indexes which rooms hold each surface model.
- **Substitution fallback** (`find_substitute_candidates` + `substitute_transform`): when a base table's own room is missing (8 tables in no scene) or infeasible, re-anchor the whole base pack onto a DIFFERENT adequately-sized table (sorted dims + area ≥ pack + margin) in another room. 14 substituted (jar 9 incl. the 8 uncovered, cabinet 3, clutter 1, lid 1), all ok.
- **THE deep-debug — spawn-collision with room on-surface obstacles** (surfaced as 6 failing liquid tasks; actually a FIDELITY bug across the whole axis): base is a CLEAN table, but the room's anchor table ALREADY holds objects (laptop, potted plant, cash register). The merge injects base objects at their base XY → they SPAWN overlapping those → `env.reset()`+the first `og.sim.step()` PhysX violently EJECTS them (a goblet spawned floating +80 mm, tilted 30°) BEFORE the live `clear_on_surface_obstacles` removes the obstacle. Flat objects self-right; tall fragiles topple (`all_fragiles_upright` false → LTL doomed → infeasible: clutter 0029/0049/0051); a bulky fixture (cash register 17 cm from the can) deep-penetrates → **NaN quaternion → load crash** (lid 0027/0029's "deterministic NaN"). PROVEN deterministic (3/3 identical runs), transform math identity-correct (the JSON pose is upright), tilt appears only at sim-load + correlates with proximity to a removed on-table obstacle; `location` (no room obstacles) settles the SAME objects fine. **Fix** = drop the room's on-surface obstacles within the anchor footprint OFFLINE in `build_merged_scene_info` (`_clear_room_spawn_obstacles` + `env_geom.surface_world_box`) BEFORE the scene is built → injected objects spawn onto a clean surface (same final scene; the live pass stays as a backstop). 117/194 prior "ok" tasks had on-table obstacles → layouts may have drifted → **full-axis regen** (all 200, no reuse, 2 tmux sessions ~87 min) → 200/200 ok.
- **Sink false-positive fix**: the sink check compared AABB-bottom to the surface AABB TOP, but a checkout-counter's raised back panel puts its AABB top ~23 cm above the real surface → a can resting perfectly on the counter read "sunk 235 mm". Now measured SPAWN-relative (`spawn_z` pre-settle; sunk = settled > `SINK_TOL` below spawn), surface-shape-independent — still catches real settle-sinks.
- **`finalize_base._build_active_objects(objects=…)`**: env restricts LTL active-object resolution to injected objects + the anchor table, so a category pattern (`desk_*`) binds the anchor and not the room's OTHER same-category furniture (a real room has several desks → false support-proposition violation otherwise).
- **driver `--launch-gap`**: optional seconds before each worker launch (headroom for 2 concurrent single-job tmux sessions on one GPU). **Verification**: all 200 complete (scene+diag+4 mp4); 6 originally-failing + 14 substitutes + auto-flagged borderline (high obj_disp / heavy declutter) all visually reviewed; per-family `<fam>_env_vs_base.png` presentation grids saved under `snapshots_env/`. **Bench COMPLETE: base + 4 perturbation axes, 200×5 = 1000 instances.**
- **Committed-artifact hygiene**: the db's `_meta.scenes_root` + the scenes-root resolver were sanitized to a dataset-relative tail (`behavior-1k-assets/scenes`) — no absolute local path / personal dir layout (`SENTINEL-Lite-data` / `ManiGuard-data` fallbacks dropped) in the pushed code; resolution now via `$OMNIGIBSON_DATA_PATH` + repo-local `datasets/`.

## Demo data generation (cuRobo auto-collection for SFT)

NEW phase after the bench finished: auto-collect success+safe demo trajectories on
the 200 base tasks for SFT, **joint-native** via the env's cuRobo. A 正本清源 REFACTOR
(like the bench): the old pnp/cuRobo code stays as pure reference; good parts are
replicated clean into the new `maniguard/data/datagen/` 3-layer module (primitives /
families / driver), bad parts rewritten. Tracking doc = Obsidian
`ManiGuard 6fam_Data_Collection_TODO_Checklist.md`.

### Step 0 — scaffold + schema (2026-06-18, `87f8da9d`)
- New `maniguard/data/datagen/` package (3-layer scaffold). `data_format.py` = schema
  single source: 5 image streams (4 bench third-person + injected wrist), `state(8)=[arm_q(7),gripper]`,
  `actions(8)`=(b) next-achieved (DROID-style, DEFAULT) + `actions_commanded(8)`=(a) cuRobo
  command (extra), `EXTERNAL_CAM_CHOICES`, MimicGen sidecar layout, `lerobot_features()` (LeRobot v2.1).
  Extended `openpi_sft/data_configs.py` `external_cam` whitelist → {opposite,left,right,left_shoulder}.
- cuRobo settled at the StanfordVL fork `cbaf7d32` **rebuilt for torch 2.6.0** (ABI-correct,
  OmniGibson-compatible, USD-kinematics, plans OK). Bumping the fork to its HEAD `d083916`
  needs a newer OmniGibson (`KeyError: 'eyes'`) → not this phase.

### Step 1 · P1 — scene_from_task_dir + task_io (Layer-1, 2026-06-18, `e15edb2b`)
- Replicated `_build_env`'s scene-build half clean so datagen is self-contained (no import of
  the old curobo reference tree). `primitives/task_io.py` parses a base-task dump
  (load_diagnostics_row / load_scene_info / identify_task_objects / build_object_cfg, lifted from
  the reference `replay_empty_from_dataset`) — **fixes the stale `from tools.replay_empty_from_dataset`
  import** (runtime ImportError), drops the dead `_SCHEME_B_PATTERN`. `primitives/scene.py` =
  `init_omnigibson()` + `scene_from_task_dir() -> SceneBundle` (fixed support surface + spawn objects
  + Franka at dump pose + goal marker + settle); cameras deferred to P8 via `external_sensors` /
  `pre_build_hooks` seams. Dropped the family-specific dusty sponge hack (→ L2) + `no_distractors`
  (YAGNI). GPU dynamics OFF (matches pnp; project_gpu_physx_rl_not_faster). Env infra
  (`build_env_config` / `camera_setup` / `goal_region`) imported from shared `maniguard.envs|utils`.
- **Verified** on a real dump (headless GPU): 5/5 task objects present, robot loaded, 15 steps
  stable, fixed surface stays at its dumped z. The post-PASS teardown segfault is the known Isaac
  Kit shutdown race (data-safe; the L3 driver will treat data-written + teardown-segfault as success).

### Step 1 · P8 — cameras: 4 recorded-pose third-person + injected wrist (Layer-1, 2026-06-18, `d66709cc`)
- `primitives/cameras.py` fills the two `scene_from_task_dir` seams: `external_camera_configs()` (the
  bench third-person VisionSensors → `external_sensors=`) and `install_wrist_camera()` (the FrankaPanda
  `_load_sensors` monkeypatch injecting the wrist Camera under `panda_hand` at +0.05 Z so it frames the
  grasp zone and rides the eef → `pre_build_hooks=`), plus `find_wrist_sensor()` and
  `place_and_resize_cameras(env, robot, og, diag)`. Wrist patch + lookup replicated clean from the
  reference `_sft_recorder` (no import of that tree).
- **Key decision — read, don't recompute**: `place_and_resize_cameras` positions the 4 third-person
  cams from the poses RECORDED in `diagnostics["cameras"]` (via `position_diagnostics_cameras`), not a
  robot-frame recompute, so the datagen views are byte-for-byte consistent with the bench render + eval
  (the bench start state IS the baseline — nothing the user finalized is modified). Then it forces 256²
  on every VisionSensor + rebuilds the obs space.
- **Locked principle + debug lesson**: datagen starts from + holds the task's recorded init pose
  (`reset_joint_pos` from the snapshot) and moves the arm ONLY via curobo-planned joint trajectories.
  `env.step(np.zeros)` is NOT a no-op under the `joint_position_impedance` controller — it's an absolute
  command to the joint-zero pose (a contorted upright arm). Observe/render with `og.sim.render()`, never
  a stray action. (Cost a debug cycle: an early smoke fed a zero action + used an old `pipeline_runs/*`
  dump that bakes the pre-standardization `FrankaMounted` robot.)
- **Verified** render-only on `maniguard-bench/clutter_pickup/task_0000/base` (the canonical base-task
  input = `<bench>/<family>/task_NNNN/base/`, FrankaPanda+longfinger): 4 recorded cams positioned, wrist
  injected at build, 5 streams at 256², and the cam_opposite render matches the task's base
  `rollout_opposite_side_front` video frame (arm init pose + objects + goal sphere identical).

### Step 1 · P9 — Recorder: joint-native LeRobot v2.1 + MimicGen sidecar (Layer-1, 2026-06-18, `bca1d610`)
- `primitives/record.py`: `make_dataset()` (datagen LeRobot v2.1 schema = `data_format.lerobot_features`,
  reusing the bench `create_or_open_dataset` MP4-passthrough infra) + `Recorder`. Per `env.step` the
  recorder captures the 5 image streams (→ target MP4s), the ACHIEVED joint state, the cuRobo COMMANDED
  joint target, the gripper command, and a serialized `og.sim.dump_state` (the MimicGen replay hook).
  `finalize()` writes one episode — `state(8)=[arm_q,gripper]`, `actions(8)`=(b) next-achieved
  `[arm_q[t+1],gripper_cmd]`, `actions_commanded(8)`=(a) `[arm_q_cmd,gripper_cmd]` — + a per-episode
  MimicGen sidecar HDF5 (sim states + gripper action); aborts (drops MP4s) on failure.
- **The recorder only READS achieved state** — the arm is moved solely by the caller's curobo trajectory
  (`record_step(arm_q_cmd, gripper_cmd)`); joint-native (no eef-8d / sim-state joint reverse-engineering,
  which was the old `_sft_recorder`'s constraint). LeRobot 0.3.4 takes `task` as a positional arg to
  `add_frame(frame, task)` (the old writer put it in the frame dict → breaks on 0.3.4).
- **Verified** locally on a real base task: 5 video keys + state/actions/actions_commanded (8 each), 1
  episode/8 frames, sidecar with sim states + gripper action, and `actions`(b) vs `actions_commanded`(a)
  genuinely distinct on the ramped joint (Δ≈0.10, the impedance tracking lag).
- **Smoke policy**: the per-primitive `_smoke_*.py` harnesses are kept local-only (untracked, `00348ad2`
  untracked the P1/P8 ones); they remain on disk for re-running as later primitives build on them.

### Step 1 · P2 — solve_segment: cuRobo segment + salvage (Layer-1, 2026-06-18, `1ac639e9`)
- `primitives/curobo_seg.py`: `solve_segment(motion_gen, robot, eef_goal_pos, eef_goal_quat, initial_joint_pos,
  *, timeout, attach_obj, motion_constraint, ...)` → `SegmentResult{arm_traj(T,7), final_full, salvaged,
  pos/rot_err, n_waypoints}`. Replicated clean from `_solve_one_segment` (the family-agnostic solver the old
  code already cross-imported). KEEP the salvage pass (recover trajectories trajopt flags `success=False` but
  which converged within 5mm/0.03rad); DROP `eef_traj` (OSC-replay leftover — joint-native executes joints
  directly), the `[PnP]` labels, and the always-on failure probe (now opt-in `diagnose_on_fail`). ADD
  `attach_obj` + `motion_constraint` seams. Pure solver — the obstacle world, constraint levers, and
  gripper-collision toggles belong to the caller (P7); the old `_plan_transport` patch pile is NOT inherited.
- **Family-agnostic by design**: the old `pick_and_place_from_dataset` SCRIPT is clutter-specific (the lid has
  its own `pick_up_lid` / `lid_transport`; stack/dusty/jar/cabinet have NO old curobo at all), but the SOLVER
  inside it is generic — extracting it to L1 is the whole point of the refactor.
- **Verified** on a real base task: reachable 8cm lift → 31-waypoint (T,7) trajectory; far-unreachable goal →
  graceful `None`.

### Step 1 · P9 revised — raw-first recorder (2026-06-18, `04ad92e7`)
- Re-architected the recorder from direct-to-LeRobot → a **reviewable RAW form first** (user wants to eyeball
  the curobo trajectories before any SFT conversion). `Recorder.attach(env, robot, out_dir, prompt)` →
  `finalize()` writes, per trajectory: 5 MP4 streams (`image_{opposite,left,right,left_shoulder}.mp4` +
  `wrist_image.mp4`) + `traj.hdf5` (state(8), actions(8) [b], actions_commanded(8) [a], `states` [sim dumps],
  `datagen_info/gripper_action`) + `meta.json`; aborts (rmtree) on failure. MP4s match the bench rollout spec
  **byte-for-byte** — PyAV `h264`/`yuv420p` at the camera's native 256² @ 30 fps (replicated from
  `task_generation/utils/video`). **LeRobot v2.1 conversion is now a SEPARATE downstream step** (MP4
  passthrough, no re-encode); `make_dataset` + the LeRobot writer are dropped from the recorder. The
  joint-native schema is unchanged — only the output target.
- **Verified** on a real base task: 5 raw MP4s all `h264/256x256/yuv420p/30fps` (== bench), `traj.hdf5` with
  state/actions/actions_commanded (8 each) + sim states, `meta.json`, and actions(b) vs actions_commanded(a)
  distinct. First real curobo-driven raw demo comes after P6 (execution moves the arm).

### Step 1 · P7 — CuroboWorld: motion_gen + obstacle world + constraint levers (Layer-1, 2026-06-18, `2e1c72d6`)
- `primitives/obstacles.py`: `CuroboWorld(env, robot)` formalizes the cuRobo setup the P2 smoke inlined —
  `_install_mimic_patch` (fill the missing Franka finger joint at 0.04 open, replicated clean from
  `collector._patch_curobo_mimic_lookup`) + `StarterSemanticActionPrimitives._motion_generator`, plus
  `update_obstacles(ignore_objects=...)` and `LINEAR_SERVO = [0.1x5, 0.0]` (partial-pose hold: hold
  orientation + perpendicular position, free along the approach axis → linear servo) +
  `gripper_collision_disabled()`.
- **Two cuRobo-fork issues found + handled (systematic debug)**: (1) `plan_batch` crashes
  (`TypeError: 'int' object is not iterable` at `[False for _ in solve_state.batch_size]`,
  motion_gen.py:3160) when a hold_partial_pose query is invalid (`valid_query=False` — start/goal
  orientation mismatch or motion off the free axis). The old code tiptoed around it via careful seeding
  (collector.py comment names the same failure). FIX: `solve_segment` now guards `motion_constraint`
  calls — an invalid constrained query returns `None` (a plan failure), not a crash, so unattended
  collection survives. (2) `toggle_link_collision` is ABSENT on the cbaf7d32 build (older-curobo API),
  so `gripper_collision_disabled()` is a warn-once NO-OP. Verified the real collision levers on this
  build = `update_obstacles(ignore_objects=...)` (drop the target during a grasp approach) +
  `attach_obj` (held object). **P3 must use these, not the gripper-link toggle.**
- **Verified** on a real base task: CuroboWorld builds + plans (31 waypoints); an invalid constrained
  query returns `None` instead of crashing.

### Step 1 · P6 — execute_trajectory: JointController replay + gripper (Layer-1, 2026-06-18, `beb222af`)
- **Reordered P6 before P3** (dependency: a grasp can only be VALIDATED by executing the approach +
  closing the gripper + checking the hold — that execution IS P6). `primitives/execute.py`:
  `execute_trajectory(env, robot, arm_traj, *, gripper_cmd, recorder, steps_per_waypoint)` feeds each
  (T,7) waypoint into the JointController arm slot (raw radians — the impedance preset's
  `command_input_limits=None`) with a binary gripper command, and records every stepped frame
  (`record_step(arm_q_cmd, gripper_cmd)` — the only place the arm is commanded). `actuate_gripper` holds
  the arm + opens/closes the gripper for N steps (settle/close). `build_joint_action` replicated clean
  from `collector._build_action`. `OPEN=+1 / CLOSE=-1`.
- **First real cuRobo-driven RAW demo end-to-end** (scene → cameras → CuroboWorld → plan → execute → raw
  record): planned 31 waypoints, executed 62 steps (×2 per waypoint), **eef rose 11.4 cm** (commanded
  12), 62 frames into 5 MP4s + `traj.hdf5` (state (62,8)). The 5-pane montage (arm visibly lifting,
  objects + goal sphere static) was shown to the user — the reviewable raw form before LeRobot.

### Step 1 · grasp annotation — per-instance human grasp DB + review/QC tooling (2026-06-19, `6b6b3a51`..`1106720d`)
- **Grasp-source pivot**: P3's grasp source is now **per-instance human-annotated grasp frames**
  (RoboTwin-style), not GraspGen. Why: GraspGen ignores robot reachability/clutter (teacup top-40 grasps
  0/40 reachable for the table Franka; teacup body 7.3 cm > 7.0 cm gripper) and can't express semantic
  grasps (rim/handle/stem). GraspGen demoted to an optional candidate.
- **Schema (locked)** `outputs/grasp_annotation/grasp_annotations.json` (gitignored): each grasp = an
  **eef_link TARGET pose in the object-local frame** `(position, quat_xyzw)`; runtime
  `T_eef_world = object_world_pose @ pose` → straight to cuRobo IK, zero conversion. Verified exact
  (~1e-16) and object-relative (follows the object's pose in any task instance).
- **Gripper eef-local frame (sim-measured, corrects the old probe)**: fingertips/**approach = eef +Z**,
  **closing = eef -Y** (the old probe mislabelled the finger-link origins as the tips).
- `6b6b3a51` `primitives/grasp_obb.py` (dependency): OBB grasp sampler + the OG mesh/transform helpers
  (`mesh_from_og_object`, `_to_np`, `_pose_to_mat`) the annotation pipeline reuses.
- `b32e8a48` `annotation/{extract_meshes,annotate_tool}.py` — **Phase A** enumerates the distinct grasp
  targets per family (diagnostics goal + scene_ep1 model, pure JSON) and in one OG session exports an
  object-local GLB + bbox + upright orientation per target (132 across 6 families) + the longfinger
  gripper mesh in eef-frame. **Phase B** = viser web GUI (`:8080`): object shown upright + world axes +
  real gripper, Guided (click + approach preset + yaw/depth) and Free (6-DoF gizmo) modes, incremental
  save/resume, `--family` filter.
- `1106720d` `annotation/{mesh_review,validate_grasps,fix_approach_tags}.py` — annotation QC.
  **`mesh_review`** (fast, NO sim): per object, object-upright + gripper at each grasp from 3 viewpoints
  (oblique/side/top) with a fixed-inset world XYZ triad, the true approach DERIVED from the pose — seconds
  per family, the DEFAULT per-family check. **`validate_grasps`** (heavy, sim): teleports the base so
  eef_link lands exactly on each grasp (0.0 mm), renders the 4 bench cams + a 3D closeup, emits per-grasp
  montages + a per-object `_summary.png` — run later for real-scene context. **`fix_approach_tags`**
  rewrites each grasp's `approach_hint` from the actual pose (confident → corrected, ambiguous band →
  reported for human review).
- **Workflow per family**: annotate (`annotate_tool --family X`) → `fix_approach_tags --family X --apply`
  → `mesh_review --family X` (fast check) → much later `validate_grasps` (sim summary). Loader/camera
  gotchas (sim path): wrist camera / new sim cameras → Vulkan DEVICE_LOST; mid-run camera reposition
  doesn't re-render in headless (use matplotlib closeups); `og.clear()` multi-task breaks the cameras →
  one object per process. Always `python -u`; exit 139 at `og.sim.stop()` is benign.
- **Status**: clutter 5/39 targets annotated + closed-loop validated (25 grasps at 0.0 mm; goblet
  #1/#2/#3 auto-corrected top_down→side). Next: annotate the remaining 34 clutter targets.

### Step 2 · executor — generic engine + clutter template, first real dataset (2026-06-20, `1c0f88c3`..)
- **Generic executor ⟂ family skeleton**, decoupled by one interface (`MotionSegment` +
  `FamilySkeleton`): `executor/` (contracts/engine/gate/geometry/grasp_select/variation) plans /
  executes / gates / records / scales ANY family; `families/clutter.py` only declares the motion
  segments. Adding a family = new task semantics, everything else reused.
- **clutter boxy skeleton**: `pre_grasp → descend(ignore_clutter, world-collision-off) → lift(to
  dynamic ≥3cm clearance) → transport(over goal) → to_goal(aim held-object centre at goal-sphere
  centre)`. Grasps from the annotation DB (`grasp_db.py`, object-local @ live pose).
- **Gate = HARD filter** (§0.1 revised): real-time LTL (`TaskLTLMonitor`) + success (held-in-goal)
  every executed step; a violation voids the demo. Never collect "success but not safe".
- **5 diversity dims, one master seed/variant**: grasp · cuRobo trajopt seed (`torch.manual_seed`)
  · lift height (1.0–1.5× clearance) · standoff · above_xy. `--target N` loops draws until N
  successes; pristine scene `load_state` reset before every variant; gap-free `traj_NNN`.
- **3 integration fixes**: LINEAR_SERVO→unconstrained fallback (this cuRobo build rejects the
  partial-pose query); grasp descent collision-off; settle + over-lift for PD undershoot.
- **Tooling**: `sweep.py` (one subprocess per task, sharded/parallel, save-isolated), `reader.py`
  (LeRobot conversion entry — videos/numbers separate, conversion needs no sim/replay), `review.py`
  (per-task montage MP4). Output `outputs/datagen/<dataset>/<bench_family>/<task>/traj_NNN/` =
  5 MP4 + traj.hdf5 + meta.json; bench-consistent naming + 256²/30fps/h264.
- **First real dataset `v1`**: clutter_pickup task_0000–0004 × 50 = **250 demos, 1.77 h (~25s/demo)**,
  integrity 0 issues (frame-aligned, gap-free, all success+safe), review quality confirmed. Full
  pipeline doc = `docs/datagen/pipeline.md`. GraspGen path (`primitives/grasp.py`,`graspgen.py`)
  demoted + kept local-only. Next: cabinet family.

### Step 3 · cabinet family — first goal_conditions family, task_0000 success+safe (2026-06-21, `64e566f2`..)
- **Goal** `inside(target,drawer) & closed(cabinet)`; **5-phase skeleton** (`families/cabinet.py`):
  close drawer → relocate in-path blockers → open to max → place target in drawer → close drawer.
  Only the family declares segments; the generic executor plans/executes/gates/records it, reusing
  the SAME `eval.goal_checker` success + `safety_monitor` LTL as teleop/eval.
- **Two motion building blocks** (generic, backward-compatible — clutter unchanged): `Mode.FREE` =
  cuRobo collision-aware transit; `Mode.SERVO` = pure straight-line IK for every deterministic
  straight contact (drawer push/pull, the place up-over-down).
- **Key debug — `Mode.LINEAR` is broken on this cuRobo fork**: its partial-pose query always fails
  and silently falls back to an unconstrained salvage solve that DRIFTS the eef off the straight
  line (carried object dropped off the table edge; grasp missed). Fix: every straight contact uses
  SERVO. SERVO itself skewed the redundant wrist until `solve_ik` was changed to pick the IK branch
  NEAREST the seed (not cuRobo's first-valid) → continuous, un-skewed straight servos.
- **Relocation heuristic** (`cabinet_geom.blocker_placement`): obstacle → far `-p` table edge first,
  target → near `+p` edge, BOTH pushed flush to the edge to fully clear the opening corridor; the
  target is additionally slid `+opening` off the robot base's straight-ahead line, else the re-grasp
  folds the arm too tight for cuRobo (≈0.31 m straight-ahead was unsolvable; ≈0.42 m diagonal works).
- **Place into the deep drawer**: deep top-down grasp (grips the solid base, not the thin post);
  inverted-U — lift the bottom over the rim (hard-verified), translate to the cavity centre =
  OmniGibson's **fillable meta-link** live xy (ground truth; handle-inclusive AABB estimates bias
  toward the opening), **roll the finger-rail ⊥ the opening** so the long flat housing clears the
  upper drawer's handle on the way down, lower onto the interior floor (no free-fall — a 0.21 m
  object dropped 0.3–0.5 m tumbles).
- **Close approach + termination**: after release, lift the empty gripper straight UP out of the
  cavity then translate over the handle xy at that height, so the cuRobo plan only descends straight
  onto the handle (never scrapes the cabinet door). The final close ENDS at the closing push — the
  success state — so no recording past it (like clutter, the sequence ends at success; the gate runs
  the shared success check there). LTL gated every executed step throughout.
- **Status**: `task_0000` (paper_towel_holder into a tall chest's deep drawer) collects a clean
  success+safe demo end-to-end (`traj.hdf5`). Next: scale across grasps/draws, then task_0001+.
- **Scale-up (diversity + robustness + review)**: two per-demo diversity bands sampled from the
  variant seed — lift-over-rim carry height (0.06–0.12 m) + relocate landing spot along the edge
  (0.18–0.30 m); `task_0000` collected **5/5 success+safe** with distinct draws. Dropped the
  fine-grained stall detector (false-positived on cuRobo ease-in) in favour of the reliable shared
  success+LTL gate as the only judge, plus a generic engine **step-limit backstop** (`max_steps`
  3600 ≈ 2 min). Review montage (`review.py`) made third-person-camera selectable (`--cam`) — still
  one family-agnostic tool; cabinet reviews from `opposite`, clutter unchanged on `left_shoulder`.
- **Stability pass (task_0000, pre-scale)**: place grasp changed from one confident pick (deepest
  top-down, side discarded) to **fair sampling** over the full reachable list (top-down AND side,
  per-draw = a 3rd diversity axis); cuRobo reachability is the only "is it possible" prefilter, the
  rollout + gate is the real judge. Found `move_transit` (relocate's cabinet-avoiding FREE plan)
  ~50% plan_fail — a generic engine **cuRobo plan retry** (`plan_tries=2`) only patched it; the real
  fix was lifting the relocated blocker **higher before the transit** (`LIFT_CLEAR` 0.05 → 0.18 m,
  roomy high lane) → move_transit **50% → 100%** (8/8). Remaining failures are now all in the place
  phase = the intended fair grasp test (bad grasps fail + drop). `review.py` now defaults to no-wrist
  (`--wrist` to add it); annotate tool's jump list is searchable + alphabetical.
- **Stability test task_0001–0005 → pipeline does NOT generalize + infra fixes (2026-06-22)**: annotated the
  10 new objects (+ `fix_approach_tags` = 65 corrections) and swept 0001-0005. Findings: **task_0001
  bottle_of_tequila 0/3** (all target grasps cuRobo-unreachable at spawn — tall bottle out of reach);
  **task_0002 graduated_cylinder 0/3** (7 attempts all fail: place-pick approach knocks things over, place_lift
  below_z/stuck, place_over IK fail). The place logic was tuned to task_0000's solid-base+thin-post shape; the
  new tall/thin objects fail — **NEXT = diagnose per-object via the review videos, don't blindly scale**. Infra
  fixed (all committed): (a) **HANG** — `variants_stream` spun forever (`itertools.count()` + empty reach) when
  --score dropped every grasp, blocking the whole sweep with no timeout → now returns 0/N cleanly; `score_grasps`
  got a 2-try retry. (b) **per-family logs** — sweep logs/summary moved under `<dataset>/<bench_family>/_logs/`
  (shared `_logs/` clobbered across families, task names repeat). (c) sweep **resume-to-N** (re-run tops up the
  deficit, stops at N) + `--max-attempts` + UNDER-TARGET report; `DATAGEN_HANG_WATCHDOG` faulthandler watchdog
  (found the hang). (d) **2-process-on-1-GPU deadlocks** for cabinet's cuRobo-heavy load → single process only.
- **Cabinet placement: whack-a-mole wall → systematic REDESIGN (spec + plan), uncommitted (2026-06-22)**:
  a long debug session on the placement (per-object via review videos) kept shifting the failure with each
  SERVO/heuristic patch (relocate near-side rewrite, footprint-aware blocker halves, adaptive place-grasp
  height filter, grasp-balance centrality, fit_yaw/edge_yaw early-vs-late, toward-robot cavity shift, low-height
  roll, fail0 `place_pre_grasp`/`pick_pre` no-longer-ignore-target, fail1 obstacle-balance grasp + `pick_lift`
  empty-grip verify, `place_gid or`→`is not None`, recorder keeps each failed attempt as `traj_*_failN`). All
  landed in `cabinet.py`/`cabinet_geom.py`/`record.py` but **uncommitted**. Hit the architecture-questioning
  threshold and **stopped patching**. Root cause found: **cuRobo was told to `ignore_objects=(cab,)` in EVERY
  segment**, so the FREE moves plan THROUGH the open drawer and knock it (the "gripper bends into the door");
  VERIFIED OmniGibson cuRobo `update_obstacles` adds every link's collision mesh at its LIVE pose, so the open
  drawer IS modellable — cuRobo can avoid it, it was just told not to. **Redesign = waypoint-anchored**: every
  move is a must-reach-waypoint skeleton — **门-shape pick-and-place with 4 mandatory waypoints**
  (`D_grasp`/`D_up_src`/`D_up_dst`/`D_place`) reached by **pure IK**; **cuRobo only on path-irrelevant free-space
  connectors** (avoiding the live cabinet+drawer); handle push/pull; the fragile `fit_yaw`/`rail_clear`/
  `toward_robot` fold into one **early LOW cuRobo reorient** (avoid cab, upright — done at z≈0.7 not the z≈1
  reach ceiling) + the `D_up_dst` pose; cuRobo connectors get an IK fallback (lift→across→descend) on plan_fail.
  Also flagged: the **12 far-target tasks (0001/0003/0007/0013/0015/0018/0021/0023/0026/0028/0031/0034) are still
  UNFIXED** (target spawned ~1.0-1.1m out of reach) → must be respawned to both-front; and a **diagnostics-only
  target↔obstacle role-swap** fallback for ungraspable targets (verified on task_0004, which is now swapped to
  "place the shaker"). SPEC `docs/superpowers/specs/2026-06-22-cabinet-datagen-waypoint-architecture-design.md`
  + PLAN `docs/superpowers/plans/2026-06-22-cabinet-datagen-waypoint-architecture.md` (7 tasks T1-T7) written +
  approved. **NEXT = execute the plan from T1** (12-task both-front conversion is independent + first); task_0000
  is the regression baseline. See [[project_maniguard_cabinet_family]].
- **Cabinet bench DATA renewed: T1 spawn-fix + T1.5 finalize/perturbation renew (2026-06-23)**: finalized the
  benchmark cabinet family (35 tasks) after the spawn edits — base + 4 perturbations + 5 manifests all consistent,
  all `ok`. NEW tooling: `tools/cabinet_rerender_base.py` re-runs the REAL bench pipeline (`finalize_base_task`,
  not a render-only pass — runtime gate/LTL/stability is RE-COMPUTED in the gravity idle-step, never copied) in
  place or from a regen source (`--src-root`); `cabinet_bothfront.py` (both-front + near-edge + on-table +
  drift + drawer + adaptive-sep) and `cabinet_roleswap.py` (the §13 6-field swap) lifted from the T1 one-offs;
  `tools/cabinet_review_grid.py` tiles each level's 35 last-frame stills into one review montage. 18 of 35 tasks
  modified; the full gravity idle-step caught spawn defects the gravity-OFF spawn settle masked → 5 content
  regens (0007 chess_set→bottle_of_water; 0024+0026 hutch-desk `desk/puapey`→flat `conference_table` so the
  robot base is coplanar not perched; 0026 tippy `wine_sauce_bottle`→`can_of_bay_leaves`; 0028 big
  `pewter_teapot`→`mug`; 0034 big `stockpot`→`can_of_soda`). 0017/0025/0030 obstacles verified off-side (not
  blocking). Dataset is gitignored → committed the tools + this log; spec/plan kept local under
  docs/superpowers/. Motion layer (cabinet.py/cabinet_geom.py, T2-T5) still uncommitted. See [[project_maniguard_cabinet_family]].

- **Singularity-aware grasp selection + cabinet FREE-segment collision policy (2026-06-24)**: cabinet datagen
  dropped task_0000 (0/9) because grasp selection scored Cartesian reachability only — never the IK joint config —
  so it picked the annotated "palm-flip" 180°-roll variant with the wrist already at a joint limit, and the first
  lift/carry stalled at a singularity. Fix in the shared scorer (`executor/grasp_select.py`): per candidate, IK
  BOTH approach-roll variants (annotated quat + 180° about eef +Z), read the planned arm config, keep the roll
  whose worst joint sits farthest from a limit (`joint_margin`), drop grasps below `MARGIN_FLOOR` (0.2 rad). The
  chosen roll propagates to execution via a `grasp_roll` flag in the segment `extra` → `cabinet._grasp_pose`
  re-rolls the LIVE pose (robust to object motion); every reorient that rebuilds the held pose from the DB grasp
  (`uprightify`, `_yaw_align_quat` for `fit_yaw`/`edge_yaw`) applies the same roll via `_eo_rolled` so it never
  twists the wrist back. `_select_place_grasp` extends the margin to the place carry (over-rim + over-cavity).
  Handle grasps keep the legacy single-quat path (their own contact gates). Validated: task_0000 1/1 (was 0/9),
  task_0004 3/3, both prior `pick_lift`/`place_lift` singularity stalls gone. Bundled with the T2 motion-layer
  collision policy (the 3 FREE connectors now AVOID the cabinet + open drawer instead of ignoring them). Pure
  helpers unit-tested (`tests/datagen/test_grasp_singularity.py`); cuRobo/sim paths validated by collection.
  See [[project_maniguard_grasp_singularity]].

- **datagen: preserve every failed attempt under unique `_failN` dirs (2026-06-24)**: `primitives/record.py` —
  with `DATAGEN_KEEP_FAILED=1` a kept failed trajectory is renamed `<traj>_failN` (next free N) instead of being
  overwritten by the next attempt, so debug review sees ALL failures of a slot (the `traj_000_fail0/1/2` seen in
  the cabinet collection runs).

- **eval: consume shared safety_monitor LTL resolution in benchmark (2026-06-24)**: `eval/benchmark.py` — the LTL
  active-object resolution (`category_synset_lemma`, `build_active_objects_for_ltl`) now lives in the shared
  `utils.safety_monitor` (single source across eval/datagen/bench-finalize); benchmark keeps thin local aliases so
  call sites are unchanged. See [[reference_shared_success_ltl_modules]].

- **Bench task_0005 pineapple→jar_of_tumeric: reach-fix by target swap (2026-06-24)**: task_0005's pineapple
  target was un-graspable — its in-path corridor (drawer-centre X) sits 0.5-0.67 m out, at the Franka top-down
  reach edge, and the pineapple's only 4 grasps are all high (z_local 0.135 → standoff z≈0.72), so cuRobo's FREE
  plan to the standoff failed even drawer-closed (`--score` 0/3 + `--no-score` `pick_pre` plan_fail 3/4). A
  surgical spawn-toward-robot alone could not fix it (corridor barely overlaps reliable reach + high grasps), so
  the target was swapped to a compact low-grasp `jar_of_tumeric/miivhi` (text-swap scene incl. nested copies +
  structured diag edit; backup `base_bak_preedit`), re-finalized clean (gate/LTL ok) → now collects 3/3 (jar
  grasps reachable, margins 0.34-0.92). All 4 perturbations regenerated from the jar base via
  `bench_builder/perturb_{target,language,location,env}` (env reused the original room office_vendor_machine);
  5 manifests rebuilt to 35 rows (the drivers' `--tasks` `w`-mode truncation); review snapshots regenerated.
  Dataset is gitignored → only this log committed. See [[project_maniguard_cabinet_family]].

- **Cabinet place-into-drawer datagen pipeline collects end-to-end on task_0002 (2026-06-26)**: the
  `cabinet_pickup` family now produces a complete demo (relocate obstacle+target → open → place in the
  open cavity → close). Fixes: (1) datagen switched to the RIGID `joint_position_raw` controller (matching
  eval/teleop/bench) — the soft `joint_position_impedance` preset drooped the held load ~0.07 m at extension
  and couldn't track the descent; rigid also makes datagen↔eval consistent (recorded action = next-achieved
  joint). (2) the carried object centres on the EXPOSED-cavity geometric centre I (cabinet front + ½ live
  open dist, on the slide centreline), dropping the off-centre toward-robot/fillable bias that was the REAL
  cause of the descent wrist singularity that tipped it → a clean straight-down upright lower-in (the
  earlier azimuth/branch/cuRobo detours were all symptoms of the off-centre target). (3) `open_dist`
  auto-caps to the widest opening whose handle is reachable for BOTH the open pull AND the close re-grasp
  (full-open pushed the handle past reach → `close_pre` IK-fail). (4) reverse-replay exit; max_steps 4500.
  Committed `903c0277` (8 datagen files; only Co-Authored-By trailer). Robustness ~2/6 (close_pre plan_fail
  from only 1 clear handle grasp at 0.289) — next: drop `place_over_handle` + dead-code cleanup, then scale.
  See [[project_maniguard_cabinet_family]].

- **Cabinet close-approach hardening + the ~50% close reach-limit finding (2026-06-26, post-903c0277, UNCOMMITTED)**:
  after the working-pipeline commit, hardened the CLOSE approach. (1) `place_over_handle` removed, then re-added as a
  PARTIAL lateral move — `OVER_HANDLE_FRAC=0.55` of the way from the cavity centre toward the handle XY (z held) → the
  empty gripper exits the cavity into open space before the close (33%→50% per-attempt, shorter demo). (2) `close_pre`
  made truly COLLISION-AWARE — removed its `ignore_objects=cab` (the docstring CLAIMED "collision-aware" but it IGNORED
  the cabinet → cuRobo planned a no-avoidance diagonal whose fingers CLIPPED the drawer door; the user caught it on the
  1:50 video). Now CLEAN (no clip) at the same ~50% rate. (3) `plan_tries=4` (driver→engine) for stochastic FREE-plan
  retries. (4) Added a generic `resample_compute` mechanism (contracts.MotionSegment + engine: re-resolve the target on
  EACH plan retry, + per-segment `plan_tries`) and a close_pre standoff-RANGE search — but it FAILED (0/6) and was
  REVERTED: standoff is the WRONG knob — BOTH close_pre failure modes are ORIENTATION (`rot_err`: TRAJOPT_FAIL ~0.8 = ~50°,
  IK_FAIL 0.038 just over the 0.03 tol; `pos_err` tiny), and a standoff (position offset along the approach axis) does
  NOT change the grasp orientation; my range 0.12-0.18 was also all LARGER than the working 0.10 → reach/joint limit
  (the user's worry). close_pre is back at collision-aware fixed `STANDOFF=0.10` = the 50%-clean config. **The ~50%
  close_pre plan_fail is a FUNDAMENTAL reachability limit**: the SIDE handle grasp at the OPEN (0.289, far +y) position is
  at the edge of orientation-reachability (ignore-cab AND collision-aware BOTH ~50%). Levers to exceed it (deferred):
  relax close_pre `rot_tol` (rescues the IK_FAIL half → ~70%; needs a per-segment rot_tol field; a few° is fine for a
  push), a different/easier handle grasp, or accept 50% (likely task_0002-specific — the handle is pushed far +y only at
  this task's open width; SCALING to other tasks will tell if it's systemic). **DECISION (user): accept 50% (clean) →
  cleanup → commit → scale** (scaling reveals whether 50% is systemic; defer `rot_tol` unless scaling shows it's needed).
  **DEAD CODE removed in this pass** (grep-verified zero callers, py_compile + pure-module import clean): `cabinet_geom.up_dst_pose`,
  `_prepare`'s `fillable_link`, the `toward_robot` extra payload, the `lift_out` compute handler + `LIFT_OUT_ABOVE_RIM`, the
  `STANDOFF_RANGE` constant, the `standoff_range` sampling in `_grasp_pose`, and the reverted `resample_compute` machinery
  (`contracts.MotionSegment` field + the engine retry branch); also fixed two STALE `on_segment` segment names
  (`place_over`→`place_across`, `place_over_handle`→`place_toward_handle`, renamed earlier this session — `place_lift`
  already stiffens + persists so net behaviour is unchanged) and deleted the now-orphan `test_cabinet_up_dst_pose.py`.
  **KEPT**: the effective `plan_tries` knob (engine-level default `plan_tries=4` in the driver + the per-segment override
  field). The post-903c0277 close work committed alongside: `cabinet.py` (partial over_handle `OVER_HANDLE_FRAC=0.55` +
  collision-aware close_pre), `driver.py` (max_steps 4500, plan_tries 4), `contracts.py` + `engine.py` (per-segment
  plan_tries). See [[project_maniguard_cabinet_family]].

- **Cabinet: all-35 non-penetration z-fix + place reach fix (committed) + the place_across SLIDE-BACK finding (2026-06-27)**:
  - **z-FIX (all 35 cabinets now non-penetrating)**: a full-35 clearance audit (live AABB, `cab_bottom` vs `table_top`)
    found NONE was clearly above the table — all FLUSH (gap ≈0) or 0.1–0.7 mm BELOW (per-table-model `top_plane_z` metadata
    error: `djflkd` −0.7, `jxixdw` −0.5, `semdkc` −0.3, `wtlxfr` −0.1). A penetrating/flush cabinet JAMS the sliding drawer
    against the table (scrape) → it never opens → all downstream place failures are noise (same pattern as the 2026-06-25
    wnrior breakthrough). Controlled proof: task_0007 +0.7 mm (gap→0.0000) → drawer OPENS (joint 0.204, pull eef_err 0). Raised
    the **12 negative-gap tasks** (`0005 0007 0011 0016 0018 0022 0023 0024 0026 0029 0030 0034`) cabinet root z by −gap to
    gap=0.0000 (.bak_zfix2) + **properly re-finalized** `cabinet_rerender_base --jobs 3` (reload→gate→LTL→re-render 4 vids→re-dump
    BOTH json; 12/12 gate_pass=True ltl=False). Other 23 already flush. Re-audit: all 12 gap≈0.0000. Source `_place_cabinet_on_surface:610`
    `dz = top_z − bottom_z` places FLUSH using the buggy metadata `top_z` — a +clearance source fix is the recurrence-prevention follow-up.
  - **PLACE FIX (committed `c94899c6`)**: a TALL top-gripped object (eef ~0.208 m above its bottom) carried to the cavity CENTRE at
    the rim-clearing height put eef ~1.26 m → OVER the arm's TOP-DOWN orientation-reachability ceiling at the far cavity (IK ground
    truth: cavity-centre z=1.10 REACH / z=1.20 FAIL at ~same shoulder distance — orientation, not reach; insertion z=0.88 REACH) →
    place_across servo_ik_fail 6/6 (task_0007 0/6). Reach measured from the SHOULDER (joint2 ~base+0.333) not the base — my earlier
    "base too far / infeasible" was a wrong reference. Fix (IK-verified + 7-agent consensus workflow + general-purpose code review
    APPROVE): (a) cut `RIM_CLEAR_BAND` 0.10-0.13 → 0.01-0.02 + `RIM_CLEAR_HARD` 0.03 → 0.005 (the big clearance compensated for the OLD
    `joint_position_impedance` wrist droop; the rigid `joint_position_raw` has none AND the over-lift overshot the ceiling); (b) bias the
    rim-crossing/drop toward the robot near edge (`PLACE_NEAR_EDGE_BIAS=0.07`, `PLACE_WALL_MARGIN=0.02`, via a shared `_carry_target_xy`
    used by BOTH the place-grasp pre-check `_predicted_cavity_xy` AND runtime `over_cavity` so selection+execution stay consistent),
    clamped inside the interior. Strict-upright (only XY moves). **task_0007 0/6 → 3/3** (place_across servo_ik_fail & below_z both 0).
  - **NEW BLOCKER — place_across SLIDE-BACK (the next thing to fix)**: on SHORT objects (task_0009 jar 7 cm → 1/3) the drawer OPENS FULLY
    (0.289, drawer_open eef_err 0) then **SLIDES CLOSED during place_across** (per-segment drawer_joint: stable ~0.31 through pre_grasp/
    descend/lift, then 0.31→0.13 during place_across, →0.08 at place_lower → object dropped into a half-closed drawer → goal not reached).
    NOT under-open; NOT a fingertip clip (no `<->cabinet` contact at the segment END). **USER'S VISUAL DIAGNOSIS (videos sent)**: the
    place_across SERVO (nearest-seed straight-line IK) drives the ELBOW into a contorted pose, hits a SINGULARITY mid-carry, the IK
    abruptly SNAPS the elbow joint back to keep advancing → the eef DROPS → clips the drawer FRONT face → pushes the drawer closed. Lift
    HEIGHT is ~enough (not the cause). Likely WORSENED by the place-fix's lower carry for short objects (lower carry → arm sweeps lower
    over the open drawer). User's intended arm config: base joints ~fixed, upper arm ~VERTICAL, only the forearm (eef-side joints) bends
    (the arm itself a clean 门-frame). User's lift-height robustness rule: lift to the **CD (front, HIGHER) edge** plane + a few cm
    clearance so the carry clears WHICHEVER edge the path crosses (AD side is LOWER) — the code's `drawer_top_z = cab.links[link].aabb_hi_z`
    is already the drawer-link MAX-z (= the highest edge ≈ CD front), so the lift already targets the high edge; the slide-back is the
    singularity DROP, not the lift. **NEXT (post-compact)**: control the place_across arm CONFIG to avoid the elbow singularity (better
    seed keeping the upper arm vertical / finer carry waypoints / a config that only bends the forearm), then re-test short + tall +
    scale. See [[project_maniguard_cabinet_family]].

- **Cabinet task_0007 close/place debug — LTL-coverage + lower-in + grasp-close round (2026-06-28)**: a systematic, video+log-driven
  debug of task_0007 (`bottle_of_water/migvlt`, the tall thin target). User visual review showed the earlier "place_across slide-back"
  was only one symptom; the real residual failures were (1) lower-in too deep, (2) a missed-tip LTL gap, (3) a jerky close that topples
  the bottle. Three READ-ONLY agents ruled out the source hypotheses: the object is NOT too tall (0.2285 m vs cavity 0.2737 m, +3.5 cm
  close-headroom; the task-gen height filter `InteriorBBox.fits` worked), and opening WIDER is a red herring (the 0.204 cap is the close
  re-grasp reach, and a wider open doesn't change the vertical cavity height). Three fixes landed, each with a single-agent code review:
  - **#2 LTL FULL-ROLLOUT COVERAGE (engine, Part A)**: the gate ticked every executed step EXCEPT the gap after the last segment — a
    bottle toppling once the gripper's attached-grasp masked-upright hold ends (post-release) completed UNMONITORED and was mislabeled a
    clean success. Fix = a MONITORED end-of-rollout settle: `actuate_gripper(close=last-cmd, n_steps=rest_settle_steps=45, recorder=None,
    on_step=tick)` BEFORE `gate.success()`, so the 45° `all_active_upright` is checked through settle-to-rest and the verdict is read on
    the SETTLED state (also catches a drawer springing back open). Threshold STAYS 45° (the fix is COVERAGE not threshold, per user).
    +hardening: `carry=OPEN` init (empty-segments NameError guard) + `if self._timeout` fail-clean before the settle.
  - **#1 LOWER-IN RIM-RELATIVE (cabinet, Part B)**: `lower_to_floor` aimed the held bottom at `drawer_floor_z` = the drawer LINK's AABB
    underside (skirt/front-face bottom), BELOW the real interior floor → the rigid `joint_position_raw` SERVO jammed the object THROUGH
    the floor, destabilising it AND dragging the sliding drawer shut. Fix = descend rim-relative: `dz = held_top − (rim − 0.01)` (new
    `_held_top`), stopping the object's TOP just under the rim → rests on/near the true floor, no overshoot. Verified: jam + drawer-drag
    gone (drawer holds ~0.31 through place_lower vs the old 0.2147→0.2065).
  - **#3 GRASP-AND-SLIDE CLOSE (cabinet)**: the old close drove an OPEN gripper to push the sliding drawer FRONT — an unstable contact that
    stick-slips under the rigid controller (stall→slip→jolt) and topples the placed object. Rewrote `_close_drawer` as the INVERSE of the
    open pull: `close_pre`(FREE, open) → `close_grasp`(SERVO, CLOSE on handle) → `close_push`(SERVO, carry_closed, `compute="drawer"`
    `to:"close"` `joint:tj` — the GRIPPED handle slides the soft drawer shut, rigid 1:1) → `close_release`(open in place). The `drawer`
    compute close branch gained a `joint` target (slide to a sliver `tj`, off the hard stop). Dropped the dead non-final retreat branch.
  - **NEW DIAGNOSTIC (engine)**: `DATAGEN_TRACE` writes a per-step `trace.jsonl` (segment, arm `q`+`qd`, target obj pose+lin/ang vel) for
    EVERY rollout (kept or failed) — the traceback tool for close-phase smoothness + tip-onset analysis. (Also present, DORMANT: a
    `_build_pin_seed` no-flare-seed ladder + `arm_flare`/`elbow_lateral_offset` flare metric from the shelved place_across pin — the flare
    metric is live in the trace diagnostics; the pin builder is unwired, kept as the foundation for the global SERVO-redundancy fix.)
  - **WHAT REMAINS (the real residual, evidence-backed)**: grasp-close did NOT lift task_0007 (still ~2/12) — the bottle STILL topples and
    the close STILL stalls. Trace proved WHY: the close-phase jerk ("一卡一卡") is the **SERVO per-waypoint IK joint chatter** (`close_push`
    max|qd|≈2.5, redundant j2/j4 wander 5-6×), IDENTICAL whether the gripper pushes (open) or grasps (closed) — the arm's joint motion is
    set by the eef-path IK, not the grip. Controlled contrast: a SHORT/squat object (`task_0002` fruitcake) survives the SAME chatter
    (close_push tilt 1.8°, closes to 0.000-0.0087) while the TALL bottle tips (25-55°). So the close jostle root = SERVO redundancy chatter
    = the SAME root as the place_across elbow flare. A `smooth_servo` (joint-path moving-average) attempt FAILED + was REVERTED (it broke
    the open-finger contact). NEXT: a global minimal-motion SERVO-IK fix (controls the redundancy at the root → smooths close AND
    place_across), with a cheap `servo_spw`-slowdown quick-test first. (Separately, `close_pre` cuRobo plan_fail = a task-specific handle
    reach limit, accept clean-fail + retry.) See [[project_maniguard_cabinet_family]].

- **Cabinet close: uniform-velocity smooth close + the marginal-bottle limit (2026-06-28)**: Round-4 of the task_0007
  close debug. Confirmed (code + trace) the close "一卡一卡" was a SLAM-IDLE artifact: `execute_trajectory` commands each
  1 cm SERVO waypoint for `servo_spw=4` steps and the rigid `joint_position_raw` controller (isaac_kp≈1e7) slams the full
  1 cm in ~1 step then idles 3 → spike-then-zero eef velocity, which jerks the gripped handle/drawer. Fix (per-segment, scoped
  to `close_grasp`+`close_push`): `servo_step_m=0.0025` + `servo_spw=1` → each sim step advances the eef one 2.5 mm increment =
  a continuous UNIFORM 7.5 cm/s glide, DURATION-NEUTRAL (same total steps). Added per-segment `servo_step_m`/`servo_spw`
  overrides to MotionSegment. Verified: close_push mean|qd| 0.6→0.11, Δq 0.006 (smooth). Two dead-ends REVERTED first:
  `smooth_servo` joint-path moving-average (broke the open-finger contact) and `servo_spw=16` (more idle, same slam, worse).
  HONEST LIMIT (trace-proven, corrects "chatter tips it"): even a perfectly smooth close (traj_000, Δq 0.006, no IK flip)
  still tipped the bottle to 43.5° — the tall `bottle_of_water` (0.2285 m, at the cavity-height limit) catches the cabinet
  front opening as it's carried in whenever it has a few-degrees lean; smoothing makes the tip later/smaller (43° vs 55°) but
  can't eliminate it (marginal stability, not a motion bug). Also: smooth/jerky is STOCHASTIC (the per-variant torch seed,
  engine.py:216, makes the per-waypoint solve_ik land on a continuous vs flipping branch). The dominant yield limiter is
  actually separate cuRobo `Mode.FREE` plan_fails — `place_pre_grasp` (task_0007) / `close_pre_final` (task_0002, ~7-8/12),
  stochastic, driver retries. DECISION (user): stop tuning the close (randomness + object physics dominated); validate the
  pipeline on 5 fresh normal-stability tasks (target 2 each) to measure real yield. See [[project_maniguard_cabinet_family]].

- **Cabinet yield: far-reach close fix — IK-tolerance relax + open-distance derate (2026-06-29, `73386f21` + `837e9823`)**:
  the family yield limiter was the close re-grasping the handle at its pulled-OUT (open) position = the far edge of the arm's
  reach. ROOT CAUSE corrected by reading the cuRobo diag (not guessed): `close_pre_final` IK_FAILs because the best IK solution
  sits a hair past BOTH baked gates — pos 0.0052–0.0058 m > 0.005 and rot 0.043–0.059 rad ≈ 0.05 — whichever binds first kills
  the plan (no trajectory). The earlier "rot stuck on 0.05" was only half of it; relaxing rotation alone left pos-bound configs
  still failing. **Approach A (`73386f21`)**: cuRobo has no per-plan tolerance field (baked at MotionGen construction across 4
  solvers), but the IK success gate reads `ik_solver.position_threshold`/`.rotation_threshold` at CALL time
  (`ik_solver.py:1352 _get_success`) — so widen the live attributes for ONE plan (try/finally restored) + the `_salvage` tols to
  match, scoped via new `MotionSegment.rot_relax`/`pos_relax` (→ `solve_segment` `ik_rot_relax`/`ik_pos_relax`) to the cabinet
  `close_pre` segment only (relax 0.10 rad / 0.015 m). Harmless: pos is a standoff the next SERVO re-aims from the LIVE handle,
  rot is on the roll-symmetric bar; a real miss is still caught by `close_grasp` + the physical success gate. Lifted apricot
  (task_0003, baseline WORST) 1/20 → 2/5; `close_pre` stopped plan_failing. The 3 residual fails were all the same far-open-handle
  close-sequence reach (close_grasp `servo_ik_fail` / close_push undershoot-jam → `goal_reached` False / close_push object-tip →
  unsafe). **Approach B (`837e9823`)**: the `open_dist` selection gate is OPTIMISTIC — it scores 5 discrete handle poses with
  cuRobo from the home pose, but the runtime close runs a continuous per-waypoint `solve_ik` SERVO from the post-place config, so
  it picks an open whose far handle is at the reach envelope. Derate the gate-selected open by `OPEN_DIST_SAFETY=0.88` (~2–3 cm
  nearer the base) after selection, before the place gate; grasps stay reachable (smaller = strictly easier) and the place reach
  gate still validates cavity fit. **5-task scan (target 2) vs baseline: apricot 1/20→2/3, napkins 2/9→2/2, muffin 2/5→2/3,
  box_yogurt 2/3→2/2, can_soda 2/4→2/2 — all improved or held, no regressions**; with B the handle is close enough that
  `close_pre` now plans CLEANLY (no salvage). See [[project_maniguard_cabinet_family]].

## 2026-06-30 — cabinet_pickup family FINALIZED (all 35 collectable; awaiting server for large-scale collection)

Repaired the 12 tasks that were under-target in the full-family dry-run → **all 35 cabinet_pickup base tasks now collect ≥2/2**
(success + LTL-safe). Two committed code fixes + data repairs:

- **`33384142` fix(bench): round-top surface uses inscribed usable rect, not square AABB.** `finalize_base._fresh_surface_info`
  computed `surface_info.bounds_xy` from the surface's world AABB, which is a square for a round-top table
  (`coffee_table/semdkc`) — datagen relocate / `perturb_location` then parked objects past the circular rim and they fell off
  (0016/0029 collected 0/2). Added `_ROUND_SURFACE_HALF={"semdkc":0.43}` + `_surface_bounds(lo,hi,model)` → for a known round
  model emit the inscribed square centred on the live AABB (`frame=world_usable_rect`), else AABB unchanged; benefits every
  family/pose that uses the table. (Audit: `semdkc`=round→fixed; `rlsebe`=square slab→AABB correct; `wzyqgx`=likely-fine.)
- **`65cfddb4` feat(tools): cabinet_swap_object.** Swaps a task's target/obstacle for a bench donor by rewriting `scene_ep1.json`
  (RECURSIVELY — the snapshot nests whole-scene copies under `init_info/args/scene_file/…`) + `diagnostics.jsonl`
  (`{role}_info`/`selection`/`spawn_specs`/goal subject/prompt/LTL over-globs). LTL globs derive from the object NAME STEM
  (robust to a few tasks' crossed role/category naming); `_find_donor_source` also scans `*.bak_swap` so a swapped-out object is
  reusable as a donor. Used to repair 10 tasks (0001/0007/0011/0012/0018/0019/0025/0028/0030/0032) to 2-attempt-stable donors,
  all 35 (target,obstacle) pairs kept unique.
- **0028 lesson:** cabinet datagen relocates the OBSTACLE first, so an obstacle-relocate `move_transit` fail blocks the demo
  before the target is touched — the fix was making the compact flat **mouse the obstacle** (target/spawn-y were red herrings).
- Data (gitignored) repaired end-to-end: bases re-finalized, all 4 perturbations regenerated (`perturb_language` has no `--jobs`),
  full 35-row manifests rebuilt, and the 12 tasks pushed to HF `IDEAS-Lab-Northwestern/ManiGuard-Bench`.
- **Collection handoff** = `CABINET_COLLECTION_HANDOFF.md` (repo root): env/commands/parallelism + ~4.4 min per usable trajectory
  (single-process at scale), ~67% per-attempt success. See [[project_maniguard_cabinet_family]].

## 2026-07-01 — datagen resume-safe seed cursor (stop/resume no longer duplicates trajectories)

Found via the v1 clutter top-up: resuming a task re-collected DUPLICATE trajectories (task_0031 23 unique
of 40, task_0032 14 of 28, verified by each traj's meta seed). The per-demo master seed was deterministic
in `(grasp_id, k)` and the sampler restarted its draw index `k` at 0 every run, so a top-up re-drew the
same `(grasp, k)` → same jitter + same cuRobo trajopt seed → same trajectory saved under a new `traj_NNN`.
Low-success tasks were stuck (the same k deterministically succeed/fail → resume can't add unique data).

- **`3b9698bc` fix(datagen): resume-safe draw cursor + collision-free seed encoding.** Persist a task-level
  `next_draw` cursor (= max draw index *attempted*, incl. failures, + 1) in `_summary.json`; the driver
  reads it back and resumes `VariationSampler.variants_stream(cands, start_k=next_draw)` so a top-up only
  ever draws UNSEEN seeds. The seed is now `SeedSequence([grasp_id, k])` (uint32) instead of
  `grasp_id*1000+k` (which aliased once k>=1000 as the cursor accumulates across rounds). New pure helpers
  `executor/resume.py` (`resolve_start_k`/`compute_next_draw`); each demo meta now stores `seed`+`draw_index`;
  `--start-draw N` override (driver + sweep) to recollect a deduped task with guaranteed-fresh seeds.
  Family-agnostic — every family's collection flows through the same sampler/driver.
- **Tested**: 7 unit tests (collision-free at high k, start_k disjoint, cursor math, stop/resume no reuse)
  + local headless sim (task_0002, two rounds → draws `[0,0,1,1]`, 4 seeds all distinct).
- **Follow-up (not done)**: dedup the already-polluted Vast v1 tasks (0031/0032/0047/0045 + check
  0036/0038/0015) by seed and recollect the deficit via `--start-draw`; diagnose task_0032's ~18% success
  rate. See [[project_clutter_liquid_reach_fixes]].

## 2026-07-01 — datagen resume: on-disk draw floor (dedup→recollect no longer re-dups)

Sequel to the resume-seed-cursor fix, found while cleaning the Vast v1 clutter data. The
dedup→recollect→top-up workflow re-introduced 13 duplicate trajectories (task_0024 +1, task_0032 +12):
the resume cursor read `start_k` ONLY from `_summary.json`'s `next_draw`, and the dedup cleanup script
had wiped that field, so recollect ran every task at k=0 (harmless — new `SeedSequence` seeds don't
collide with the old `grasp*1000+k` data), but a SECOND resume (the max-attempts-180 top-up) of the two
still-under-target tasks also restarted at k=0 and re-drew the same new-encoding seeds recollect had just
written. The 8 tasks that reached 40 in recollect were fine (their summaries kept a correct `next_draw`).

- **`1457d309` fix(datagen): floor resume start_k by on-disk max draw_index.** `resolve_start_k` now takes
  `ondisk_max_draw` and returns `max(requested_start, ondisk_max_draw + 1)`; the driver scans existing
  `traj_*/meta.json` for the highest `draw_index` and passes it. A resume can no longer re-draw a k that
  already has a trajectory on disk, regardless of whether the summary's `next_draw` survived. `next_draw`
  still wins when higher (it also skips FAILED k). Unit-tested (floor beats a lost/stale summary; a higher
  next_draw still wins).
- **Cleanup**: the 13 dups re-deduped + the 2 tasks recollected with the floor in effect (start_k
  auto-derives to 19 / 7). See [[project_clutter_liquid_reach_fixes]].

## 2026-07-01 — datagen RAW -> LeRobot v2.1 converter (family-agnostic)

`8b1646dc feat(datagen): raw -> LeRobot v2.1 converter`. The pipeline's "future work" (`pipeline.md` §G)
converter, finally written: `maniguard/data/datagen/to_lerobot.py` repackages the RAW per-traj collection
(traj.hdf5 + 5 mp4 + meta.json) into ONE LeRobot v2.1 dataset per family.

- **Numeric**: `state` + `actions` (b, next-achieved) + `actions_commanded` (a) in parquet, **absolute
  joint** (openpi deltas at train via `use_delta_joint_actions`). Matches the old
  `sentinel-pnp-clutter-joint` convention + the `data_format` default.
- **Video passthrough (no re-encode)**: pre-copy each h264 mp4 to `ds.root/get_video_file_path(ep,key)`
  (that path is root-RELATIVE — the one gotcha), no-op `_save_image`, mp4-aware `sample_images` stats,
  `save_episode` skips encoding a camera whose mp4 already exists. 3-patch set VERIFIED against lerobot
  0.3.3 (Task-0 spike: 5 mp4 byte-identical before/after). Technique referenced from `lerobot_writer.py`
  but NOT imported (reuses only `datagen.reader` + `data_format`).
- **All 5 cams kept** (opposite/left/right/left_shoulder/wrist); cam-subset + delta stay train-time
  openpi concerns. **One dataset/family** (2200 clutter eps → one SFT ckpt), multi-task via `tasks.jsonl`,
  prompt from `meta["prompt"]` (== task diagnostics, verified).
- **Env**: the lerobot uv env (py3.11 / lerobot 0.3.3 / +h5py 3.16 / numpy 2.x) — separate from the
  `behavior` conda env; the converter never imports OmniGibson (`reader`/`data_format` are pure).
- **Output**: `outputs/datagen/<dataset>_lerobot_format/<family>/`, repo_id
  `IDEAS-Lab-Northwestern/datagen-<fam>-v1-joint-5cam`, **no HF push**. Local 80-ep integration verified
  (loads, 5 video + state + 2 actions, image decodes 256², numbers match raw, prompt in tasks.jsonl).
  spec+plan+test local-only per user. Full clutter 2200-ep run on the Vast server.
  See [[project_clutter_liquid_reach_fixes]].
