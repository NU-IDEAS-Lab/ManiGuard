# SENTINEL-Lite — Dev Log

## Action items (deferred)

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

## 2026-05-27 — `feat/omnisafe-integration`: Phase 5 — observability + configs

`SafetyCallback` now records `info["cost"]` per step: accumulates per-rollout sums into `safety/ep_cost_mean` and `safety/ep_cost_max` (flushed in `_on_rollout_end`), matching the `rollout/ep_cost_mean` convention logged by `ConstrainedPPO.collect_rollouts`. `safety/ltl_violations` unchanged. Starter YAML hyperparams added at `configs/rl/safe/{ppo_lag,focops,cpo}.yaml` derived from omnisafe's published defaults adapted for `PickAndLiftPrivilegedTask` (shorter episodes, smaller obs). All 32 tests still pass.

## 2026-05-27 — `feat/omnisafe-integration`: Phase 4 — CPO + TRPO infrastructure

`sentinel/rl/safe/trpo_utils.py`: `flat_params`, `set_flat_params`, `flat_grad`, `conjugate_gradients` (CG solver with Tikhonov damping, validated against a known 2×2 SPD system). `sentinel/rl/policies/constrained_actor_critic.py` gains `get_action_distribution(obs)` (actor-only forward, skips value heads; used by the FVP to avoid computing the two value heads on each of 20+ CG iterations). `sentinel/rl/algorithms/cpo.py:CPO(ConstrainedPPO)`: uses `FOCOPSDictRolloutBuffer` for old-dist storage; separate `_critic_optimizer` (Adam) that only touches value heads; `train()` does: full-batch policy gradient + cost gradient → CG natural gradients `x_a = F⁻¹g` and `x_b = F⁻¹b` via Pearlmutter double-backward FVP (must be outside `th.no_grad()` — caught during debugging) → three-case QP projection (`feasible_trpo`, `projected/projected_infeasible`, `feasibility`) → KL backtracking line search → standard Adam minibatch value updates. Logs `train/cpo_optim_case`, `train/cpo_accept`, `train/cpo_constraint_violation`. Save/load persists `_critic_optimizer` state to sidecar `.._critic_opt.pt`. 8/8 new tests pass. 32/32 total.

## 2026-05-27 — `feat/omnisafe-integration`: Phase 3 — FOCOPS

FOCOPS: First-Order Constrained Optimization in Policy Space (ported from omnisafe's FOCOPS, Apache-2.0). Policy stores `_last_distribution` as a side-effect of both `forward` and `evaluate_actions`, making the current Gaussian available to the training hook. `sentinel/rl/buffers/focops_rollout_buffer.py:FOCOPSDictRolloutBuffer` extends `ConstrainedDictRolloutBuffer` with `mu_old`/`log_std_old` arrays shaped `(buffer_size, n_envs, act_dim)`. `ConstrainedPPO.train()` gains a `_compute_policy_loss(rollout_data, log_prob, advantages, clip_range) → (loss, clip_frac)` hook (default: PPO clipped surrogate). `FOCOPS(ConstrainedPPO)` overrides `_setup_model` (swap to FOCOPSBuffer + build Lagrange), `collect_rollouts` (extract + store mu/log_std per step), `_compute_policy_loss` (KL-projected loss: `E[(KL − ratio·adv/ζ) × I[KL ≤ η]]`). `_before_update` and `_compute_adv_surrogate` identical to PPO-Lag. Also stores `cost_gamma`/`cost_gae_lambda` on `ConstrainedPPO` (needed by `_setup_model` override). 8/8 new tests pass. 24/24 total.

## 2026-05-27 — `feat/omnisafe-integration`: Phase 2 — PPO-Lag

PPO-Lagrangian algorithm (ported from omnisafe's `PPOLag`, Apache-2.0). `sentinel/rl/safe/lagrange.py:Lagrange` is a scalar `nn.Parameter` λ with a dedicated torch optimizer; `update_lagrange_multiplier(ep_cost)` does `λ ← λ + lr × (ep_cost − cost_limit)` with a non-negativity clamp; supports softplus parameterisation and upper-bound clamp; includes `state_dict_extra` / `load_state_dict_extra` for save/load. `sentinel/rl/algorithms/ppo_lag.py:PPOLag(ConstrainedPPO)` overrides `_before_update` (call `lagrange.update_lagrange_multiplier`, log `train/lagrangian_multiplier`) and `_compute_adv_surrogate` (`(adv_r − λ·adv_c)/(1+λ)`). Save/load persists λ state to a sidecar `_lagrange.pt`. `sentinel/rl/cli/common.py` gains `add_lagrange_args` and `add_cost_args` groups. CLI entry point mirrors `ppo_proprio_goal.py` shape. 9/9 new tests in `tests/test_ppo_lag.py` pass (λ update direction, clamping, softplus, state-dict roundtrip, surrogate math, end-to-end with ConstantCost, save/load). 16/16 total.

## 2026-05-27 — `feat/omnisafe-integration`: Phase 1 — two-critic SB3 PPO base

Three building blocks for the upcoming PPO-Lag / FOCOPS / CPO ports. `sentinel/rl/policies/constrained_actor_critic.py:MultiInputConstrainedActorCriticPolicy` is an SB3 dict policy with a parallel `cost_value_net` (Linear on the shared vf latent by default; opt-in separate features extractor via `separate_cost_value_extractor=True`). `forward` / `evaluate_actions` return 4-tuples — second value is the cost-value. Save/load round-trips via `_get_constructor_parameters`. `sentinel/rl/buffers/constrained_rollout_buffer.py:ConstrainedDictRolloutBuffer` adds the parallel `cost_{rewards,values,returns,advantages}` arrays + a second GAE pass with its own `(cost_gamma, cost_gae_lambda)`; yields `ConstrainedDictRolloutBufferSamples` with both streams. `sentinel/rl/safe/constrained_ppo.py:ConstrainedPPO` subclasses SB3 PPO and overrides `collect_rollouts` (read `info["cost"]`, bootstrap cost-value on truncation, call extended `buffer.add`) and `train` (4-tuple unpack, `_compute_adv_surrogate(adv_r, adv_c)` hook for subclasses, cost-critic MSE loss weighted by `vf_cost_coef`). Also exposes a `_before_update` hook for the upcoming Lagrange step. No subclass yet; `ConstrainedPPO._compute_adv_surrogate` raises `NotImplementedError`. 7/7 tests in `tests/test_constrained_ppo.py` pass (buffer GAE, policy optimizer wiring, end-to-end rollout+train, save/load roundtrip). Vanilla PPO path unchanged.

## 2026-05-27 — `feat/omnisafe-integration`: Phase 0 — cost-function base + cost-injecting vec env wrapper

New `sentinel/rl/costs/` package — `BaseCostFunction` ABC mirrors OG's `BaseRewardFunction` lifecycle (`reset(task, env)` + `_step(task, env, action) → (cost, info)`). `ZeroCost` is the MVP stub; `ConstantCost` exists so unit tests can exercise the cost path. New `sentinel/rl/envs/cost_wrapper.py:CostInjectingVecEnvWrapper` sits between `SentinelSB3VectorEnvironment` and SB3, sums per-env cost functions on each step, writes `info["cost"]` + `info["cost_breakdown"]`. Handles sim-fault recovery (zero cost on skipped steps) and episode boundaries (reset cost-fn state across all parallel slots). `tests/test_cost_wrapper.py` covers per-env isolation, sim-fault, and validation — 6/6 pass under `SENTINEL_SKIP_OMNIGIBSON_PATCH=1`. No algorithm consumes the cost channel yet; vanilla PPO path bit-for-bit unchanged. Branch base: `4ca7ed8c` (`feat/behavior-main-isaac5`).

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
