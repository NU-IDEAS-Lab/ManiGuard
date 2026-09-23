"""Single-task orchestration for scripted demonstration collection.

Reconstruct a frozen task, initialize its family skeleton and shared executor,
sample grasp and motion variants, and record attempts that pass the configured
goal, safety, and family acceptance checks. Collection can use a bounded number
of variants per grasp or a target number of accepted demonstrations.

Use datagen.sweep to collect across multiple tasks in separate processes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def run_task(task_dir, *, family: str = "clutter", dataset: str = "demos", grasp_ids=None,
             n_per_grasp: int = 1, target: int | None = None, max_attempts: int | None = None,
             score: bool = False, out_root: str = "outputs/datagen",
             headless: bool = True, timeout: float = 5.0, steps_per_waypoint: int = 2,
             limit_demos=None, grasping_mode: str | None = None, start_draw: int | None = None,
             horizon_override: str | None = None):
    """Build one base task, then run the family skeleton x variants through the generic engine,
    recording each success+safe demo. Returns the list of DemoResults.

    ``target`` (preferred for collection): keep drawing variants until this many successes (or
    ``max_attempts``, default ``target*4``). Else ``n_per_grasp`` bounded variants. One OG
    session per process (og.clear() multi-task reload breaks cameras)."""
    import json
    import os
    import time

    if os.environ.get("DATAGEN_HANG_WATCHDOG"):          # debug: dump the main-thread Python stack every
        import faulthandler  # 20 s (a watchdog thread runs even if main is in a
        faulthandler.dump_traceback_later(20, repeat=True)   # native C call → shows the Python frame that hangs)

    from maniguard._omnigibson_patches import _patch_franka_longfinger
    from maniguard.data.datagen.executor.contracts import TaskContext
    from maniguard.data.datagen.executor.engine import DemoEngine
    from maniguard.data.datagen.executor.gate import build_gate
    from maniguard.data.datagen.executor.variation import VariationSampler
    from maniguard.data.datagen.families import FAMILY
    from maniguard.data.datagen.primitives import cameras, obstacles, record
    from maniguard.data.datagen.primitives import scene as scenemod

    t0 = time.time()
    task_dir = Path(task_dir)
    skeleton = FAMILY[family]()                        # Initialize the family before reading its grasping-mode default.
    # liquid/particle tasks (diagnostics selection.system_name, e.g. "water") REQUIRE GPU dynamics +
    # flatcache OFF or the PhysX water system NaN-segfaults at init; dry tasks stay on the CPU pipeline.
    # Must be decided BEFORE init_omnigibson (gm macros take effect only before `import omnigibson`).
    needs_gpu = scenemod.task_needs_gpu_dynamics(task_dir)
    if needs_gpu:
        print("[driver] liquid/particle task -> gm.USE_GPU_DYNAMICS=True, ENABLE_FLATCACHE=False", flush=True)
    og = scenemod.init_omnigibson(headless=headless, needs_gpu_dynamics=needs_gpu)   # Initialize the selected physics configuration.
    ag_mode = grasping_mode or skeleton.grasping_mode()   # CLI override takes precedence over the family default.
    print(f"[driver] grasping_mode={ag_mode!r} (family default={skeleton.grasping_mode()!r}, "
          f"cli_override={grasping_mode!r})", flush=True)
    bundle = scenemod.scene_from_task_dir(
        task_dir, external_sensors=cameras.external_camera_configs(),
        grasping_mode=ag_mode,
        pre_build_hooks=[_patch_franka_longfinger])
    env, robot = bundle.env, bundle.robot
    cameras.place_and_resize_cameras(env, robot, og, bundle.diagnostics)

    # Horizon variant (e.g. cabinet firsthalf): substitute this task's goal + prompt before
    # anything reads them. `bundle.diagnostics` is the single source for BOTH -- build_gate
    # takes the goal from it and the engine takes the recorded prompt from it -- so one
    # substitution here keeps success checking and language consistent, and consistent with
    # eval, which applies the same table to its scene_info. Omitted (the default) = the
    # task's stored full-horizon goal and instruction.
    if horizon_override:
        from maniguard.eval.horizon_override import apply_horizon_override
        # Key includes the perturbation level (…/task_0019/base): a variant hard-codes concrete
        # object instance names, and whether those survive a perturbation is a per-task property
        # we must not assume. eval keys on the same string (its scene_info["name"]).
        _hkey = f"{task_dir.parent.parent.name}/{task_dir.parent.name}/{task_dir.name}"
        bundle.diagnostics = apply_horizon_override(bundle.diagnostics, horizon_override, _hkey)
        print(f"[driver] horizon variant {_hkey} <- {horizon_override}\n"
              f"[driver]   prompt: {bundle.diagnostics.get('prompt', '')}", flush=True)

    # goal_region families (clutter) carry a sphere spec; goal_conditions families (cabinet)
    # don't — resolve the target + support from the spec or diagnostics accordingly.
    # (`target_obj` is the object; `target` is the --target demo count.)
    spec = bundle.goal_spec
    if spec is not None:
        target_name = spec.target_name
        goal_center, goal_radius = np.asarray(spec.center_world, float), float(spec.radius_m)
        surface_name = spec.support_name
    else:
        ti = bundle.diagnostics.get("target_info")
        if ti:
            target_name = ti["name"]
        elif bundle.diagnostics.get("dust_system"):
            # dusty: target = the SOURCE carrier instance (food rides it; dest never grasped)
            from maniguard.data.datagen.families.dusty import source_cat_model
            cat, model = source_cat_model(bundle.diagnostics)
            target_name = next(o.name for o in env.scene.objects
                               if getattr(o, "category", None) == cat
                               and getattr(o, "model", None) == model)
            # let the skeleton find scene_ep1.json (its dust-restore guard reads the saved group)
            bundle.diagnostics["_task_dir"] = str(task_dir)
        else:
            raise ValueError(f"{task_dir} has neither goal_region nor target_info")
        goal_center, goal_radius = np.zeros(3), 0.0          # unused by goal_conditions families
        surface_name = getattr(bundle.surface, "name", None)
    target_obj = env.scene.object_registry("name", target_name)
    target_key = f"{target_obj.category}/{target_obj.model}"
    ctx = TaskContext(
        env=env, robot=robot, target=target_obj, target_key=target_key,
        target_name=target_name, goal_center=goal_center, goal_radius=goal_radius,
        support=bundle.surface, diagnostics=bundle.diagnostics)

    world = obstacles.CuroboWorld(env, robot)
    gate = build_gate(env, bundle.diagnostics, surface_name=surface_name)
    skeleton.select_grasps(ctx, world, robot)   # Pre-filter family-internal auxiliary grasps.
    engine = DemoEngine(env, robot, world, timeout=timeout, steps_per_waypoint=steps_per_waypoint,
                        max_steps=4500, plan_tries=4)   # Bound rollout length and retry failed plans up to four times.
                        # Each retry samples fresh planner seeds; successful plans return immediately.
    recorder = record.Recorder()      # Record simulation states, padding variable-length dumps.

    cands = skeleton.grasp_candidates(ctx)
    if grasp_ids is not None:
        cands = [c for c in cands if c.id in set(grasp_ids)]
    fam_name = skeleton.name                                           # "clutter" (skeleton key)
    bench_family = task_dir.parent.parent.name                        # "clutter_pickup" (= bench dir name)
    task_name = task_dir.parent.name                                   # task_0000
    src_task = f"{bench_family}/{task_name}"                           # clutter_pickup/task_0000
    print(f"[driver] {src_task} target={target_key} grasps={[c.id for c in cands]} "
          f"goal_r={goal_radius:.3f}", flush=True)

    if score:
        from maniguard.data.datagen.executor.grasp_select import score_grasps
        # score_grasps drops the target from the collision world; a family may add MORE objects to drop
        # (stack: the pile above the buried bottom target, gone by the time it is grasped) via score_drop_extra.
        drop_target = [target_obj, *skeleton.score_drop_extra(ctx)]
        mf = skeleton.score_margin_floor()
        kw = {} if mf is None else {"margin_floor": float(mf)}
        cands = score_grasps(world, robot, drop_target, cands,
                             prefer_top_down=skeleton.relocate_prefer_top_down(),
                             prefer_wrist_dir=skeleton.relocate_open_dir(ctx), **kw)

    sampler = VariationSampler(n_per_grasp=n_per_grasp)

    # output dir uses the BENCH family name (clutter_pickup) so it matches the bench dataset layout
    out_base = Path(out_root) / dataset / bench_family / task_name
    out_base.mkdir(parents=True, exist_ok=True)

    # resume cursor: a fresh run starts drawing at k=0; a top-up resumes from the prior run's next_draw
    # so it only ever draws UNSEEN seeds (the seed is deterministic per (grasp_id, k) → restarting at 0
    # would re-collect duplicate trajectories). --start-draw forces the cursor (recollect a deduped task).
    from maniguard.data.datagen.executor.resume import compute_next_draw, resolve_start_k
    summary_path = out_base / "_summary.json"
    prev_summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    # Floor the resume cursor at the highest draw_index stored on disk.
    # Trajectories without draw_index do not contribute to this bound.
    # This protects against a missing or incomplete summary cursor.
    ondisk_max_draw = -1
    for _mp in out_base.glob("traj_*/meta.json"):
        try:
            _di = json.loads(_mp.read_text()).get("draw_index")
        except Exception:  # noqa: BLE001
            _di = None
        if _di is not None:
            ondisk_max_draw = max(ondisk_max_draw, int(_di))
    start_k = resolve_start_k(prev_summary, start_draw, ondisk_max_draw)

    if target:
        max_att = max_attempts or target * 4                          # give up if a task can't reach target
        variant_iter = sampler.variants_stream(cands, start_k=start_k)
        print(f"[driver] target={target} demos (max {max_att} attempts), score={score}, "
              f"start_draw={start_k}", flush=True)
    else:
        variant_iter = sampler.variants(cands)
        print(f"[driver] {n_per_grasp}/grasp bounded variants, score={score}", flush=True)
    # resume / top-up: every KEPT demo carries a traj.hdf5, so prior successes count TOWARD the target.
    # A re-run then collects only the DEFICIT (target - existing) and stops the moment N is reached,
    # instead of collecting `target` MORE on top of what's already there.
    n_existing = sum(1 for p in out_base.glob("traj_*") if (p / "traj.hdf5").exists())
    idx = n_existing
    if n_existing:
        print(f"[driver] resume: {n_existing} existing demo(s) count toward target={target}", flush=True)

    # pristine scene snapshot — RESTORED before every variant (each demo moves the target /
    # disturbs clutter; without this, variant 2+ start from a corrupted scene).
    init_state = og.sim.dump_state(serialized=True)
    results = []
    n_att = 0
    last_run_draw = None                                 # highest draw index actually attempted -> resume cursor
    for g, params in variant_iter:
        n_have = n_existing + sum(r.ok for r in results)             # existing + this run -> resume toward N
        if target and (n_have >= target or n_att >= max_att):
            break
        if limit_demos and n_have >= limit_demos:
            break
        n_att += 1
        last_run_draw = params.draw_index                # attempted (incl. failures); cursor skips past all tried k
        og.sim.load_state(init_state, serialized=True)
        robot.keep_still()
        for _ in range(3):
            og.sim.step()
        traj = f"traj_{idx:03d}"
        out_dir = out_base / traj
        meta = {"family": fam_name, "source_task": src_task, "task": task_name, "traj": traj,
                "target_key": target_key, "grasp_id": g.id, "approach": g.approach,
                "seed": params.seed, "draw_index": params.draw_index, "standoff_m": round(params.standoff_m, 4),
                "lift_clearance_mult": round(params.lift_clearance_mult, 3),
                "jitter": params.jitter, "grasp_score": round(getattr(g, "score", 0.0), 3)}
        segs = skeleton.derive_segments(ctx, g, params)
        res = engine.run(ctx, skeleton, segs, gate, recorder, out_dir=out_dir, seed=params.seed, meta=meta)
        n_have2 = n_existing + sum(r.ok for r in results) + (1 if res.ok else 0)
        print(f"[driver] {traj} g{g.id} seed{params.seed}: ok={res.ok} fail={res.fail_stage} "
              f"{res.detail} [{n_have2}/{target or '∞'} att={n_att}]", flush=True)
        results.append(res)
        if res.ok:
            idx += 1                                                  # only kept demos consume a number (gap-free)

    n_this = sum(r.ok for r in results)                  # collected THIS run
    n_total = n_existing + n_this                        # total kept on disk (incl. prior runs)
    elapsed = time.time() - t0
    summary = {"source_task": src_task, "task": task_name, "target_key": target_key,
               "target": target, "n_success": n_total, "n_collected_this_run": n_this,
               "n_attempts": n_att, "next_draw": compute_next_draw(last_run_draw, start_k),
               "reached_target": (target is None or n_total >= target),
               "elapsed_s": round(elapsed, 1), "dataset": dataset}
    (out_base / "_summary.json").write_text(json.dumps(summary, indent=2))
    status = "REACHED" if (target is None or n_total >= target) else "UNDER-TARGET"
    print(f"[driver] DONE {status} {n_total}/{target or '∞'} kept ({n_this} this run, "
          f"{n_att} attempts) in {elapsed / 60:.1f} min -> {out_base}", flush=True)
    try:
        og.sim.stop()
    except Exception:  # noqa: BLE001
        pass
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--family", default="clutter")
    ap.add_argument("--dataset", default="demos")
    ap.add_argument("--grasp-ids", type=int, nargs="*", default=None)
    ap.add_argument("--n-per-grasp", type=int, default=1, help="variants per grasp (jittered)")
    ap.add_argument("--target", type=int, default=None, help="collect until N success+safe demos")
    ap.add_argument("--max-attempts", type=int, default=None, help="attempt cap (default target*4)")
    ap.add_argument("--score", action="store_true", help="cuRobo-score + rank grasps first")
    ap.add_argument("--limit-demos", type=int, default=None)
    ap.add_argument("--steps-per-waypoint", type=int, default=2)
    ap.add_argument("--grasping-mode", choices=["physical", "assisted", "sticky"], default=None,
                    help="override the family-default AG mode (e.g. force 'sticky' for a target that "
                         "is un-graspable by force closure); default None = use the family default")
    ap.add_argument("--start-draw", type=int, default=None,
                    help="force the resume draw cursor (overrides the summary's next_draw); use to "
                         "recollect a deduped task with guaranteed-fresh seeds")
    ap.add_argument("--horizon-override", default=None,
                    help="JSON table of task-horizon variants (e.g. configs/firsthalf/"
                         "cabinet_task0019.json): substitutes this task's goal_conditions + prompt "
                         "so a truncated-horizon demo is gated and captioned as its own task. "
                         "Omit for the shipped full-horizon task.")
    a = ap.parse_args()
    run_task(a.task_dir, family=a.family, dataset=a.dataset, grasp_ids=a.grasp_ids,
             n_per_grasp=a.n_per_grasp, target=a.target, max_attempts=a.max_attempts,
             score=a.score, steps_per_waypoint=a.steps_per_waypoint, limit_demos=a.limit_demos,
             grasping_mode=a.grasping_mode, start_draw=a.start_draw,
             horizon_override=a.horizon_override)
    # All summaries and trajectory files are written before process exit. Bypass
    # torch/Isaac extension teardown, which can segfault during interpreter shutdown.
    import os
    os._exit(0)
