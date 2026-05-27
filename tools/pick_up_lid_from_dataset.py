"""Pick up a lid and place it onto a container via a variant transport.

Counterpart to ``tools/pick_and_place_from_dataset.py``, but for the
lid_transport task family. The pipeline is:

  Phase A — search for a graspable pose on the lid (cuRobo + OBB sampler,
            or a single SFT-prior candidate via --phase-a-grasp-from-dataset)
  Phase A replay — kinematic replay of the approach + AG re-engage,
                   captured into the SFT recorder
  Phase B — variant loop: sample (lift_extra_z, hover_clearance), plan +
            execute lid -> above container via _plan_transport(skip_descend=True),
            open gripper in place, settle for ``--post-release-settle`` steps
            while polling LidSnapper, declare success when AttachedTo fires.

Each successful variant commits one LeRobot v2.1 episode + thin HDF5 +
result.json. The LTL monitor and Phase A cache machinery from pnp
(variant 0 captures the frames, variants 1..N-1 restore the post-Phase-A
sim state and inject the cached frames) work the same way here.

Output layout under ``--out-dir``::

    variant_NN/
        result.json
        rollout.hdf5         (thin: state, action, sim_states, datagen_info)
        rollout_image_*.mp4  (symlinks into LeRobot videos/ when --lerobot-* set)
        trajectory.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task-dir", type=Path, required=True,
                   help="Task folder (must contain scene_ep<N>.json + diagnostics.jsonl)")
    p.add_argument("--episode", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override output dir (default: <task-dir>/pick_up_lid)")
    p.add_argument("--gui", action="store_true")

    # Phase A
    p.add_argument("--max-candidates", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pick-timeout", type=float, default=180.0)
    p.add_argument("--max-reach", type=float, default=0.95)
    p.add_argument("--max-obj-to-eef-after-hold", type=float, default=0.20)
    p.add_argument("--transport-timeout", type=float, default=60.0)
    p.add_argument("--ik-precheck", action="store_true")
    p.add_argument("--phase-a-grasp-from-dataset", default=None,
                   help="If set, load a single grasp candidate from a prior "
                        "SFT dataset matching this task, skipping the OBB "
                        "sampler. Falls back to OBB on miss.")

    # SFT
    p.add_argument("--record-sft", action="store_true",
                   help="Capture state/action/MP4 frames via tools._sft_recorder.")
    p.add_argument("--record-resolution", type=int, default=256)
    p.add_argument("--video-fps", type=int, default=30)

    # Variant loop
    p.add_argument("--n-transport-variants", type=int, default=1,
                   help="Number of (lift_extra, hover_clearance) samples per "
                        "(task, seed). Variant 0 runs Phase A search + replay "
                        "and caches the frames; variants 1..N-1 restore post-"
                        "Phase-A sim state and inject the cached frames.")
    p.add_argument("--lift-extra-z-min", type=float, default=0.00,
                   help="Min ADDITIONAL z-lift above the hover height (m). "
                        "0 means lift goes directly to hover height; positive "
                        "values overshoot then come down to hover.")
    p.add_argument("--lift-extra-z-max", type=float, default=0.10)
    p.add_argument("--hover-clearance-base", type=float, default=0.10,
                   help="Fixed baseline clearance above the container's world "
                        "AABB top (m) — matches lid_transport_from_dataset's "
                        "hardcoded 0.10. The variant-sampled clearance is "
                        "added on top.")
    p.add_argument("--hover-clearance-min", type=float, default=0.00,
                   help="Min ADDITIONAL clearance (m) above the baseline. "
                        "Effective clearance = base + sampled.")
    p.add_argument("--hover-clearance-max", type=float, default=0.10)
    p.add_argument("--post-release-settle", type=int, default=60,
                   help="Sim steps to settle physics after opening the "
                        "gripper; LidSnapper.try_snap is invoked on every step.")
    p.add_argument("--release-retreat-z", type=float, default=0.10,
                   help="Vertical retreat (m) after a successful snap-attach.")

    # Lid placement
    p.add_argument("--lid-at-edge", type=float, default=None, metavar="OVERHANG_FRAC",
                   help="Move the lid to the near edge of the support before "
                        "Phase A. OVERHANG_FRAC in [0,1] = fraction of the "
                        "lid's local half-extent past the edge.")
    p.add_argument("--lid-edge-pause", type=float, default=0.0,
                   help="Seconds to idle after placing the lid at the edge.")
    p.add_argument("--skip-rotate", action="store_true",
                   help="Drop the Phase B 'rotate' segment that aligns the "
                        "lid's M-link with the container's F-link. "
                        "LidSnapper teleport-aligns at contact, so the "
                        "release pose orientation only has to land on the "
                        "container, not match the F-link. Useful when "
                        "rotate is cuRobo-infeasible (e.g. wide --lid-at-edge).")
    p.add_argument("--lid-mass", type=float, default=None,
                   help="If set, override the lid's root-link mass (kg) "
                        "after env build. Workaround for JointController "
                        "PD under-tracking under held-object payload. "
                        "Set to e.g. 0.001 to confirm tracking with a "
                        "near-massless lid.")
    p.add_argument("--lift-z-rel-cap", type=float, default=None,
                   help="If set, hard-cap the Phase B lift_z_rel value "
                        "to this many meters above the post-grasp lid Z. "
                        "Useful when the unconstrained lift (which "
                        "raises the lid to clear the container top) "
                        "demands an IK solution outside the post-grasp "
                        "arm's joint manifold. Tradeoff: a small cap may "
                        "make the subsequent hover translation collide "
                        "with the container.")
    p.add_argument("--max-held-candidates", type=int, default=1,
                   help="Number of HELD grasp candidates Phase A should "
                        "collect. When > 1, if Phase B variants fail for "
                        "the first grasp, the tool restores pre-Phase-A "
                        "state and tries the next held grasp. Default 1 "
                        "preserves legacy behavior.")

    # LeRobot live-write
    p.add_argument("--lerobot-repo-id", default=None,
                   help="If set, write each successful variant as a LeRobot "
                        "v2.1 episode at --lerobot-root.")
    p.add_argument("--lerobot-root", default=None)
    p.add_argument("--lerobot-prompt-template",
                   default="pick up the lid and place it on the {container_clean}",
                   help="Substitutions: {target} (lid name), {target_clean} "
                        "(lid name suffix stripped), {container} (raw), "
                        "{container_clean} (suffix stripped).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-variant runner
# ---------------------------------------------------------------------------


def _run_one_variant(
    env, og, robot, lid, container, snapper, primitives, *,
    home_joint_q, lid_init_pos, lid_init_quat, approach_traj, grasp,
    phase_a_wall_s, phase_a_breakdown,
    args, task_dir, out_dir, camera_names,
    variant_idx, n_variants,
    lift_extra, hover_clearance,
    lerobot_dataset=None,
    ltl_monitor=None,
    phase_a_cache=None,
    cache_phase_a=False,
) -> dict:
    """Phase A replay (or cache injection) + Phase B (lift -> hover ->
    release + snap). Returns a result_payload dict."""
    import torch as th

    from tools.pick_and_place_from_dataset import (
        _plan_transport, _replay_holding, _record_phase_a_replay,
    )
    from tools.lid_transport_from_dataset import (
        _lid_goal_world_pose, _open_gripper_in_place, _retreat_eef_upward,
    )
    from omnigibson.object_states import AttachedTo

    sft_recorder = None
    lerobot_writer = None
    if args.record_sft:
        from tools._sft_recorder import SFTRecorder
        if lerobot_dataset is not None:
            from sentinel.data.lerobot_writer import (
                LeRobotEpisodeWriter, episode_prompt,
            )
            lerobot_writer = LeRobotEpisodeWriter(lerobot_dataset)
            # Templated with {container_clean} so the prompt mentions the
            # actual container the agent is closing.
            from sentinel.data.lerobot_writer import clean_target
            prompt = args.lerobot_prompt_template.format(
                target=lid.name, target_clean=clean_target(lid.name),
                container=container.name,
                container_clean=clean_target(container.name),
            )
        else:
            prompt = None
        sft_recorder = SFTRecorder(
            out_dir, resolution=args.record_resolution, fps=args.video_fps,
            lerobot_writer=lerobot_writer, lerobot_prompt=prompt,
            ltl_monitor=ltl_monitor,
        )
        sft_recorder.attach(env, env.robots[0])

    result_payload: dict = {
        "task_dir": str(task_dir),
        "lid_name": lid.name,
        "container_name": container.name,
        "variant_idx": variant_idx,
        "n_variants": n_variants,
        "lift_extra_z": lift_extra,
        "hover_clearance": hover_clearance,
        "phase_a": {
            "held": True,
            "wall_s": round(phase_a_wall_s, 2),
            "obj_to_eef": float(grasp["obj_to_eef"]),
            "breakdown": phase_a_breakdown,
        },
        "phase_b": {"planned": False, "executed": False, "success": False},
        "phase_c": {"attached": False, "snap_step": None},
    }

    # Helper: dump a sim-state snapshot to ``<out_dir>/<name>.npy``.
    # Downstream tools can ``np.load`` the file and call
    # ``og.sim.load_state(th.as_tensor(arr), serialized=True)`` to
    # restore the scene at a known phase boundary, then run variants
    # from there without re-running earlier phases.
    def _save_phase_snapshot(name: str) -> None:
        try:
            st = og.sim.dump_state(serialized=True)
            arr = (st.cpu().numpy() if hasattr(st, "cpu")
                   else np.asarray(st))
            path = out_dir / f"{name}.npy"
            np.save(path, arr)
            print(f"[Lid v{variant_idx:02d}] saved {name} state -> {path} "
                  f"(shape={arr.shape})", flush=True)
        except Exception as exc:
            print(f"[Lid v{variant_idx:02d}] {name} snapshot raised "
                  f"{type(exc).__name__}: {exc}", flush=True)

    try:
        # Phase A end: the variant is called from main() AFTER the
        # per-grasp re-engage replay has put the sim into the held
        # configuration. Snapshot here so downstream can start at
        # "lid held, arm in grasp pose."
        _save_phase_snapshot("phase_a_end")
        # ----- SFT Phase A capture / replay-from-cache -----
        if sft_recorder is not None:
            if phase_a_cache is None:
                if cache_phase_a:
                    sft_recorder.start_phase_a_cache()
                _record_phase_a_replay(
                    env, og, robot, lid,
                    home_joint_q=home_joint_q,
                    target_init_pos=lid_init_pos,
                    target_init_quat=lid_init_quat,
                    approach_traj=approach_traj,
                    sft_recorder=sft_recorder,
                    deadline=time.time() + args.pick_timeout,
                )
                if cache_phase_a:
                    captured = sft_recorder.end_phase_a_cache()
                    result_payload["_captured_phase_a"] = captured
                    result_payload["_post_phase_a_state"] = og.sim.dump_state()
            else:
                for frame in phase_a_cache:
                    sft_recorder.append_cached_step(frame)
                print(f"[Lid v{variant_idx:02d}] Phase A: replayed "
                      f"{len(phase_a_cache)} cached frames (no physics)",
                      flush=True)

        # ----- Phase B PLAN -----
        # Goal pose = lid root pose aligned so its M-link sits on the
        # container's F-link, plus an additional hover_clearance in z.
        lid_goal_xyz, lid_goal_quat = _lid_goal_world_pose(
            lid, container, clearance_z=hover_clearance,
        )
        _, c_hi_t = container.aabb
        c_hi_z = float(c_hi_t.cpu()[2] if hasattr(c_hi_t, "cpu") else c_hi_t[2])
        lid_now_pos_t = lid.get_position_orientation()[0]
        lid_now_z = float(lid_now_pos_t.cpu()[2]
                          if hasattr(lid_now_pos_t, "cpu") else lid_now_pos_t[2])
        # Effective clearance = fixed baseline (--hover-clearance-base,
        # default 0.10m, matching lid_transport's original hover height)
        # PLUS the variant-sampled extra. Below ~0.10m the lid release
        # drop is too short for it to seat onto the container.
        effective_clearance = float(args.hover_clearance_base) + float(hover_clearance)
        hover_z_rel = (c_hi_z + effective_clearance) - lid_now_z
        # Lift target z = hover height + lift_extra. _plan_transport's lift
        # waypoint takes the lid up to lift_z, hover to (target_goal at lift_z),
        # then descend — but skip_descend=True means we stop at hover height.
        lift_z_rel = hover_z_rel + float(lift_extra)
        if args.lift_z_rel_cap is not None:
            old_lift = lift_z_rel
            lift_z_rel = min(lift_z_rel, float(args.lift_z_rel_cap))
            if lift_z_rel < old_lift:
                print(f"[Lid v{variant_idx:02d}] lift_z_rel capped: "
                      f"{old_lift:.3f} -> {lift_z_rel:.3f}", flush=True)
        print(f"[Lid v{variant_idx:02d}] Phase B PLAN: "
              f"lift_extra_z={lift_extra:.3f}  hover_extra={hover_clearance:.3f}  "
              f"(effective_clearance={effective_clearance:.3f}  "
              f"lift_z_rel={lift_z_rel:+.3f}  hover_z_rel={hover_z_rel:+.3f})",
              flush=True)
        result_payload["phase_b"]["lift_z_rel"] = round(lift_z_rel, 4)
        result_payload["phase_b"]["hover_z_rel"] = round(hover_z_rel, 4)

        t_plan = time.time()
        seg_timings: list = []
        seg_pairs = _plan_transport(
            primitives, robot, lid, lid_goal_xyz,
            transport_timeout=args.transport_timeout,
            lift_z=lift_z_rel,
            hover_z=hover_z_rel,
            goal_target_quat_world=lid_goal_quat,
            skip_descend=True,
            skip_rotate=bool(args.skip_rotate),
            seg_timings=seg_timings,
        )
        plan_wall = time.time() - t_plan
        result_payload["phase_b"]["plan_wall_s"] = round(plan_wall, 2)
        result_payload["phase_b"]["plan_segments"] = seg_timings
        if seg_pairs is None or len(seg_pairs) == 0:
            failing = next((s["label"] for s in seg_timings if not s["ok"]), "?")
            result_payload["fail_step"] = f"phase_b_plan:{failing}"
            print(f"[Lid v{variant_idx:02d}] Phase B PLAN FAILED at {failing}",
                  flush=True)
            return result_payload
        result_payload["phase_b"]["planned"] = True

        # ----- Phase B EXECUTE -----
        transport_arm_traj = th.cat([arm for _, arm, _ in seg_pairs], dim=0)
        replay_deadline = time.time() + max(args.transport_timeout * 4, 240.0)
        t_exec = time.time()
        ok_replay = _replay_holding(
            env, og, robot, lid, transport_arm_traj,
            deadline=replay_deadline,
            sft_recorder=sft_recorder,
            segment_breakdown=[(lbl, len(arm_t))
                               for lbl, arm_t, _ in seg_pairs],
        )
        exec_wall = time.time() - t_exec
        result_payload["phase_b"]["execute_wall_s"] = round(exec_wall, 2)
        result_payload["phase_b"]["executed"] = bool(ok_replay)
        if not ok_replay:
            result_payload.setdefault("fail_step", "phase_b_execute")
        else:
            # Phase B end: lid is hovering above the container with the
            # gripper still closed (Phase C hasn't released yet). This
            # is the right entry-point for "vary the release/snap stage."
            _save_phase_snapshot("phase_b_end")

        # ----- Phase C — open + retreat + settle/snap -----
        # The gripper must be OUT of the lid's drop path before the snap
        # window: otherwise the falling lid bounces off the open fingers
        # and never seats onto the container. Order:
        #   1. open gripper in place (6 steps)
        #   2. retreat the eef vertically (gripper clears the lid)
        #   3. settle window — lid drops freely, LidSnapper polls each step
        print(f"[Lid v{variant_idx:02d}] Phase C: open + retreat + settle/snap",
              flush=True)
        _open_gripper_in_place(env, robot, sft_recorder=sft_recorder, n_steps=6)
        _retreat_eef_upward(
            env, og, robot, primitives=primitives,
            sft_recorder=sft_recorder,
            retreat_z=args.release_retreat_z,
        )
        attached = False
        snap_step = None
        settle_n = int(args.post_release_settle)
        for i in range(settle_n):
            og.sim.step()
            try:
                snapper.try_snap(robot=robot, verbose=False)
            except Exception as e:  # noqa: BLE001
                print(f"[Lid v{variant_idx:02d}] snap try {i}: {e}", flush=True)
            if lid.states[AttachedTo].get_value(container):
                attached = True
                snap_step = i + 1
                print(f"[Lid v{variant_idx:02d}] ATTACHED at step {snap_step}",
                      flush=True)
                break
        result_payload["phase_c"]["attached"] = bool(attached)
        result_payload["phase_c"]["snap_step"] = snap_step

        if not attached:
            result_payload.setdefault("fail_step", "phase_c_no_attach")
            print(f"[Lid v{variant_idx:02d}] LidSnapper did not attach within "
                  f"{settle_n} settle steps", flush=True)

        # Success = AttachedTo fired.
        result_payload["phase_b"]["success"] = bool(attached)
        print(f"[Lid v{variant_idx:02d}] → "
              f"{'SUCCESS' if attached else 'MISS'}", flush=True)
        # Phase C end: the lid has been released (and ideally snapped
        # onto the container). Useful as a final-state checkpoint for
        # downstream stages that pick up from a placed lid.
        _save_phase_snapshot("phase_c_end")
    finally:
        if sft_recorder is not None:
            sft_recorder.finalize(success=bool(result_payload["phase_b"]["success"]),
                                  attrs={
                "task_dir": str(task_dir),
                "lid_name": lid.name,
                "container_name": container.name,
                "seed": int(args.seed),
                "variant_idx": int(variant_idx),
                "lift_extra_z": float(lift_extra),
                "hover_clearance": float(hover_clearance),
                "phase_a_held": True,
                "phase_c_attached": bool(result_payload["phase_c"]["attached"]),
            })
        if ltl_monitor is not None:
            s = ltl_monitor.summary()
            result_payload["ltl"] = {
                "formula": s["formula"],
                "violated": bool(s["violated"]),
                "violation_step": s["violation_step"],
                "violation_count": int(s["violation_count"]),
                "total_steps_monitored": int(s["total_steps_monitored"]),
            }
        (out_dir / "result.json").write_text(json.dumps(
            {k: v for k, v in result_payload.items() if not k.startswith("_")},
            indent=2,
        ))
    return result_payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    task_dir = args.task_dir.resolve()
    if not task_dir.is_dir():
        raise SystemExit(f"task-dir not found: {task_dir}")

    # Default to outputs/pick_up_lid/<task-name>/ — keeps runtime
    # artifacts out of the dataset tree per project convention.
    # task_dir typically looks like .../<task-name>/base, so use the
    # parent's name as the per-task subdirectory label.
    default_out = (Path("outputs/pick_up_lid")
                   / (task_dir.parent.name if task_dir.name == "base"
                      else task_dir.name))
    out_dir = (args.out_dir or default_out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch as th  # noqa: F401 (used by per-grasp re-engage loop)
    from tools.pick_and_place_from_dataset import (
        _init_omnigibson, _phase_a_pick,
    )
    from tools.replay_empty_from_dataset import _load_diagnostics_row
    from tools.lid_transport_from_dataset import _identify_lid_container

    og = _init_omnigibson(headless=not args.gui)

    if args.record_sft:
        from tools._sft_recorder import install_wrist_camera_patch
        install_wrist_camera_patch()

    # Load diagnostics + scene info to resolve lid + container names.
    diagnostics = _load_diagnostics_row(task_dir, args.episode)
    scene_p = (task_dir / f"scene_ep{args.episode}_replay.json"
               if (task_dir / f"scene_ep{args.episode}_replay.json").is_file()
               else task_dir / f"scene_ep{args.episode}.json")
    with open(scene_p, "r") as fh:
        scene_info = json.load(fh)
    lid_name, container_name = _identify_lid_container(scene_info, diagnostics)

    # Build env (mimics pnp's _build_env for the lid task — uses the same
    # scene loader; the lid_transport pipeline's distractor handling isn't
    # needed here because we never drop task objects). We reuse pnp's helper
    # by tricking it: pnp's _build_env will set target=goal_region's
    # target_name, which for lid_transport is the container. We then override.
    from tools.pick_and_place_from_dataset import _build_env
    env, og, _diagnostics, camera_names = _build_env(
        task_dir, args.episode, no_distractors=False,
        record_sft=args.record_sft, record_resolution=args.record_resolution,
    )

    lerobot_dataset = None
    if args.lerobot_repo_id:
        if not args.record_sft:
            raise SystemExit("--lerobot-repo-id requires --record-sft")
        from sentinel.data.lerobot_writer import create_or_open_dataset
        lerobot_dataset = create_or_open_dataset(
            repo_id=args.lerobot_repo_id, root=args.lerobot_root,
            fps=args.video_fps, resolution=args.record_resolution,
        )
        print(f"[Lid] LeRobot dataset opened at {lerobot_dataset.root}  "
              f"(starting from episode {lerobot_dataset.meta.total_episodes})",
              flush=True)

    # LTL monitor (auto-attach when the task ships ltl_safety).
    ltl_monitor = None
    ltl_safety_spec = diagnostics.get("ltl_safety") or {}
    if args.record_sft and ltl_safety_spec:
        try:
            from sentinel.utils.safety_monitor import TaskLTLMonitor
            cat_to_synset = {
                spec["category"]: spec["synset"].split(".")[0]
                for spec in diagnostics.get("selection", {}).get("spawn_specs", [])
            }
            active_by_inst = {}
            synset_counts = {}
            for obj in env.scene.objects:
                cat = getattr(obj, "category", None)
                short = cat_to_synset.get(cat)
                if short is None:
                    continue
                idx = synset_counts.get(short, 0)
                active_by_inst[f"{short}_{idx}"] = obj
                synset_counts[short] = idx + 1
            ltl_monitor = TaskLTLMonitor(
                env, ltl_safety=ltl_safety_spec,
                activity_name=diagnostics.get("activity_name", ""),
                scene_model=diagnostics.get("scene_model"),
                active_objects_by_inst=active_by_inst,
            )
            print(f"[Lid] LTL monitor attached "
                  f"(active_objects: {sorted(active_by_inst.keys())})",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[Lid] LTL monitor init failed ({e}); continuing without LTL",
                  flush=True)
            ltl_monitor = None

    # Resolve lid + container in the loaded scene.
    lid = env.scene.object_registry("name", lid_name)
    container = env.scene.object_registry("name", container_name)
    if lid is None or container is None:
        raise RuntimeError(
            f"missing lid={lid_name!r} or container={container_name!r}")

    if args.lid_mass is not None:
        try:
            old_mass = float(lid.root_link.mass)
            lid.root_link.mass = float(args.lid_mass)
            print(f"[Lid] mass override: {old_mass:.3f} -> "
                  f"{float(args.lid_mass)} kg", flush=True)
        except Exception as exc:
            print(f"[Lid] mass override raised "
                  f"{type(exc).__name__}: {exc}", flush=True)

    # Optional: position the lid at the table edge for the edge-test scenario.
    if args.lid_at_edge is not None:
        _place_lid_at_edge(env, lid, container, diagnostics, args)

    # LidSnapper — discovers (lid, container) pair via meta-links.
    from sentinel.utils.lid_attach import LidSnapper
    snapper = LidSnapper(env)

    # cuRobo primitives.
    from omnigibson.action_primitives.starter_semantic_action_primitives import (
        StarterSemanticActionPrimitives,
    )
    primitives = StarterSemanticActionPrimitives(env, env.robots[0],
                                                 enable_head_tracking=False)

    # Snapshot pre-Phase-A state for variant restoration when no cache.
    pre_phase_a_state = og.sim.dump_state()

    # Phase A search — runs once.
    robot = env.robots[0]
    home_joint_q = robot.get_joint_positions().clone()
    lid_init_pos, lid_init_quat = lid.get_position_orientation()
    lid_init_pos = lid_init_pos.clone()
    lid_init_quat = lid_init_quat.clone()

    # Snapshot pre-Phase-A scene so we can restore between grasp
    # candidates (and between variants of a given grasp).
    pre_phase_a_state = og.sim.dump_state()

    t0 = time.time()
    pick_deadline = t0 + args.pick_timeout
    held_grasps, phase_a_timings = _phase_a_pick(
        env, og, primitives, lid, args, pick_deadline,
    )
    phase_a_wall = time.time() - t0
    n_cands = len(phase_a_timings)
    n_held = sum(1 for t in phase_a_timings if t.get("held"))
    phase_a_breakdown = {"n_cands_tried": n_cands, "n_held": n_held}
    print(f"[Lid] Phase A wall={phase_a_wall:.1f}s  "
          f"cands_tried={n_cands}  held={n_held}", flush=True)

    if not held_grasps:
        print("[Lid] Phase A FAILED — no graspable lid", flush=True)
        (out_dir / "result.json").write_text(json.dumps({
            "task_dir": str(task_dir),
            "lid_name": lid_name,
            "container_name": container_name,
            "phase_a": {"held": False, "wall_s": round(phase_a_wall, 2),
                        "breakdown": phase_a_breakdown},
            "fail_step": "phase_a",
        }, indent=2))
        return

    # ===== Per-grasp fallback loop =====
    # For each HELD candidate, run the full variant loop. If all variants
    # fail for the current grasp (e.g. Phase B's lift segment IK_FAILs
    # because the post-grasp configuration is near-singular), restore the
    # pre-Phase-A scene and try the next grasp candidate. Breaks out the
    # outer loop on the first grasp that produces at least one successful
    # variant.
    n_variants = max(1, args.n_transport_variants)
    rng = np.random.default_rng(args.seed)
    variant_summary: list = []
    n_grasps = len(held_grasps)
    print(f"[Lid] Phase A held {n_grasps} candidate(s); will try each in "
          f"sequence until one produces a successful Phase B variant.",
          flush=True)

    for gi, grasp in enumerate(held_grasps):
        approach_traj = grasp["approach_traj"]
        print(f"\n[Lid] ====== GRASP CANDIDATE {gi+1}/{n_grasps} ======",
              flush=True)
        # Always restore pre-Phase-A state + physically replay the
        # approach for THIS grasp. Required because Phase A's
        # collect_valid_grasps leaves the scene holding only the LAST
        # successful candidate (not held_grasps[0]) when collecting
        # multiple, so even gi=0 needs an explicit re-grasp to land in
        # the right held state. Variant 0 of each grasp then enters
        # _run_one_variant from the correct "held by THIS grasp"
        # configuration.
        print(f"[Lid] Restoring pre-Phase-A state for grasp #{gi+1} "
              f"and replaying its approach trajectory", flush=True)
        og.sim.load_state(pre_phase_a_state)
        og.sim.step()
        from sentinel.rl.grasps.collector import (
            run_grasp_attempt, GraspCollectorConfig,
        )
        _arm = robot.default_arm
        _arm_ctrl_idx = robot.arm_control_idx[_arm]
        _gripper_ctrl_idx = robot.gripper_control_idx[_arm]
        _open_q = robot.joint_upper_limits[_gripper_ctrl_idx].clone()
        _zero_arm_cmd = th.zeros(
            len(robot.arm_action_idx[_arm]), dtype=th.float32)
        _cfg_r = GraspCollectorConfig(num_target_grasps=1)
        _t_deadline = time.time() + 90.0
        _result = run_grasp_attempt(
            env, robot, lid, lid_init_pos, lid_init_quat,
            joint_traj=th.as_tensor(approach_traj, dtype=th.float32),
            cfg=_cfg_r,
            open_gripper_q=_open_q,
            zero_arm_cmd=_zero_arm_cmd,
            arm_control_idx=_arm_ctrl_idx,
            gripper_control_idx=_gripper_ctrl_idx,
            initial_joint_pos=home_joint_q,
            deadline=_t_deadline,
        )
        if _result is None:
            print(f"[Lid] Grasp #{gi+1} re-engage FAILED at approach "
                  f"replay; skipping to next candidate.", flush=True)
            variant_summary.append({
                "grasp_idx": gi,
                "variant_idx": None,
                "success": False,
                "fail_step": "approach_replay_no_ag",
            })
            continue
        # Diagnostic: print eef pose immediately after re-engage. If
        # this shows the HELD pose (~z=0.8 above the lid) but the
        # subsequent _plan_transport lift override reads ~z=1.5, then
        # something between this point and _plan_transport resets the
        # joints. If this also shows ~z=1.5, then run_grasp_attempt's
        # "result" is reporting a held config that doesn't reflect
        # the actual sim state.
        _re_pos, _re_quat = robot.eef_links[_arm].get_position_orientation()
        print(f"[Lid] Grasp #{gi+1} re-engaged AG; post-re-engage "
              f"eef pos={_re_pos.cpu().numpy().tolist()} "
              f"quat={_re_quat.cpu().numpy().tolist()}",
              flush=True)

        # Fresh per-grasp variant caches.
        phase_a_cache: list | None = None
        post_phase_a_state = None
        grasp_succeeded_any = False
        for vi in range(n_variants):
            if n_variants == 1:
                v_out_dir = (out_dir if n_grasps == 1
                             else out_dir / f"grasp_{gi:02d}")
                v_out_dir.mkdir(parents=True, exist_ok=True)
                lift_extra = 0.0
                hover_clearance = 0.10
            else:
                v_subdir = (f"variant_{vi:02d}" if n_grasps == 1
                            else f"grasp_{gi:02d}/variant_{vi:02d}")
                v_out_dir = out_dir / v_subdir
                v_out_dir.mkdir(parents=True, exist_ok=True)
                lift_extra = float(rng.uniform(args.lift_extra_z_min,
                                               args.lift_extra_z_max))
                hover_clearance = float(rng.uniform(args.hover_clearance_min,
                                                    args.hover_clearance_max))

            if vi > 0:
                if post_phase_a_state is not None and phase_a_cache is not None:
                    og.sim.load_state(post_phase_a_state)
                    og.sim.step()
                    if ltl_monitor is not None and ltl_monitor._monitor is not None:
                        ltl_monitor.reset()
                        for f in phase_a_cache:
                            ap = f.get("ap_labels")
                            if ap:
                                ltl_monitor._monitor.step(ap)
                else:
                    og.sim.load_state(pre_phase_a_state)
                    og.sim.step()

            print(f"\n[Lid] === Grasp {gi+1}/{n_grasps}  "
                  f"Variant {vi+1}/{n_variants}  "
                  f"lift_extra={lift_extra:.3f}  hover_clearance={hover_clearance:.3f}  "
                  f"out_dir={v_out_dir.name} ===", flush=True)
            v_result = _run_one_variant(
                env, og, robot, lid, container, snapper, primitives,
                home_joint_q=home_joint_q,
                lid_init_pos=lid_init_pos, lid_init_quat=lid_init_quat,
                approach_traj=approach_traj, grasp=grasp,
                phase_a_wall_s=phase_a_wall,
                phase_a_breakdown=phase_a_breakdown,
                args=args, task_dir=task_dir, out_dir=v_out_dir,
                camera_names=camera_names,
                variant_idx=vi, n_variants=n_variants,
                lift_extra=lift_extra, hover_clearance=hover_clearance,
                lerobot_dataset=lerobot_dataset,
                ltl_monitor=ltl_monitor,
                phase_a_cache=phase_a_cache,
                cache_phase_a=(vi == 0),
            )
            if vi == 0:
                phase_a_cache = v_result.pop("_captured_phase_a", None)
                post_phase_a_state = v_result.pop("_post_phase_a_state", None)
                if phase_a_cache is not None:
                    print(f"[Lid] Phase A cache captured for grasp #{gi+1}: "
                          f"{len(phase_a_cache)} frames "
                          f"(variants 1..{n_variants-1} skip physics replay)",
                          flush=True)
            succ = bool(v_result.get("phase_b", {}).get("success"))
            grasp_succeeded_any = grasp_succeeded_any or succ
            variant_summary.append({
                "grasp_idx": gi,
                "variant_idx": vi,
                "lift_extra_z": lift_extra,
                "hover_clearance": hover_clearance,
                "success": succ,
                "fail_step": v_result.get("fail_step"),
                "out_dir": str(v_out_dir),
            })

        if grasp_succeeded_any:
            print(f"\n[Lid] Grasp #{gi+1} produced ≥1 successful variant; "
                  f"stopping fallback search.", flush=True)
            break
        else:
            if gi + 1 < n_grasps:
                print(f"\n[Lid] Grasp #{gi+1} produced 0 successful variants; "
                      f"falling back to grasp #{gi+2}", flush=True)

    n_succ = sum(1 for v in variant_summary if v["success"])
    (out_dir / "variants_summary.json").write_text(json.dumps({
        "n_variants": n_variants, "n_succ": n_succ,
        "variants": variant_summary,
    }, indent=2))
    print(f"[Lid] variants: {n_succ}/{n_variants} succeeded; wrote "
          f"{out_dir / 'variants_summary.json'}", flush=True)


def _place_lid_at_edge(env, lid, container, diagnostics, args) -> None:
    """Move the lid to the near edge of the support surface (closest to
    the robot) with an OVERHANG_FRAC. Mirrors lid_transport_from_dataset's
    --lid-at-edge path exactly — uses the full robot-local/world transform
    so the lid lands correctly when the robot is rotated relative to
    world (which is the common case).
    """
    import torch as th
    import omnigibson as _og
    from sentinel.utils.goal_region import _local_xy_to_world, _world_to_local

    sb = diagnostics.get("goal_region", {}).get("support_bounds_robot_local_xy")
    if sb is None:
        raise RuntimeError("--lid-at-edge requires goal_region."
                           "support_bounds_robot_local_xy in diagnostics")
    (lx0, ly0), (lx1, ly1) = sb

    robot = env.robots[0]
    r_pos_t, r_quat_t = robot.get_position_orientation()
    r_pos = tuple(float(v) for v in (r_pos_t.cpu()
                  if hasattr(r_pos_t, "cpu") else r_pos_t))
    r_quat = tuple(float(v) for v in (r_quat_t.cpu()
                   if hasattr(r_quat_t, "cpu") else r_quat_t))

    # Build lid AABB in robot-local frame by transforming all 8 world
    # corners (world AABB is axis-aligned in world).
    l_lo_t, l_hi_t = lid.aabb
    l_lo = [float(v) for v in (l_lo_t.cpu()
            if hasattr(l_lo_t, "cpu") else l_lo_t)]
    l_hi = [float(v) for v in (l_hi_t.cpu()
            if hasattr(l_hi_t, "cpu") else l_hi_t)]
    xs_l, ys_l = [], []
    for x in (l_lo[0], l_hi[0]):
        for y in (l_lo[1], l_hi[1]):
            for z in (l_lo[2], l_hi[2]):
                lx, ly, _lz = _world_to_local(r_pos, r_quat, (x, y, z))
                xs_l.append(float(lx)); ys_l.append(float(ly))
    lid_local_xmin, lid_local_xmax = min(xs_l), max(xs_l)
    lid_local_ymin, lid_local_ymax = min(ys_l), max(ys_l)
    lid_local_half_x = 0.5 * (lid_local_xmax - lid_local_xmin)
    lid_local_cx = 0.5 * (lid_local_xmin + lid_local_xmax)
    lid_local_cy = 0.5 * (lid_local_ymin + lid_local_ymax)

    overhang = float(args.lid_at_edge)
    target_local_x = lx0 + (1.0 - 2.0 * overhang) * lid_local_half_x
    target_local_y = lid_local_cy  # keep current local-y

    # Convert (target_local - current_local) delta into a true world delta
    # via the same rotated transform used elsewhere in the pipeline.
    delta_local = (target_local_x - lid_local_cx,
                   target_local_y - lid_local_cy)
    wx_a, wy_a = _local_xy_to_world(r_pos, r_quat, 0.0, 0.0)
    wx_b, wy_b = _local_xy_to_world(r_pos, r_quat,
                                    delta_local[0], delta_local[1])
    dwx, dwy = wx_b - wx_a, wy_b - wy_a

    cur_pos_t, cur_quat_t = lid.get_position_orientation()
    cur_pos_np = np.asarray(cur_pos_t.cpu()
                            if hasattr(cur_pos_t, "cpu") else cur_pos_t,
                            dtype=np.float64)
    new_root = cur_pos_np.copy()
    new_root[0] += dwx
    new_root[1] += dwy

    print(f"[Lid] --lid-at-edge: support_bounds_local_xy="
          f"[({lx0:.3f},{ly0:.3f}),({lx1:.3f},{ly1:.3f})] "
          f"overhang_frac={overhang:.2f}  (near edge = lx0)", flush=True)
    print(f"[Lid]   lid local AABB: x=[{lid_local_xmin:.3f},"
          f"{lid_local_xmax:.3f}] y=[{lid_local_ymin:.3f},"
          f"{lid_local_ymax:.3f}] half_x={lid_local_half_x:.3f}",
          flush=True)
    print(f"[Lid]   target local center=({target_local_x:.3f},"
          f"{target_local_y:.3f}); world delta=({dwx:+.3f},{dwy:+.3f})",
          flush=True)
    print(f"[Lid]   lid root: {cur_pos_np.tolist()} -> "
          f"{new_root.tolist()}", flush=True)

    lid.set_position_orientation(
        position=th.as_tensor(new_root, dtype=th.float32),
        orientation=cur_quat_t,
    )
    lid.root_link.set_linear_velocity(th.zeros(3))
    lid.root_link.set_angular_velocity(th.zeros(3))
    for _ in range(30):
        _og.sim.step()

    # Verify the lid actually overhangs the placeable edge by the requested
    # amount. We re-read the lid AABB after settle, transform 8 corners to
    # robot-local, and compare lid_local_xmin to lx0.
    #
    # Geometry: target_local_x = lx0 + (1 - 2*overhang) * half_x, so the lid's
    # near edge should be at  lx0 - 2*overhang*half_x + half_x - half_x
    #                       = lx0 - 2*overhang*half_x + 0  (since half_x is
    # half-width and target is the center)... explicitly:
    #   lid_near_local_x  = target_local_x - half_x
    #                     = lx0 + (1 - 2*overhang) * half_x - half_x
    #                     = lx0 - 2*overhang*half_x
    #   expected_overhang = lx0 - lid_near_local_x = 2*overhang*half_x
    expected_overhang = 2.0 * overhang * lid_local_half_x
    l_lo_t2, l_hi_t2 = lid.aabb
    l_lo2 = [float(v) for v in (l_lo_t2.cpu()
             if hasattr(l_lo_t2, "cpu") else l_lo_t2)]
    l_hi2 = [float(v) for v in (l_hi_t2.cpu()
             if hasattr(l_hi_t2, "cpu") else l_hi_t2)]
    xs_l2, ys_l2 = [], []
    for x in (l_lo2[0], l_hi2[0]):
        for y in (l_lo2[1], l_hi2[1]):
            for z in (l_lo2[2], l_hi2[2]):
                lx, ly, _ = _world_to_local(r_pos, r_quat, (x, y, z))
                xs_l2.append(float(lx)); ys_l2.append(float(ly))
    lid_near_local_x = min(xs_l2)
    lid_far_local_x = max(xs_l2)
    actual_overhang = lx0 - lid_near_local_x

    # Place the placeable-edge in world frame so it can be cross-checked
    # against the viewer. The edge is the vertical line x=lx0 in robot-local;
    # plot the two endpoints at y=ly0 and y=ly1 transformed to world.
    edge_w_a = _local_xy_to_world(r_pos, r_quat, lx0, ly0)
    edge_w_b = _local_xy_to_world(r_pos, r_quat, lx0, ly1)

    print(f"[Lid]   placeable near-edge (lx0={lx0:.3f}) in world: "
          f"({edge_w_a[0]:.3f},{edge_w_a[1]:.3f}) -> "
          f"({edge_w_b[0]:.3f},{edge_w_b[1]:.3f})", flush=True)
    print(f"[Lid]   post-settle lid local-x=[{lid_near_local_x:.3f},"
          f"{lid_far_local_x:.3f}]  near-edge vs placeable: "
          f"lid_near - lx0 = {lid_near_local_x - lx0:+.3f} m",
          flush=True)
    overhang_ok = abs(actual_overhang - expected_overhang) < 0.005
    print(f"[Lid]   overhang check: expected={expected_overhang:+.3f} m "
          f"actual={actual_overhang:+.3f} m  diff="
          f"{actual_overhang - expected_overhang:+.4f} m  "
          f"-> {'OK' if overhang_ok else 'MISMATCH'}", flush=True)

    if args.lid_edge_pause > 0:
        import time as _t
        t_end = _t.time() + args.lid_edge_pause
        print(f"[Lid] --lid-edge-pause: idling {args.lid_edge_pause}s "
              "for visual inspection ...", flush=True)
        while _t.time() < t_end:
            _og.sim.step()


if __name__ == "__main__":
    main()
