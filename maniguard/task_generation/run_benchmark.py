#!/usr/bin/env python
"""Run scene-generation pipelines in separate subprocesses.

Enumerate eligible scenes, invoke the selected generator, and aggregate
reported artifacts and diagnostics under a run directory.

Examples:
    python -m maniguard.task_generation.run_benchmark --pipeline table --scenes Rs_int --steps 300
    python -m maniguard.task_generation.run_benchmark --pipeline transfer --no-strict-gate
    python -m maniguard.task_generation.run_benchmark --pipeline stack --stack-height medium"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

log = logging.getLogger(__name__)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
_DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs", "benchmark_runs")

_STACK_SCRIPT = os.path.join(_SCRIPT_DIR, "stack_scene_pipeline.py")

_PIPELINE_SCRIPTS = {
    "table": os.path.join(_SCRIPT_DIR, "clutter_scene_pipeline.py"),
    # Empty-scene cabinet pickup: cabinet placed on a generated surface,
    # target/obstacle in front of (or beside) the drawer's swept zone.
    "cabinet_pickup": os.path.join(_SCRIPT_DIR, "cabinet_pickup_pipeline.py"),
    "transfer": os.path.join(_SCRIPT_DIR, "transfer_scene_pipeline.py"),
    "dusty_transfer": os.path.join(_SCRIPT_DIR, "dusty_transfer_pipeline.py"),
    "stack": _STACK_SCRIPT,
    "stack_same": _STACK_SCRIPT,
    "stack_flat": _STACK_SCRIPT,
    "stack_receptacle": _STACK_SCRIPT,
    "lid_transport": os.path.join(_SCRIPT_DIR, "lid_transport_pipeline.py"),
    "jar_transport": os.path.join(_SCRIPT_DIR, "jar_transport_pipeline.py"),
    "wet_transport": os.path.join(_SCRIPT_DIR, "wet_transport_pipeline.py"),
    "liquid_transport": os.path.join(_SCRIPT_DIR, "liquid_transport_pipeline.py"),
}

# Scenes excluded per pipeline type.
_EXCLUDED_SCENES = {
    "table": frozenset({
        "Benevolence_0_int",         # bathroom only
        "grocery_store_convenience", # no table-like surface in sim
        "hall_arch_wood",            # public restroom
        "hall_train_station",        # train station restroom
        "school_gym",                # gymnasium, no tables
    }),
    # No cabinet-specific scene exclusions are configured in this mapping.
    "transfer": frozenset({
        "Benevolence_0_int",
        "grocery_store_convenience",
        "hall_arch_wood",
        "hall_train_station",
        "school_gym",
    }),
    # dusty_transfer inherits transfer's surface needs.
    "dusty_transfer": frozenset({
        "Benevolence_0_int",
        "grocery_store_convenience",
        "hall_arch_wood",
        "hall_train_station",
        "school_gym",
    }),
    "stack": frozenset({
        "Benevolence_0_int",
        "grocery_store_convenience",
        "hall_arch_wood",
        "hall_train_station",
        "school_gym",
    }),
    "stack_same": frozenset({
        "Benevolence_0_int",
        "grocery_store_convenience",
        "hall_arch_wood",
        "hall_train_station",
        "school_gym",
    }),
    "stack_flat": frozenset({
        "Benevolence_0_int",
        "grocery_store_convenience",
        "hall_arch_wood",
        "hall_train_station",
        "school_gym",
    }),
    "stack_receptacle": frozenset({
        "Benevolence_0_int",
        "grocery_store_convenience",
        "hall_arch_wood",
        "hall_train_station",
        "school_gym",
    }),
}


def _discover_scenes(scenes_dir, pipeline):
    """Return sorted list of scene model names, excluding unsuitable ones."""
    if not os.path.isdir(scenes_dir):
        print(f"[Benchmark] ERROR: scenes directory not found: {scenes_dir}")
        sys.exit(1)
    all_scenes = sorted(os.listdir(scenes_dir))
    excluded = _EXCLUDED_SCENES.get(pipeline, frozenset())
    eligible = [s for s in all_scenes if s not in excluded]
    return eligible


def _artifact_paths(run_dir, episodes, pipeline=None):
    """Required snapshot/video files in the scene or standalone episode layout."""
    paths = [os.path.join(run_dir, "diagnostics.jsonl")]
    nested = pipeline in ("cabinet_pickup", "jar_transport") or os.path.isdir(
        os.path.join(run_dir, "snapshots", "ep001")
    )
    for ep in range(1, episodes + 1):
        root = os.path.join(run_dir, "snapshots", f"ep{ep:03d}") if nested else run_dir
        paths.append(os.path.join(root, f"scene_ep{ep}.json"))
        views = [os.path.join(root, f"rollout_{view}_ep{ep}.mp4") for view in
                 ("opposite_side_front", "left_overview", "right_overview", "left_shoulder")]
        legacy = os.path.join(root, f"rollout_ep{ep}.mp4")
        paths.extend(views if any(os.path.isfile(v) for v in views) or not os.path.isfile(legacy) else [legacy])
    return paths


def _read_episode_diagnostics(run_dir, episodes):
    path = os.path.join(run_dir, "diagnostics.jsonl")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    # Generators append diagnostics when retrying in the same directory.
    rows = rows[-episodes:]
    if len(rows) != episodes or any(not isinstance(r, dict) for r in rows):
        raise ValueError(f"expected {episodes} episode records")
    ids = [r.get("episode") for r in rows]
    if any(type(ep) is not int for ep in ids) or sorted(ids) != list(range(1, episodes + 1)):
        raise ValueError("episode records are missing or duplicated")
    if any(type(r.get(key)) is not bool for r in rows for key in ("gate_pass", "ltl_violated")):
        raise ValueError("episode gate/safety verdicts must be Boolean")
    return rows


def _file_stamp(path):
    stat = os.stat(path)
    return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size


def _validate_scene_artifacts(run_dir, episodes, pipeline=None, previous_files=None):
    errors = []
    for path in _artifact_paths(run_dir, episodes, pipeline):
        label = os.path.relpath(path, run_dir)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            errors.append(label)
        elif previous_files is not None and previous_files.get(path) == _file_stamp(path):
            errors.append(f"unchanged artifact from earlier attempt: {label}")
    diag = os.path.join(run_dir, "diagnostics.jsonl")
    if os.path.isfile(diag):
        try:
            _read_episode_diagnostics(run_dir, episodes)
        except (OSError, ValueError) as exc:
            errors.append(f"invalid diagnostics.jsonl: {exc}")
    return errors


def _spot_preflight_or_exit():
    from maniguard.utils.ltl_utils import get_spot_runtime_status

    status = get_spot_runtime_status(require_buddy=True)
    print(f"[Benchmark] Python: {status['python_executable']}")
    print(f"[Benchmark] Spot module: {status['module_path']}")
    if not status["valid"]:
        print(f"[Benchmark] ERROR: {status['error']}")
        sys.exit(1)


def _run_scene(scene_model, args, output_dir, scene_index=0):
    """Run the pipeline on a single scene (or auto-select) in a subprocess."""
    label = scene_model or f"trial_{scene_index}"
    run_dir = os.path.join(output_dir, label)
    os.makedirs(run_dir, exist_ok=True)

    # Vary seed per scene/trial so each gets different randomization.
    scene_seed = args.seed + scene_index

    pipeline_script = _PIPELINE_SCRIPTS[args.pipeline]
    cmd = [
        sys.executable, "-m", "maniguard.task_generation." + os.path.basename(pipeline_script)[:-3],
        "--episodes", str(args.episodes),
        "--steps", str(args.steps),
        "--seed", str(scene_seed),
        "--run-dir", run_dir,
        "--save-video",
        "--video-fps", str(args.video_fps),
    ]
    standalone = args.pipeline in ("cabinet_pickup", "jar_transport")
    if not standalone:
        cmd.extend(["--mount-gap-m", str(args.mount_gap_m),
                    "--strict-gate" if args.strict_gate else "--no-strict-gate"])
    if scene_model and not standalone:
        cmd.extend(["--scene-model", scene_model])
    # Pipeline-specific flags.
    if args.pipeline == "table":
        cmd.extend(["--clutter-density", args.density])
        if args.randomize:
            cmd.append("--randomize")
    if args.pipeline == "transfer":
        if args.food_model:
            cmd.extend(["--food-model", args.food_model])
        if args.source_model:
            cmd.extend(["--source-model", args.source_model])
        if args.dest_model:
            cmd.extend(["--dest-model", args.dest_model])
        if args.goal_predicate:
            cmd.extend(["--goal-predicate", args.goal_predicate])
    if args.pipeline.startswith("stack"):
        # Derive --stack-mode from the pipeline name (stack_flat -> flat, etc.)
        if "_" in args.pipeline:
            stack_mode = args.pipeline.split("_", 1)[1]
        else:
            stack_mode = "same"
        cmd.extend(["--stack-mode", stack_mode])
        if args.stack_height:
            cmd.extend(["--stack-height", args.stack_height])
        if args.target_model:
            cmd.extend(["--target-model", args.target_model])
        if args.stack_model:
            cmd.extend(["--stack-model", args.stack_model])

    log_path = os.path.join(run_dir, "stdout.log")
    result = {
        "scene": label,
        "status": "unknown",
        "duration_s": 0,
        "gate_pass": False,
        "ltl_violated": None,
        "error": "",
        "run_dir": run_dir,
    }

    print(f"\n{'='*70}")
    print(f"[Benchmark] Starting: {label}")
    print(f"[Benchmark] Run dir:  {run_dir}")
    print(f"[Benchmark] Timeout:  {args.timeout}s")
    print(f"{'='*70}")

    previous_files = {
        os.path.join(root, name): _file_stamp(os.path.join(root, name))
        for root, _, names in os.walk(run_dir) for name in names
        if os.path.isfile(os.path.join(root, name))
    }
    t0 = time.time()
    try:
        with open(log_path, "w") as log_file:
            proc = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                cwd=_PROJECT_ROOT,
            )
        elapsed = time.time() - t0
        result["duration_s"] = round(elapsed, 1)

        missing_artifacts = _validate_scene_artifacts(
            run_dir, args.episodes, args.pipeline, previous_files,
        )

        if proc.returncode == 0 and not missing_artifacts:
            result["status"] = "success"
        elif proc.returncode == -11 and not missing_artifacts:
            # Accept freshly completed artifacts while retaining the exit error;
            # the return code alone does not establish where the crash occurred.
            result["status"] = "success"
            result["error"] = "exit -11 after complete, fresh artifacts"
        elif missing_artifacts:
            result["status"] = "failed"
            result["error"] = f"missing artifacts: {', '.join(missing_artifacts)}"
        else:
            result["status"] = "failed"
            result["error"] = f"exit code {proc.returncode}"

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        result["duration_s"] = round(elapsed, 1)
        result["status"] = "timeout"
        result["error"] = f"exceeded {args.timeout}s"

    except Exception as e:
        elapsed = time.time() - t0
        result["duration_s"] = round(elapsed, 1)
        result["status"] = "error"
        result["error"] = str(e)

    # Report every episode in this attempt, not just the final row.
    if result["status"] == "success":
        entries = _read_episode_diagnostics(run_dir, args.episodes)
        result["gate_pass"] = all(r["gate_pass"] for r in entries)
        result["ltl_violated"] = any(r["ltl_violated"] for r in entries)

    status_icon = {"success": "OK", "failed": "FAIL", "timeout": "TIME", "error": "ERR"}.get(
        result["status"], "?"
    )
    print(f"[Benchmark] {status_icon}: {scene_model} "
          f"({result['duration_s']}s, gate={result['gate_pass']}, ltl_violated={result['ltl_violated']})")

    return result


def _write_summary(results, output_dir):
    """Write CSV summary and print a table."""
    csv_path = os.path.join(output_dir, "summary.csv")
    fieldnames = ["scene", "status", "duration_s", "gate_pass", "ltl_violated", "error", "run_dir"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Print summary table.
    print(f"\n{'='*90}")
    print(f"  BENCHMARK SUMMARY — {len(results)} scenes")
    print(f"{'='*90}")
    print(f"  {'Scene':<40} {'Status':<10} {'Time(s)':<10} {'Gate':<8} {'LTL Viol.':<10}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
    for r in results:
        print(f"  {r['scene']:<40} {r['status']:<10} {r['duration_s']:<10} "
              f"{'pass' if r['gate_pass'] else 'fail':<8} {r['ltl_violated']!s:<10}")

    n_success = sum(1 for r in results if r["status"] == "success")
    n_gate = sum(1 for r in results if r["gate_pass"])
    n_timeout = sum(1 for r in results if r["status"] == "timeout")
    n_failed = sum(1 for r in results if r["status"] in ("failed", "error"))
    total_time = sum(r["duration_s"] for r in results)

    print(f"\n  Success: {n_success}/{len(results)}  |  Gate pass: {n_gate}/{len(results)}  |  "
          f"Timeout: {n_timeout}  |  Failed: {n_failed}  |  Total time: {total_time:.0f}s")
    print(f"  CSV saved: {csv_path}")
    print(f"{'='*90}\n")


def parse_args():
    p = argparse.ArgumentParser(description="Run clutter scene pipeline benchmark on all eligible scenes")
    p.add_argument("--pipeline", default="table", choices=list(_PIPELINE_SCRIPTS),
                   help="Task-generation pipeline to run")
    p.add_argument("--scenes", nargs="*", default=None,
                   help="Specific scenes to run. If omitted, each trial auto-selects.")
    p.add_argument("--num-trials", type=int, default=None,
                   help="Number of trials when auto-selecting scenes (default: 10)")
    p.add_argument("--exclude", nargs="*", default=None,
                   help="Additional scenes to exclude (only with --scenes)")
    p.add_argument("--timeout", type=int, default=900,
                   help="Timeout per scene in seconds (default: 900 = 15min)")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--density", default="medium", choices=["low", "medium", "high", "ultra"])
    p.add_argument("--mount-gap-m", type=float, default=0.10)
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--strict-gate", dest="strict_gate", action="store_true")
    p.add_argument("--no-strict-gate", dest="strict_gate", action="store_false")
    p.set_defaults(strict_gate=False)
    p.add_argument("--randomize", action="store_true",
                   help="Randomize target, fragile, and clutter object types each episode")
    # Transfer pipeline flags.
    p.add_argument("--food-model", default=None, help="(transfer) Override food model id")
    p.add_argument("--source-model", default=None, help="(transfer) Override source container model id")
    p.add_argument("--dest-model", default=None, help="(transfer) Override dest container model id")
    p.add_argument("--goal-predicate", default=None, help="(transfer) Override goal predicate")
    # Stack pipeline flags.
    p.add_argument("--stack-height", default=None, help="(stack) Stack height preset")
    p.add_argument("--target-model", default=None, help="(stack) Override target (bottom) model id")
    p.add_argument("--stack-model", default=None, help="(stack) Override stack-item model id")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: outputs/benchmark_runs/<timestamp>)")
    p.add_argument("--resume", default=None,
                   help="Resume a previous benchmark run directory (skip completed scenes)")
    args = p.parse_args()
    if args.pipeline in ("cabinet_pickup", "jar_transport") and args.scenes:
        p.error("This pipeline constructs empty-scene tasks; use --num-trials instead of --scenes")
    for name in ("episodes", "timeout", "video_fps"):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.steps < 0 or (args.num_trials is not None and args.num_trials <= 0):
        p.error("--steps must be nonnegative and --num-trials must be positive")
    return args


def _find_completed_scenes(output_dir, episodes):
    """Resume only directories with complete artifacts and episode diagnostics."""
    if not os.path.isdir(output_dir):
        return set()
    statuses = {}
    summary = os.path.join(output_dir, "summary.csv")
    if os.path.isfile(summary):
        with open(summary, newline="") as f:
            statuses = {r["scene"]: r.get("status") for r in csv.DictReader(f)}
    return {
        name for name in os.listdir(output_dir)
        if statuses.get(name, "success") == "success"
        and os.path.isdir(os.path.join(output_dir, name))
        and not _validate_scene_artifacts(os.path.join(output_dir, name), episodes)
    }


def main():
    args = parse_args()
    _spot_preflight_or_exit()

    scenes_dir = os.path.join(
        os.environ.get("OMNIGIBSON_DATA_PATH", os.path.join(_PROJECT_ROOT, "behavior-1k", "datasets")),
        "behavior-1k-assets", "scenes",
    )

    # Determine output directory.
    if args.resume:
        output_dir = args.resume
    elif args.output_dir:
        output_dir = args.output_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(_DEFAULT_OUTPUT_DIR, f"benchmark_{ts}")
    os.makedirs(output_dir, exist_ok=True)

    # Determine scene list.  When --scenes is given, run those specific scenes
    # with --scene-model.  Otherwise, let each subprocess auto-select a scene.
    if args.scenes:
        scenes = args.scenes
        if args.exclude:
            scenes = [s for s in scenes if s not in set(args.exclude)]
    else:
        scenes = None  # auto-select mode

    # Keep original indices when resuming: the index determines the seed.
    if scenes is None:
        num_trials = args.num_trials if args.num_trials is not None else 10
        scenes = [None] * num_trials
        print(f"[Benchmark] Auto-select mode: {num_trials} trials")
    planned = list(enumerate(scenes))
    completed = _find_completed_scenes(output_dir, args.episodes) if args.resume else set()
    pending = [(i, scene) for i, scene in planned if (scene or f"trial_{i}") not in completed]

    print(f"[Benchmark] Output: {output_dir}")
    print(f"[Benchmark] Trials: {len(pending)} to run")
    print(f"[Benchmark] Config: episodes={args.episodes}, steps={args.steps}, "
          f"density={args.density}, timeout={args.timeout}s")

    # Save run config.
    config_path = os.path.join(output_dir, "benchmark_config.json")
    config_data = {
        "pipeline": args.pipeline,
        "scenes": [s for s in scenes if s is not None],
        "episodes": args.episodes,
        "num_trials": len(scenes) if all(s is None for s in scenes) else None,
        "steps": args.steps,
        "seed": args.seed,
        "timeout": args.timeout,
        "strict_gate": args.strict_gate,
        "mount_gap_m": args.mount_gap_m,
        "timestamp": datetime.now().isoformat(),
    }
    if args.pipeline == "table":
        config_data["density"] = args.density
        config_data["randomize"] = args.randomize
    if args.pipeline == "transfer":
        config_data.update({
            "food_model": args.food_model,
            "source_model": args.source_model,
            "dest_model": args.dest_model,
            "goal_predicate": args.goal_predicate,
        })
    if args.pipeline.startswith("stack"):
        config_data.update({
            "stack_height": args.stack_height,
            "target_model": args.target_model,
            "stack_model": args.stack_model,
        })
    if args.resume and os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as f:
            previous_config = json.load(f)
        mismatched = [k for k in config_data if k != "timestamp" and k in previous_config
                      and previous_config[k] != config_data[k]]
        if mismatched:
            raise ValueError(f"Resume configuration differs in {mismatched}; use the original run settings")
    else:
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

    previous_rows = {}
    summary_path = os.path.join(output_dir, "summary.csv")
    if args.resume and os.path.isfile(summary_path):
        with open(summary_path, newline="") as f:
            previous_rows = {r["scene"]: r for r in csv.DictReader(f)}
    results = []
    for idx, scene in planned:
        label = scene or f"trial_{idx}"
        if label in completed:
            run_dir = os.path.join(output_dir, label)
            entries = _read_episode_diagnostics(run_dir, args.episodes)
            results.append({
                "scene": label, "status": "success",
                "duration_s": float(previous_rows.get(label, {}).get("duration_s") or 0),
                "gate_pass": all(r["gate_pass"] for r in entries),
                "ltl_violated": any(r["ltl_violated"] for r in entries),
                "error": previous_rows.get(label, {}).get("error", ""), "run_dir": run_dir,
            })
        else:
            print(f"\n[Benchmark] Trial {idx + 1}/{len(planned)}")
            results.append(_run_scene(scene, args, output_dir, scene_index=idx))
        _write_summary(results, output_dir)

    print(f"\n[Benchmark] Done. Results at: {output_dir}")
    return 1 if any(r["status"] != "success" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
