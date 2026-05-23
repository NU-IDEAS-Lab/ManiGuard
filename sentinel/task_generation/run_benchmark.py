#!/usr/bin/env python
"""Run a scene generation pipeline on all eligible scenes as a benchmark.

Spawns one subprocess per scene to avoid GPU memory accumulation.
Records videos, saves scene JSON snapshots, and produces a summary report.

Usage:
    python -m sentinel.task_generation.run_benchmark
    python -m sentinel.task_generation.run_benchmark --pipeline cabinet
    python -m sentinel.task_generation.run_benchmark --pipeline transfer --no-strict-gate
    python -m sentinel.task_generation.run_benchmark --pipeline stack --stack-height medium
    python -m sentinel.task_generation.run_benchmark --scenes Rs_int Merom_1_int --timeout 600
    python -m sentinel.task_generation.run_benchmark --density high --steps 500 --episodes 2
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
import logging

log = logging.getLogger(__name__)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
_DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs", "benchmark_runs")

_STACK_SCRIPT = os.path.join(_SCRIPT_DIR, "stack_scene_pipeline.py")

_PIPELINE_SCRIPTS = {
    "table": os.path.join(_SCRIPT_DIR, "clutter_scene_pipeline.py"),
    # Empty-scene cabinet pickup: cabinet placed on a generated surface,
    # target/obstacle in front of (or beside) the drawer's swept zone.
    "cabinet_pickup": os.path.join(_SCRIPT_DIR, "cabinet_pickup_pipeline.py"),
    "transfer": os.path.join(_SCRIPT_DIR, "transfer_scene_pipeline.py"),
    "stack": _STACK_SCRIPT,
    "stack_same": _STACK_SCRIPT,
    "stack_flat": _STACK_SCRIPT,
    "stack_receptacle": _STACK_SCRIPT,
    "lid_transport": os.path.join(_SCRIPT_DIR, "lid_transport_pipeline.py"),
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
    # cabinet_pickup is empty-scene and managed via cabinet_pickup_pipeline.py's
    # own --task-id batch flag; it doesn't participate in run_benchmark's
    # per-scene loop. Leaving the key out so we don't accidentally route a
    # scene-based subprocess to an empty-scene pipeline.
    # Transfer and stack pipelines need the same table-like surfaces as table.
    "transfer": frozenset({
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


def _expected_rollout_paths(run_dir, episodes):
    return [
        os.path.join(run_dir, f"rollout_ep{ep}.mp4")
        for ep in range(1, episodes + 1)
    ]


def _validate_scene_artifacts(run_dir, episodes):
    missing = []
    diagnostics_path = os.path.join(run_dir, "diagnostics.jsonl")
    if not os.path.isfile(diagnostics_path):
        missing.append("diagnostics.jsonl")
    for video_path in _expected_rollout_paths(run_dir, episodes):
        if not os.path.isfile(video_path):
            missing.append(os.path.basename(video_path))
    return missing


def _spot_preflight_or_exit():
    from sentinel.utils.ltl_utils import get_spot_runtime_status

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
        sys.executable, pipeline_script,
        "--episodes", str(args.episodes),
        "--steps", str(args.steps),
        "--seed", str(scene_seed),
        "--mount-gap-m", str(args.mount_gap_m),
        "--run-dir", run_dir,
        "--save-video",
        "--video-fps", str(args.video_fps),
        "--strict-gate" if args.strict_gate else "--no-strict-gate",
    ]
    if scene_model:
        cmd.extend(["--scene-model", scene_model])
    # Pipeline-specific flags.
    if args.pipeline in ("table", "cabinet"):
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
        if args.target_synset:
            cmd.extend(["--target-synset", args.target_synset])
        if args.stack_synset:
            cmd.extend(["--stack-synset", args.stack_synset])

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

        missing_artifacts = _validate_scene_artifacts(run_dir, args.episodes)

        if proc.returncode == 0 and not missing_artifacts:
            result["status"] = "success"
        elif proc.returncode == -11 and not missing_artifacts:
            # SIGSEGV during Isaac Sim shutdown is benign if the pipeline
            # wrote diagnostics (meaning it completed its real work).
            result["status"] = "success"
            result["error"] = "clean exit (shutdown segfault ignored)"
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

    # Parse diagnostics.jsonl if it exists to extract gate/ltl info.
    diagnostics_path = os.path.join(run_dir, "diagnostics.jsonl")
    if os.path.isfile(diagnostics_path):
        try:
            with open(diagnostics_path, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if "gate_pass" in entry:
                        result["gate_pass"] = entry["gate_pass"]
                    if "ltl_violated" in entry:
                        result["ltl_violated"] = entry["ltl_violated"]
        except Exception as exc:
            log.warning("run_benchmark: diagnostics read from %s failed: %s", diagnostics_path, exc)
            pass

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
              f"{'pass' if r['gate_pass'] else 'fail':<8} {str(r['ltl_violated']):<10}")

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
                   help="Pipeline type: 'table' (tabletop clutter) or 'cabinet' (cabinet clutter)")
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
    p.add_argument("--target-synset", default=None, help="(stack) Override target synset")
    p.add_argument("--stack-synset", default=None, help="(stack) Override stack synset")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: outputs/benchmark_runs/<timestamp>)")
    p.add_argument("--resume", default=None,
                   help="Resume a previous benchmark run directory (skip completed scenes)")
    return p.parse_args()


def _find_completed_scenes(output_dir, episodes):
    """Find scenes that already completed successfully in a previous run."""
    completed = set()
    summary_path = os.path.join(output_dir, "summary.csv")
    if os.path.isfile(summary_path):
        with open(summary_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "success":
                    completed.add(row["scene"])
    # Also check individual scene dirs for diagnostics.
    if os.path.isdir(output_dir):
        for scene_dir in os.listdir(output_dir):
            scene_run_dir = os.path.join(output_dir, scene_dir)
            diag = os.path.join(scene_run_dir, "diagnostics.jsonl")
            if os.path.isfile(diag) and not _validate_scene_artifacts(scene_run_dir, episodes):
                try:
                    with open(diag, "r") as f:
                        for line in f:
                            entry = json.loads(line.strip())
                            if entry.get("gate_pass"):
                                completed.add(scene_dir)
                except Exception as exc:
                    log.warning("run_benchmark: diagnostics scan of %s failed: %s", diag, exc)
                    pass
    return completed


def main():
    args = parse_args()
    _spot_preflight_or_exit()

    scenes_dir = os.path.join(
        _PROJECT_ROOT, "datasets", "behavior-1k-assets", "scenes",
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

    # In auto-select mode, build a trial list of None entries.
    if scenes is None:
        num_trials = args.num_trials or 10
        scenes = [None] * num_trials
        print(f"[Benchmark] Auto-select mode: {num_trials} trials")
    else:
        # If resuming, skip already-completed scenes.
        completed = set()
        if args.resume:
            completed = _find_completed_scenes(output_dir, args.episodes)
            if completed:
                print(f"[Benchmark] Resuming — skipping {len(completed)} completed scenes")
            scenes = [s for s in scenes if s not in completed]

    print(f"[Benchmark] Output: {output_dir}")
    print(f"[Benchmark] Trials: {len(scenes)} to run")
    print(f"[Benchmark] Config: episodes={args.episodes}, steps={args.steps}, "
          f"density={args.density}, timeout={args.timeout}s")

    # Save run config.
    config_path = os.path.join(output_dir, "benchmark_config.json")
    config_data = {
        "pipeline": args.pipeline,
        "scenes": [s for s in scenes if s is not None],
        "episodes": args.episodes,
        "steps": args.steps,
        "seed": args.seed,
        "timeout": args.timeout,
        "strict_gate": args.strict_gate,
        "mount_gap_m": args.mount_gap_m,
        "timestamp": datetime.now().isoformat(),
    }
    if args.pipeline in ("table", "cabinet"):
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
            "target_synset": args.target_synset,
            "stack_synset": args.stack_synset,
        })
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    results = []
    for idx, scene in enumerate(scenes):
        print(f"\n[Benchmark] Progress: {idx + 1}/{len(scenes)}")
        result = _run_scene(scene, args, output_dir, scene_index=idx)
        results.append(result)
        # Write incremental summary after each scene so progress is visible.
        _write_summary(results, output_dir)

    print(f"\n[Benchmark] Done. Results at: {output_dir}")


if __name__ == "__main__":
    main()
