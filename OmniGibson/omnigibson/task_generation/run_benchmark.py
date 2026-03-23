#!/usr/bin/env python
"""Run a scene generation pipeline on all eligible scenes as a benchmark.

Spawns one subprocess per scene to avoid GPU memory accumulation.
Records videos, saves scene JSON snapshots, and produces a summary report.

Usage:
    python -m omnigibson.task_generation.run_benchmark
    python -m omnigibson.task_generation.run_benchmark --pipeline cabinet
    python -m omnigibson.task_generation.run_benchmark --pipeline transfer --no-strict-gate
    python -m omnigibson.task_generation.run_benchmark --pipeline stack --stack-height medium
    python -m omnigibson.task_generation.run_benchmark --scenes Rs_int Merom_1_int --timeout 600
    python -m omnigibson.task_generation.run_benchmark --density high --steps 500 --episodes 2
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
_DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs", "benchmark_runs")

_PIPELINE_SCRIPTS = {
    "table": os.path.join(_SCRIPT_DIR, "clutter_scene_pipeline.py"),
    "cabinet": os.path.join(_SCRIPT_DIR, "cabinet_clutter_pipeline.py"),
    "transfer": os.path.join(_SCRIPT_DIR, "transfer_scene_pipeline.py"),
    "stack": os.path.join(_SCRIPT_DIR, "stack_scene_pipeline.py"),
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
    "cabinet": frozenset({
        "Benevolence_0_int",  # bathroom only
        "gates_bedroom",      # no cabinets
        "hall_arch_wood",     # public restroom
        "hall_train_station", # train station restroom
        "hall_glass_ceiling", # no cabinets
    }),
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


def _run_scene(scene_model, args, output_dir, scene_index=0):
    """Run the pipeline on a single scene in a subprocess. Returns a result dict."""
    run_dir = os.path.join(output_dir, scene_model)
    os.makedirs(run_dir, exist_ok=True)

    # Vary seed per scene so each gets different randomization.
    scene_seed = args.seed + scene_index

    pipeline_script = _PIPELINE_SCRIPTS[args.pipeline]
    cmd = [
        sys.executable, pipeline_script,
        "--scene-model", scene_model,
        "--episodes", str(args.episodes),
        "--steps", str(args.steps),
        "--seed", str(scene_seed),
        "--mount-gap-m", str(args.mount_gap_m),
        "--run-dir", run_dir,
        "--save-video",
        "--video-fps", str(args.video_fps),
        "--strict-gate" if args.strict_gate else "--no-strict-gate",
    ]

    # Pipeline-specific flags.
    if args.pipeline in ("table", "cabinet"):
        cmd.extend(["--clutter-density", args.density])
        if args.randomize:
            cmd.append("--randomize")
    if args.pipeline == "transfer":
        if args.food_synset:
            cmd.extend(["--food-synset", args.food_synset])
        if args.source_synset:
            cmd.extend(["--source-synset", args.source_synset])
        if args.dest_synset:
            cmd.extend(["--dest-synset", args.dest_synset])
        if args.goal_predicate:
            cmd.extend(["--goal-predicate", args.goal_predicate])
    if args.pipeline == "stack":
        if args.stack_height:
            cmd.extend(["--stack-height", args.stack_height])
        if args.target_synset:
            cmd.extend(["--target-synset", args.target_synset])
        if args.stack_synset:
            cmd.extend(["--stack-synset", args.stack_synset])

    log_path = os.path.join(run_dir, "stdout.log")
    diagnostics_path = os.path.join(run_dir, "diagnostics.jsonl")

    result = {
        "scene": scene_model,
        "status": "unknown",
        "duration_s": 0,
        "gate_pass": False,
        "ltl_violated": None,
        "error": "",
        "run_dir": run_dir,
    }

    print(f"\n{'='*70}")
    print(f"[Benchmark] Starting: {scene_model}")
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

        if proc.returncode == 0:
            result["status"] = "success"
        elif proc.returncode == -11 and os.path.isfile(diagnostics_path):
            # SIGSEGV during Isaac Sim shutdown is benign if the pipeline
            # wrote diagnostics (meaning it completed its real work).
            result["status"] = "success"
            result["error"] = "clean exit (shutdown segfault ignored)"
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
    if os.path.isfile(diagnostics_path):
        try:
            with open(diagnostics_path, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if "gate_pass" in entry:
                        result["gate_pass"] = entry["gate_pass"]
                    if "ltl_violated" in entry:
                        result["ltl_violated"] = entry["ltl_violated"]
        except Exception:
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
                   help="Specific scenes to run (default: all eligible)")
    p.add_argument("--exclude", nargs="*", default=None,
                   help="Additional scenes to exclude")
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
    p.add_argument("--food-synset", default=None, help="(transfer) Override food synset")
    p.add_argument("--source-synset", default=None, help="(transfer) Override source synset")
    p.add_argument("--dest-synset", default=None, help="(transfer) Override dest synset")
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


def _find_completed_scenes(output_dir):
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
            diag = os.path.join(output_dir, scene_dir, "diagnostics.jsonl")
            if os.path.isfile(diag):
                try:
                    with open(diag, "r") as f:
                        for line in f:
                            entry = json.loads(line.strip())
                            if entry.get("gate_pass"):
                                completed.add(scene_dir)
                except Exception:
                    pass
    return completed


def main():
    args = parse_args()

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

    # Determine scene list.
    if args.scenes:
        scenes = args.scenes
    else:
        scenes = _discover_scenes(scenes_dir, args.pipeline)

    if args.exclude:
        scenes = [s for s in scenes if s not in set(args.exclude)]

    # If resuming, skip already-completed scenes.
    completed = set()
    if args.resume:
        completed = _find_completed_scenes(output_dir)
        if completed:
            print(f"[Benchmark] Resuming — skipping {len(completed)} completed scenes")
        scenes = [s for s in scenes if s not in completed]

    print(f"[Benchmark] Output: {output_dir}")
    print(f"[Benchmark] Scenes: {len(scenes)} to run"
          f"{f' ({len(completed)} already completed)' if completed else ''}")
    print(f"[Benchmark] Config: episodes={args.episodes}, steps={args.steps}, "
          f"density={args.density}, timeout={args.timeout}s")

    # Save run config.
    config_path = os.path.join(output_dir, "benchmark_config.json")
    config_data = {
        "pipeline": args.pipeline,
        "scenes": scenes,
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
            "food_synset": args.food_synset,
            "source_synset": args.source_synset,
            "dest_synset": args.dest_synset,
            "goal_predicate": args.goal_predicate,
        })
    if args.pipeline == "stack":
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
