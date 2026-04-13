from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from omnigibson.task_generation.support_surface_profiles import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROFILE_PATH,
    current_timestamp_utc,
    load_support_surface_profiles,
    make_empty_support_surface_profiles_document,
    save_support_surface_profiles,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_OMNIGIBSON_ROOT = _PROJECT_ROOT / "OmniGibson"
_SURFACE_CATALOG_PATH = Path(__file__).resolve().with_name("surface_catalog.json")
_MILESTONE_PATH = _PROJECT_ROOT / "SURFACE_PROFILE_MILESTONES.md"
_SECTION_BEGIN = "<!-- BEGIN SURFACE_PROFILE_PROGRESS -->"
_SECTION_END = "<!-- END SURFACE_PROFILE_PROGRESS -->"


def parse_args():
    parser = argparse.ArgumentParser(description="Run support-surface profiler in category batches.")
    parser.add_argument("--categories", nargs="*", default=None, help="Optional subset of categories to run")
    parser.add_argument(
        "--order",
        choices=("size_asc", "size_desc", "alpha"),
        default="size_asc",
        help="Category execution order",
    )
    parser.add_argument("--grid-step-m", type=float, default=0.03)
    parser.add_argument("--output-json", default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--batch-root", default=None, help="Root directory for per-category batch outputs")
    parser.add_argument("--progress-json", default=None, help="Machine-readable progress log path")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reset-output-json", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--showcase-gui", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def _load_surface_catalog() -> dict:
    with open(_SURFACE_CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_batch_root(args) -> Path:
    if args.batch_root:
        root = Path(args.batch_root)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(DEFAULT_OUTPUT_ROOT) / f"catalog_batches_{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_progress_json(args, batch_root: Path) -> Path:
    if args.progress_json:
        return Path(args.progress_json)
    return batch_root / "batch_progress.json"


def _category_sequence(catalog: dict, selected: list[str] | None, order: str) -> list[str]:
    categories = sorted(catalog)
    if selected:
        selected_set = set(selected)
        missing = sorted(selected_set - set(categories))
        if missing:
            raise ValueError(f"Unknown categories: {missing}")
        categories = [cat for cat in categories if cat in selected_set]

    if order == "alpha":
        return categories
    reverse = order == "size_desc"
    return sorted(categories, key=lambda cat: (len(catalog[cat]), cat), reverse=reverse)


def _load_progress(progress_path: Path, catalog: dict, batch_root: Path, profile_json: Path) -> dict:
    if progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as f:
            progress = json.load(f)
    else:
        progress = {
            "batch_root": os.path.relpath(batch_root, _PROJECT_ROOT),
            "profile_json": os.path.relpath(profile_json, _PROJECT_ROOT),
            "surface_catalog": os.path.relpath(_SURFACE_CATALOG_PATH, _PROJECT_ROOT),
            "started_at": current_timestamp_utc(),
            "updated_at": current_timestamp_utc(),
            "categories": {},
            "history": [],
        }

    for category, models in catalog.items():
        progress["categories"].setdefault(
            category,
            {
                "catalog_model_count": len(models),
                "status": "pending",
                "profiled_model_count": 0,
                "candidate_count": 0,
                "last_run_dir": None,
                "last_summary_jsonl": None,
                "last_started_at": None,
                "last_completed_at": None,
                "returncode": None,
                "last_command": None,
            },
        )
    return progress


def _refresh_category_counts(progress: dict, profile_doc: dict) -> None:
    profiles = profile_doc.get("profiles", {})
    for category, meta in progress["categories"].items():
        entries = list((profiles.get(category) or {}).values())
        meta["profiled_model_count"] = len(entries)
        meta["candidate_count"] = sum(bool(entry.get("candidate_for_generation")) for entry in entries)
        if meta["status"] != "failed":
            if meta["profiled_model_count"] >= meta["catalog_model_count"]:
                meta["status"] = "completed"
            elif meta["profiled_model_count"] > 0:
                meta["status"] = "partial"
            else:
                meta["status"] = "pending"


def _render_progress_section(progress: dict, profile_doc: dict) -> str:
    _refresh_category_counts(progress, profile_doc)
    categories = progress["categories"]
    total_categories = len(categories)
    done_categories = sum(meta["status"] == "completed" for meta in categories.values())
    failed_categories = sum(meta["status"] == "failed" for meta in categories.values())
    total_catalog_models = sum(meta["catalog_model_count"] for meta in categories.values())
    total_profiled_models = sum(meta["profiled_model_count"] for meta in categories.values())
    total_candidates = sum(meta["candidate_count"] for meta in categories.values())

    lines = [
        _SECTION_BEGIN,
        "## Surface Profile Progress",
        "",
        f"- Last updated: `{progress['updated_at']}`",
        f"- Batch root: `{progress['batch_root']}`",
        f"- Profile JSON: `{progress['profile_json']}`",
        f"- Category completion: `{done_categories}/{total_categories}` completed, `{failed_categories}` failed",
        f"- Model completion: `{total_profiled_models}/{total_catalog_models}` profiled, `{total_candidates}` current candidates",
        "",
        "| category | catalog | profiled | candidates | status | latest run |",
        "|---|---:|---:|---:|---|---|",
    ]
    for category in sorted(categories):
        meta = categories[category]
        run_dir = meta["last_run_dir"] or "-"
        lines.append(
            f"| `{category}` | {meta['catalog_model_count']} | {meta['profiled_model_count']} | "
            f"{meta['candidate_count']} | `{meta['status']}` | `{run_dir}` |"
        )

    if progress["history"]:
        lines.extend([
            "",
            "### Batch History",
        ])
        for item in progress["history"]:
            lines.append(
                f"- `{item['completed_at']}` `{item['category']}` -> `{item['status']}` "
                f"(profiled={item['profiled_model_count']}/{item['catalog_model_count']}, "
                f"candidates={item['candidate_count']}, run_dir=`{item['run_dir']}`)"
            )

    lines.append(_SECTION_END)
    return "\n".join(lines) + "\n"


def _write_progress(progress_path: Path, progress: dict) -> None:
    progress["updated_at"] = current_timestamp_utc()
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=True)
        f.write("\n")


def _update_milestone(progress: dict, profile_doc: dict) -> None:
    milestone_text = _MILESTONE_PATH.read_text(encoding="utf-8") if _MILESTONE_PATH.exists() else "# Surface Profile Milestones\n\n"
    section = _render_progress_section(progress, profile_doc)
    if _SECTION_BEGIN in milestone_text and _SECTION_END in milestone_text:
        prefix = milestone_text.split(_SECTION_BEGIN, 1)[0]
        suffix = milestone_text.split(_SECTION_END, 1)[1]
        new_text = prefix + section + suffix.lstrip("\n")
    else:
        if not milestone_text.endswith("\n"):
            milestone_text += "\n"
        new_text = milestone_text + "\n" + section
    _MILESTONE_PATH.write_text(new_text, encoding="utf-8")


def _run_category(args, category: str, batch_root: Path, output_json: Path) -> tuple[int, list[str], Path]:
    run_dir = batch_root / category
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "omnigibson.task_generation.support_surface_profiler",
        "--category",
        category,
        "--grid-step-m",
        str(args.grid_step_m),
        "--run-dir",
        str(run_dir),
        "--output-json",
        str(output_json),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.skip_plots:
        cmd.append("--skip-plots")
    if args.showcase_gui:
        cmd.append("--showcase-gui")

    print(f"[Batch] Running category `{category}`")
    print("[Batch] Command:", " ".join(cmd))
    sys.stdout.flush()
    completed = subprocess.run(cmd, cwd=str(_OMNIGIBSON_ROOT), check=False)
    return completed.returncode, cmd, run_dir


def main():
    args = parse_args()
    catalog = _load_surface_catalog()
    batch_root = _resolve_batch_root(args)
    progress_json = _resolve_progress_json(args, batch_root)
    output_json = Path(args.output_json).resolve()

    if args.reset_output_json:
        save_support_surface_profiles(make_empty_support_surface_profiles_document(), str(output_json))

    progress = _load_progress(progress_json, catalog, batch_root, output_json)
    categories = _category_sequence(catalog, args.categories, args.order)
    print(f"[Batch] Categories to run: {categories}")
    print(f"[Batch] Batch root: {batch_root}")
    print(f"[Batch] Progress JSON: {progress_json}")
    print(f"[Batch] Output JSON: {output_json}")
    sys.stdout.flush()

    exit_code = 0
    for category in categories:
        meta = progress["categories"][category]
        meta["last_started_at"] = current_timestamp_utc()
        returncode, cmd, run_dir = _run_category(args, category, batch_root, output_json)
        meta["last_completed_at"] = current_timestamp_utc()
        meta["last_run_dir"] = os.path.relpath(run_dir, _PROJECT_ROOT)
        meta["last_summary_jsonl"] = os.path.relpath(run_dir / "summary.jsonl", _PROJECT_ROOT)
        meta["returncode"] = int(returncode)
        meta["last_command"] = cmd

        profile_doc = load_support_surface_profiles(str(output_json), use_cache=False)
        _refresh_category_counts(progress, profile_doc)
        meta["status"] = "completed" if returncode == 0 else "failed"

        progress["history"].append(
            {
                "category": category,
                "status": meta["status"],
                "catalog_model_count": meta["catalog_model_count"],
                "profiled_model_count": meta["profiled_model_count"],
                "candidate_count": meta["candidate_count"],
                "run_dir": meta["last_run_dir"],
                "completed_at": meta["last_completed_at"],
            }
        )
        _write_progress(progress_json, progress)
        _update_milestone(progress, profile_doc)

        if returncode != 0:
            exit_code = 1
            print(f"[Batch] Category `{category}` failed with return code {returncode}")
            sys.stdout.flush()
            if args.stop_on_error:
                break

    final_doc = load_support_surface_profiles(str(output_json), use_cache=False)
    _write_progress(progress_json, progress)
    _update_milestone(progress, final_doc)
    print(f"[Batch] Finished with exit_code={exit_code}")
    sys.stdout.flush()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
