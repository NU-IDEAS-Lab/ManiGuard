#!/usr/bin/env python3
"""Materialize one online-required perturbation variant in isolation and return a JSON result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from sentinel.data.perturbation_scaling import materialize_online_variant_in_place


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--activity-root", required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--online-steps", type=int, default=60)
    parser.add_argument("--online-video-fps", type=int, default=30)
    parser.add_argument("--attempt-artifacts-dir", default=None)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = materialize_online_variant_in_place(
        family=str(args.family),
        task_dir=Path(args.task_dir).expanduser().resolve(),
        activity_root=Path(args.activity_root).expanduser().resolve(),
        headless=bool(args.headless),
        online_steps=int(args.online_steps),
        online_video_fps=int(args.online_video_fps),
        attempt_artifacts_dir=Path(args.attempt_artifacts_dir).expanduser().resolve() if args.attempt_artifacts_dir else None,
    )
    Path(args.output_path).write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    exit_code = int(main())
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
