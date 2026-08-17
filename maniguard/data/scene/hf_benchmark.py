"""Resolve a benchmark source (local path OR HF dataset repo_id) to a local dir.

Scene loading expects a real filesystem (OmniGibson passes ``scene_file``
straight to Isaac Sim), so streaming isn't an option. HF datasets are
snapshot-downloaded into the standard ``huggingface_hub`` cache and
iterated from there. Calls are idempotent: ``snapshot_download`` skips
files whose server-side hash matches the cache, so repeated eval runs
only pay bandwidth for changed scenes.

Usage:
    from maniguard.data.scene.hf_benchmark import resolve_benchmark_root
    path = resolve_benchmark_root(args.benchmark_root, revision="main")
    # path.is_dir() == True; discover_scenes(path) proceeds unchanged.

Private datasets: the user must already be logged in
(``huggingface-cli login`` or ``HF_TOKEN`` env var); no explicit auth
handling here.
"""

from __future__ import annotations

from pathlib import Path


def resolve_benchmark_root(source: str, revision: str = "main") -> Path:
    """Turn a benchmark source string into a local directory path.

    Args:
        source: Either an absolute/relative local path, or a HuggingFace
            dataset repo_id in the form ``<owner>/<name>``.
        revision: Git revision to snapshot when ``source`` is a repo_id.
            Defaults to ``main`` (always latest). Pass a commit SHA or
            tag name for reproducibility.

    Returns:
        Path to a directory containing per-scene subdirectories
        (``<scene_name>/scene_ep1.json`` + ``diagnostics.jsonl``).
    """
    candidate = Path(source).expanduser()
    if candidate.is_dir():
        return candidate.resolve()

    # Heuristic: anything containing '/' and not a real directory is a
    # repo_id. Paths with '/' that don't exist locally would also error
    # here, but that's the right signal.
    if "/" not in source:
        raise FileNotFoundError(
            f"Benchmark source is neither a local directory nor a "
            f"'<owner>/<name>' HF repo_id: {source!r}"
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to fetch benchmark datasets "
            f"from HF; install it in this env ({exc})."
        ) from exc

    local_dir = snapshot_download(
        repo_id=source,
        repo_type="dataset",
        revision=revision,
    )
    return Path(local_dir)


def is_hf_repo_id(source: str) -> bool:
    """Quick check that ``source`` looks like a HF repo_id, not a path."""
    if Path(source).expanduser().exists():
        return False
    return "/" in source and not source.startswith(("/", "./", "../"))
