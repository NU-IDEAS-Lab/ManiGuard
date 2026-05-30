"""Shared HF-push logic for the SFT checkpoint uploaders.

Both ``hf_push_watcher.py`` (runs alongside training, uploads each checkpoint as
soon as it finalizes) and ``hf_push.py`` (one-shot, run after training to backfill
anything missing) import this module, so they share ONE notion of:

  * which local step dirs are finalized (safe to upload),
  * how a local step maps to its HF directory name (relabel the 0-indexed final
    step to a round integer),
  * whether a step is ALREADY completely on HF (so it is skipped, never
    re-uploaded).

"Already pushed" is decided against HF itself (the authoritative source), not a
local marker file, because the two uploaders are independent processes. A step
counts as complete iff every file under the local ``<step>/params/`` exists under
the remote ``<remote_label>/params/`` -- a filename-set fingerprint, so a half
-finished upload (interrupted mid-commit) is detected as incomplete and retried.

``train_state/`` (optimizer state) is never uploaded; ``params/`` + ``assets/``
(norm stats) are, which is what inference / further-LoRA needs.
"""

from __future__ import annotations

import os

from huggingface_hub import HfApi

# Optimizer state -- not needed for inference, never uploaded.
IGNORE_PATTERNS = ["train_state/**", "train_state"]


def remote_label(step: int, num_train_steps: int) -> str:
    """Map a local step dir name to its HF directory name.

    openpi saves the final checkpoint 0-indexed (``num_train_steps - 1``); we
    relabel it to the round integer ``num_train_steps`` on HF. All other (period)
    checkpoints keep their step number.
    """
    if step == num_train_steps - 1:
        return str(num_train_steps)
    return str(step)


def finalized_steps(ckpt_dir: str) -> list[int]:
    """Local step dirs that are fully written and safe to upload.

    A step dir qualifies iff it (a) is a plain integer dir, (b) has no sibling
    ``*.orbax-checkpoint-tmp-*`` staging dir for that step, and (c) contains a
    non-empty ``params/`` subdir. orbax writes to a tmp dir then renames to the
    final ``<step>/``, so an integer dir with a populated ``params/`` and no tmp
    sibling is finalized.
    """
    if not os.path.isdir(ckpt_dir):
        return []
    entries = os.listdir(ckpt_dir)
    has_tmp = any(".orbax-checkpoint-tmp-" in e for e in entries)
    steps = []
    for e in entries:
        if not e.isdigit():
            continue
        params = os.path.join(ckpt_dir, e, "params")
        if os.path.isdir(params) and os.listdir(params) and not has_tmp:
            steps.append(int(e))
    return sorted(steps)


def _local_param_files(local_step_dir: str) -> set[str]:
    """Relative file paths under ``<step>/params/`` (the upload fingerprint)."""
    root = os.path.join(local_step_dir, "params")
    out: set[str] = set()
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            out.add(os.path.relpath(os.path.join(dirpath, fn), root))
    return out


def is_pushed_complete(api: HfApi, repo: str, local_step_dir: str, label: str) -> bool:
    """True iff every local ``params/`` file already exists on HF under ``<label>/params/``.

    Filename-set comparison against the live repo -- a partial/interrupted upload
    (missing files) reads as incomplete, so it is retried rather than skipped.
    """
    local = _local_param_files(local_step_dir)
    if not local:
        return False
    try:
        remote_all = set(api.list_repo_files(repo_id=repo, repo_type="model"))
    except Exception:
        return False  # repo may not exist yet -> not pushed
    prefix = f"{label}/params/"
    remote = {f[len(prefix):] for f in remote_all if f.startswith(prefix)}
    return local.issubset(remote)


def push_step(api: HfApi, repo: str, local_step_dir: str, label: str) -> None:
    """Upload one checkpoint's params/ + assets/ to ``<repo>/<label>/`` (skip train_state)."""
    api.upload_folder(
        repo_id=repo,
        repo_type="model",
        folder_path=local_step_dir,
        path_in_repo=label,
        ignore_patterns=IGNORE_PATTERNS,
        commit_message=f"checkpoint {label}",
    )
