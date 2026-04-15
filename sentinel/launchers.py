"""Thin launcher wrappers that run RLinf's training entry points after
importing `sentinel` (which registers our Sentinel-specific
TrainConfigs into RLinf's registry at import time).

Invoke via ``python -m sentinel.launchers.sft_main ...`` style or
via the launcher shells in ``tools/``. All CLI args after the script
name are forwarded verbatim to the wrapped hydra main, so hydra
overrides (e.g. ``runner.max_steps=1``) work unchanged.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import sentinel.rlinf.patches  # noqa: F401  # patch RLinf surfaces
import sentinel.openpi.configs  # noqa: F401  # register TrainConfigs

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RLINF_ROOT = _REPO_ROOT / "RLinf"


def _run_rlinf_entry(relative_path: str) -> None:
    """Run an RLinf training script as if launched via ``python path``.

    We rely on runpy so Hydra's ``@hydra.main`` decorator picks up
    ``sys.argv`` the same way it would from a plain python invocation.
    """
    script = _RLINF_ROOT / relative_path
    if not script.is_file():
        raise FileNotFoundError(f"RLinf entry script not found: {script}")
    # Expose both the repo root and RLinf root on sys.path, mirroring
    # run_embodiment*.sh's PYTHONPATH export.
    for p in (str(_REPO_ROOT), str(_RLINF_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    # Hydra reads sys.argv[1:]; sys.argv[0] is scrubbed to the script path.
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")


def sft_main() -> None:
    """Entry point for SFT: RLinf/examples/sft/train_vla_sft.py.

    Upstream renamed train_embodied_sft.py -> train_vla_sft.py when it
    split VLA (Pi0.5 etc.) from VLM SFT in late 2026.
    """
    _configure_runtime_env()
    _run_rlinf_entry("examples/sft/train_vla_sft.py")


def rl_main() -> None:
    """Entry point for RL (PPO/GRPO): RLinf/examples/embodiment/train_embodied_agent.py."""
    _configure_runtime_env()
    _run_rlinf_entry("examples/embodiment/train_embodied_agent.py")


def _configure_runtime_env() -> None:
    """Match the exports run_embodiment_sft.sh / run_embodiment.sh do."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


if __name__ == "__main__":  # pragma: no cover
    cmd = sys.argv.pop(1) if len(sys.argv) > 1 else ""
    if cmd == "sft":
        sft_main()
    elif cmd == "rl":
        rl_main()
    else:
        sys.exit(
            "Usage: python -m sentinel.launchers {sft|rl} [hydra overrides...]"
        )
