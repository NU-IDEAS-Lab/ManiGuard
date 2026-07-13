"""Prompt construction helpers for eval scene language instructions.

``episode_prompt`` fills a prompt template with a scene's target-object name
(and a cleaned, human-readable variant), used by the benchmark runner to build
the language instruction shown to the policy for each eval scene.
"""
from __future__ import annotations

import re

_TRAIL_INSTANCE_RE = re.compile(r"_\d+$")


def clean_target(name: str) -> str:
    """``teacup_178`` -> ``teacup``; idempotent if no trailing instance id."""
    return _TRAIL_INSTANCE_RE.sub("", name).replace("_", " ").strip()


def episode_prompt(target_name: str, template: str, override: str | None = None) -> str:
    if override is not None:
        return override
    return template.format(target=target_name, target_clean=clean_target(target_name))
