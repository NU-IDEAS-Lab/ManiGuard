"""Prompt construction helpers for eval scene language instructions.

``episode_prompt`` fills a prompt template with a scene's target-object name
(and a cleaned, human-readable variant), used by the benchmark runner to build
the language instruction shown to the policy for each eval scene.
"""
from __future__ import annotations

import json
import re

_TRAIL_INSTANCE_RE = re.compile(r"_\d+$")


def clean_target(name: str) -> str:
    """``teacup_178`` -> ``teacup``; idempotent if no trailing instance id."""
    return _TRAIL_INSTANCE_RE.sub("", name).replace("_", " ").strip()


def episode_prompt(target_name: str, template: str, override: str | None = None) -> str:
    if override is not None:
        return override
    return template.format(target=target_name, target_clean=clean_target(target_name))


_ABLATION_CACHE: dict[str, dict] = {}


def ablation_prompt(base_prompt: str, map_path: str, condition: str) -> str:
    """Swap a scene's instruction for its prompt-ablation variant.

    The Q2 study varies ONLY how the safety constraint is conveyed
    (``no_instruction`` / ``natural_language`` / ``ltl``) while the task instruction
    and the underlying LTL automaton stay fixed. The variants are pre-generated per
    instruction into a JSON map (``configs/ablation_prompt/*.json``) that the SFT
    datasets are rewritten from as well, so training and eval see byte-identical
    prompts — the whole point of the ablation.

    The scene's own instruction is the lookup key. A miss raises: silently falling
    back to the unmodified prompt would run the wrong condition while still
    reporting the requested one, which is exactly the failure this study cannot afford.
    """
    if map_path not in _ABLATION_CACHE:
        with open(map_path, encoding="utf-8") as f:
            _ABLATION_CACHE[map_path] = json.load(f)["instruction_map"]
    table = _ABLATION_CACHE[map_path]

    key = base_prompt.strip()
    if key not in table:
        raise KeyError(
            f"prompt_condition={condition!r} but this scene's instruction is not in "
            f"{map_path}:\n  {key!r}\n"
            "The map is built from the ablation's task set (clutter base); evaluating a "
            "scene outside it would silently mix conditions."
        )
    entry = table[key]
    if condition not in entry:
        raise KeyError(f"unknown prompt condition {condition!r}; have {sorted(entry)}")
    return entry[condition]
