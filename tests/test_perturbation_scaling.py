from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from maniguard.data.perturbation_scaling import (
    TaskBundle,
    apply_object_model_swap,
    apply_env_remap,
    apply_position_jitter,
    build_prompt_variants,
    list_generation_specs,
    load_task_bundle,
    scale_base_task_set,
    validate_variant_prompts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REP_ROOT = REPO_ROOT / "outputs" / "benchmark_runs-datasets" / "20260417_taskgen7fam_large_v1" / "accepted"
REP_TASKS = {
    "table": REP_ROOT / "table" / "task_0046",
    "liquid_transport": REP_ROOT / "liquid_transport" / "task_0020",
    "stack_same": REP_ROOT / "stack_same" / "task_0002",
    "lid_transport_liquid": REPO_ROOT / "outputs" / "benchmark_base_task_sets_reviewed" / "final_unique_accepted" / "lid_transport_liquid" / "task_0000",
}
ENV_ROOT = REPO_ROOT / "outputs" / "benchmark_base_task_sets_reviewed" / "final_accepted"
GOAL_REGION_ROOT = REPO_ROOT / "outputs" / "benchmark_base_task_sets_reviewed" / "final_unique_accepted-goal_region_sphere-full"
GOAL_REGION_REP_TASKS = {
    "table": GOAL_REGION_ROOT / "table" / "task_0000",
    "liquid_transport": GOAL_REGION_ROOT / "liquid_transport" / "task_0006",
    "stack_same": GOAL_REGION_ROOT / "stack_same" / "task_0000",
    "stack_flat": GOAL_REGION_ROOT / "stack_flat" / "task_0000",
    "lid_transport_food": GOAL_REGION_ROOT / "lid_transport_food" / "task_0000",
    "lid_transport_liquid": GOAL_REGION_ROOT / "lid_transport_liquid" / "task_0000",
    "transfer": GOAL_REGION_ROOT / "transfer" / "task_0000",
}


def _require_task(path: Path) -> None:
    if not path.is_dir():
        pytest.skip(f"Representative task missing: {path}")


def _load_script_module(module_name: str, rel_path: str):
    mod_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("family", sorted(REP_TASKS))
def test_load_task_bundle_on_representatives(family: str) -> None:
    task_dir = REP_TASKS[family]
    _require_task(task_dir)
    bundle = load_task_bundle(task_dir)
    assert bundle.family == family
    assert isinstance(bundle.prompt, str) and bundle.prompt
    assert isinstance(bundle.task_roles, dict) and bundle.task_roles
    assert isinstance(bundle.diagnostics.get("goal_conditions"), (list, dict))


def test_list_generation_specs_object_emits_model_and_appearance() -> None:
    task_dir = REP_TASKS["table"]
    _require_task(task_dir)
    bundle = load_task_bundle(task_dir)
    specs = list_generation_specs(bundle, perturbation_kind="object", global_seed=7, variant_count=1)
    subtypes = {(spec["role"], spec["subtype"], bool(spec["requires_online"])) for spec in specs}
    assert ("target", "model_swap", True) in subtypes
    assert ("target", "appearance", False) in subtypes


def test_transfer_variants_use_destination_neutral_wording() -> None:
    diagnostics = {
        "pipeline": "transfer",
        "selection": {
            "food_synset": "potato.n.01",
            "source_synset": "chopping_board.n.01",
            "dest_synset": "stockpot.n.01",
        },
    }

    prompts = build_prompt_variants(
        canonical="Transfer the potato from the chopping board to the stockpot.",
        diagnostics=diagnostics,
        scene_info={},
    )

    assert prompts == [
        "Transfer the potato from the chopping board to the stockpot.",
        "Move the potato from the chopping board to the stockpot.",
        "Relocate the potato from the chopping board to the stockpot.",
        "Bring the potato from the chopping board to the stockpot.",
    ]
    assert validate_variant_prompts("transfer", prompts) == []


@pytest.mark.parametrize("bad_prompt", [
    "Put the potato into the stockpot.",
    "Put the potato onto the plate.",
    "Carry the potato from the plate to the bowl.",
    "Move the potato from the plate to the bowl without spilling.",
])
def test_transfer_validation_rejects_unsafe_or_relation_specific_language(bad_prompt: str) -> None:
    errors = validate_variant_prompts("transfer", ["Transfer the potato from the plate to the bowl.", bad_prompt])
    assert errors


def test_lid_variants_keep_lid_task_language_without_liquid_terms() -> None:
    diagnostics = {
        "pipeline": "lid_transport_liquid",
        "selection": {"container_synset": "kettle.n.01"},
    }

    prompts = build_prompt_variants(
        canonical="Place the lid on the filled kettle, then lift the filled kettle upward.",
        diagnostics=diagnostics,
        scene_info={},
    )

    assert prompts == [
        "Place the lid on the filled kettle, then lift the filled kettle upward.",
        "Put the lid on the filled kettle, then lift the filled kettle.",
        "Place the lid on the filled kettle, then raise the filled kettle upward.",
        "Cover the filled kettle with the lid, then lift the filled kettle.",
    ]
    assert validate_variant_prompts("lid_transport_liquid", prompts) == []


def test_list_generation_specs_rejects_removed_task_kind() -> None:
    task_dir = REP_TASKS["table"]
    _require_task(task_dir)
    bundle = load_task_bundle(task_dir)
    with pytest.raises(ValueError, match="Unsupported perturbation kind"):
        list_generation_specs(bundle, perturbation_kind="task", global_seed=9, variant_count=1)


def test_list_generation_specs_position_emits_jitter() -> None:
    task_dir = REP_TASKS["table"]
    _require_task(task_dir)
    bundle = load_task_bundle(task_dir)
    specs = list_generation_specs(bundle, perturbation_kind="position", global_seed=11, variant_count=1)
    assert len(specs) == 1
    assert specs[0]["kind"] == "position"
    assert specs[0]["subtype"] == "jitter"
    assert specs[0]["requires_online"] is False


def test_list_generation_specs_env_emits_env_swap() -> None:
    task_dir = REPO_ROOT / "outputs" / "benchmark_base_task_sets_reviewed" / "final_unique_accepted" / "table" / "task_0000"
    _require_task(task_dir)
    if not ENV_ROOT.is_dir():
        pytest.skip(f"Env donor root missing: {ENV_ROOT}")
    from maniguard.data.perturbation_scaling import build_env_inventory

    bundle = load_task_bundle(task_dir)
    specs = list_generation_specs(
        bundle,
        perturbation_kind="env",
        global_seed=13,
        variant_count=1,
        env_inventory=build_env_inventory(ENV_ROOT),
    )
    assert len(specs) == 1
    assert specs[0]["kind"] == "env"
    assert specs[0]["subtype"] == "env_swap"
    assert specs[0]["requires_online"] is False


def test_scale_base_task_set_writes_base_and_static_variants(tmp_path: Path) -> None:
    task_dir = REP_TASKS["table"]
    _require_task(task_dir)

    base_root = tmp_path / "base_root" / "table" / "task_0046"
    base_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(task_dir, base_root)

    output_root = tmp_path / "published"
    summary = scale_base_task_set(
        base_root=tmp_path / "base_root",
        env_source_root=None,
        output_root=output_root,
        perturbation_kind="semantic",
        seed=17,
        variant_count=1,
        dry_run=False,
    )

    assert summary["total_base_tasks"] == 1
    task_output_dir = output_root / "table" / "task_0046"
    base_dir = task_output_dir / "base"
    assert base_dir.is_dir()
    assert (base_dir / "scene_ep1.json").is_file()
    assert (base_dir / "diagnostics.jsonl").is_file()
    semantic_dir = task_output_dir / "semantic"
    variant_dirs = [path for path in semantic_dir.iterdir() if path.is_dir()]
    assert len(variant_dirs) == 1
    assert (variant_dirs[0] / "scene_ep1.json").is_file()
    assert (variant_dirs[0] / "diagnostics.jsonl").is_file()
    assert not (variant_dirs[0] / "validator_report.json").exists()
    assert any(item["variant_subtype"] == "paraphrase" for item in summary["outputs"] if not item.get("skipped"))


def test_scale_base_task_set_writes_position_variants(tmp_path: Path) -> None:
    task_dir = REP_TASKS["table"]
    _require_task(task_dir)

    base_root = tmp_path / "base_root" / "table" / "task_0046"
    base_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(task_dir, base_root)

    output_root = tmp_path / "published"
    summary = scale_base_task_set(
        base_root=tmp_path / "base_root",
        env_source_root=None,
        output_root=output_root,
        perturbation_kind="position",
        seed=23,
        variant_count=1,
        dry_run=False,
    )

    assert summary["total_base_tasks"] == 1
    position_dir = output_root / "table" / "task_0046" / "position"
    variant_dirs = [path for path in position_dir.iterdir() if path.is_dir()]
    assert len(variant_dirs) == 1
    assert (variant_dirs[0] / "scene_ep1.json").is_file()
    assert (variant_dirs[0] / "diagnostics.jsonl").is_file()


def test_scale_base_task_set_online_specs_use_materializer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    task_dir = REP_TASKS["table"]
    _require_task(task_dir)

    base_root = tmp_path / "base_root" / "table" / "task_0046"
    base_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(task_dir, base_root)

    materialized: list[Path] = []

    def _fake_materialize_online_variant(*, family: str, output_dir: Path, activity_root: Path, headless: bool, online_steps: int, online_video_fps: int, **kwargs):
        materialized.append(output_dir)
        (output_dir / "rollout_left_shoulder_ep1.mp4").write_bytes(b"fake-video")
        return {
            "task_dir": str(output_dir),
            "family": family,
            "online_accept_ok": True,
            "failed_checks": [],
            "returncode": 0,
        }

    monkeypatch.setattr("maniguard.data.perturbation_scaling._run_online_materialization_subprocess", _fake_materialize_online_variant)

    output_root = tmp_path / "published"
    summary = scale_base_task_set(
        base_root=tmp_path / "base_root",
        env_source_root=None,
        output_root=output_root,
        perturbation_kind="object",
        seed=17,
        variant_count=1,
        dry_run=False,
    )

    online_outputs = [item for item in summary["outputs"] if item["requires_online"] and not item.get("skipped")]
    static_outputs = [item for item in summary["outputs"] if not item["requires_online"] and not item.get("skipped")]
    assert online_outputs
    assert static_outputs
    assert materialized
    for path in materialized:
        assert (path / "rollout_left_shoulder_ep1.mp4").is_file()


def test_scale_base_task_set_retries_online_variants_until_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    task_dir = REP_TASKS["table"]
    _require_task(task_dir)

    base_root = tmp_path / "base_root" / "table" / "task_0046"
    base_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(task_dir, base_root)

    calls = {"count": 0}

    def _fake_worker(*, family: str, output_dir: Path, activity_root: Path, headless: bool, online_steps: int, online_video_fps: int, **kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            return {
                "task_dir": str(output_dir),
                "family": family,
                "online_accept_ok": False,
                "failed_checks": [f"attempt_{calls['count']}_failed"],
                "returncode": 1,
            }
        (output_dir / "rollout_left_shoulder_ep1.mp4").write_bytes(b"fake-video")
        return {
            "task_dir": str(output_dir),
            "family": family,
            "online_accept_ok": True,
            "failed_checks": [],
            "returncode": 0,
        }

    monkeypatch.setattr("maniguard.data.perturbation_scaling._run_online_materialization_subprocess", _fake_worker)
    monkeypatch.setattr(
        "maniguard.data.perturbation_scaling._resample_online_spec",
        lambda bundle, spec, global_seed, attempt_idx, attempt_limit: {**spec, "model": f"candidate_{attempt_idx}"},
    )

    summary = scale_base_task_set(
        base_root=tmp_path / "base_root",
        env_source_root=None,
        output_root=tmp_path / "published",
        perturbation_kind="object",
        seed=17,
        variant_count=1,
        variant_attempt_limit=5,
        dry_run=False,
    )

    online_outputs = [item for item in summary["outputs"] if item["requires_online"] and not item.get("skipped")]
    assert online_outputs
    assert online_outputs[0]["attempts_used"] == 3


def test_scale_base_task_set_skips_variant_after_attempt_budget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    task_dir = REP_TASKS["table"]
    _require_task(task_dir)

    base_root = tmp_path / "base_root" / "table" / "task_0046"
    base_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(task_dir, base_root)

    def _fake_worker(*, family: str, output_dir: Path, activity_root: Path, headless: bool, online_steps: int, online_video_fps: int, **kwargs):
        return {
            "task_dir": str(output_dir),
            "family": family,
            "online_accept_ok": False,
            "failed_checks": ["runtime_validation"],
            "returncode": 1,
        }

    monkeypatch.setattr("maniguard.data.perturbation_scaling._run_online_materialization_subprocess", _fake_worker)
    monkeypatch.setattr(
        "maniguard.data.perturbation_scaling._resample_online_spec",
        lambda bundle, spec, global_seed, attempt_idx, attempt_limit: {**spec, "model": f"candidate_{attempt_idx}"},
    )

    summary = scale_base_task_set(
        base_root=tmp_path / "base_root",
        env_source_root=None,
        output_root=tmp_path / "published",
        perturbation_kind="object",
        seed=17,
        variant_count=1,
        variant_attempt_limit=5,
        dry_run=False,
    )

    skipped = [item for item in summary["outputs"] if item.get("skipped")]
    assert skipped
    assert skipped[0]["skip_reason"] == "variant_attempt_budget_exhausted"
    assert skipped[0]["attempt_limit"] == 5
    assert skipped[0]["attempts_used"] == 5


def test_render_task_variants_collects_new_layout(tmp_path: Path) -> None:
    task_dir = REP_TASKS["table"]
    _require_task(task_dir)

    published_task = tmp_path / "published" / "table" / "task_0046"
    base_dir = published_task / "base"
    semantic_dir = published_task / "semantic" / "table__task_0046__semantic__instr_01"
    removed_task_dir = published_task / "task" / "table__task_0046__task__legacy"
    semantic_dir.mkdir(parents=True, exist_ok=True)
    removed_task_dir.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    for target in (base_dir, semantic_dir, removed_task_dir):
        shutil.copy2(task_dir / "scene_ep1.json", target / "scene_ep1.json")
        shutil.copy2(task_dir / "diagnostics.jsonl", target / "diagnostics.jsonl")

    render_mod = _load_script_module("render_task_variants_mod", "scripts/render_task_variants.py")
    items = render_mod.collect_render_items(
        tmp_path / "published",
        families={"table"},
        task_ids={"task_0046"},
        kinds={"semantic"},
        include_base=True,
    )
    assert len(items) == 2
    assert {item["kind"] for item in items} == {"base", "semantic"}


def test_scale_base_task_set_writes_env_variants(tmp_path: Path) -> None:
    task_dir = REPO_ROOT / "outputs" / "benchmark_base_task_sets_reviewed" / "final_unique_accepted" / "table" / "task_0000"
    _require_task(task_dir)
    if not ENV_ROOT.is_dir():
        pytest.skip(f"Env donor root missing: {ENV_ROOT}")

    base_root = tmp_path / "base_root" / "table" / "task_0000"
    base_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(task_dir, base_root)

    output_root = tmp_path / "published"
    summary = scale_base_task_set(
        base_root=tmp_path / "base_root",
        env_source_root=ENV_ROOT,
        output_root=output_root,
        perturbation_kind="env",
        seed=19,
        variant_count=1,
        dry_run=False,
    )

    assert summary["total_base_tasks"] == 1
    env_dir = output_root / "table" / "task_0000" / "env"
    variant_dirs = [path for path in env_dir.iterdir() if path.is_dir()]
    assert len(variant_dirs) == 1
    assert (variant_dirs[0] / "scene_ep1.json").is_file()
    assert (variant_dirs[0] / "diagnostics.jsonl").is_file()


def test_load_task_bundle_preserves_goal_region_prompt() -> None:
    task_dir = GOAL_REGION_REP_TASKS["table"]
    _require_task(task_dir)
    bundle = load_task_bundle(task_dir)
    assert isinstance(bundle.diagnostics.get("goal_region"), dict)
    assert "green goal sphere" in bundle.prompt


def test_scale_base_task_set_preserves_green_sphere_base_without_videos(tmp_path: Path) -> None:
    task_dir = GOAL_REGION_REP_TASKS["table"]
    _require_task(task_dir)

    base_root = tmp_path / "base_root" / "table" / "task_0000"
    base_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(task_dir, base_root)
    (base_root / "rollout_right_overview_ep1.mp4").write_bytes(b"fake-video")

    output_root = tmp_path / "published"
    summary = scale_base_task_set(
        base_root=tmp_path / "base_root",
        env_source_root=None,
        output_root=output_root,
        perturbation_kind="semantic",
        seed=29,
        variant_count=1,
        dry_run=False,
    )

    assert summary["total_base_tasks"] == 1
    base_dir = output_root / "table" / "task_0000" / "base"
    diag = json.loads((base_dir / "diagnostics.jsonl").read_text().splitlines()[0])
    assert isinstance(diag.get("goal_region"), dict)
    assert "green goal sphere" in str(diag.get("prompt") or "")
    assert not (base_dir / "rollout_right_overview_ep1.mp4").exists()


def test_position_variant_recomputes_goal_region_for_green_sphere_base() -> None:
    task_dir = GOAL_REGION_REP_TASKS["table"]
    _require_task(task_dir)
    bundle = load_task_bundle(task_dir)
    original_center = list((bundle.diagnostics.get("goal_region") or {}).get("center_world") or [])
    derived = apply_position_jitter(bundle, variant_id="table__task_0000__position__jitter", seed=31)
    goal_region = derived.diagnostics.get("goal_region")
    assert isinstance(goal_region, dict)
    assert "green goal sphere" in str(derived.diagnostics.get("prompt") or "")
    assert list(goal_region.get("center_world") or []) != original_center


def test_object_variant_preserves_goal_region_for_green_sphere_base() -> None:
    task_dir = GOAL_REGION_REP_TASKS["table"]
    _require_task(task_dir)
    bundle = load_task_bundle(task_dir)
    selection = bundle.diagnostics.get("selection") or {}
    derived = apply_object_model_swap(
        bundle,
        role="target",
        synset=str(selection.get("target_synset") or ""),
        variant_id="table__task_0000__object__target_model",
        seed=37,
    )
    goal_region = derived.diagnostics.get("goal_region")
    assert isinstance(goal_region, dict)
    assert "green goal sphere" in str(derived.diagnostics.get("prompt") or "")


def test_env_variant_recomputes_goal_region_for_green_sphere_base() -> None:
    task_dir = GOAL_REGION_REP_TASKS["table"]
    donor_task_dir = ENV_ROOT / "table" / "task_0000"
    _require_task(task_dir)
    _require_task(donor_task_dir)
    bundle = load_task_bundle(task_dir)
    donor_bundle = load_task_bundle(donor_task_dir)
    original_center = list((bundle.diagnostics.get("goal_region") or {}).get("center_world") or [])
    derived = apply_env_remap(bundle, donor_bundle=donor_bundle, variant_id="table__task_0000__env__swap")
    goal_region = derived.diagnostics.get("goal_region")
    assert isinstance(goal_region, dict)
    assert "green goal sphere" in str(derived.diagnostics.get("prompt") or "")
    assert list(goal_region.get("center_world") or []) != original_center


def test_transfer_green_sphere_root_still_has_no_goal_region() -> None:
    task_dir = GOAL_REGION_REP_TASKS["transfer"]
    _require_task(task_dir)
    bundle = load_task_bundle(task_dir)
    assert bundle.diagnostics.get("goal_region") is None
    assert "green goal sphere" not in bundle.prompt


def test_transfer_object_specs_only_modify_food() -> None:
    task_dir = GOAL_REGION_REP_TASKS["transfer"]
    _require_task(task_dir)
    bundle = load_task_bundle(task_dir)
    specs = list_generation_specs(bundle, perturbation_kind="object", global_seed=0, variant_count=1)
    roles = {str(spec.get("role")) for spec in specs if spec.get("kind") == "object"}
    assert roles == {"food"}
