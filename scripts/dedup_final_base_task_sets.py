#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT_ROOT = Path(
    "outputs/benchmark_base_task_sets_reviewed/final_accepted"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/benchmark_base_task_sets_reviewed/final_unique_accepted"
)
DEFAULT_GLOBAL_SEED = 20260422
DEFAULT_POS_DECIMALS = 3
DEFAULT_QUAT_DECIMALS = 3
SUPPORT_XY_MARGIN_M = 0.08
SYNSET_CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "chopping_board": ("chopping_board", "cutting_board"),
}


@dataclass(frozen=True)
class Bundle:
    family: str
    task_name: str
    task_dir: Path
    diagnostics: dict[str, Any]
    scene: dict[str, Any]
    object_registry: dict[str, Any]
    init_args: dict[str, dict[str, Any]]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_bundle(family: str, task_dir: Path) -> Bundle:
    diagnostics_path = task_dir / "diagnostics.jsonl"
    scene_path = task_dir / "scene_ep1.json"
    diagnostics_lines = diagnostics_path.read_text().splitlines()
    if len(diagnostics_lines) != 1:
        raise ValueError(
            f"{task_dir}: expected exactly one diagnostics row, got {len(diagnostics_lines)}"
        )
    diagnostics = json.loads(diagnostics_lines[0])
    scene = _load_json(scene_path)
    object_registry = scene["state"]["registry"]["object_registry"]
    init_info = scene["objects_info"]["init_info"]
    init_args = {name: entry.get("args", {}) for name, entry in init_info.items()}
    return Bundle(
        family=family,
        task_name=task_dir.name,
        task_dir=task_dir,
        diagnostics=diagnostics,
        scene=scene,
        object_registry=object_registry,
        init_args=init_args,
    )


def _normalize_quaternion(q: list[float] | tuple[float, ...]) -> tuple[float, float, float, float]:
    x, y, z, w = (float(v) for v in q)
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    if norm == 0.0:
        raise ValueError("zero-norm quaternion")
    return (x / norm, y / norm, z / norm, w / norm)


def _quat_conjugate(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = q
    return (-x, -y, -z, w)


def _quat_multiply(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _rotate_vec_by_quat(
    vec: tuple[float, float, float],
    quat: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    vx, vy, vz = vec
    qvec = (vx, vy, vz, 0.0)
    rotated = _quat_multiply(_quat_multiply(quat, qvec), _quat_conjugate(quat))
    return (rotated[0], rotated[1], rotated[2])


def _canonicalize_quaternion(
    q: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    neg = tuple(-v for v in q)
    return neg if neg > q else q


def _round_vec(vec: tuple[float, float, float], decimals: int) -> tuple[float, float, float]:
    return tuple(round(v, decimals) for v in vec)


def _round_quat(
    quat: tuple[float, float, float, float],
    decimals: int,
) -> tuple[float, float, float, float]:
    return tuple(round(v, decimals) for v in quat)


def _robot_frame_pose(
    bundle: Bundle,
    object_name: str,
    pos_decimals: int,
    quat_decimals: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    base_pose = bundle.diagnostics["robot_mount"]["base_pose"]
    base_pos = tuple(float(v) for v in base_pose["position"])
    base_ori = _normalize_quaternion(base_pose["orientation"])
    base_inv = _quat_conjugate(base_ori)

    object_state = bundle.object_registry[object_name]["root_link"]
    obj_pos = tuple(float(v) for v in object_state["pos"])
    obj_ori = _normalize_quaternion(object_state["ori"])

    delta = (
        obj_pos[0] - base_pos[0],
        obj_pos[1] - base_pos[1],
        obj_pos[2] - base_pos[2],
    )
    rel_pos = _rotate_vec_by_quat(delta, base_inv)
    rel_ori = _quat_multiply(base_inv, obj_ori)
    rel_ori = _canonicalize_quaternion(_normalize_quaternion(rel_ori))

    return _round_vec(rel_pos, pos_decimals), _round_quat(rel_ori, quat_decimals)


def _synset_to_category(synset: str) -> str:
    return str(synset).split(".")[0]


def _candidate_categories_for_synset(synset: str) -> tuple[str, ...]:
    prefix = _synset_to_category(synset)
    return SYNSET_CATEGORY_ALIASES.get(prefix, (prefix,))


def _support_bounds_xy(bundle: Bundle) -> tuple[float, float, float, float] | None:
    bounds = bundle.diagnostics.get("support_selection", {}).get("surface_bounds_xy")
    if not bounds:
        return None
    (xmin, ymin), (xmax, ymax) = bounds
    return float(xmin), float(ymin), float(xmax), float(ymax)


def _inside_xy_bounds(
    pos: list[float] | tuple[float, ...],
    bounds: tuple[float, float, float, float] | None,
    margin: float = SUPPORT_XY_MARGIN_M,
) -> bool:
    if bounds is None:
        return True
    x, y = float(pos[0]), float(pos[1])
    xmin, ymin, xmax, ymax = bounds
    return (xmin - margin) <= x <= (xmax + margin) and (ymin - margin) <= y <= (ymax + margin)


def _distance_sq(
    a: list[float] | tuple[float, ...],
    b: list[float] | tuple[float, ...],
) -> float:
    return sum((float(a[i]) - float(b[i])) ** 2 for i in range(3))


def _pick_names_by_category(
    bundle: Bundle,
    category: str | tuple[str, ...],
    expected_count: int,
    *,
    exclude: set[str] | None = None,
    anchor_pos: list[float] | tuple[float, ...] | None = None,
) -> list[str]:
    exclude = exclude or set()
    bounds = _support_bounds_xy(bundle)

    categories = (category,) if isinstance(category, str) else category

    def _candidates(with_bounds: bool) -> list[str]:
        names: list[str] = []
        for name, args in bundle.init_args.items():
            if name in exclude:
                continue
            if args.get("category") not in categories:
                continue
            if name not in bundle.object_registry:
                continue
            pos = bundle.object_registry[name]["root_link"]["pos"]
            if with_bounds and not _inside_xy_bounds(pos, bounds):
                continue
            names.append(name)
        return names

    candidates = _candidates(with_bounds=True)
    if len(candidates) < expected_count:
        candidates = _candidates(with_bounds=False)
    if len(candidates) < expected_count:
        raise ValueError(
            f"{bundle.task_dir}: expected at least {expected_count} candidates for category={categories}, "
            f"got {len(candidates)}"
        )

    if anchor_pos is None and bounds is not None:
        xmin, ymin, xmax, ymax = bounds
        anchor_pos = ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, 0.0)

    if anchor_pos is None:
        candidates.sort()
    else:
        candidates.sort(
            key=lambda name: (
                _distance_sq(bundle.object_registry[name]["root_link"]["pos"], anchor_pos),
                name,
            )
        )
    return candidates[:expected_count]


def _model_token(bundle: Bundle, object_name: str) -> tuple[str | None, str | None]:
    args = bundle.init_args[object_name]
    return (args.get("category"), args.get("model"))


def _make_token(
    bundle: Bundle,
    role: str,
    object_name: str,
    pos_decimals: int,
    quat_decimals: int,
) -> tuple[Any, ...]:
    rel_pos, rel_ori = _robot_frame_pose(bundle, object_name, pos_decimals, quat_decimals)
    category, model = _model_token(bundle, object_name)
    return (role, category, model, rel_pos, rel_ori)


def _extract_tokens_for_family(
    bundle: Bundle,
    pos_decimals: int,
    quat_decimals: int,
) -> tuple[tuple[Any, ...], ...]:
    family = bundle.family
    diagnostics = bundle.diagnostics

    if family in {"table", "liquid_transport"}:
        tokens = [
            _make_token(
                bundle,
                str(item.get("role", "object")),
                str(item["scene_object_name"]),
                pos_decimals,
                quat_decimals,
            )
            for item in diagnostics["active_object_summary"]
        ]
        return tuple(sorted(tokens))

    if family == "transfer":
        goal = diagnostics["goal_conditions"][0]
        food_name = str(goal["subject"])
        dest_name = str(goal["reference"])
        source_category = _candidate_categories_for_synset(
            diagnostics["selection"]["source_synset"]
        )
        source_name = _pick_names_by_category(
            bundle,
            source_category,
            1,
            exclude={food_name, dest_name},
            anchor_pos=bundle.object_registry[food_name]["root_link"]["pos"],
        )[0]
        tokens = [
            _make_token(bundle, "food", food_name, pos_decimals, quat_decimals),
            _make_token(bundle, "source", source_name, pos_decimals, quat_decimals),
            _make_token(bundle, "destination", dest_name, pos_decimals, quat_decimals),
        ]
        return tuple(sorted(tokens))

    if family == "lid_transport_food":
        goal = diagnostics["goal_conditions"]
        lid_name = next(str(item["subject"]) for item in goal if str(item["subject"]) != "robot")
        container_name = next(
            str(item["reference"]) for item in goal if str(item["reference"]) != "robot"
        )
        food_category = _candidate_categories_for_synset(
            diagnostics["selection"]["food_synset"]
        )
        food_name = _pick_names_by_category(
            bundle,
            food_category,
            1,
            exclude={lid_name, container_name},
            anchor_pos=bundle.object_registry[container_name]["root_link"]["pos"],
        )[0]
        tokens = [
            _make_token(bundle, "container", container_name, pos_decimals, quat_decimals),
            _make_token(bundle, "lid", lid_name, pos_decimals, quat_decimals),
            _make_token(bundle, "food", food_name, pos_decimals, quat_decimals),
        ]
        return tuple(sorted(tokens))

    if family == "lid_transport_liquid":
        goal = diagnostics["goal_conditions"]
        lid_name = next(str(item["subject"]) for item in goal if str(item["subject"]) != "robot")
        container_name = next(
            str(item["reference"]) for item in goal if str(item["reference"]) != "robot"
        )
        tokens = [
            _make_token(bundle, "container", container_name, pos_decimals, quat_decimals),
            _make_token(bundle, "lid", lid_name, pos_decimals, quat_decimals),
        ]
        return tuple(sorted(tokens))

    if family == "stack_flat":
        target_category = _candidate_categories_for_synset(
            diagnostics["selection"]["target_synset"]
        )
        stack_category = _candidate_categories_for_synset(
            diagnostics["selection"]["stack_synset"]
        )
        stack_above = int(diagnostics["selection"]["stack_above"])
        target_name = _pick_names_by_category(bundle, target_category, 1)[0]
        stack_names = _pick_names_by_category(
            bundle,
            stack_category,
            stack_above,
            exclude={target_name},
            anchor_pos=bundle.object_registry[target_name]["root_link"]["pos"],
        )
        tokens = [_make_token(bundle, "target", target_name, pos_decimals, quat_decimals)]
        tokens.extend(
            _make_token(bundle, "stack", name, pos_decimals, quat_decimals)
            for name in stack_names
        )
        return tuple(sorted(tokens))

    if family == "stack_same":
        bowl_category = _candidate_categories_for_synset(
            diagnostics["selection"]["target_synset"]
        )
        total_bowls = int(diagnostics["selection"]["stack_above"]) + 1
        bowl_names = _pick_names_by_category(bundle, bowl_category, total_bowls)
        tokens = [
            _make_token(bundle, "bowl", name, pos_decimals, quat_decimals) for name in bowl_names
        ]
        return tuple(sorted(tokens))

    raise ValueError(f"Unsupported family: {family}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _signature_json(signature: tuple[tuple[Any, ...], ...]) -> str:
    return json.dumps(_jsonable(signature), ensure_ascii=True, separators=(",", ":"))


def _signature_hash(signature_json: str) -> str:
    return hashlib.sha256(signature_json.encode("utf-8")).hexdigest()


def _group_seed(family: str, signature_json: str, global_seed: int) -> int:
    seed_material = json.dumps(
        {
            "family": family,
            "signature": json.loads(signature_json),
            "global_seed": global_seed,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _build_manifest(
    input_root: Path,
    output_root: Path,
    *,
    seed: int,
    pos_decimals: int,
    quat_decimals: int,
) -> dict[str, Any]:
    families = sorted(p.name for p in input_root.iterdir() if p.is_dir())
    manifest: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "global_seed": seed,
        "signature_policy": {
            "frame": "robot_base",
            "liquid_signature": "rigid_only",
            "include_orientation": True,
            "include_object_model": True,
            "position_decimals": pos_decimals,
            "quaternion_decimals": quat_decimals,
            "support_identity_ignored": True,
            "scene_model_ignored": True,
        },
        "families": {},
    }

    total_before = 0
    total_after = 0

    for family in families:
        family_dir = input_root / family
        task_dirs = sorted(p for p in family_dir.iterdir() if p.is_dir())
        total_before += len(task_dirs)

        grouped: dict[str, list[str]] = {}
        for task_dir in task_dirs:
            bundle = _load_bundle(family, task_dir)
            signature = _extract_tokens_for_family(bundle, pos_decimals, quat_decimals)
            signature_json = _signature_json(signature)
            grouped.setdefault(signature_json, []).append(task_dir.name)

        kept_tasks: list[str] = []
        duplicate_groups: list[dict[str, Any]] = []

        for signature_json in sorted(grouped):
            members = sorted(grouped[signature_json])
            if len(members) == 1:
                representative = members[0]
            else:
                rng = random.Random(_group_seed(family, signature_json, seed))
                representative = rng.choice(members)
            kept_tasks.append(representative)
            duplicate_groups.append(
                {
                    "signature_sha256": _signature_hash(signature_json),
                    "member_count": len(members),
                    "members": members,
                    "representative": representative,
                }
            )

        kept_tasks.sort()
        total_after += len(kept_tasks)

        manifest["families"][family] = {
            "input_count": len(task_dirs),
            "unique_count": len(kept_tasks),
            "dropped_count": len(task_dirs) - len(kept_tasks),
            "kept_tasks": kept_tasks,
            "duplicate_groups": [group for group in duplicate_groups if group["member_count"] > 1],
        }

    manifest["total_input_count"] = total_before
    manifest["total_unique_count"] = total_after
    manifest["total_dropped_count"] = total_before - total_after
    return manifest


def _materialize_output(
    input_root: Path,
    output_root: Path,
    manifest: dict[str, Any],
    overwrite: bool,
) -> None:
    if output_root.exists():
        if not overwrite:
            raise SystemExit(
                f"Output root already exists: {output_root}. "
                "Pass --overwrite to rebuild it."
            )
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    for family, info in manifest["families"].items():
        family_output = output_root / family
        family_output.mkdir(parents=True, exist_ok=True)
        for task_name in info["kept_tasks"]:
            shutil.copytree(input_root / family / task_name, family_output / task_name)

    manifest_path = output_root / "dedup_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def _print_summary(manifest: dict[str, Any]) -> None:
    print("Dedup summary")
    print(f"  input_root:  {manifest['input_root']}")
    print(f"  output_root: {manifest['output_root']}")
    print(
        "  totals:"
        f" before={manifest['total_input_count']}"
        f" unique={manifest['total_unique_count']}"
        f" dropped={manifest['total_dropped_count']}"
    )
    for family, info in manifest["families"].items():
        print(
            f"  {family}: before={info['input_count']} "
            f"unique={info['unique_count']} dropped={info['dropped_count']}"
        )
        if info["duplicate_groups"]:
            sample = info["duplicate_groups"][0]
            print(
                "    sample_duplicate_group: "
                f"members={sample['members']} representative={sample['representative']}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate human-reviewed final base task sets per family using "
            "robot-frame rigid-object signatures, then materialize a unique output root."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Source reviewed task root (default: {DEFAULT_INPUT_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Destination unique task root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_GLOBAL_SEED,
        help=f"Global seed for deterministic representative choice (default: {DEFAULT_GLOBAL_SEED})",
    )
    parser.add_argument(
        "--pos-decimals",
        type=int,
        default=DEFAULT_POS_DECIMALS,
        help=f"Robot-frame relative position rounding decimals (default: {DEFAULT_POS_DECIMALS})",
    )
    parser.add_argument(
        "--quat-decimals",
        type=int,
        default=DEFAULT_QUAT_DECIMALS,
        help=f"Robot-frame relative quaternion rounding decimals (default: {DEFAULT_QUAT_DECIMALS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute signatures and print a summary without creating output files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild the output root if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if not input_root.is_dir():
        raise SystemExit(f"Input root does not exist: {input_root}")

    manifest = _build_manifest(
        input_root,
        output_root,
        seed=args.seed,
        pos_decimals=args.pos_decimals,
        quat_decimals=args.quat_decimals,
    )
    _print_summary(manifest)
    if args.dry_run:
        return 0

    _materialize_output(
        input_root,
        output_root,
        manifest,
        overwrite=args.overwrite,
    )
    print(f"Wrote deduplicated output root: {output_root}")
    print(f"Wrote manifest: {output_root / 'dedup_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
