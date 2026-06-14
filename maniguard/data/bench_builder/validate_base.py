"""Exhaustive OFFLINE QC for a finalized ManiGuard-Bench base task.

``validate_base_task`` reads a finalized ``maniguard-bench/<fam>/task_NNNN/base/`` dir and checks
every property the bench must guarantee — WITHOUT loading OmniGibson (reads the saved
``scene_ep1.json`` + ``diagnostics.jsonl`` + the 4 mp4 files). "Loads cleanly into the sim" is
already evidenced by finalize having rendered the 4 videos; this pass is the independent
artifact-level gate the user reviews (alongside the videos) before locking the base set.

Checks (design doc §8 P1.2): robot identity (FrankaPanda), init pose A, mount (base_z =
support_top + offset, from the finalize provenance block), task objects present + none fallen,
goal-region marker present (5 families) / absent (dusty), 4 cameras = canonical poses, LTL Tier-A
(every proposition pattern resolves to a real object via the benchmark-equivalent matcher), 4
non-empty 256² videos, diagnostics family fields preserved.
"""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path

from maniguard.utils.camera_setup import EXTERNAL_CAMERA_NAMES
from maniguard.utils.robot_pose import BENCH_INIT_QPOS, ROBOT_MOUNT_OFFSET

POSE_TOL = 1e-2          # rad
BASE_Z_TOL = 5e-3        # m
RESOLUTION = 256
N_FRAMES = 60
# Per-family diagnostics fields that finalize must preserve (besides the universal ones).
FAMILY_DIAG_FIELDS = {
    "jar_transport": ["jar_info", "item_info"],
    "cabinet_pickup": ["cabinet_info"],
    "stack_retrieve": ["stack_mode"],
    "dusty_transfer": ["dust_system"],
    "clutter_pickup": ["clutter_info"],   # derived in finalize
    "lid_transport": ["lid_info"],        # derived in finalize
}
# Owned-schema fields the finalizer always writes (task-def carried + fresh in-sim). The goal spec
# is goal_conditions (universal); goal_region (the sphere marker) is conditional — cabinet has none
# (goal = inside-cabinet + closed) so it is NOT required here (the data-driven marker check handles it).
UNIVERSAL_DIAG_FIELDS = ["surface", "prompt", "selection", "ltl_safety", "cameras", "goal_conditions",
                         "surface_info", "gate_pass", "ltl_violated", "steps_executed", "bench"]


def _load_diag(base_dir: Path) -> dict:
    return json.JSONDecoder().raw_decode((base_dir / "diagnostics.jsonl").read_text(encoding="utf-8").lstrip())[0]


def _robot_entry(header: dict):
    init = header.get("objects_info", {}).get("init_info", {})
    reg = header.get("state", {}).get("registry", {}).get("object_registry", {})
    for name, info in init.items():
        cn = info.get("class_name", "")
        if info.get("class_module", "").startswith("omnigibson.robots.") or cn.endswith(("Robot", "Mounted", "Panda")):
            return name, info, reg.get(name, {})
    return None, None, None


_TAXONOMY = None


def _category_lemma(category: str) -> str:
    """OG category -> BDDL synset lemma (e.g. cocktail_glass -> goblet), via the bddl object
    taxonomy DIRECTLY (same bridge eval uses, but no eval/imageio/og import). '' if no synset.
    Does NOT swallow a missing-taxonomy error — a silent '' would manufacture false LTL failures.
    """
    global _TAXONOMY
    if not category:
        return ""
    if _TAXONOMY is None:
        from bddl.object_taxonomy import ObjectTaxonomy
        _TAXONOMY = ObjectTaxonomy()
    syn = _TAXONOMY.get_synset_from_category(category)
    return syn.split(".n.")[0] if syn else ""


def _as_list(v):
    return v if isinstance(v, list) else ([v] if isinstance(v, str) else [])


def _resolve_pattern(pat: str, names, cat2lemma) -> tuple[list[str], bool]:
    """Objects matching one LTL pattern (category / taxonomy-lemma / name fnmatch); + whether it
    is a `.n.` synset pattern (eligible for eval's surface fallback)."""
    prefix = pat[:-2] if pat.endswith("_*") else pat
    if prefix.startswith("agent"):
        return ["__robot__"], False
    base = prefix.split(".n.")[0]
    matched = [n for n, c in names if c == base or cat2lemma.get(c) == base]
    matched += [n for n, c in names if n not in matched and fnmatch.fnmatch(n, pat)]
    return matched, (".n." in prefix)


def _resolve_ltl(ltl: dict, init: dict, surface_name: str | None) -> list[tuple[bool, str]]:
    """Resolution check at the PROPOSITION level (eval semantics: a proposition is vacuous only if
    its `over` subject set is empty across ALL its patterns — a multi-category `over` with one
    absent category is fine). Returns (is_fail, message): over-empty = fail (vacuous safety);
    a relative_to/surface pattern resolvable only via the surface fallback (or not at all) = warn.
    """
    names = [(n, info.get("args", {}).get("category", "")) for n, info in init.items()]
    cat2lemma = {c: _category_lemma(c) for _, c in names if c}
    out: list[tuple[bool, str]] = []
    for prop_name, prop in (ltl.get("propositions") or {}).items():
        over_pats = _as_list(prop.get("over"))
        if over_pats:
            over_hit: set[str] = set()
            for p in over_pats:
                over_hit.update(_resolve_pattern(p, names, cat2lemma)[0])
            if not over_hit:
                out.append((True, f"{prop_name}: over {over_pats} -> 0 objects (vacuous safety)"))
        for key in ("relative_to", "surface"):
            for p in _as_list(prop.get(key)):
                matched, is_synset = _resolve_pattern(p, names, cat2lemma)
                if matched:
                    continue
                if is_synset and surface_name:
                    out.append((False, f"{prop_name}: {key} {p!r} resolves ONLY via surface fallback"))
                else:
                    out.append((False, f"{prop_name}: {key} {p!r} -> 0 objects"))
    return out


def _probe_mp4(path: Path):
    """(width, height, n_frames) by decoding; None if unreadable."""
    try:
        import av
        with av.open(str(path)) as c:
            stream = c.streams.video[0]
            w, h = stream.codec_context.width, stream.codec_context.height
            n = sum(1 for _ in c.decode(stream))
        return int(w), int(h), int(n)
    except Exception:
        return None


def validate_base_task(out_base_dir, *, family: str, episode: int = 1) -> dict:
    """Offline QC of a finalized base task. Returns a dict with per-check results + status."""
    out_base_dir = Path(out_base_dir)
    checks: dict[str, bool] = {}
    warnings: list[str] = []
    fails: list[str] = []

    scene_path = out_base_dir / f"scene_ep{episode}.json"
    if not scene_path.exists():
        return {"task": out_base_dir.parent.name, "family": family, "status": "fail",
                "fails": [f"missing {scene_path.name}"], "warnings": [], "checks": {}}
    header = json.loads(scene_path.read_text(encoding="utf-8"))
    diag = _load_diag(out_base_dir)
    init = header.get("objects_info", {}).get("init_info", {})
    bench = diag.get("bench", {})

    # 1. robot identity (invariant #1): FrankaPanda (longfinger is guaranteed by the import patch on load)
    rname, rinfo, rstate = _robot_entry(header)
    checks["robot_FrankaPanda"] = rinfo is not None and rinfo.get("class_name") == "FrankaPanda"
    if not checks["robot_FrankaPanda"]:
        fails.append(f"robot class != FrankaPanda ({rinfo and rinfo.get('class_name')})")

    # 2. init pose A (invariant #5)
    jp = rstate.get("joint_pos") if rstate else None
    checks["pose_A"] = jp is not None and len(jp) == len(BENCH_INIT_QPOS) and max(
        abs(float(a) - b) for a, b in zip(jp, BENCH_INIT_QPOS)) < POSE_TOL
    if not checks["pose_A"]:
        fails.append("joint_pos != BENCH_INIT_QPOS")

    # 3. mount: base_z == support_top + offset (from finalize provenance)
    base_z = float(rstate["root_link"]["pos"][2]) if rstate and rstate.get("root_link", {}).get("pos") else None
    support_top = bench.get("support_top")
    if base_z is not None and support_top is not None:
        checks["mount"] = abs(base_z - (support_top + ROBOT_MOUNT_OFFSET)) < BASE_Z_TOL
        if not checks["mount"]:
            fails.append(f"base_z {base_z:.4f} != support_top+{ROBOT_MOUNT_OFFSET} ({support_top + ROBOT_MOUNT_OFFSET:.4f})")
    else:
        checks["mount"] = False
        warnings.append("mount uncheckable (no base_z / provenance support_top)")

    # 4. software config declared correctly (raw + assisted) in the saved robot init_info
    args = rinfo.get("args", {}) if rinfo else {}
    arm0 = args.get("controller_config", {}).get("arm_0", {})
    checks["config_canonical"] = (
        args.get("grasping_mode") == "assisted"
        and args.get("action_normalize") is False
        and arm0.get("name") == "JointController"
        and arm0.get("command_input_limits") is None
    )
    if not checks["config_canonical"]:
        warnings.append("robot config not canonical raw+assisted")

    # 5. objects: surface present + task-object count sane + none fallen far below the surface
    non_robot = {n: info for n, info in init.items() if n != rname}
    surface = diag.get("surface")
    # surface is either an object name (clutter/lid/stack/dusty, e.g. "desk_qpuflh_2") or a
    # "category/model" string (jar/cabinet, e.g. "desk/obvwds"); resolve via name, the
    # goal_region's support_name, or a category match.
    gr_support = diag.get("goal_region", {}).get("support_name")
    surf_cat = surface.split("/")[0] if surface and "/" in surface else None
    surf_present = bool(
        (surface in init)
        or (gr_support and gr_support in init)
        or (surf_cat and any(info.get("args", {}).get("category", "") == surf_cat for info in init.values()))
    )
    checks["surface_present"] = surf_present
    if not surf_present:
        warnings.append(f"surface {surface!r} not resolvable in snapshot")
    spawn = diag.get("selection", {}).get("spawn_specs", []) or []
    n_task = sum(int(s.get("count", 1)) for s in spawn)
    # data-driven: a task has a goal marker iff its goal_region declares one (clutter/lid/jar/stack
    # do; cabinet's goal is inside-cabinet+closed with NO marker, dusty has none) — no hardcoded list.
    goal_region = diag.get("goal_region") or {}
    marker_name = goal_region.get("marker_name")
    marker_expected = bool(marker_name)
    expected_non_robot = 1 + n_task + (1 if marker_expected else 0)  # surface + task objs + marker
    checks["object_count"] = len(non_robot) == expected_non_robot
    if not checks["object_count"]:
        warnings.append(f"non-robot objs {len(non_robot)} != expected {expected_non_robot} (surface+{n_task}+marker)")
    reg = header.get("state", {}).get("registry", {}).get("object_registry", {})
    # The support surface and the goal marker are NOT manipulands: a surface's origin can sit
    # well below its own top plane (a tall bar's centre is ~0.6 m under its top), so exclude
    # them from the "fallen task object" check — only real task objects must stay on the surface.
    # The surface entity may be named anything (jar: "desk_ep1_1", cabinet: "support_surface"), and
    # the goal-marker families resolve it via gr_support. For the "category/model" surface form
    # (jar/cabinet) with no goal_region (cabinet), resolve the entity by BOTH category and model so
    # a same-category manipuland is never excluded by accident.
    structural = {c for c in (surface, gr_support, marker_name) if c and c in init}
    if surf_cat:
        surf_model = surface.split("/", 1)[1] if "/" in surface else None
        for n, info in init.items():
            a = info.get("args", {})
            if a.get("category") == surf_cat and (surf_model is None or a.get("model") == surf_model):
                structural.add(n)
    if support_top is not None:
        fallen = [n for n in non_robot
                  if n not in structural
                  and reg.get(n, {}).get("root_link", {}).get("pos")
                  and float(reg[n]["root_link"]["pos"][2]) < support_top - 0.3]
        checks["no_fallen"] = not fallen
        if fallen:
            fails.append(f"task objects fallen below surface: {fallen}")
    else:
        checks["no_fallen"] = True

    # 6. goal-region marker: required iff the task declares one (data-driven, see marker_expected)
    has_marker = bool(marker_name) and marker_name in init
    if marker_expected:
        checks["goal_marker"] = has_marker
        if not has_marker:
            fails.append(f"declared goal marker {marker_name!r} missing from scene")
    else:
        checks["goal_marker"] = True  # no marker declared (cabinet/dusty) -> nothing to check

    # 7. cameras (invariant #4): 4 canonical poses, correct sensor names, lookat above the surface
    cams = diag.get("cameras", []) or []
    cam_names = {c.get("sensor_name") for c in cams}
    checks["cameras"] = len(cams) == 4 and cam_names == set(EXTERNAL_CAMERA_NAMES)
    if not checks["cameras"]:
        fails.append(f"cameras {sorted(cam_names)} != {sorted(EXTERNAL_CAMERA_NAMES)}")
    elif support_top is not None:
        low = [c["sensor_name"] for c in cams
               if c.get("lookat") and float(c["lookat"][2]) < support_top - 0.3]
        if low:
            warnings.append(f"camera lookat below surface: {low}")

    # 8. LTL Tier-A: every proposition's `over` resolves to >=1 real object (proposition-level)
    ltl_problems = _resolve_ltl(diag.get("ltl_safety") or {}, init, surface)
    checks["ltl_resolves"] = not any(is_fail for is_fail, _ in ltl_problems)
    for is_fail, msg in ltl_problems:
        (fails if is_fail else warnings).append(f"LTL {msg}")

    # 9. videos: 4 non-empty, 256x256, N frames, decodable
    probes = {}
    for v in EXTERNAL_CAMERA_NAMES:
        label = {"cam_opposite": "opposite_side_front", "cam_left": "left_overview",
                 "cam_right": "right_overview", "cam_left_shoulder": "left_shoulder"}[v]
        mp4 = out_base_dir / f"rollout_{label}_ep{episode}.mp4"
        probes[label] = _probe_mp4(mp4) if mp4.exists() and mp4.stat().st_size > 0 else None
    bad = [k for k, pr in probes.items() if pr is None or (pr[0], pr[1]) != (RESOLUTION, RESOLUTION)]
    checks["videos"] = len(probes) == 4 and not bad
    if bad:
        fails.append(f"bad/missing videos: {bad}")
    short = [k for k, pr in probes.items() if pr and pr[2] != N_FRAMES]
    if short:
        warnings.append(f"video frame count != {N_FRAMES}: {[(k, probes[k][2]) for k in short]}")

    # 10. diagnostics integrity: universal + family-specific fields preserved
    missing_fields = [f for f in UNIVERSAL_DIAG_FIELDS + FAMILY_DIAG_FIELDS.get(family, []) if f not in diag]
    checks["diag_fields"] = not missing_fields
    if missing_fields:
        fails.append(f"diagnostics missing fields: {missing_fields}")

    # 11. fresh bench in-sim verdicts: spawn-gate passed + init scene not already LTL-violating
    checks["gate_pass"] = bool(diag.get("gate_pass"))
    if not checks["gate_pass"]:
        fails.append(f"gate_pass=False ({(diag.get('bench') or {}).get('gate')})")
    checks["ltl_not_violated"] = not bool(diag.get("ltl_violated"))
    if not checks["ltl_not_violated"]:
        fails.append("ltl_violated=True (init scene violates LTL over the idle-step)")

    status = "fail" if fails else ("warn" if warnings else "ok")
    return {
        "task": out_base_dir.parent.name,
        "family": family,
        "status": status,
        "checks": checks,
        "fails": fails,
        "warnings": warnings,
    }
