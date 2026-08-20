"""Shared perturbation infrastructure for ManiGuard-Bench.

Every bench task — `base` or any perturbation level (`target` / `language` /
`location` / `env`) — is just a **task instance**: a directory with
`scene_ep1.json` + `diagnostics.jsonl` + 4 review videos. A consumer (the bench
renderer or the eval client) loads ANY instance the same way:

    cfg = build_og_config(scene_file, diag)   # adaptive: Scene vs room scene
    env = og.Environment(cfg); env.reset()
    apply_perturbation(env, diag)             # <- the only post-load branch

`apply_perturbation` is the single, uniform post-load hook. It reads the
instance's optional ``diagnostics["perturbation"]`` block and applies whatever
that level needs that cannot be baked into `scene_ep1.json`:

* ``target``   — recolor the target object (``diffuse_tint`` is an asset/USD
                 material property; OmniGibson does NOT serialize it into the
                 scene snapshot, so it must be re-applied on every load).
* ``location`` / ``env`` — object moves / room geometry are already baked into
                 `scene_ep1.json` (or the scene config), so this is a no-op;
                 the block is provenance only.
* ``language`` — only the prompt changes, no sim effect → no-op.
* ``base`` / absent — no-op.

So the consumer never branches on "is this a perturbation"; it always calls
``apply_perturbation`` and the dispatch is data-driven by ``kind``.

This module deliberately imports NOTHING from the legacy perturbation code
(``perturbation_scaling.py`` / ``perturbation_runtime.py``); the palette and the
material-override idiom are re-derived here.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

# Vivid, high-saturation candidate colors spread evenly around the hue wheel so
# every recolored target reads as a bold, instantly-distinguishable OOD color
# (no dark/muted entries). The recolor FORCES the object to ~this color via
# albedo_add + diffuse_tint (see ``apply_recolor``), so it shows regardless of
# the object's original brightness/texture.
APPEARANCE_COLOR_PALETTE: tuple[tuple[float, float, float], ...] = (
    (1.00, 0.13, 0.13),  # red     #FF2222
    (1.00, 0.53, 0.00),  # orange  #FF8800
    (1.00, 0.83, 0.00),  # yellow  #FFD400
    (0.16, 0.78, 0.16),  # green   #28C828
    (0.00, 0.75, 0.91),  # cyan    #00C0E8
    (0.78, 0.16, 0.85),  # magenta #C828D8
)

# Per-family "target" object — the manipuland whose appearance the `target`
# level recolors (your §4b definitions). Each family lists the candidate
# spawn-spec ROLE(s) (first that resolves wins); the concrete scene object is
# then found by that role's category. lid carries two provenances — the capped
# container is role "container" in older tasks, "target" in newer ones.
TARGET_ROLE: dict[str, tuple[str, ...]] = {
    "jar_transport": ("target",),            # hinged_jar
    "cabinet_pickup": ("target",),           # place target
    "clutter_pickup": ("target",),           # grasp target
    "stack_retrieve": ("target",),           # bottom object
    "lid_transport": ("container", "target"),  # the container being capped
    "dusty_transfer": ("source",),           # source container
}


def derive_seed(global_seed: int, *parts: Any) -> int:
    """Deterministic 32-bit seed from arbitrary string parts (sha256)."""
    payload = "|".join([str(int(global_seed))] + [str(p) for p in parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


# ---------------------------------------------------------------------------
# Target resolution + color selection
# ---------------------------------------------------------------------------

def _spawn_role_category(diag: dict) -> dict[str, str]:
    return {
        s.get("role"): s.get("category")
        for s in ((diag.get("selection") or {}).get("spawn_specs") or [])
    }


def resolve_target_category(diag: dict, family: str) -> str | None:
    """The category of the family's recolor target — the first candidate role
    (per ``TARGET_ROLE``) that resolves to a spawned category."""
    r2c = _spawn_role_category(diag)
    for role in TARGET_ROLE.get(family) or ():
        cat = r2c.get(role)
        if cat:
            return cat
    return None


def find_object_by_category(env, category: str):
    """First live scene object whose ``category`` matches."""
    if not category:
        return None
    for obj in env.scene.objects:
        if getattr(obj, "category", "") == category:
            return obj
    return None


def _grasp_reference(diag: dict):
    """The object name the goal grasps (``grasping robot <ref>``), searched
    recursively through goal_conditions. For stack-retrieve this is the BOTTOM
    object being pulled out; for clutter the grasp target."""
    def scan(node):
        if isinstance(node, dict):
            if node.get("predicate") == "grasping" and node.get("subject") == "robot":
                return node.get("reference")
            for v in node.values():
                r = scan(v)
                if r:
                    return r
        elif isinstance(node, list):
            for v in node:
                r = scan(v)
                if r:
                    return r
        return None
    return scan(diag.get("goal_conditions"))


def _obj_z(obj) -> float:
    p = obj.get_position_orientation()[0]
    return float(p[2].item() if hasattr(p[2], "item") else p[2])


def resolve_target_object(env, diag: dict, family: str):
    """The SPECIFIC live target object to recolor. Resolves the target category
    (per family role), then the concrete object. When several objects share that
    category — stack-SAME tasks stack identical objects, so target and stack are
    the same category — disambiguate to the actual manipuland: the goal's grasp
    reference (e.g. the bottom bowl ``bowl_45``), else the BOTTOM of the stack
    (lowest z, the retrieved object). Picking the first category match recolored
    the TOP of the stack instead of the bottom."""
    cat = resolve_target_category(diag, family)
    if not cat:
        return None
    matches = [o for o in env.scene.objects if getattr(o, "category", "") == cat]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    ref = _grasp_reference(diag)
    if ref:
        by_name = {o.name: o for o in matches}
        if ref in by_name:
            return by_name[ref]
    return min(matches, key=_obj_z)


_MIN_TINT_DIST = 0.35  # a tint this far (RGB) from the original reads as clearly different


def pick_tint(orig_rgb, task_index: int) -> list[float]:
    """Pick a saturated palette tint by CYCLING on the task index, so a family's
    variants rotate through visibly distinct colors (task 0→red, 1→orange,
    2→yellow, 3→green, 4→cyan, 5→magenta, then wraps). A light guard skips a
    palette color that happens to sit near the target's own color (rare — most
    manipulands are neutral), advancing to the next index so the recolor is always
    clearly out-of-distribution. Deterministic in the task index.
    """
    orig = np.asarray(orig_rgb, dtype=np.float32).reshape(3)
    n = len(APPEARANCE_COLOR_PALETTE)
    for k in range(n):
        c = APPEARANCE_COLOR_PALETTE[(int(task_index) + k) % n]
        if float(np.linalg.norm(np.asarray(c, dtype=np.float32) - orig)) >= _MIN_TINT_DIST:
            return [round(float(x), 4) for x in c]
    # every palette color sits near the original (very unlikely) → take the farthest
    idx = int(np.argmax([float(np.linalg.norm(np.asarray(c, dtype=np.float32) - orig))
                         for c in APPEARANCE_COLOR_PALETTE]))
    return [round(float(x), 4) for x in APPEARANCE_COLOR_PALETTE[idx]]


# ---------------------------------------------------------------------------
# Material recolor (re-derived from the legacy diffuse_tint idiom)
# ---------------------------------------------------------------------------

def _iter_materials(obj) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for m in (getattr(obj, "materials", []) or []):
        key = str(getattr(m, "prim_path", id(m)))
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


def average_object_color(obj) -> list[float] | None:
    """Mean diffuse color of the object's materials, NORMALIZED to 0-1.

    ``MaterialPrim.average_diffuse_color`` reports an 8-bit-style 0-255 color
    (e.g. mid-grey ~120); the palette + tint are 0-1, so we must rescale or the
    farthest-color distance is dominated by magnitude, not hue.
    """
    vals = []
    for m in _iter_materials(obj):
        c = getattr(m, "average_diffuse_color", None)
        if c is not None:
            arr = c.tolist() if hasattr(c, "tolist") else list(c)
            vals.append(np.asarray(arr, dtype=np.float32).reshape(-1)[:3])
    if not vals:
        return None
    mean = np.mean(vals, axis=0)
    if float(mean.max()) > 1.5:  # 0-255 scale -> normalize to 0-1
        mean = mean / 255.0
    return [round(float(x), 4) for x in mean]


def luminance(rgb) -> float:
    r, g, b = (float(x) for x in np.asarray(rgb, dtype=np.float32).reshape(3))
    return 0.299 * r + 0.587 * g + 0.114 * b


def albedo_add_for(orig_rgb) -> float:
    """The additive lift that raises a (possibly dark/textured) albedo to ~1 so
    the tint then renders as the full vivid color. ``1 - luminance(original)``."""
    return round(max(0.0, min(1.0, 1.0 - luminance(orig_rgb))), 4)


def apply_recolor(obj, tint_rgb, albedo_add: float = 0.0) -> int:
    """FORCE every material of ``obj`` to the vivid color, regardless of the
    object's original brightness/texture, via the engine's own recolor inputs:

        final_albedo = diffuse_tint * (orig_albedo + albedo_add)

    ``albedo_add`` lifts a dark/textured albedo up to ~1 (washing out the
    original color), then ``diffuse_tint`` colors it — so even a near-black tray
    becomes the vivid tint. This is exactly how OmniGibson recolors objects for
    Frozen / Cooked / Burnt states (``StatefulObject._update_texture_change``),
    so it is robust across asset materials. A plain multiplicative tint cannot
    brighten a dark albedo; the additive term is what makes this universal.
    Returns how many materials were recolored.
    """
    import torch as th

    tint = th.tensor(np.asarray(tint_rgb, dtype=np.float32).reshape(3), dtype=th.float32)
    n = 0
    for m in _iter_materials(obj):
        applied = False
        try:
            if hasattr(m, "albedo_add"):
                m.albedo_add = float(albedo_add)
            if hasattr(m, "diffuse_tint"):
                m.diffuse_tint = tint
                applied = True
            elif hasattr(m, "diffuse_color_constant"):  # primitives (no texture/tint slot)
                m.diffuse_color_constant = tint
                applied = True
        except Exception:
            pass
        n += int(applied)
    return n


# ---------------------------------------------------------------------------
# The uniform post-load hook
# ---------------------------------------------------------------------------

def apply_perturbation(env, diag: dict) -> dict:
    """Apply whatever the instance's ``perturbation`` block needs at load time.

    The ONE post-load branch every consumer runs after building + resetting the
    env. Data-driven by ``perturbation.kind``; a no-op for base and for levels
    whose change is already baked into the scene. Returns a small status dict
    for logging/QC.
    """
    pert = diag.get("perturbation") or {}
    kind = pert.get("kind")
    if not kind or kind in ("base", "language", "location", "env"):
        return {"kind": kind or "base", "applied": False}

    if kind == "target":
        rc = pert.get("recolor") or {}
        obj = (env.scene.object_registry("name", rc.get("object"))
               if rc.get("object") else None)
        if obj is None:
            obj = find_object_by_category(env, rc.get("category"))
        if obj is None or not rc.get("diffuse_tint"):
            return {"kind": kind, "applied": False, "reason": "target not resolved"}
        n = apply_recolor(obj, rc["diffuse_tint"], rc.get("albedo_add", 0.0))
        import omnigibson as og
        og.sim.step()
        return {"kind": kind, "applied": n > 0, "object": obj.name, "n_materials": n}

    return {"kind": kind, "applied": False, "reason": f"unknown kind {kind!r}"}
