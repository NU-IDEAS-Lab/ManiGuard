"""Shared perturbation metadata and runtime appearance overrides.

Task instances store a scene snapshot and diagnostics. Builder runs can also
write four review videos. After loading a snapshot, consumers should call
apply_perturbation(env, diagnostics) to apply material properties that are not
serialized in the scene snapshot.

Target variants store their recoloring parameters in diagnostics. Environment
and location variants store geometry changes in the scene snapshot; language
variants store the rewritten prompt in diagnostics. These three kinds require
no additional action from this hook. Missing or base perturbation metadata is
also a no-op.
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

# Candidate spawn-spec roles for each family's appearance target. The first
# role with a category is used; lid tasks accept container or target.
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
    """Resolve the live manipuland used for appearance perturbation.

    Match the family's target category. When several objects match, prefer
    the goal's grasp reference; otherwise choose the object with the lowest
    world z coordinate. This selects the bottom object in same-category stacks.
    """
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
    """Select a palette color deterministically from the task index.

    Cycle through the palette starting at task_index, skipping colors whose
    RGB distance from orig_rgb is below _MIN_TINT_DIST. If none qualifies,
    return the farthest palette color.
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
# Material recolor
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
    """Set material tint and additive albedo adjustment where supported.

    For textured materials, the renderer combines diffuse_tint with the
    original albedo and albedo_add. The additive term brightens dark materials.
    Primitive materials can instead receive diffuse_color_constant.
    Return the number of materials whose color property was set successfully.
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
