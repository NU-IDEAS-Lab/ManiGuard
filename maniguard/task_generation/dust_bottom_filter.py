#!/usr/bin/env python3
"""Filter serialized dust particles to a container's lowest particle layer.

For snapshots with one dust group, retain particles at z <= min(z) + tol
and rebuild the associated arrays and counts. Leave sufficiently flat
particle distributions unchanged. The operation reads saved scene JSON and
does not simulate or revalidate the task.

The CLI previews changes by default. --apply writes the filtered snapshot;
use the backup option when retaining the input file is required.

Example:
    python -m maniguard.task_generation.dust_bottom_filter --scene /path/to/task/base/scene_ep1.json"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

_DUST_SYSTEM_NAME = "dust"
# Keep particles within this many metres of the lowest particle (the
# container bottom plane). The default tolerance is 20 mm.
DEFAULT_TOL_M = 0.020
# Containers whose dust z-spread is below this are treated as flat — there
# is no meaningful bottom-vs-wall split, so keep every particle.
DEFAULT_FLAT_SPREAD_M = 0.030


def refine_scene_dust_to_bottom(
    scene_ep_path,
    tol=DEFAULT_TOL_M,
    flat_spread=DEFAULT_FLAT_SPREAD_M,
    apply=False,
    backup_dir=None,
):
    """Keep only the bottom-plane (``z <= z_min + tol``) dust particles in
    one ``scene_ep1.json``.

    Edits the file in place when ``apply`` is True, rebuilding the single
    dust group's parallel arrays + counts so the snapshot stays
    self-consistent. Single dust group only (each dusty task dusts one
    dest). Returns a human-readable status string; no exception on the
    no-dust / flat / nothing-to-drop cases (those are normal skips).
    """
    # Label with the last two path components (e.g. ``task_0000/base``) so
    # batch output stays legible when every scene file is named the same.
    _d = os.path.dirname(scene_ep_path)
    name = os.path.join(os.path.basename(os.path.dirname(_d)),
                        os.path.basename(_d)) or scene_ep_path
    with open(scene_ep_path) as f:
        d = json.load(f)

    sysreg = (
        d.get("state", {}).get("registry", {}).get("system_registry", {})
    )
    if _DUST_SYSTEM_NAME not in sysreg:
        return f"{name}: no dust system — skip"
    dust = sysreg[_DUST_SYSTEM_NAME]
    pos = dust.get("positions", [])
    n0 = len(pos)
    if n0 == 0:
        return f"{name}: 0 particles — skip"
    if dust.get("n_groups", 1) != 1 or len(dust.get("groups", {})) != 1:
        return (
            f"{name}: !! {dust.get('n_groups')} groups (expected 1) — "
            f"SKIP, handle manually"
        )

    zs = [xyz[2] for xyz in pos]
    spread = max(zs) - min(zs)
    if spread < flat_spread:
        return (
            f"{name}: flat (spread {spread * 1000:.1f}mm < "
            f"{flat_spread * 1000:.0f}mm) — keep all {n0}"
        )

    z_min = min(zs)
    keep = [i for i in range(n0) if zs[i] <= z_min + tol]
    nk = len(keep)
    if nk == 0:
        return f"{name}: keep-set empty?! — SKIP"
    if nk == n0:
        return f"{name}: tol keeps all {n0} (no wall particles) — no change"

    gname, g = next(iter(dust["groups"].items()))
    idns = g["particle_idns"]
    refs = g["particle_attached_references"]

    if not apply:
        return (
            f"{name}: {gname} spread={spread * 1000:.1f}mm  "
            f"{n0} -> {nk}  (would drop {n0 - nk} wall/rim)"
        )

    # -- rebuild every parallel array + counter consistently --------------
    dust["positions"] = [dust["positions"][i] for i in keep]
    dust["orientations"] = [dust["orientations"][i] for i in keep]
    dust["scales"] = [dust["scales"][i] for i in keep]
    dust["n_particles"] = nk
    kept_idns = [idns[i] for i in keep]
    dust["particle_counter"] = max(kept_idns) + 1
    g["n_particles"] = nk
    g["particle_idns"] = kept_idns
    g["particle_indices"] = list(range(nk))
    g["particle_attached_references"] = [refs[i] for i in keep]

    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        safe = name.replace(os.sep, "_")
        shutil.copy2(
            scene_ep_path,
            os.path.join(backup_dir, f"{safe}_scene_ep1.json"),
        )
    with open(scene_ep_path, "w") as f:
        json.dump(d, f)
    return (
        f"{name}: {gname} APPLIED  {n0} -> {nk}  "
        f"(dropped {n0 - nk} wall/rim"
        + ("; backup saved" if backup_dir else "")
        + ")"
    )


def _discover_scene_files(root):
    """Find every ``scene_ep1.json`` under a task-set root.

    Handles both the bench layout (``task_*/base/scene_ep1.json`` or
    ``task_*/<subdir>/scene_ep1.json``) and a flat ``task_*/scene_ep1.json``.
    """
    hits = sorted(glob.glob(os.path.join(root, "task_*", "**", "scene_ep1.json"),
                            recursive=True))
    if hits:
        return hits
    return sorted(glob.glob(os.path.join(root, "**", "scene_ep1.json"),
                            recursive=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", default=None,
                     help="Task-set root; refine every task_*/.../scene_ep1.json under it.")
    src.add_argument("--scene", default=None,
                     help="Single scene_ep1.json to refine.")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL_M,
                    help=f"Keep particles within this many metres of the "
                         f"lowest one (default {DEFAULT_TOL_M}).")
    ap.add_argument("--flat-spread", type=float, default=DEFAULT_FLAT_SPREAD_M,
                    help=f"Skip (keep all) when dust z-spread is below this "
                         f"(default {DEFAULT_FLAT_SPREAD_M}).")
    ap.add_argument("--apply", action="store_true",
                    help="Write changes in place (default: dry-run).")
    ap.add_argument("--backup-dir", default=None,
                    help="If set, copy each pre-edit scene_ep1.json here.")
    args = ap.parse_args()

    if args.scene:
        scenes = [args.scene]
    else:
        scenes = _discover_scene_files(args.root)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"{mode} | tol={args.tol * 1000:.0f}mm flat<{args.flat_spread * 1000:.0f}mm "
          f"| {len(scenes)} scene(s)\n")
    changed = 0
    for s in scenes:
        msg = refine_scene_dust_to_bottom(
            s, tol=args.tol, flat_spread=args.flat_spread,
            apply=args.apply, backup_dir=args.backup_dir,
        )
        print("  " + msg)
        if "APPLIED" in msg:
            changed += 1
    if args.apply:
        print(f"\napplied to {changed} scene(s)"
              + (f" (backups in {args.backup_dir})" if args.backup_dir else ""))


if __name__ == "__main__":
    main()
