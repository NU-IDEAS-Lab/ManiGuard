# Grasp annotation (steps 1–2)

The pipeline does not auto-generate grasps — the grasp source is **per-instance human
annotation**, authored once per object and shared by all six families. This page covers
the two prep steps: extract the meshes, then annotate grasps on them.

## 1. Mesh extraction

OmniGibson assets are *encrypted USD* — meshes can only be read through OG. Extract each
grasp target's mesh **once** so the annotation tool (step 2) loads instantly with no
sim. Code: `annotation/extract_meshes.py`.

```bash
# meshes + metadata DB for one or all families
python -u -m maniguard.data.datagen.annotation.extract_meshes --families clutter_pickup
python -u -m maniguard.data.datagen.annotation.extract_meshes --families all
python -u -m maniguard.data.datagen.annotation.extract_meshes --gripper   # the gripper mesh (once)
```

Output: `outputs/grasp_annotation/{meshes/*.glb, gripper_longfinger.glb, mesh_db.json}`.

??? note "▸ What it does in detail"
    Enumerates the distinct `(category, model)` grasp **targets** across the bench
    families (pure JSON: each task's `diagnostics` goal target + `scene_ep1` model), then
    in **one OG session** spawns every target `visual_only` in a grid, exports its
    **object-local** GLB, and records `bbox_size` + the **upright world orientation** (how
    it stands in the scene). Also dumps the longfinger gripper mesh in the **eef_link
    frame** (`--gripper`). Distractors are never grasped, so not extracted. The cabinet's
    articulated handle is extracted separately (`extract_cabinet` → `cabinet_geom.json`),
    since all cabinet tasks share one cabinet model.

## 2. Grasp annotation

Each grasp is stored as an **`eef_link` target pose** in the object-**local** frame
`(position, quat_xyzw)`, in **one JSON database shared by all six families**. At
runtime `T_eef_world = T_object_world @ T_grasp_local` feeds straight to cuRobo IK —
**zero conversion** (verified object-relative and exact to ~1e-16). Because a grasp lives
on the *object model*, it is **family-agnostic**: switching family only changes which
objects you filter to annotate (`--family`).

The tool is a [viser](https://viser.studio) web GUI. It shows the object **upright** with
the world axes and the real longfinger gripper drawn at the draft pose; **Enter** saves
each grasp incrementally (reopening auto-resumes, never overwrites).

```bash
python -m maniguard.data.datagen.annotation.annotate_tool --family clutter   # → localhost:8080
```

<img src="../annotation_gui.png" alt="Grasp annotation GUI" style="max-width:100%; border-radius:6px;">
<p style="text-align:center;"><em>The annotation GUI: the longfinger gripper drawn at the draft grasp, with the guided panel and the 6-DoF gizmo.</em></p>

Two input modes, both producing the **same** record:

- **Guided** (default): click a surface point + pick an approach preset (`top_down` /
  `side_*`) + yaw/depth sliders. Fastest for regular grasps.
- **Free** (toggle): a 6-DoF drag gizmo for odd semantic grasps (rim / handle / stem).

??? note "▸ The grasp database — frame & schema"
    - **File**: `outputs/grasp_annotation/grasp_annotations.json` (**gitignored** — it is
      user data). **Key**: `"category/model"` (per-object, not per-task).
    - **Gripper eef-local frame** (sim-measured): fingertips / **approach = eef +Z**,
      **closing (between fingers) = eef −Y**. The GUI draws the gripper using this frame.
    - **Prerequisite**: `mesh_db.json` (step 1 output; the GUI renders objects from it).

    ```json
    { "objects": { "<category>/<model>": {
        "bbox_size": [..], "upright_orientation_xyzw": [..], "mesh": "meshes/...glb",
        "grasps": [ { "id": 0, "position": [..], "orientation_xyzw": [..],
                      "approach_hint": "top_down|side", "label": "" } ] } } }
    ```

??? example "▸ QC trio — after annotating"
    Recommended cadence per family: annotate → `fix_approach_tags --apply` → `mesh_review`
    (seconds-scale self-check) → confirm → next family. Defer the heavy `validate_grasps`
    sim to a later consolidated pass.

    1. **`fix_approach_tags.py`** (no sim) rewrites each grasp's `approach_hint` from its
       **actual stored pose** (the GUI dropdown is often left wrong): `top_down 0–60° /
       side 60–120° / bottom_up 120–180°`. Default dry-run; `--apply` writes back.
    2. **`mesh_review.py`** (no sim, seconds/object) writes one PNG per object — object +
       gripper point clouds at each grasp from 3 viewpoints, approach derived from the pose.
    3. **`validate_grasps.py`** (heavy sim, one object per process) teleports the base so
       `eef_link` lands **exactly** on the grasp and renders the 4 bench cameras + a 3D
       closeup — confirming the grasp lands on the intended part.

    ```bash
    python    -m maniguard.data.datagen.annotation.fix_approach_tags --family clutter --apply
    python    -m maniguard.data.datagen.annotation.mesh_review       --family clutter
    python -u -m maniguard.data.datagen.annotation.validate_grasps   --object <cat>/<model>
    ```

??? note "▸ Programmatic top-down grasps (sticky-grasp families)"
    For **sticky-grasp** families where force-closure is impossible (e.g. a cabinet slab
    wider than the gripper), `generate_topdown_grasps.py` synthesizes straight-down grasps
    (approach = world −Z, a fan of yaws, centred over the top face) — avoiding the failure
    modes of hand-annotated edge grasps on slabs (tip-over, wrist droop, palm-flip). Default
    dry-run; `--apply` writes into the DB.

    ```bash
    python -m maniguard.data.datagen.annotation.generate_topdown_grasps --cabinet-all --n-yaw 6 --apply
    ```

**Next step →** [Collection](collection.md)
