# Grasp annotation

The scripted [datagen pipeline](pipeline.md) does not auto-generate grasps — its
grasp source is **per-instance human annotation**. Each object that ever needs to
be grasped (a task target, a relocated obstacle, a cabinet handle) gets its grasp
poses annotated once against its mesh and stored in **one JSON database shared by
all six families**. At collection time the executor looks each grasp up by
`category/model` and feeds it to cuRobo.

Because a grasp is annotated on the *object model*, it is **family-agnostic**:
switching family only changes which objects you filter to annotate (`--family`).

Code: `maniguard/data/datagen/annotation/`. Data: `outputs/grasp_annotation/`
(**gitignored** — it is user data).

## The shared grasp database

- **File**: `outputs/grasp_annotation/grasp_annotations.json`
- **Key**: `"category/model"` (per-object, *not* per-task/per-family) — all six
  families read and extend the same file.
- **Each grasp** = a target pose for the **`eef_link`**, stored in the
  **object-local frame**: `position` + `orientation_xyzw`.
- **Runtime bridge** (`grasp_db.py`): `T_eef_world = T_object_world @ T_grasp_local`
  → fed straight to cuRobo IK, **zero conversion** (verified object-relative and
  exact to ~1e-16; `validate_grasps` confirms 0.0 mm in sim).
- **Gripper eef-local frame** (sim-measured): fingertips / **approach = eef +Z**,
  **closing (between fingers) = eef −Y**. The GUI draws the real longfinger
  gripper at the draft pose using this frame.
- **Prerequisite mesh DB**: `outputs/grasp_annotation/mesh_db.json` (Phase A
  output; the GUI renders objects from it). Cabinet also uses `cabinet_geom.json`
  (handle / drawer geometry).

## Environment

```bash
conda activate behavior
export PYTHONPATH=/path/to/ManiGuard               # repo root
# sim steps (Phase A / validate_grasps) also need:
export OMNIGIBSON_HEADLESS=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json   # point at a valid local ICD
export CUDA_VISIBLE_DEVICES=0
```

## Workflow

```
Phase A: extract meshes (sim, one-time)  →  Phase B: GUI annotate (no sim)  →  QC trio (mostly no sim)
  extract_meshes / extract_cabinet          annotate_tool                     fix_approach_tags → mesh_review → validate_grasps
```

## Phase A — mesh extraction (needs sim, one-time)

OmniGibson assets are encrypted USD — meshes can only be read through OG. This
step enumerates each family's grasp **targets**, spawns them, and exports each
object's local mesh → GLB + bbox + upright orientation into `mesh_db.json`
(distractors are not grasped, so not extracted).

```bash
python -u -m maniguard.data.datagen.annotation.extract_meshes --families clutter_pickup   # [--limit N] [--gripper]
# the cabinet's articulated handle (all cabinet tasks share one cabinet model):
python -u -m maniguard.data.datagen.annotation.extract_cabinet
```

## Phase B — GUI annotation (`annotate_tool`, no sim)

```bash
python -m maniguard.data.datagen.annotation.annotate_tool --family clutter
```

- The terminal prints **`http://localhost:8080`** — open it for the
  [viser](https://viser.studio) GUI. The object is shown **upright** (its scene
  orientation) with the world XYZ axes and the longfinger gripper drawn at the
  current draft grasp.
- **Two input modes, both producing the same stored record:**
  - **Guided** (default): click a surface point + pick an approach preset
    (`top_down` / `side_*`) + drag the yaw/depth sliders. Fastest for regular
    top/side grasps.
  - **Free** (toggle): a 6-DoF drag gizmo on the gripper for odd semantic grasps
    (rim / handle / stem).
- **Save** with **Enter** (or "✚ save grasp"): each grasp is written
  **incrementally** to `grasp_annotations.json`; reopening **auto-resumes** and
  never overwrites existing grasps. One object may carry several grasps.
- `--family X` only filters which objects are listed; omit it for all objects.

## QC trio — after annotating

1. **`fix_approach_tags.py`** (no sim) rewrites each grasp's `approach_hint` from
   its **actual stored pose** (the GUI dropdown is often left wrong). Bands:
   `top_down 0–60° / side 60–120° / bottom_up 120–180°`; ambiguous cases are
   reported, not changed. Default dry-run; `--apply` writes back.
   ```bash
   python -m maniguard.data.datagen.annotation.fix_approach_tags --family clutter --apply
   ```
2. **`mesh_review.py`** (no sim, seconds/object — the point-cloud review) writes
   **one PNG per object**: the object point cloud + the gripper point cloud drawn
   at each annotated grasp, from 3 viewpoints (oblique / side / top) with a fixed
   world-XYZ triad. The labelled approach is derived from the stored pose (not the
   dropdown). Use it immediately after annotating to eyeball that the grasps sit
   where intended.
   ```bash
   python -m maniguard.data.datagen.annotation.mesh_review --family clutter   # [--object cat/model]
   # → outputs/grasp_annotation/mesh_review/<cat__model>.png
   ```
3. **`validate_grasps.py`** (heavy sim, run later in a time block) loads each
   grasp into its source task, teleports the robot base so `eef_link` lands
   **exactly** on the grasp (no IK ambiguity), hides the non-gripper arm, and
   renders the 4 bench cameras + a matplotlib 3D closeup — confirming the grasp
   really lands on the intended part. Run **one object per process**.
   ```bash
   python -u -m maniguard.data.datagen.annotation.validate_grasps --object <cat>/<model>   # key = DB category/model; [--limit N]
   ```

**Recommended cadence:** annotate a family → `fix_approach_tags --apply` →
`mesh_review` (seconds-scale self-check) → confirm → next family. Defer the heavy
`validate_grasps` sim to a later consolidated pass (mesh-vs-sim cross-check).

## Optional — programmatic top-down grasps

For **sticky-grasp** families where force-closure is impossible (e.g. a cabinet
slab wider than the gripper; sticky grasping engages on first contact),
`generate_topdown_grasps.py` synthesizes straight-down grasps: approach = world
−Z, a fan of yaws about the vertical axis, centred over the object's top face.
Vertical descent avoids the failure modes of hand-annotated edge grasps on slabs
(reorient about a tilted axis → tip-over, off-centre → wrist droop, palm-flip →
wrist near limit). Default dry-run; `--apply` writes into the DB.

```bash
python -m maniguard.data.datagen.annotation.generate_topdown_grasps --cabinet-all --n-yaw 6 --apply
```

## Gotchas

- **`validate_grasps` is one-object-per-process**: `og.clear()` across tasks
  breaks the cameras, so pass `--object` and run singly (headless mid-run camera
  reposition doesn't re-render, hence the matplotlib closeup).
- Sim steps run with `python -u`; the `og.sim.stop()` **exit 139** at teardown is
  a benign segfault.
- The DB is **gitignored** user data — to move to another machine, either re-run
  the annotation or sync `outputs/grasp_annotation/` (mesh GLBs included).
- After re-annotating, the consumer (the datagen executor via `grasp_db.py`)
  needs no change — the next collection reads the new DB automatically.

## Per-family quick reference

```bash
# 1) extract meshes (sim, once)          2) GUI annotate (browser)          3) QC trio
python -u -m maniguard.data.datagen.annotation.extract_meshes --families <FAM>_pickup
python    -m maniguard.data.datagen.annotation.annotate_tool  --family <FAM>       #  → localhost:8080, Enter to save
python    -m maniguard.data.datagen.annotation.fix_approach_tags --family <FAM> --apply
python    -m maniguard.data.datagen.annotation.mesh_review       --family <FAM>    # point-cloud self-check
python -u -m maniguard.data.datagen.annotation.validate_grasps   --object <cat>/<model>   # later, one object per process
```
