"""Phase B — viser grasp-annotation tool.

Loads the Phase-A mesh DB and lets you annotate, per object, the eef_link TARGET pose(s)
to grasp it. The object is shown UPRIGHT (its scene orientation) with the world XYZ axes
and the real longfinger gripper rendered at the current draft grasp. Two input modes,
both producing the SAME stored record (object-local eef-target, locked schema §4.2):

  mode③ (default): click a point on the object + pick an approach preset + yaw/depth
                   sliders  ->  fast, for regular top-down/side grasps.
  mode① (toggle "free drag"): a 6-DoF gizmo on the gripper  ->  for odd semantic grasps.

Open the printed http://localhost:8080 in a browser. Saves incrementally + resumes from
an existing grasp_annotations.json.

  conda activate behavior
  PYTHONPATH=$HOME/project/ManiGuard python -m maniguard.data.datagen.annotation.annotate_tool
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation as Rot

ANN_DIR = Path("outputs/grasp_annotation")
MESH_DB = ANN_DIR / "mesh_db.json"
GRIPPER = ANN_DIR / "gripper_longfinger.glb"
ANN_OUT = ANN_DIR / "grasp_annotations.json"

# approach = unit vector (world/display frame) the gripper travels along to reach the grasp.
APPROACHES = {
    "top_down": [0.0, 0.0, -1.0],
    "side_X+": [1.0, 0.0, 0.0], "side_X-": [-1.0, 0.0, 0.0],
    "side_Y+": [0.0, 1.0, 0.0], "side_Y-": [0.0, -1.0, 0.0],
}

# FAMILY_STEMS + the multi-family membership test live in family_membership (shared, no heavy deps).
from maniguard.data.datagen.annotation.family_membership import FAMILY_STEMS, obj_in_family


def _xyzw_to_wxyz(q):
    q = np.asarray(q, float)
    return np.array([q[3], q[0], q[1], q[2]])


def _wxyz_to_xyzw(q):
    q = np.asarray(q, float)
    return np.array([q[1], q[2], q[3], q[0]])


def _load_mesh(path) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))
    return m


def _eef_R_from_approach(a, yaw_rad):
    """eef rotation (display frame) for approach ``a`` + roll ``yaw`` about it.

    Gripper eef-local frame (sim-measured, NOT the old probe — the probe mislabelled the
    finger-link origins as the tips): **fingertips / approach = eef +Z**, **closing
    (between fingers) = eef -Y**. So the gripper travels toward the object along its +Z;
    to grasp along world approach ``a`` we set eef +Z = a (fingertips lead toward object).
    """
    a = np.asarray(a, float); a /= np.linalg.norm(a) + 1e-9
    z_col = a                                             # eef +Z = approach (fingertips)
    ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    perp = ref - (ref @ a) * a
    perp /= np.linalg.norm(perp) + 1e-9
    closing = Rot.from_rotvec(yaw_rad * a).apply(perp)    # roll about approach
    y_col = -closing                                      # eef -Y = closing
    x_col = np.cross(y_col, z_col)
    x_col /= np.linalg.norm(x_col) + 1e-9
    y_col = np.cross(z_col, x_col)
    return np.stack([x_col, y_col, z_col], axis=1)


class App:
    def __init__(self, families=None):
        self.db = json.load(open(MESH_DB))
        keys = list(self.db["objects"])
        if families:
            keys = [k for k in keys if obj_in_family(self.db["objects"][k], families)]
        self.keys = sorted(keys)               # alphabetical so the jump list / prev-next are findable
        self.gripper = _load_mesh(GRIPPER)
        self.ann = self._load_ann()
        self.i = 0
        # current draft (display frame)
        self.click_pt = None
        self.disp_mesh = None
        self.T_up = np.eye(4)

        self.server = viser.ViserServer()
        self._build_gui()
        self._gizmo = None
        self.show(0)

    # ---- persistence -------------------------------------------------------
    def _load_ann(self):
        if ANN_OUT.exists():
            return json.load(open(ANN_OUT))
        return {"schema_version": "1.0", "gripper": self.db["gripper"],
                "convention": self.db["convention"], "objects": {}}

    def _save(self):
        json.dump(self.ann, open(ANN_OUT, "w"), indent=2)

    def _rec(self, key):
        if key not in self.ann["objects"]:
            o = self.db["objects"][key]
            self.ann["objects"][key] = {
                "category": o["category"], "model": o["model"],
                "bbox_size": o["bbox_size"],
                "upright_orientation_xyzw": o["upright_orientation_xyzw"],
                "mesh": o["mesh"], "grasps": []}
        return self.ann["objects"][key]

    # ---- GUI ---------------------------------------------------------------
    def _build_gui(self):
        g = self.server.gui
        self.lbl = g.add_text("object", initial_value="", disabled=True)
        self.prog = g.add_text("progress", initial_value="", disabled=True)
        with g.add_folder("navigate"):
            g.add_button("◀ prev").on_click(lambda _: self.show(self.i - 1))
            g.add_button("next ▶").on_click(lambda _: self.show(self.i + 1))
            # type a substring to filter the jump list (case-insensitive); the dropdown stays
            # alphabetical. _filtering guards the jump callback so re-setting options while typing
            # doesn't fire a stray jump — only an explicit pick navigates.
            self._filtering = False
            self.search = g.add_text("search", initial_value="")
            self.search.on_update(lambda _: self._filter_jump())
            self.jump = g.add_dropdown("jump to", options=self.keys)
            self.jump.on_update(
                lambda _: None if self._filtering else self.show(self.keys.index(self.jump.value)))
        with g.add_folder("Guided — click point + presets"):
            self.appr = g.add_dropdown("approach", options=list(APPROACHES),
                                       initial_value="top_down")
            self.yaw = g.add_slider("yaw°", -180, 180, 5, 0)
            self.depth = g.add_slider("depth (m)", -0.02, 0.18, 0.005, 0.06)
            for w in (self.appr, self.yaw, self.depth):
                w.on_update(lambda _: self._recompute())
        self.freedrag = g.add_checkbox("Free — 6-DoF drag gizmo", initial_value=False)
        self.freedrag.on_update(lambda _: self._toggle_gizmo())
        with g.add_folder("grasps"):
            self.glist = g.add_text("saved", initial_value="0", disabled=True)
            self.hint = g.add_dropdown("approach_hint", options=["top_down", "side"],
                                       initial_value="top_down")
            self.label = g.add_text("label", initial_value="")
            g.add_button("✚ save grasp (Enter)").on_click(lambda _: self._save_grasp())
            g.add_button("✖ delete last").on_click(lambda _: self._del_last())
        # Enter-to-save: commit the finished pose without moving the mouse to the button. A
        # command-palette command (also reachable via Ctrl/Cmd+K) bound to the Enter hotkey.
        # viser's client uses Mantine useHotkeys with the default tagsToIgnore
        # (INPUT/TEXTAREA/SELECT), so hitting Enter WHILE typing in the search / label fields
        # does NOT misfire a save — only Enter with focus off a text input triggers it.
        g.add_command("save grasp", hotkey="enter").on_trigger(lambda _: self._save_grasp())

    def _filter_jump(self):
        """Filter the jump dropdown to keys containing the search substring (empty => all), and
        FOLLOW the filter: if the shown object is filtered away, display the first match (viser won't
        re-fire on_update for an already-selected single match, so we navigate explicitly here)."""
        q = self.search.value.strip().lower()
        opts = [k for k in self.keys if q in k.lower()] if q else list(self.keys)
        if not opts:
            return                              # no match: keep the current list, don't blank it
        self._filtering = True                  # suppress the jump callback while reassigning options
        self.jump.options = opts
        self._filtering = False
        cur = self.keys[self.i]
        if cur not in opts:                     # current object narrowed away -> jump to first match
            self.show(self.keys.index(opts[0]))

    # ---- per-object display ------------------------------------------------
    def show(self, i):
        self.i = i % len(self.keys)
        key = self.keys[self.i]
        o = self.db["objects"][key]
        self.lbl.value = key
        # keep the jump dropdown highlighting the shown object (guarded so it doesn't re-fire show);
        # only if it's a current option (prev/next can land outside an active search filter).
        jump = getattr(self, "jump", None)
        if jump is not None and key in jump.options and jump.value != key:
            self._filtering = True
            jump.value = key
            self._filtering = False
        self.prog.value = (f"{self.i + 1}/{len(self.keys)}  "
                           f"[{len(self.ann['objects'].get(key, {}).get('grasps', []))} saved]")
        self.server.scene.reset()
        self.click_pt = None
        if self._gizmo is not None:
            self._gizmo = None
        self.freedrag.value = False

        R_up = Rot.from_quat(o["upright_orientation_xyzw"]).as_matrix()
        self.T_up = np.eye(4); self.T_up[:3, :3] = R_up
        mesh = _load_mesh(ANN_DIR / o["mesh"])
        self.disp_mesh = mesh.copy(); self.disp_mesh.apply_transform(self.T_up)

        self.server.scene.add_frame("/world", axes_length=0.12, axes_radius=0.004)
        mh = self.server.scene.add_mesh_trimesh(
            "/obj", mesh, wxyz=_xyzw_to_wxyz(Rot.from_matrix(R_up).as_quat()),
            position=(0, 0, 0))
        mh.on_click(self._on_click)
        self._render_grasp_count(key)
        # default draft: a top-down grasp at the object's top-centre
        c = self.disp_mesh.vertices.mean(0)
        top = self.disp_mesh.vertices[:, 2].max()
        self.click_pt = np.array([c[0], c[1], top])
        self.appr.value = "top_down"
        self._recompute()

    def _render_grasp_count(self, key):
        self.glist.value = str(len(self.ann["objects"].get(key, {}).get("grasps", [])))

    # ---- click (mode③) -----------------------------------------------------
    def _on_click(self, event):
        ro = np.asarray(getattr(event, "ray_origin", None), float) \
            if getattr(event, "ray_origin", None) is not None else None
        rd = np.asarray(getattr(event, "ray_direction", None), float) \
            if getattr(event, "ray_direction", None) is not None else None
        if ro is None or rd is None or self.disp_mesh is None:
            return
        locs, idx_ray, _ = self.disp_mesh.ray.intersects_location(
            ray_origins=ro[None], ray_directions=rd[None])
        if len(locs) == 0:
            return
        # nearest hit along the ray
        d = ((locs - ro) * rd).sum(1)
        self.click_pt = locs[np.argmin(np.where(d > 0, d, np.inf))]
        self._recompute()

    # ---- recompute draft eef pose + render gripper -------------------------
    def _draft_eef(self):
        """(eef_pos, R_eef) in display frame from mode③ params + click point."""
        a = np.asarray(APPROACHES[self.appr.value], float)
        R = _eef_R_from_approach(a, np.radians(self.yaw.value))
        pos = self.click_pt - self.depth.value * (a / (np.linalg.norm(a) + 1e-9))
        return pos, R

    def _recompute(self):
        if self.click_pt is None or self.freedrag.value:
            return
        pos, R = self._draft_eef()
        self._show_gripper(pos, R)

    def _show_gripper(self, pos, R):
        self._draft = (np.asarray(pos, float), np.asarray(R, float))
        self.server.scene.add_mesh_trimesh(
            "/gripper", self.gripper,
            wxyz=_xyzw_to_wxyz(Rot.from_matrix(R).as_quat()), position=tuple(pos))

    # ---- mode① free-drag gizmo --------------------------------------------
    def _toggle_gizmo(self):
        if self.freedrag.value:
            pos, R = getattr(self, "_draft", (self.click_pt, np.eye(3)))
            self._gizmo = self.server.scene.add_transform_controls(
                "/gizmo", scale=0.15,
                wxyz=_xyzw_to_wxyz(Rot.from_matrix(R).as_quat()), position=tuple(pos))

            def _upd(_):
                p = np.asarray(self._gizmo.position, float)
                Rg = Rot.from_quat(_wxyz_to_xyzw(self._gizmo.wxyz)).as_matrix()
                self._show_gripper(p, Rg)
            self._gizmo.on_update(_upd)
        else:
            self._gizmo = None
            self.server.scene.add_frame("/gizmo", show_axes=False)  # clear
            self._recompute()

    # ---- save / delete -----------------------------------------------------
    def _save_grasp(self):
        if not hasattr(self, "_draft"):
            return
        pos, R = self._draft
        T_disp = np.eye(4); T_disp[:3, :3] = R; T_disp[:3, 3] = pos
        T_local = np.linalg.inv(self.T_up) @ T_disp          # display -> object-local
        q = Rot.from_matrix(T_local[:3, :3]).as_quat()       # xyzw
        key = self.keys[self.i]
        rec = self._rec(key)
        rec["grasps"].append({
            "id": len(rec["grasps"]),
            "position": [float(v) for v in T_local[:3, 3]],
            "orientation_xyzw": [float(v) for v in q],
            "approach_hint": self.hint.value,
            "label": self.label.value,
            "source": "freedrag" if self.freedrag.value else "click",
            "validated": None})
        self._save()
        self._render_grasp_count(key)
        self.prog.value = (f"{self.i + 1}/{len(self.keys)}  "
                           f"[{len(rec['grasps'])} saved]  ✓ saved grasp {rec['grasps'][-1]['id']}")

    def _del_last(self):
        key = self.keys[self.i]
        rec = self.ann["objects"].get(key)
        if rec and rec["grasps"]:
            rec["grasps"].pop()
            self._save()
            self._render_grasp_count(key)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", nargs="+", default=None, choices=list(FAMILY_STEMS),
                    help="only annotate objects whose source task is in these families "
                         "(e.g. --family clutter)")
    args = ap.parse_args()
    app = App(families=args.family)
    suffix = f" (families={args.family})" if args.family else ""
    print(f"\n[annotate] {len(app.keys)} objects loaded{suffix}. "
          f"Open the URL above in a browser.\n"
          f"[annotate] saving to {ANN_OUT}\n", flush=True)
    while True:
        time.sleep(2.0)


if __name__ == "__main__":
    main()
