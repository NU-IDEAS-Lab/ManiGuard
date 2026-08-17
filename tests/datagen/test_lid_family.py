"""lid_transport family unit tests (pure — no OmniGibson/cuRobo)."""
from maniguard.data.datagen.annotation.extract_meshes import _lid_instance_names


def _diag(container_role: str, cat: str) -> dict:
    return {"lid_info": {"container": {"category": cat, "model": "m1"},
                         "lid": {"model": "l1"}},
            "selection": {"spawn_specs": [
                {"role": container_role, "category": cat, "model": "m1"},
                {"role": "lid", "category": "lid", "model": "l1"}]}}


def _scene(names_cat_model: dict) -> dict:
    return {"objects_info": {"init_info": {
        n: {"args": {"category": c, "model": m}} for n, (c, m) in names_cat_model.items()}}}


def test_lid_instance_names_old_generation():
    diag = _diag("container", "tupperware")
    scene = _scene({"lid_43": ("lid", "l1"), "tupperware_44": ("tupperware", "m1"),
                    "orange_45": ("orange", "x")})
    assert _lid_instance_names(diag, scene) == ["lid_43", "tupperware_44"]


def test_lid_instance_names_new_generation():
    diag = _diag("target", "hingeless_jar")
    scene = _scene({"lid_lid_ep1_1": ("lid", "l1"),
                    "target_hingeless_jar_ep1_1": ("hingeless_jar", "m1")})
    assert _lid_instance_names(diag, scene) == [
        "lid_lid_ep1_1", "target_hingeless_jar_ep1_1"]


def test_lid_instance_names_cap_category():
    diag = {"lid_info": {"container": {"category": "jug", "model": "m1"},
                         "lid": {"model": "l1"}},
            "selection": {"spawn_specs": [
                {"role": "target", "category": "jug", "model": "m1"},
                {"role": "lid", "category": "cap", "model": "l1"}]}}
    scene = _scene({"lid_cap_ep1_1": ("cap", "l1"), "target_jug_ep1_1": ("jug", "m1")})
    assert _lid_instance_names(diag, scene) == ["lid_cap_ep1_1", "target_jug_ep1_1"]


# ---- Task 2: pure helpers -------------------------------------------------------
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from maniguard.data.datagen.families.lid import geometry_class, insertion_dir_from_grasp, release_pose_for_lid


def test_geometry_class():
    assert geometry_class("kettle", "lid") == "handle"
    assert geometry_class("teapot", "lid") == "handle"
    assert geometry_class("bottle_of_whiskey", "cap") == "cap"
    assert geometry_class("tupperware", "lid") == "plain"


def test_insertion_dir_horizontal_grasp():
    # approach = eef +Z; rotate +Z onto world +X => side grasp inserting along +X
    q = Rot.from_euler("y", 90, degrees=True).as_quat()
    d = insertion_dir_from_grasp(np.asarray(q))
    assert np.allclose(d, [1.0, 0.0, 0.0], atol=1e-6)
    assert abs(np.linalg.norm(d) - 1.0) < 1e-9


def test_insertion_dir_rejects_vertical():
    # straight-down approach (+Z -> world -Z): no horizontal insertion direction
    q = Rot.from_euler("x", 180, degrees=True).as_quat()
    with pytest.raises(ValueError):
        insertion_dir_from_grasp(np.asarray(q))


def test_release_pose_offsets_eef_by_m_link():
    f = np.array([1.0, 2.0, 0.5])
    eef = np.array([1.1, 2.0, 0.8])
    m = np.array([1.0, 2.0, 0.7])
    out = release_pose_for_lid(f, eef, m, clear_z=0.015)
    assert np.allclose(out, [1.1, 2.0, 0.615])


# ---- Task 4: assembly-aware held gating -----------------------------------------
def test_build_goal_checker_sets_assembly_lid_only_for_lid_family():
    from maniguard.eval.goal_checker import build_goal_checker
    region = {"mode": "held_intersection", "shape": "sphere", "family": "lid_transport_food",
              "target_name": "tupperware_44", "support_name": "desk_0",
              "marker_name": "goal_region__tupperware_44",
              "center_world": [0.0, 0.0, 1.0], "radius_m": 0.1,
              "color_rgba": [0.0, 1.0, 0.0, 0.35], "target_width_m": 0.2,
              "anchor_local_xy": [0.0, 0.0],
              "pack_bbox_robot_local_xy": [[-0.1, -0.1], [0.1, 0.1]],
              "support_bounds_robot_local_xy": [[-0.5, -0.5], [0.5, 0.5]],
              "clamped_to_support_bounds": False}
    lid_diag = {"goal_region": region, "lid_info": {"mode": "food"},
                "goal_conditions": [
                    {"predicate": "ontop", "subject": "lid_43", "reference": "tupperware_44"},
                    {"predicate": "grasping", "subject": "robot", "reference": "tupperware_44"}]}
    c = build_goal_checker(lid_diag)
    assert c.assembly_lid_name == "lid_43"
    # any non-lid goal_region family (clutter/stack): field stays None
    clutter_diag = {"goal_region": region}
    assert build_goal_checker(clutter_diag).assembly_lid_name is None
