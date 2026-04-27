from sentinel.utils.goal_region import (
    GoalRegionSpec,
    build_task_prompt,
    object_intersects_goal_region,
    resolve_goal_region_entities,
)


class _FakeObj:
    def __init__(self, name, aabb_min, aabb_max):
        self.name = name
        self.aabb = (aabb_min, aabb_max)


def test_build_task_prompt_table_mentions_green_sphere():
    scene_info = {
        "objects_info": {"init_info": {"desk_1": {"args": {"category": "desk"}}}},
    }
    diagnostics = {
        "surface": "desk_1",
        "selection": {"target_synset": "goblet.n.01"},
    }
    goal_region = {
        "family": "table",
        "target_name": "cocktail_glass_1",
        "support_name": "desk_1",
        "marker_name": "goal_region__cocktail_glass_1",
        "mode": "held_intersection",
        "shape": "sphere",
        "center_world": [0.0, 0.0, 0.0],
        "radius_m": 0.1,
        "color_rgba": [0.1, 0.8, 0.2, 0.35],
        "target_width_m": 0.1,
        "anchor_local_xy": [0.0, 0.2],
        "pack_bbox_robot_local_xy": [[0.0, 0.0], [0.1, 0.1]],
        "support_bounds_robot_local_xy": [[-1.0, -1.0], [1.0, 1.0]],
        "clamped_to_support_bounds": False,
    }
    prompt = build_task_prompt(scene_info, diagnostics, goal_region=goal_region)
    assert "green goal sphere" in prompt
    assert "goblet" in prompt


def test_resolve_goal_region_entities_table_and_transfer():
    scene_info = {
        "objects_info": {
            "init_info": {
                "desk_1": {"args": {"category": "desk"}},
                "cocktail_glass_1": {"args": {"category": "cocktail_glass"}},
                "plate_1": {"args": {"category": "plate"}},
                "potato_1": {"args": {"category": "potato"}},
                "stockpot_1": {"args": {"category": "stockpot"}},
            }
        }
    }
    table_diag = {
        "surface": "desk_1",
        "selection": {"target_synset": "goblet.n.01"},
        "active_object_summary": [
            {"scene_object_name": "cocktail_glass_1", "role": "target"},
            {"scene_object_name": "plate_1", "role": "fragile"},
        ],
        "goal_conditions": [{"predicate": "grasping", "subject": "robot", "reference": "cocktail_glass_1"}],
    }
    entities = resolve_goal_region_entities(scene_info, table_diag)
    assert entities is not None
    assert entities.family == "table"
    assert entities.target_name == "cocktail_glass_1"
    assert entities.pack_object_names == ("cocktail_glass_1", "plate_1")

    transfer_diag = {
        "pipeline": "transfer",
        "surface": "desk_1",
        "selection": {
            "food_synset": "potato.n.01",
            "source_synset": "plate.n.04",
            "dest_synset": "stockpot.n.01",
        },
        "goal_conditions": [{"predicate": "inside", "subject": "potato_1", "reference": "stockpot_1"}],
    }
    assert resolve_goal_region_entities(scene_info, transfer_diag) is None


def test_object_intersects_goal_region_uses_aabb_overlap():
    spec = GoalRegionSpec(
        mode="held_intersection",
        shape="sphere",
        family="table",
        target_name="cup_1",
        support_name="desk_1",
        marker_name="goal_region__cup_1",
        center_world=(0.3, 0.0, 0.55),
        radius_m=0.05,
        color_rgba=(0.1, 0.8, 0.2, 0.35),
        target_width_m=0.05,
        anchor_local_xy=(0.0, 0.2),
        pack_bbox_robot_local_xy=((0.0, 0.0), (0.1, 0.1)),
        support_bounds_robot_local_xy=((-1.0, -1.0), (1.0, 1.0)),
        clamped_to_support_bounds=False,
    )
    near = _FakeObj("cup_1", (0.26, -0.02, 0.50), (0.34, 0.02, 0.60))
    far = _FakeObj("cup_1", (0.40, -0.02, 0.50), (0.48, 0.02, 0.60))
    assert object_intersects_goal_region(near, spec) is True
    assert object_intersects_goal_region(far, spec) is False
