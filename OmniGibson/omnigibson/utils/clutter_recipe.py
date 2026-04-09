from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ClutterSceneRecipe:
    scene_model: str
    task_name: str
    clutter_level: str
    seed: int
    placement_constraints: Dict[str, float] = field(default_factory=dict)
    mount_policy: Dict[str, object] = field(default_factory=lambda: {"enable_auto_mount": True})
    runtime_mode: str = "cached"
    target_binding: Dict[str, object] = field(default_factory=dict)
    mount_result: Optional[Dict[str, Tuple[float, ...]]] = None

    def validate(self) -> None:
        if self.runtime_mode != "cached":
            raise ValueError(f"MVP runner only supports runtime_mode='cached', got {self.runtime_mode}")
        if self.clutter_level not in {"low", "high"}:
            raise ValueError(f"clutter_level must be low/high, got {self.clutter_level}")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ClutterSceneRecipe":
        recipe = cls(
            scene_model=str(data["scene_model"]),
            task_name=str(data["task_name"]),
            clutter_level=str(data["clutter_level"]),
            seed=int(data["seed"]),
            placement_constraints=dict(data.get("placement_constraints", {})),
            mount_policy=dict(data.get("mount_policy", {"enable_auto_mount": True})),
            runtime_mode=str(data.get("runtime_mode", "cached")),
            target_binding=dict(data.get("target_binding", {})),
            mount_result=data.get("mount_result"),
        )
        recipe.validate()
        return recipe


def load_clutter_scene_recipe(path: str) -> ClutterSceneRecipe:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ClutterSceneRecipe.from_dict(data)


def save_clutter_scene_recipe(recipe: ClutterSceneRecipe, path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(recipe.to_dict(), f, indent=2)
