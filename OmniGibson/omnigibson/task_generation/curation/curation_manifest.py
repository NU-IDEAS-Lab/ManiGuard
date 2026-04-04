"""Helpers for scene-level curation manifests used by task generation pipelines.

The manifest is intentionally pure-Python so it can be tested without loading
the full OmniGibson stack. A compact manifest may specify only problematic
scenes; unspecified scenes inherit top-level defaults.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional


DEFAULT_ACTIVITY_PREFIX = "auto_clutter_on"
DEFAULT_STATUS = "keep"
_VALID_STATUSES = frozenset({"keep", "repair", "defer"})
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _candidate_paths(value: str, roots: tuple[Path, ...]) -> list[Path]:
    path = Path(value)
    if path.is_absolute():
        return [path.resolve()]

    candidates = []
    seen = set()
    for root in roots:
        resolved = (root / path).resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(resolved)
    return candidates


def _resolve_path(
    value: Optional[str],
    *,
    roots: tuple[Path, ...],
) -> Optional[str]:
    if not value:
        return None
    candidates = _candidate_paths(value, roots)
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def _manifest_resolution_roots(manifest_path: Path) -> tuple[Path, ...]:
    """Prefer the checkout that contains the manifest before falling back."""
    roots = [Path.cwd()]
    roots.extend(manifest_path.parents)
    roots.append(_PROJECT_ROOT)
    return tuple(roots)


def _merge_scene_data(defaults: Mapping[str, Any], scene_data: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    merged.update(scene_data)
    return merged


@dataclass(frozen=True)
class SceneCurationEntry:
    scene_model: str
    benchmark_run_dir: str
    activity_prefix: str = DEFAULT_ACTIVITY_PREFIX
    status: str = DEFAULT_STATUS
    repair_mode: str = "snapshot_first"
    canonical_episode: Optional[int] = 1
    issue_tags: tuple[str, ...] = ()
    review_note: str = ""
    defer_reason: str = ""
    support_category: Optional[str] = None
    support_room: Optional[str] = None
    surface_name: Optional[str] = None
    remove_other_object_categories: tuple[str, ...] = ()
    surface_bounds_override_xy: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
    obstacle_bounds_override_xy: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
    preferred_edge: Optional[str] = None
    mount_anchor_offset_m: Optional[float] = None
    mount_base_pose_xyyaw: Optional[tuple[float, float, float]] = None
    mount_workspace_front_m: Optional[float] = None
    mount_workspace_side_m: Optional[float] = None
    mount_workspace_rear_m: Optional[float] = None
    pin_support_base: bool = False
    post_mount_settle_steps: Optional[int] = None
    clutter_density: Optional[str] = None
    require_support_and_upright_after_pack: bool = False
    use_resident_surface_obstacles: bool = False
    require_resident_surface_stability: bool = False
    pack_jitter_xy: Optional[float] = None
    pack_min_clearance: Optional[float] = None
    pack_clearance_step_m: Optional[float] = None
    pack_clearance_floor_m: Optional[float] = None
    pack_clearance_search_mode: Optional[str] = None
    zone_edge_margin_m: Optional[float] = None
    obstacle_keepout_margin_m: Optional[float] = None
    obstacle_side_clearance_m: Optional[float] = None
    perimeter_clear_margin_m: Optional[float] = None
    mount_gap_m: Optional[float] = None
    save_scene_even_if_gate_fails: bool = False
    video_viewer_only: bool = False
    video_candidate_mode: Optional[str] = None
    video_camera_eye: Optional[tuple[float, float, float]] = None
    video_camera_lookat: Optional[tuple[float, float, float]] = None
    video_candidate_views: tuple[dict[str, Any], ...] = ()
    video_final_view: Optional[str] = None
    support_clear_mode: Optional[str] = None
    perimeter_clear_mode: Optional[str] = None
    snapshot_path: Optional[str] = None

    @property
    def activity_name(self) -> str:
        return f"{self.activity_prefix}_{self.scene_model}"

    @property
    def scene_dir(self) -> str:
        return str(Path(self.benchmark_run_dir) / self.scene_model)

    def available_snapshot_paths(self) -> tuple[str, ...]:
        scene_dir = Path(self.scene_dir)
        if not scene_dir.is_dir():
            return ()
        return tuple(
            str(path.resolve())
            for path in sorted(scene_dir.glob("scene_ep*.json"))
        )

    def snapshot_candidates(self, episode: Optional[int] = None) -> tuple[str, ...]:
        if self.snapshot_path:
            return (self.snapshot_path,)
        ep = episode if episode is not None else self.canonical_episode
        if ep is None:
            return self.available_snapshot_paths()
        return (str((Path(self.scene_dir) / f"scene_ep{ep}.json").resolve()),)

    def resolve_snapshot_path(self, episode: Optional[int] = None) -> Optional[str]:
        for candidate in self.snapshot_candidates(episode=episode):
            if Path(candidate).is_file():
                return candidate
        return None

    def to_runtime_overrides(self) -> Dict[str, Any]:
        return {
            "support_category": self.support_category,
            "support_room": self.support_room,
            "surface_name": self.surface_name,
            "remove_other_object_categories": self.remove_other_object_categories,
            "surface_bounds_override_xy": self.surface_bounds_override_xy,
            "obstacle_bounds_override_xy": self.obstacle_bounds_override_xy,
            "preferred_edge": self.preferred_edge,
            "mount_anchor_offset_m": self.mount_anchor_offset_m,
            "mount_base_pose_xyyaw": self.mount_base_pose_xyyaw,
            "mount_workspace_front_m": self.mount_workspace_front_m,
            "mount_workspace_side_m": self.mount_workspace_side_m,
            "mount_workspace_rear_m": self.mount_workspace_rear_m,
            "pin_support_base": self.pin_support_base,
            "post_mount_settle_steps": self.post_mount_settle_steps,
            "clutter_density": self.clutter_density,
            "require_support_and_upright_after_pack": self.require_support_and_upright_after_pack,
            "use_resident_surface_obstacles": self.use_resident_surface_obstacles,
            "require_resident_surface_stability": self.require_resident_surface_stability,
            "pack_jitter_xy": self.pack_jitter_xy,
            "pack_min_clearance": self.pack_min_clearance,
            "pack_clearance_step_m": self.pack_clearance_step_m,
            "pack_clearance_floor_m": self.pack_clearance_floor_m,
            "pack_clearance_search_mode": self.pack_clearance_search_mode,
            "zone_edge_margin_m": self.zone_edge_margin_m,
            "obstacle_keepout_margin_m": self.obstacle_keepout_margin_m,
            "obstacle_side_clearance_m": self.obstacle_side_clearance_m,
            "perimeter_clear_margin_m": self.perimeter_clear_margin_m,
            "mount_gap_m": self.mount_gap_m,
            "save_scene_even_if_gate_fails": self.save_scene_even_if_gate_fails,
            "video_viewer_only": self.video_viewer_only,
            "video_candidate_mode": self.video_candidate_mode,
            "video_camera_eye": self.video_camera_eye,
            "video_camera_lookat": self.video_camera_lookat,
            "video_candidate_views": self.video_candidate_views,
            "video_final_view": self.video_final_view,
            "support_clear_mode": self.support_clear_mode,
            "perimeter_clear_mode": self.perimeter_clear_mode,
        }

    @classmethod
    def from_dict(
        cls,
        scene_model: str,
        data: Mapping[str, Any],
        *,
        benchmark_run_dir: str,
        activity_prefix: str,
        base_dir: Path,
    ) -> "SceneCurationEntry":
        status = data.get("status", DEFAULT_STATUS)
        if status not in _VALID_STATUSES:
            raise ValueError(f"Unsupported curation status '{status}' for scene '{scene_model}'")

        def _triplet(name: str) -> Optional[tuple[float, float, float]]:
            raw = data.get(name)
            if raw is None:
                return None
            if len(raw) != 3:
                raise ValueError(f"{name} must be a 3-element sequence for scene '{scene_model}'")
            return tuple(float(v) for v in raw)

        def _candidate_views() -> tuple[dict[str, Any], ...]:
            views = []
            for idx, raw in enumerate(data.get("video_candidate_views", ())):
                if not isinstance(raw, Mapping):
                    raise ValueError(f"video_candidate_views[{idx}] must be a mapping for scene '{scene_model}'")
                label = str(raw.get("label", "")).strip()
                if not label:
                    raise ValueError(f"video_candidate_views[{idx}] is missing a non-empty label for scene '{scene_model}'")
                eye = raw.get("eye")
                lookat = raw.get("lookat")
                if eye is None or lookat is None:
                    raise ValueError(
                        f"video_candidate_views[{idx}] must define both eye and lookat for scene '{scene_model}'"
                    )
                if len(eye) != 3 or len(lookat) != 3:
                    raise ValueError(
                        f"video_candidate_views[{idx}] eye/lookat must be length 3 for scene '{scene_model}'"
                    )
                views.append({
                    "label": label,
                    "eye": tuple(float(v) for v in eye),
                    "lookat": tuple(float(v) for v in lookat),
                })
            return tuple(views)

        def _string_list(name: str) -> tuple[str, ...]:
            raw = data.get(name)
            if raw is None:
                return ()
            if not isinstance(raw, (list, tuple)):
                raise ValueError(f"{name} must be a list for scene '{scene_model}'")
            return tuple(str(item).strip() for item in raw if str(item).strip())

        def _bounds2d(name: str) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
            raw = data.get(name)
            if raw is None:
                return None
            if len(raw) != 2 or len(raw[0]) != 2 or len(raw[1]) != 2:
                raise ValueError(f"{name} must be [[x0, y0], [x1, y1]] for scene '{scene_model}'")
            return (
                (float(raw[0][0]), float(raw[0][1])),
                (float(raw[1][0]), float(raw[1][1])),
            )

        issue_tags = tuple(str(tag) for tag in data.get("issue_tags", ()))
        canonical_episode = data.get("canonical_episode", 1)
        if canonical_episode is not None:
            canonical_episode = int(canonical_episode)

        return cls(
            scene_model=scene_model,
            benchmark_run_dir=benchmark_run_dir,
            activity_prefix=str(data.get("activity_prefix", activity_prefix)),
            status=status,
            repair_mode=str(data.get("repair_mode", "snapshot_first")),
            canonical_episode=canonical_episode,
            issue_tags=issue_tags,
            review_note=str(data.get("review_note", "")),
            defer_reason=str(data.get("defer_reason", "")),
            support_category=data.get("support_category"),
            support_room=data.get("support_room"),
            surface_name=data.get("surface_name"),
            remove_other_object_categories=_string_list("remove_other_object_categories"),
            surface_bounds_override_xy=_bounds2d("surface_bounds_override_xy"),
            obstacle_bounds_override_xy=_bounds2d("obstacle_bounds_override_xy"),
            preferred_edge=data.get("preferred_edge"),
            mount_anchor_offset_m=float(data["mount_anchor_offset_m"])
            if data.get("mount_anchor_offset_m") is not None else None,
            mount_base_pose_xyyaw=_triplet("mount_base_pose_xyyaw"),
            mount_workspace_front_m=float(data["mount_workspace_front_m"])
            if data.get("mount_workspace_front_m") is not None else None,
            mount_workspace_side_m=float(data["mount_workspace_side_m"])
            if data.get("mount_workspace_side_m") is not None else None,
            mount_workspace_rear_m=float(data["mount_workspace_rear_m"])
            if data.get("mount_workspace_rear_m") is not None else None,
            pin_support_base=bool(data.get("pin_support_base", False)),
            post_mount_settle_steps=int(data["post_mount_settle_steps"])
            if data.get("post_mount_settle_steps") is not None else None,
            clutter_density=data.get("clutter_density"),
            require_support_and_upright_after_pack=bool(data.get("require_support_and_upright_after_pack", False)),
            use_resident_surface_obstacles=bool(data.get("use_resident_surface_obstacles", False)),
            require_resident_surface_stability=bool(data.get("require_resident_surface_stability", False)),
            pack_jitter_xy=float(data["pack_jitter_xy"]) if data.get("pack_jitter_xy") is not None else None,
            pack_min_clearance=float(data["pack_min_clearance"]) if data.get("pack_min_clearance") is not None else None,
            pack_clearance_step_m=float(data["pack_clearance_step_m"]) if data.get("pack_clearance_step_m") is not None else None,
            pack_clearance_floor_m=float(data["pack_clearance_floor_m"]) if data.get("pack_clearance_floor_m") is not None else None,
            pack_clearance_search_mode=str(data["pack_clearance_search_mode"]) if data.get("pack_clearance_search_mode") is not None else None,
            zone_edge_margin_m=float(data["zone_edge_margin_m"]) if data.get("zone_edge_margin_m") is not None else None,
            obstacle_keepout_margin_m=float(data["obstacle_keepout_margin_m"])
            if data.get("obstacle_keepout_margin_m") is not None else None,
            obstacle_side_clearance_m=float(data["obstacle_side_clearance_m"])
            if data.get("obstacle_side_clearance_m") is not None else None,
            perimeter_clear_margin_m=float(data["perimeter_clear_margin_m"])
            if data.get("perimeter_clear_margin_m") is not None else None,
            mount_gap_m=float(data["mount_gap_m"]) if data.get("mount_gap_m") is not None else None,
            save_scene_even_if_gate_fails=bool(data.get("save_scene_even_if_gate_fails", False)),
            video_viewer_only=bool(data.get("video_viewer_only", False)),
            video_candidate_mode=str(data["video_candidate_mode"]).strip()
            if data.get("video_candidate_mode") else None,
            video_camera_eye=_triplet("video_camera_eye"),
            video_camera_lookat=_triplet("video_camera_lookat"),
            video_candidate_views=_candidate_views(),
            video_final_view=str(data["video_final_view"]).strip() if data.get("video_final_view") else None,
            support_clear_mode=str(data["support_clear_mode"]).strip()
            if data.get("support_clear_mode") else None,
            perimeter_clear_mode=str(data["perimeter_clear_mode"]).strip()
            if data.get("perimeter_clear_mode") else None,
            snapshot_path=_resolve_path(
                data.get("snapshot_path"),
                roots=(base_dir, _PROJECT_ROOT, Path.cwd()),
            ),
        )


@dataclass(frozen=True)
class CurationManifest:
    source_path: str
    benchmark_run_dir: str
    activity_prefix: str = DEFAULT_ACTIVITY_PREFIX
    defaults: Dict[str, Any] = field(default_factory=dict)
    scenes: Dict[str, SceneCurationEntry] = field(default_factory=dict)

    def get_scene_entry(self, scene_model: str) -> SceneCurationEntry:
        if scene_model in self.scenes:
            return self.scenes[scene_model]
        data = dict(self.defaults)
        return SceneCurationEntry.from_dict(
            scene_model,
            data,
            benchmark_run_dir=self.benchmark_run_dir,
            activity_prefix=self.activity_prefix,
            base_dir=Path(self.source_path).resolve().parent,
        )


def load_curation_manifest(path: str) -> CurationManifest:
    manifest_path = Path(path).resolve()
    with open(manifest_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    benchmark_run_dir = _resolve_path(
        raw["benchmark_run_dir"],
        roots=_manifest_resolution_roots(manifest_path),
    )
    if benchmark_run_dir is None:
        raise ValueError("benchmark_run_dir is required in the curation manifest")

    activity_prefix = str(raw.get("activity_prefix", DEFAULT_ACTIVITY_PREFIX))
    defaults = dict(raw.get("defaults", {}))
    scenes_raw = dict(raw.get("scenes", {}))

    scenes = {}
    for scene_model, scene_data in scenes_raw.items():
        merged = _merge_scene_data(defaults, scene_data)
        scenes[scene_model] = SceneCurationEntry.from_dict(
            scene_model,
            merged,
            benchmark_run_dir=benchmark_run_dir,
            activity_prefix=activity_prefix,
            base_dir=manifest_path.parent,
        )

    return CurationManifest(
        source_path=str(manifest_path),
        benchmark_run_dir=benchmark_run_dir,
        activity_prefix=activity_prefix,
        defaults=defaults,
        scenes=scenes,
    )


def apply_scene_entry_to_args(args: MutableMapping[str, Any] | Any, entry: SceneCurationEntry) -> None:
    """Apply runtime-relevant entry values onto an argparse namespace-like object."""
    updates = {
        "clutter_density": entry.clutter_density,
        "pack_jitter_xy": entry.pack_jitter_xy,
        "pack_min_clearance": entry.pack_min_clearance,
        "pack_clearance_step_m": entry.pack_clearance_step_m,
        "pack_clearance_floor_m": entry.pack_clearance_floor_m,
        "pack_clearance_search_mode": entry.pack_clearance_search_mode,
        "use_resident_surface_obstacles": entry.use_resident_surface_obstacles,
        "require_resident_surface_stability": entry.require_resident_surface_stability,
        "zone_edge_margin_m": entry.zone_edge_margin_m,
        "obstacle_keepout_margin_m": entry.obstacle_keepout_margin_m,
        "obstacle_side_clearance_m": entry.obstacle_side_clearance_m,
        "perimeter_clear_margin_m": entry.perimeter_clear_margin_m,
        "mount_gap_m": entry.mount_gap_m,
        "mount_anchor_offset_m": entry.mount_anchor_offset_m,
        "mount_base_pose_xyyaw": entry.mount_base_pose_xyyaw,
        "mount_workspace_front_m": entry.mount_workspace_front_m,
        "mount_workspace_side_m": entry.mount_workspace_side_m,
        "mount_workspace_rear_m": entry.mount_workspace_rear_m,
        "pin_support_base": entry.pin_support_base,
        "post_mount_settle_steps": entry.post_mount_settle_steps,
        "remove_other_object_categories": entry.remove_other_object_categories,
        "video_viewer_only": entry.video_viewer_only,
        "video_candidate_mode": entry.video_candidate_mode,
        "video_candidate_views": entry.video_candidate_views,
        "video_final_view": entry.video_final_view,
        "support_clear_mode": entry.support_clear_mode,
        "perimeter_clear_mode": entry.perimeter_clear_mode,
    }
    for key, value in updates.items():
        if value is None:
            continue
        if isinstance(args, MutableMapping):
            args[key] = value
        else:
            setattr(args, key, value)
