"""Signal detection for source demonstrations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch as th

from omnigibson.object_states import Inside, OnTop, Open

from _common import robot_action_offset


@dataclass
class PickSignal:
    type: str = "pick"
    frame_idx: int = 0
    object_name: str = ""
    object_category: str = ""
    contact_link: str = ""


@dataclass
class PlaceSignal:
    type: str = "place"
    release_frame_idx: int = 0
    contact_frame_idx: int = 0
    released_object_name: str = ""
    released_object_category: str = ""
    target_object_name: str = ""
    target_object_category: str = ""
    target_contact_link: str = ""


@dataclass
class OpenSignal:
    type: str = "open"
    contact_frame_idx: int = 0
    release_frame_idx: int = 0
    object_name: str = ""
    object_category: str = ""
    contact_link: str = ""


@dataclass
class CloseSignal:
    type: str = "close"
    contact_frame_idx: int = 0
    release_frame_idx: int = 0
    object_name: str = ""
    object_category: str = ""
    contact_link: str = ""


class AnnotationManager:
    """Detect pick, place, open, and close events during HDF5 playback."""

    def __init__(
        self,
        env,
        *,
        robot_id: int = 0,
        invert_gripper_action: bool = False,
        check_gripper_open_during_place: bool = True,
        min_finger_contacts_for_pick: int = 1,
    ):
        self.env = env
        self.robot = env.robots[robot_id]
        self.action_offset = robot_action_offset(env, robot_id)
        self.invert_gripper_action = invert_gripper_action
        self.check_gripper_open_during_place = check_gripper_open_during_place
        self.min_finger_contacts_for_pick = min_finger_contacts_for_pick
        self.prev_states: dict[str, dict[str, Any]] = {}
        self.signals: list[dict[str, Any]] = []
        self.pick_is_active = False

    def reset(self) -> None:
        self.prev_states = {}
        self.signals = []
        self.pick_is_active = False

    def gripper_action_is_closing(self, gripper_action: float) -> bool:
        return gripper_action < 0.5 if not self.invert_gripper_action else gripper_action > 0.5

    def step(self, frame_idx: int, action: th.Tensor, env: Any = None) -> None:
        arm = self.robot.default_arm
        gripper_idx = self.robot.gripper_action_idx[arm]
        gripper_action = float(action[self.action_offset + gripper_idx].item())

        if self.gripper_action_is_closing(gripper_action):
            if not self.pick_is_active:
                picked_obj = self._detect_pick_from_contacts()
                if picked_obj is not None:
                    self.signals.append(
                        asdict(
                            PickSignal(
                                frame_idx=frame_idx,
                                object_name=picked_obj.name,
                                object_category=getattr(picked_obj, "category", ""),
                            )
                        )
                    )
                    print(f"[Annotate] PICK frame={frame_idx} object={picked_obj.name}")
                    self.pick_is_active = True
        else:
            for state in self.prev_states.values():
                state["picked"] = False
            self.pick_is_active = False

        for obj in self.env.scene.objects:
            if obj == self.robot:
                continue
            obj_name = obj.name
            curr_on_top = self._get_object_on_top_of(obj) if OnTop in obj.states else None
            curr_inside = self._get_object_inside_of(obj) if Inside in obj.states else None
            curr_open = obj.states[Open].get_value() if Open in obj.states else None

            if obj_name not in self.prev_states:
                self.prev_states[obj_name] = {
                    "on_top_of": curr_on_top,
                    "inside_of": curr_inside,
                    "open": curr_open,
                    "picked": False,
                    "placed_on": None,
                }
                continue

            prev = self.prev_states[obj_name]
            was_supported = prev["on_top_of"] is not None or prev["inside_of"] is not None
            is_supported = curr_on_top is not None or curr_inside is not None
            gripper_open = not self.gripper_action_is_closing(gripper_action)
            if not was_supported and is_supported and (gripper_open or not self.check_gripper_open_during_place):
                target_obj = curr_on_top if curr_on_top is not None else curr_inside
                if prev.get("placed_on") != target_obj:
                    self.signals.append(
                        asdict(
                            PlaceSignal(
                                release_frame_idx=frame_idx,
                                contact_frame_idx=frame_idx,
                                released_object_name=obj.name,
                                released_object_category=getattr(obj, "category", ""),
                                target_object_name=target_obj.name,
                                target_object_category=getattr(target_obj, "category", ""),
                            )
                        )
                    )
                    prev["placed_on"] = target_obj
                    self.pick_is_active = False
                    print(f"[Annotate] PLACE frame={frame_idx} object={obj.name} target={target_obj.name}")

            if curr_open is not None and prev["open"] is not None and curr_open != prev["open"]:
                contact_link = self._get_moving_link(obj)
                if contact_link is not None:
                    signal_cls = OpenSignal if curr_open else CloseSignal
                    self.signals.append(
                        asdict(
                            signal_cls(
                                contact_frame_idx=frame_idx,
                                release_frame_idx=frame_idx,
                                object_name=obj.name,
                                object_category=getattr(obj, "category", ""),
                                contact_link=contact_link,
                            )
                        )
                    )
                    kind = "OPEN" if curr_open else "CLOSE"
                    print(f"[Annotate] {kind} frame={frame_idx} object={obj.name} link={contact_link}")

            self.prev_states[obj_name] = {
                "on_top_of": curr_on_top,
                "inside_of": curr_inside,
                "open": curr_open,
                "picked": prev.get("picked", False),
                "placed_on": prev.get("placed_on"),
            }

    def _get_object_on_top_of(self, obj):
        for other in self.env.scene.objects:
            if other == obj or other == self.robot:
                continue
            try:
                if obj.states[OnTop].get_value(other):
                    return other
            except Exception:
                pass
        return None

    def _get_object_inside_of(self, obj):
        for other in self.env.scene.objects:
            if other == obj or other == self.robot:
                continue
            try:
                if obj.states[Inside].get_value(other):
                    return other
            except Exception:
                pass
        return None

    def _detect_pick_from_contacts(self):
        arm = self.robot.default_arm
        contact_prims, robot_contact_links = self.robot._find_gripper_contacts(arm=arm)
        if not contact_prims:
            return None

        finger_link_paths = {link.prim_path for link in self.robot.finger_links[arm]}
        for contact_prim_path in contact_prims:
            try:
                contact_obj_prim_path = "/".join(contact_prim_path.split("/")[:4])
                obj = self.env.scene.object_registry("prim_path", contact_obj_prim_path, None)
                if obj is None:
                    continue
                if self.prev_states.get(obj.name, {}).get("picked", False):
                    continue
                link_name = contact_prim_path.split("/")[-1]
                if link_name != obj.root_link_name:
                    continue
                fingers_in_contact = finger_link_paths.intersection(robot_contact_links[contact_prim_path])
                if len(fingers_in_contact) >= self.min_finger_contacts_for_pick:
                    self.prev_states.setdefault(obj.name, {})["picked"] = True
                    return obj
            except Exception as exc:
                print(f"[Annotate] WARNING: failed to parse contact {contact_prim_path}: {exc}")
        return None

    def _get_moving_link(self, obj):
        if getattr(obj, "n_joints", 0) <= 0:
            return None
        max_vel = 0.0
        max_joint = None
        for joint in obj.joints.values():
            _, vel, _ = joint.get_state()
            abs_vel = abs(vel.item()) if vel.ndim == 0 else th.max(th.abs(vel)).item()
            if abs_vel > max_vel:
                max_vel = abs_vel
                max_joint = joint
        if max_vel <= 0.01 or max_joint is None:
            return None
        body1 = getattr(max_joint, "body1", None)
        return body1.split("/")[-1] if body1 else None

    def get_annotations(self) -> list[dict[str, Any]]:
        return self.signals

    def episode_start_callback(self, episode_id: int, env: Any) -> None:
        self.reset()
