"""Eager lid/cap → container snap-attach helper.

When the robot has placed a lid (or cap) physically *touching* its
canonical container and is no longer grasping it, this helper
repositions the lid's male attachment meta-link directly onto the
container's female meta-link and calls ``set_value(container, True)`` on
the canonical ``AttachedTo`` state. The 7-step OmniGibson orchestration
creates the fixed joint, after which the lid follows the container.

Discovery is automatic: at construction time, walk all loaded objects
and pair each lid/cap that has an ``AttachedTo`` male meta-link with the
container that holds the matching female meta-link. Pairs that don't
match (operator placed two unrelated lids) are simply ignored.

Usage:

    from maniguard.utils.lid_attach import LidSnapper

    snapper = LidSnapper(env)            # discovers eligible pairs once
    for step in range(N):
        env.step(action)
        snapper.try_snap(robot=robot)    # eager attach when touching + released

The snap fires when ALL of:
  * lid is not already attached to container,
  * lid is in PhysX contact with the container (any link),
  * ``robot.is_grasping(candidate_obj=lid)`` is not TRUE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


_LID_LIKE_CATEGORIES = frozenset({"lid", "cap"})


def find_M_link(obj):
    for _, link in obj.links.items():
        if getattr(link, "is_meta_link", False) and link.meta_link_id.endswith("M"):
            return link
    return None


def find_F_link(obj, m_id: str):
    f_target = m_id[:-1] + "F"
    for _, link in obj.links.items():
        if getattr(link, "is_meta_link", False) and link.meta_link_id == f_target:
            return link
    return None


def reposition_lid_onto_F(lid, container) -> bool:
    """Set lid pose so its M-link frame matches the container's F-link
    frame, then lift 1 cm in world z so the next sim step is a clean
    drop instead of an instant penetration."""
    import omnigibson.utils.transform_utils as T

    m = find_M_link(lid)
    if m is None:
        return False
    f = find_F_link(container, m.meta_link_id)
    if f is None:
        return False

    parent_pos, parent_quat = f.get_position_orientation()
    child_pos, child_quat = m.get_position_orientation()
    child_root_pos, child_root_quat = lid.get_position_orientation()

    # rel = parent ⊗ child^-1 — transform that brings child's frame to parent's.
    rel_pos, rel_quat = T.mat2pose(
        T.pose2mat((parent_pos, parent_quat))
        @ T.pose_inv(T.pose2mat((child_pos, child_quat)))
    )
    new_root_pos, new_root_quat = T.pose_transform(
        rel_pos, rel_quat, child_root_pos, child_root_quat
    )
    lid.set_position_orientation(
        position=(float(new_root_pos[0]),
                  float(new_root_pos[1]),
                  float(new_root_pos[2]) + 0.01),
        orientation=new_root_quat,
    )
    lid.keep_still()
    return True


@dataclass
class _AttachPair:
    lid: object
    container: object


class LidSnapper:
    """Discover lid↔container pairs at init; offer ``try_snap`` per-step."""

    def __init__(self, env):
        self.env = env
        from omnigibson.object_states import AttachedTo, ContactBodies

        self._AttachedTo = AttachedTo
        self._ContactBodies = ContactBodies
        self.pairs: list[_AttachPair] = self._discover(env)
        # Print so it's visible without tweaking log config — same line goes
        # to log.info for callers that have INFO routed elsewhere.
        msg = f"[LidSnapper] {len(self.pairs)} eligible (lid|cap, container) pair(s)"
        print(msg, flush=True)
        log.info(msg)
        for p in self.pairs:
            line = f"[LidSnapper]   {p.lid.name} -> {p.container.name}"
            print(line, flush=True)
            log.info(line)

    def _discover(self, env) -> list[_AttachPair]:
        AttachedTo = self._AttachedTo
        # Collect every lid/cap that has an M-link + AttachedTo state.
        lids = []
        for obj in env.scene.objects:
            cat = getattr(obj, "category", None)
            if cat not in _LID_LIKE_CATEGORIES:
                continue
            if AttachedTo not in obj.states:
                continue
            m = find_M_link(obj)
            if m is None:
                continue
            lids.append((obj, m.meta_link_id))

        # For each lid, find the container holding the matching F-link.
        # NOTE: the container does NOT need its own AttachedTo state —
        # AttachedTo lives on the M-side (the lid). We only require that
        # the container exposes the F meta-link.
        pairs: list[_AttachPair] = []
        for lid, m_id in lids:
            f_target = m_id[:-1] + "F"
            container = None
            for obj in env.scene.objects:
                if obj is lid:
                    continue
                if find_F_link(obj, m_id) is not None or any(
                    getattr(link, "is_meta_link", False)
                    and link.meta_link_id == f_target
                    for link in obj.links.values()
                ):
                    container = obj
                    break
            if container is not None:
                pairs.append(_AttachPair(lid=lid, container=container))
            else:
                log.debug("LidSnapper: no container with F-link %s for lid %s",
                          f_target, lid.name)
        return pairs

    def _is_grasped(self, robot, lid) -> bool:
        if robot is None:
            return False
        from omnigibson.controllers.controller_base import IsGraspingState
        state = robot.is_grasping(candidate_obj=lid)
        return state == IsGraspingState.TRUE

    def _lid_touching_container(self, lid, container) -> tuple[bool, int]:
        """Return ``(touching, n_container_links_in_contact)``.

        Contact lookup goes via PhysX through the ContactBodies state.
        Raises if the state isn't on the lid — that means the lid wasn't
        eligible in the first place, which discover() already filtered for.
        """
        ContactBodies = self._ContactBodies
        contact_links = lid.states[ContactBodies].get_value()
        if not contact_links:
            return False, 0
        container_link_set = set(container.links.values())
        hits = contact_links & container_link_set
        return bool(hits), len(hits)

    def try_snap(self, robot=None, verbose: bool = False) -> str | None:
        """Run one snap pass over all known pairs. Returns the name of the
        first pair attached this call, or None.

        When ``verbose=True``, prints one line per pair explaining which
        check (already-attached, still-grasped, not-touching, reposition,
        set_value) skipped the pair this call.
        """
        AttachedTo = self._AttachedTo
        import omnigibson as og
        for p in self.pairs:
            try:
                if p.lid.states[AttachedTo].get_value(p.container):
                    if verbose:
                        print(f"[LidSnapper] {p.lid.name}->{p.container.name}: "
                              f"already-attached", flush=True)
                    continue  # already attached
            except KeyError:
                if verbose:
                    print(f"[LidSnapper] {p.lid.name}->{p.container.name}: "
                          f"AttachedTo state missing (KeyError)", flush=True)
                continue
            grasped = self._is_grasped(robot, p.lid)
            if grasped:
                if verbose:
                    print(f"[LidSnapper] {p.lid.name}->{p.container.name}: "
                          f"still-grasped-by-robot", flush=True)
                continue
            touching, n_hits = self._lid_touching_container(p.lid, p.container)
            if not touching:
                if verbose:
                    print(f"[LidSnapper] {p.lid.name}->{p.container.name}: "
                          f"not-touching (n_container_links_in_contact={n_hits})",
                          flush=True)
                continue

            if not reposition_lid_onto_F(p.lid, p.container):
                if verbose:
                    print(f"[LidSnapper] {p.lid.name}->{p.container.name}: "
                          f"reposition_lid_onto_F returned False", flush=True)
                continue
            og.sim.step()  # let new poses register
            try:
                # can_joint_break=False so the FixedJoint can't snap under
                # Phase 2B's transport acceleration (otherwise the lid's
                # inertia pulling against the joint mid-motion exceeds the
                # default break_force and the lid detaches mid-trajectory).
                ok = p.lid.states[AttachedTo].set_value(
                    p.container, True, can_joint_break=False)
            except Exception as exc:
                log.warning("LidSnapper: set_value failed on %s -> %s: %s",
                            p.lid.name, p.container.name, exc)
                if verbose:
                    print(f"[LidSnapper] {p.lid.name}->{p.container.name}: "
                          f"set_value raised {type(exc).__name__}: {exc}",
                          flush=True)
                continue
            og.sim.step()
            if ok and p.lid.states[AttachedTo].get_value(p.container):
                line = f"[LidSnapper] ATTACHED {p.lid.name} -> {p.container.name}"
                print(line, flush=True)
                log.info(line)
                return p.lid.name
            if verbose:
                print(f"[LidSnapper] {p.lid.name}->{p.container.name}: "
                      f"set_value ok={ok} but AttachedTo.get_value()=False "
                      f"after step", flush=True)
        return None
