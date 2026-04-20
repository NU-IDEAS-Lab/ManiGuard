"""Tests for pack_retry_loop — uses mock callbacks, no simulator needed."""

import pytest

from sentinel.utils.pack_retry_loop import (
    PackRetryConfig,
    _build_min_clearance_schedule,
    _local_bounds_from_zone,
    _select_cull_candidate,
)
from sentinel.utils.clutter_pack_layout import ClutterObjectDescriptor, ClutterPackSpec, ClutterPackEntry


class TestBuildMinClearanceSchedule:
    def test_basic(self):
        schedule = _build_min_clearance_schedule(0.008, 0.002, 0.005)
        assert schedule[0] == 0.008
        assert schedule[-1] == 0.002
        assert len(schedule) >= 2
        # Should be monotonically decreasing.
        for i in range(1, len(schedule)):
            assert schedule[i] <= schedule[i - 1]

    def test_equal_start_floor(self):
        schedule = _build_min_clearance_schedule(0.005, 0.005, 0.001)
        assert schedule == (0.005,)

    def test_floor_above_start_raises(self):
        with pytest.raises(ValueError):
            _build_min_clearance_schedule(0.002, 0.008, 0.001)


class TestLocalBoundsFromZone:
    def test_symmetric(self):
        zone = ((1.0, 2.0), (3.0, 4.0))
        local = _local_bounds_from_zone(zone)
        assert local == ((-1.0, -1.0), (1.0, 1.0))

    def test_small_zone(self):
        zone = ((0.0, 0.0), (0.4, 0.6))
        local = _local_bounds_from_zone(zone)
        assert abs(local[0][0] - (-0.2)) < 1e-9
        assert abs(local[1][1] - 0.3) < 1e-9


class TestSelectCullCandidate:
    def _make_descriptor(self, inst_id, role):
        return ClutterObjectDescriptor(
            instance_id=inst_id,
            role=role,
            half_extent_xy=(0.04, 0.04),
            height=0.1,
        )

    def test_never_culls_target(self):
        descs = [
            self._make_descriptor("cup_1", "target"),
            self._make_descriptor("glass_1", "fragile"),
        ]
        result = _select_cull_candidate(descs, None)
        assert result is not None
        assert result[0] == "glass_1"

    def test_prefers_clutter_over_fragile(self):
        descs = [
            self._make_descriptor("cup_1", "target"),
            self._make_descriptor("glass_1", "fragile"),
            self._make_descriptor("bowl_1", "clutter"),
        ]
        result = _select_cull_candidate(descs, None)
        assert result[0] == "bowl_1"

    def test_empty_returns_none(self):
        assert _select_cull_candidate([], None) is None

    def test_only_targets_returns_none(self):
        descs = [self._make_descriptor("cup_1", "target")]
        assert _select_cull_candidate(descs, None) is None

    def test_uses_pack_radius_for_ordering(self):
        descs = [
            self._make_descriptor("cup_1", "target"),
            self._make_descriptor("bowl_1", "clutter"),
            self._make_descriptor("bowl_2", "clutter"),
        ]
        pack_spec = ClutterPackSpec(
            table_obj_name="table",
            pack_origin_world=(0.0, 0.0, 0.0),
            object_entries=(
                ClutterPackEntry(inst_id="cup_1", role="target", rel_pose=(0.0, 0.0, 0.01, 0, 0, 0, 1)),
                ClutterPackEntry(inst_id="bowl_1", role="clutter", rel_pose=(0.1, 0.0, 0.01, 0, 0, 0, 1)),
                ClutterPackEntry(inst_id="bowl_2", role="clutter", rel_pose=(0.05, 0.0, 0.01, 0, 0, 0, 1)),
            ),
            seed=42,
            template_id="test",
        )
        result = _select_cull_candidate(descs, pack_spec)
        # Should pick the outermost clutter (bowl_1 at radius 0.1).
        assert result[0] == "bowl_1"


class TestPackRetryConfig:
    def test_defaults(self):
        cfg = PackRetryConfig()
        assert cfg.pack_jitter_xy == 0.022
        assert cfg.pack_min_clearance == 0.008
        assert cfg.pack_tries_per_clearance == 10
