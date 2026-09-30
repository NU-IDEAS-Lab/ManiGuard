from omnigibson.object_states.object_state_base import AbsoluteObjectState, BooleanStateMixin

DEFAULT_DROP_FLOOR_Z = 0.0
DEFAULT_DROP_Z_MARGIN = 0.05


class Dropped(AbsoluteObjectState, BooleanStateMixin):
    """Check whether the object origin is below floor_z + z_margin.

    The thresholds are configurable per instance and can be set from the
    safety specification's proposition parameters."""

    def __init__(self, obj, floor_z=None, z_margin=None):
        super().__init__(obj)
        self.floor_z = floor_z if floor_z is not None else DEFAULT_DROP_FLOOR_Z
        self.z_margin = z_margin if z_margin is not None else DEFAULT_DROP_Z_MARGIN

    def _get_value(self):
        pos, _ = self.obj.get_position_orientation()
        return float(pos[2]) < self.floor_z + self.z_margin

    def _set_value(self, new_value):
        raise NotImplementedError("Dropped state cannot be set directly.")

    def _has_changed(self, get_value_args, value, info):
        return self.get_value() != value
