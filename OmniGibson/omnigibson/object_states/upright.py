import torch as th

from omnigibson.macros import create_module_macros
from omnigibson.object_states.object_state_base import AbsoluteObjectState, BooleanStateMixin
import omnigibson.utils.transform_utils as T

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.DEFAULT_MAX_TILT_DEG = 45.0


class Upright(AbsoluteObjectState, BooleanStateMixin):
    """
    State that checks whether an object's local up-axis is within a tilt
    threshold of the world up-axis.

    The check computes the angle between the object's rotated +Z and the world
    +Z using the object's orientation quaternion. If the angle exceeds
    ``max_tilt_deg`` the object is considered *not* upright.
    """

    def __init__(self, obj, max_tilt_deg=None):
        super().__init__(obj)
        self.max_tilt_deg = max_tilt_deg if max_tilt_deg is not None else m.DEFAULT_MAX_TILT_DEG

    def _get_value(self):
        _, quat = self.obj.get_position_orientation()
        quat_t = th.as_tensor(quat, dtype=th.float32)
        world_up = th.tensor([0.0, 0.0, 1.0], dtype=th.float32)
        up = T.quat_apply(quat_t, world_up)
        up = th.as_tensor(up, dtype=th.float32).reshape(-1)[-3:]
        up = up / (th.norm(up) + 1e-8)
        cos_angle = th.clamp(th.dot(up, world_up), -1.0, 1.0)
        angle_deg = float(th.rad2deg(th.acos(cos_angle)))
        return angle_deg <= self.max_tilt_deg

    def _set_value(self, new_value):
        raise NotImplementedError("Upright state cannot be set directly.")

    def _has_changed(self, get_value_args, value, info):
        return self.get_value() != value
