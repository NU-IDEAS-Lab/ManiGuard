from types import SimpleNamespace

import numpy as np
import pytest


def fixture_env():
    class Sensor:
        image_height = 256
        image_width = 256

        def get_obs(self):
            return {"rgb": np.zeros((self.image_height, self.image_width, 4), np.uint8)}, {}

    env = SimpleNamespace(
        external_sensors={f"cam_{name}": Sensor() for name in ("left", "right", "left_shoulder", "opposite")},
        robots=[SimpleNamespace(sensors={"panda_hand/Camera": Sensor()})],
    )
    env.robots[0].sensors["panda_hand/Camera"].prim_path = "/World/agent/panda_hand/Camera"
    env.robots[0].eef_links = {"0": SimpleNamespace(prim_path="/World/agent/panda_hand")}
    env.robots[0].default_arm = "0"
    env.load_observation_space = lambda: None
    return env


def test_five_cameras_are_required_and_real_sensor_size_is_checked():
    from maniguard.visual.cameras import configure_source_sensors, read_rgb
    from maniguard.visual.profile import load_profile

    env = fixture_env()
    sensors = configure_source_sensors(env, load_profile())
    assert len(sensors) == 5
    assert read_rgb(sensors, load_profile())["wrist_image"].shape == (640, 640, 3)
    env.external_sensors.pop("cam_left")
    with pytest.raises(ValueError, match="cam_left"):
        configure_source_sensors(env, load_profile())


def test_no_fallback_to_unrelated_robot_camera_or_black_image():
    from maniguard.visual.cameras import configure_source_sensors, read_rgb
    from maniguard.visual.profile import load_profile

    env = fixture_env()
    env.robots[0].sensors["other_camera"] = env.robots[0].sensors.pop("panda_hand/Camera")
    env.robots[0].sensors["other_camera"].prim_path = "/World/agent/base/Camera"
    with pytest.raises(ValueError, match="wrist"):
        configure_source_sensors(env, load_profile())
    env = fixture_env()
    sensors = configure_source_sensors(env, load_profile())
    sensors["wrist_image"].get_obs = lambda: ({}, {})
    with pytest.raises(ValueError, match="wrist_image"):
        read_rgb(sensors, load_profile())


def test_wrist_follows_robot_end_effector_link_not_a_hardcoded_hand_name():
    from maniguard.visual.cameras import configure_source_sensors
    from maniguard.visual.profile import load_profile

    env = fixture_env()
    wrist = env.robots[0].sensors.pop("panda_hand/Camera")
    wrist.prim_path = "/World/agent/eef_link/Camera"
    env.robots[0].sensors["agent:eef_link:Camera:0"] = wrist
    env.robots[0].eef_links = {"0": SimpleNamespace(prim_path="/World/agent/eef_link")}
    env.robots[0].default_arm = "0"
    assert configure_source_sensors(env, load_profile())["wrist_image"] is wrist


def test_headless_previews_do_not_disable_recorded_render_products():
    from maniguard.visual.cameras import disable_headless_previews

    def viewport(name, path):
        return SimpleNamespace(
            name=name,
            viewport_api=SimpleNamespace(updates_enabled=True, get_render_product_path=lambda: path),
        )

    preview = viewport("preview", "/Render/preview")
    shared = viewport("shared", "/Render/source")
    sensor = SimpleNamespace(render_product=SimpleNamespace(path="/Render/source"))
    views = [preview, shared]
    assert disable_headless_previews(views, {"image_left": sensor}, headless=False) == []
    assert preview.viewport_api.updates_enabled
    assert disable_headless_previews(views, {"image_left": sensor}, headless=True) == ["preview"]
    assert not preview.viewport_api.updates_enabled
    assert shared.viewport_api.updates_enabled
