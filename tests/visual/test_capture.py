from contextlib import nullcontext
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("hz,ticks", [(20, 6), (30, 4)])
def test_relative_command_is_submitted_once(hz, ticks):
    from maniguard.visual.capture import SourceCapture

    sim = SimpleNamespace(
        current_time=0.0,
        current_time_step_index=0,
        get_physics_dt=lambda: 1 / 120,
        get_sim_step_dt=lambda: 1 / hz,
        render_on_step=lambda value: nullcontext(),
    )
    actions = []

    def step(action):
        actions.append(action)
        sim.current_time += 1 / hz
        sim.current_time_step_index += ticks
        return "env_result"

    capture = object.__new__(SourceCapture)
    capture.env = SimpleNamespace(step=step)
    capture.sim = sim
    capture.observation_id = 0
    capture.capture = lambda index: {"observation_id": index}
    result, obs = capture.step("relative command")
    assert actions == ["relative command"]
    assert result == "env_result"
    assert obs["observation_id"] == 1
    assert sim.current_time_step_index == ticks


def test_barrier_requires_deliveries_and_never_accepts_missing_clock():
    from maniguard.visual.capture import wait_for_delivery

    clock = iter([0, 0, 1, 2, 3])
    calls = []
    assert wait_for_delivery(lambda: calls.append(1), lambda: next(clock), 3, 5) == 4
    with pytest.raises(RuntimeError, match="dispatcher"):
        wait_for_delivery(lambda: None, lambda: None, 1, 2)
    with pytest.raises(TimeoutError):
        wait_for_delivery(lambda: None, lambda: 0, 1, 2)


def test_source_capture_claims_accumulation_without_pausing_physics():
    from maniguard.visual.capture import claim_source_rendering

    calls = []

    class Orchestrator:
        def set_capture_on_play(self, enabled):
            calls.append(("capture_on_play", enabled))

        def pause(self):
            calls.append(("pause_replicator",))

    claim_source_rendering(Orchestrator())
    assert calls == [("capture_on_play", False), ("pause_replicator",)]


def test_capture_starts_new_accumulation_and_rejects_external_resets(monkeypatch):
    import maniguard.visual.capture as module

    settings = {"/rtx/externalFrameCounter": 100}

    class Settings:
        get = settings.get

        def set(self, key, value):
            settings[key] = value

    capture = object.__new__(module.SourceCapture)
    capture.settings = Settings()
    capture.frame_counter = 2
    capture.profile = {"renderer": "PathTracing", "total_spp": 128, "spp_per_update": 16}
    capture.sensors = {}
    capture.sim = SimpleNamespace(current_time=1.0, current_time_step_index=120, render=lambda: None)
    monkeypatch.setattr(module, "wait_for_delivery", lambda *args: 10)
    monkeypatch.setattr(module, "read_rgb", lambda *args: {})
    monkeypatch.setattr(module, "camera_metadata", lambda *args: {})
    monkeypatch.setattr(module, "dispatcher_time", lambda: 1.0)
    capture.capture(0)
    assert capture.frame_counter == 101

    def conflicting_render(*args):
        settings["/rtx/externalFrameCounter"] += 20
        return 10

    monkeypatch.setattr(module, "wait_for_delivery", conflicting_render)
    with pytest.raises(RuntimeError, match="accumulation"):
        capture.capture(1)


def test_dlaa_delivery_does_not_reset_or_require_pt_counter(monkeypatch):
    import maniguard.visual.capture as module
    from maniguard.visual.profile import load_profile

    settings = {"/rtx/externalFrameCounter": 100}
    capture = object.__new__(module.SourceCapture)
    capture.settings = SimpleNamespace(get=settings.get, set=lambda k, v: settings.update({k: v}))
    capture.frame_counter = 100
    capture.profile = load_profile({"renderer": "RaytracedLighting"})
    capture.sensors = {}
    capture.sim = SimpleNamespace(current_time=1.0, current_time_step_index=120, render=lambda: None)
    waits = []

    def wait(render, clock, required, maximum):
        waits.append(required)
        settings["/rtx/externalFrameCounter"] += 1
        return required

    monkeypatch.setattr(module, "wait_for_delivery", wait)
    monkeypatch.setattr(module, "read_rgb", lambda *args: {})
    monkeypatch.setattr(module, "camera_metadata", lambda *args: {})
    monkeypatch.setattr(module, "dispatcher_time", lambda: 1.0)
    result = capture.capture(0)
    assert waits == [capture.profile["render_deliveries"]]
    assert result["render_metadata"]["renderer"] == "RaytracedLighting"
    assert settings["/rtx/externalFrameCounter"] == 101


def test_route_replacement_restores_global_settings_before_pt():
    from maniguard.visual.capture import SourceCapture
    from maniguard.visual.profile import load_profile

    values = {"/rtx/rendermode": "RaytracedLighting", "/rtx/sceneDb/ambientLightIntensity": 0.2}
    settings = SimpleNamespace(
        get=values.get, set=lambda k, v: values.update({k: v}), destroy_item=lambda k: values.pop(k, None)
    )
    env = SimpleNamespace()

    def make(profile):
        cap = object.__new__(SourceCapture)
        cap.env, cap.settings = env, settings
        cap.profile, cap.material_override = load_profile(profile), None
        cap._activate()
        return cap

    dlaa = make({"renderer": "RaytracedLighting", "glass_shadows": "native"})
    assert values["/rtx/sceneDb/ambientLightIntensity"] == 1.0
    pt = make({"renderer": "PathTracing"})
    assert values["/rtx/rendermode"] == "PathTracing"
    assert values["/rtx/sceneDb/ambientLightIntensity"] == 0.2
    assert "/rtx/post/aa/op" not in values
    dlaa._activate()
    assert values["/rtx/sceneDb/ambientLightIntensity"] == 1.0
    # Closing a superseded capture must not change the active route.
    pt.close()
    assert values["/rtx/rendermode"] == "RaytracedLighting"
    dlaa.close()
    assert values == {"/rtx/rendermode": "RaytracedLighting", "/rtx/sceneDb/ambientLightIntensity": 0.2}
