"""Synchronous source capture with separate physics and render clocks."""

from __future__ import annotations

import math
import sys

from maniguard.visual.cameras import camera_metadata, configure_source_sensors, disable_headless_previews, read_rgb
from maniguard.visual.profile import apply_profile, load_profile, renderer_settings

_SYNC_SETTINGS = {
    "/app/updateOrder/checkForHydraRenderComplete": 1000,
    "/app/renderer/waitIdle": True,
    "/app/hydraEngine/waitIdle": True,
}


def prepare_source_launch() -> None:
    """Call after application CLI parsing and before the simulator launches.

    Kit reads its update order during startup. Applying it after launch cannot
    establish the same synchronization contract.
    """
    for key, value in _SYNC_SETTINGS.items():
        token = f"--{key}={str(value).lower()}"
        matching = [arg for arg in sys.argv if arg.startswith(f"--{key}=")]
        if matching and matching != [token]:
            raise ValueError(f"Conflicting source rendering startup option: {matching}")
        if not matching:
            sys.argv.append(token)


def time_contract(action_hz: float, rendering_hz: float, physics_hz: float) -> dict:
    rates = (action_hz, rendering_hz, physics_hz)
    if any(not math.isfinite(value) or value <= 0 for value in rates):
        raise ValueError("Simulation frequencies must be finite and positive")
    ratios = (physics_hz / rendering_hz, rendering_hz / action_hz)
    if any(value < 1 or not math.isclose(value, round(value), rel_tol=1e-8) for value in ratios):
        raise ValueError("Physics/render/action frequencies require positive integer timestep ratios")
    return {
        "action_hz": action_hz,
        "rendering_hz": rendering_hz,
        "physics_hz": physics_hz,
        "physics_ticks_per_action": round(physics_hz / action_hz),
        "physics_ticks_per_render": round(ratios[0]),
        "render_intervals_per_action": round(ratios[1]),
    }


def initialize_source_simulator(action_hz, rendering_hz=60, physics_hz=120, *, clear=False):
    """Launch/reset at legal headless startup rates, then set the control rate.

    OmniGibson clear() creates a new Simulator class and instance. Reinstall
    the instance-scoped timing adapter on every reset, never on upstream files.
    """
    import omnigibson as og
    from omnigibson.macros import gm

    from maniguard._omnigibson_patches import enable_source_timing

    time_contract(action_hz, rendering_hz, physics_hz)
    gm.ENABLE_HQ_RENDERING = True
    startup = {"physics_dt": 1 / physics_hz, "rendering_dt": 1 / rendering_hz, "sim_step_dt": 1 / rendering_hz}
    if clear and og.sim is not None:
        og.sim.stop()
        og.clear(**startup)
    elif og.sim is None:
        og.launch(**startup)
    enable_source_timing(og.sim)
    og.sim.set_simulation_dt(physics_dt=1 / physics_hz, rendering_dt=1 / rendering_hz, sim_step_dt=1 / action_hz)
    return og.sim


def dispatcher_time():
    """SDG delivery clock, deliberately distinct from simulator physical time."""
    import omni.graph.core as graph

    node = graph.get_node_by_path("/Render/PostProcess/SDGPipeline/PostProcessDispatcher")
    if node is None or not node.is_valid():
        return None
    denominator = node.get_attribute("outputs:referenceTimeDenominator").get()
    if not denominator:
        return None
    return node.get_attribute("outputs:referenceTimeNumerator").get() / denominator


def wait_for_delivery(render, clock, required: int, max_passes: int) -> int:
    """Require actual SDG delivery advances; a render call alone is insufficient."""
    last = clock()
    if last is None:
        raise RuntimeError("Source capture requires an active SDG dispatcher clock")
    delivered = 0
    for passes in range(1, max_passes + 1):
        render()
        current = clock()
        if current is None:
            raise RuntimeError("SDG dispatcher stopped reporting deliveries")
        if current != last:
            delivered += 1
        last = current
        if delivered >= required:
            return passes
    raise TimeoutError(f"Source render barrier delivered {delivered}/{required} frames in {max_passes} passes")


def claim_source_rendering(orchestrator) -> None:
    """Leave sensor annotators active; give source capture ownership of rendering.

    Disabling timeline coupling first keeps Replicator's pause from pausing the
    simulator. SourceCapture then owns sensor deliveries.
    """
    orchestrator.set_capture_on_play(False)
    orchestrator.pause()


class SourceCapture:
    """Capture five views after one action step, never reapply relative actions.

    A capture is indexed before its outgoing command. Slow rendering changes
    wall-clock throughput, not the amount of physics simulated by that command.
    """

    def __init__(self, env, profile=None):
        import omnigibson as og
        from omnigibson import lazy
        from omnigibson.macros import gm

        self.env = env
        self.sim = og.sim
        self.profile = load_profile(profile)
        self.settings = lazy.carb.settings.get_settings()
        for key, value in _SYNC_SETTINGS.items():
            if self.settings.get(key) != value:
                raise RuntimeError(f"Source capture requires startup setting {key}={value}")
        if getattr(env, "_automatic_reset", False):
            raise ValueError("Source capture requires automatic_reset=false to retain terminal observations")
        expected_dt = 1 / env.env_config["action_frequency"]
        if not math.isclose(self.sim.get_sim_step_dt(), expected_dt):
            raise RuntimeError("Simulator control timestep differs from environment configuration")
        self.sensors = configure_source_sensors(env, self.profile)
        # Initialize the lazy camera-parameter annotators before any capture.
        for sensor in self.sensors.values():
            _ = sensor.camera_parameters
        self.material_override = None
        self._activate()
        viewports = list(lazy.omni.kit.viewport.window.get_viewport_window_instances())
        self.previous_preview_updates = [(v.viewport_api, v.viewport_api.updates_enabled) for v in viewports]
        self.disabled_previews = disable_headless_previews(viewports, self.sensors, headless=gm.HEADLESS)
        claim_source_rendering(lazy.omni.replicator.core.orchestrator)
        self.observation_id = 0

    def _activate(self):
        """Reapply the route when playback or another capture used this environment."""
        previous = getattr(self.env, "_maniguard_active_source_capture", None)
        if previous is self:
            return
        if previous is not None:
            previous.close()
        self.previous_settings = {key: self.settings.get(key) for key in renderer_settings(self.profile)}
        self.render_report = apply_profile(self.settings, self.profile)
        if self.profile.get("glass_shadows") == "exclude_omniglass":
            from omnigibson import lazy

            from maniguard.visual.materials import GlassShadowOverride

            self.material_override = GlassShadowOverride(lazy.omni.usd.get_context().get_stage())
        self.render_report["material_overrides"] = (
            self.material_override.report if self.material_override is not None else []
        )
        self.env._maniguard_active_source_capture = self

    def close(self):
        if self.material_override is not None:
            self.material_override.close()
            self.material_override = None
        if getattr(self.env, "_maniguard_active_source_capture", None) is self:
            for key, value in self.previous_settings.items():
                if value is None:
                    self.settings.destroy_item(key)
                else:
                    self.settings.set(key, value)
            for api, enabled in getattr(self, "previous_preview_updates", []):
                api.updates_enabled = enabled
            self.env._maniguard_active_source_capture = None

    def capture(self, observation_id: int) -> dict:
        if getattr(self, "env", None) is not None:
            self._activate()
        before = float(self.sim.current_time)
        tick = int(self.sim.current_time_step_index)
        renderer = self.profile["renderer"]
        required = self.profile["render_deliveries"]
        passes = wait_for_delivery(self.sim.render, dispatcher_time, required, required * 4)
        frames = read_rgb(self.sensors, self.profile)
        metadata = camera_metadata(self.sensors)
        if float(self.sim.current_time) != before or int(self.sim.current_time_step_index) != tick:
            raise RuntimeError("Source rendering advanced physics")
        self.observation_id = observation_id
        self.last_observation = {
            "observation_id": observation_id,
            "sim_time_s": before,
            "physics_tick": tick,
            "camera_frames": frames,
            "camera_metadata": metadata,
            "render_metadata": {
                "renderer": renderer,
                "external_frame_counter": self.settings.get("/rtx/externalFrameCounter"),
                "render_passes": passes,
                "dispatcher_time": dispatcher_time(),
            },
        }

        return self.last_observation

    def step(self, action):
        before = float(self.sim.current_time)
        if hasattr(self, "last_observation") and before != self.last_observation["sim_time_s"]:
            raise RuntimeError("Physics advanced outside the source recording interval")
        tick = int(self.sim.current_time_step_index)
        dt = self.sim.get_sim_step_dt()
        expected_ticks = round(dt / self.sim.get_physics_dt())
        with self.sim.render_on_step(False):
            result = self.env.step(action)
        if (
            not math.isclose(float(self.sim.current_time) - before, dt, rel_tol=1e-6, abs_tol=1e-7)
            or int(self.sim.current_time_step_index) - tick != expected_ticks
        ):
            raise RuntimeError("Action advanced an unexpected number of physics ticks")
        return result, self.capture(self.observation_id + 1)

    def hold_step(self):
        """Advance existing controller goals without submitting a new action."""
        before = float(self.sim.current_time)
        if before != self.last_observation["sim_time_s"]:
            raise RuntimeError("Physics advanced outside the source recording interval")
        tick = int(self.sim.current_time_step_index)
        dt = self.sim.get_sim_step_dt()
        with self.sim.render_on_step(False):
            self.sim.step()
        if not math.isclose(float(self.sim.current_time) - before, dt, rel_tol=1e-6, abs_tol=1e-7) or int(
            self.sim.current_time_step_index
        ) - tick != round(dt / self.sim.get_physics_dt()):
            raise RuntimeError("Hold interval advanced an unexpected number of physics ticks")
        return self.capture(self.observation_id + 1)


def prepare_source_environment(config, source_profile="high_fidelity", *, clear=False):
    """Configure source sensors/rendering without changing the existing control clock.

    Call before Environment construction. Renderer launch preparation must precede
    importing OmniGibson in the public entrypoint.
    """
    from maniguard.visual.profile import resolve_profile_selection
    profile = resolve_profile_selection(source_profile)
    if profile is not None:
        from maniguard.utils.camera_setup import build_external_camera_configs
        settings = config.setdefault("env", {})
        settings["rendering_frequency"] = 60
        settings["external_sensors"] = build_external_camera_configs(resolution=profile["resolution"][0])
        for robot in config.get("robots", []):
            modalities = robot.get("obs_modalities", [])
            if modalities != "all":
                robot["obs_modalities"] = list(dict.fromkeys([*modalities, "rgb"]))
        initialize_source_simulator(settings.get("action_frequency", 30), 60,
                                    settings.get("physics_frequency", 120), clear=clear)
    return profile
