import pytest


@pytest.mark.parametrize("action_hz,ticks", [(20, 6), (30, 4)])
def test_exact_tick_counts(action_hz, ticks):
    from maniguard.visual.capture import time_contract

    assert time_contract(action_hz, 60, 120)["physics_ticks_per_action"] == ticks


@pytest.mark.parametrize("rates", [(25, 60, 120), (20, 60, 100), (0, 60, 120)])
def test_incompatible_time_contract_fails(rates):
    from maniguard.visual.capture import time_contract

    with pytest.raises(ValueError):
        time_contract(*rates)


def test_opt_in_patch_retains_legacy_validation_and_rounds_ratios():
    from maniguard._omnigibson_patches import enable_source_timing

    class Sim:
        def __init__(self):
            self.physics = self.rendering = self.dt = 1 / 120

        def get_physics_dt(self):
            return self.physics

        def get_rendering_dt(self):
            return self.rendering

        def get_sim_step_dt(self):
            return self.dt

        def _validate_dts(self, physics_dt, rendering_dt, sim_step_dt):
            assert rendering_dt == sim_step_dt, "legacy headless"

        def set_simulation_dt(self, physics_dt=None, rendering_dt=None, sim_step_dt=None):
            self.physics = physics_dt or self.physics
            self.rendering = rendering_dt or self.rendering
            self.dt = sim_step_dt or self.dt
            self._validate_dts(self.physics, self.rendering, self.dt)
            self._n_steps_per_loop = int(self.dt // self.rendering)

    sim, legacy = Sim(), Sim()
    enable_source_timing(sim)
    sim.set_simulation_dt(1 / 120, 1 / 60, 1 / 20)
    assert sim._n_steps_per_loop == 3
    with pytest.raises(ValueError):
        sim.set_simulation_dt(1 / 120, 1 / 60, 1 / 25)
    assert sim.dt == 1 / 20  # Validate before altering the running simulator.
    with pytest.raises(AssertionError):
        legacy.set_simulation_dt(1 / 120, 1 / 60, 1 / 20)


def test_source_simulator_reset_reinstalls_timing_after_clear(monkeypatch):
    import sys
    from types import SimpleNamespace

    from maniguard.visual.capture import initialize_source_simulator

    calls = []

    def simulator():
        return SimpleNamespace(
            stop=lambda: calls.append("stop"), set_simulation_dt=lambda **kwargs: calls.append(kwargs)
        )

    og = SimpleNamespace(sim=simulator())

    def clear(**kwargs):
        calls.append(("clear", kwargs))
        og.sim = simulator()

    og.clear = clear
    monkeypatch.setitem(sys.modules, "omnigibson", og)
    monkeypatch.setattr("maniguard._omnigibson_patches.enable_source_timing", lambda sim: calls.append(("patch", sim)))
    initialize_source_simulator(20, 60, 120, clear=True)
    assert calls[1][1]["sim_step_dt"] == 1 / 60  # Constructor's headless contract.
    assert calls[2] == ("patch", og.sim)
    assert calls[3]["sim_step_dt"] == 1 / 20  # Actual control contract after patch.
