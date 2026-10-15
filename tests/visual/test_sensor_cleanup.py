"""Camera backends must detach while their scene and render graph still exist."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_clear_detaches_lazy_camera_parameters_before_scene_deletion(monkeypatch):
    calls = []
    annotators = {"rgb": object(), "camera_params": object()}

    def detach(modality):
        assert not any(event[0] == "clear" for event in calls)
        calls.append(("detach", modality))
        annotators[modality] = None

    def clear(**kwargs):
        assert all(value is None for value in annotators.values())
        calls.append(("clear", kwargs))
        return "new simulator"

    viewport = SimpleNamespace(
        viewport_api=SimpleNamespace(set_active_camera=lambda path: calls.append(("camera", path)))
    )
    og = SimpleNamespace(
        clear=clear,
        sim=object(),
        lazy=SimpleNamespace(
            omni=SimpleNamespace(
                kit=SimpleNamespace(
                    viewport=SimpleNamespace(
                        window=SimpleNamespace(get_viewport_window_instances=lambda: iter([viewport]))
                    )
                )
            )
        ),
    )
    sensor = SimpleNamespace(_annotators=annotators, _remove_modality_from_backend=detach)
    monkeypatch.setitem(sys.modules, "omnigibson", og)
    monkeypatch.setitem(
        sys.modules, "omnigibson.sensors", SimpleNamespace(VisionSensor=SimpleNamespace(SENSORS={"camera": sensor}))
    )
    path = Path(__file__).resolve().parents[2] / "maniguard/_omnigibson_patches.py"
    spec = importlib.util.spec_from_file_location("visual_cleanup_patch", path)
    patches = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patches)
    patches._patch_visual_clear()
    wrapped = og.clear
    patches._patch_visual_clear()
    assert og.clear is wrapped
    assert og.clear(physics_dt=1 / 120) == "new simulator"
    assert calls == [
        ("camera", "/OmniverseKit_Persp"),
        ("detach", "rgb"),
        ("detach", "camera_params"),
        ("clear", {"physics_dt": 1 / 120}),
    ]
    # A real backend failure must remain visible, not be interpreted as a reset.
    sensor._remove_modality_from_backend = lambda modality: (_ for _ in ()).throw(RuntimeError("detach failed"))
    with pytest.raises(RuntimeError, match="detach failed"):
        og.clear()
