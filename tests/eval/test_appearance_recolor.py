"""Appearance overrides must affect the material inputs that render the target."""
from types import SimpleNamespace

import pytest
import torch

from maniguard.data.bench_builder.perturbation import apply_recolor


class GlassMaterial:
    """OmniGlass's inherited diffuse setters are no-ops; only color is active."""
    mdl_name = "OmniGlass.mdl"
    mtl_name = "OmniGlass"
    prim_path = "/target/glass"

    def __init__(self):
        self.color = [1., 1., 1.]
        self.ior = 1.491
        self.roughness = 0.

    @property
    def diffuse_tint(self):
        return torch.zeros(3)

    @diffuse_tint.setter
    def diffuse_tint(self, value):
        pass

    @property
    def albedo_add(self):
        return 0.

    @albedo_add.setter
    def albedo_add(self, value):
        pass


def test_mixed_glass_and_opaque_target_receives_declared_tint():
    glass = GlassMaterial()
    opaque = SimpleNamespace(prim_path="/target/metal", diffuse_tint=torch.ones(3), albedo_add=0.)
    obj = SimpleNamespace(materials=[glass, opaque, glass])

    changed = apply_recolor(obj, [1., .13, .13], albedo_add=.868)

    assert list(glass.color) == pytest.approx([1., .13, .13])
    assert list(opaque.diffuse_tint) == pytest.approx([1., .13, .13])
    assert opaque.albedo_add == pytest.approx(.868)
    assert (glass.ior, glass.roughness) == (1.491, 0.)
    assert changed == 2


def test_primitive_color_fallback_is_preserved():
    primitive = SimpleNamespace(prim_path="/target/primitive", diffuse_color_constant=torch.ones(3))
    assert apply_recolor(SimpleNamespace(materials=[primitive]), [0., .75, .91]) == 1
    assert list(primitive.diffuse_color_constant) == pytest.approx([0., .75, .91])


@pytest.mark.parametrize("selection", [None, "high_fidelity", {}])
def test_saved_scene_applies_declared_glass_tint_before_render(tmp_path, monkeypatch, selection):
    import sys
    from maniguard.data.bench_builder import render
    from maniguard.data.datagen.primitives import scene
    from maniguard.visual import capture

    glass = GlassMaterial()
    obj = SimpleNamespace(name="jar_0", materials=[glass])
    robot = SimpleNamespace()  # The saved pose must not be overwritten.
    env = SimpleNamespace(robots=[robot], reset=lambda: None,
        scene=SimpleNamespace(object_registry=lambda kind, name: obj))
    sim = SimpleNamespace(step=lambda: None, stop=lambda: None)
    monkeypatch.setitem(sys.modules, "omnigibson", SimpleNamespace(sim=sim,
        Environment=lambda **kw: env, clear=lambda: None))
    monkeypatch.setattr(scene, "init_omnigibson", lambda **kw: sys.modules["omnigibson"])
    monkeypatch.setattr(scene, "task_needs_gpu_dynamics", lambda *args: False)
    monkeypatch.setattr(capture, "prepare_source_launch", lambda: None)
    monkeypatch.setattr(render, "prepare_preview_environment", lambda *args, **kw: None)
    monkeypatch.setattr(render, "_build_og_config", lambda *args: {"env": {}})
    def observe(env, diagnostics, out_dir, **kwargs):
        assert list(glass.color) == pytest.approx([1., .13, .13])
        assert kwargs["source_profile"]["resolution"] == [640, 640]
        return diagnostics, {}
    monkeypatch.setattr(render, "render_views", observe)
    diagnostics = {"perturbation": {"kind": "target", "recolor": {
        "object": "jar_0", "diffuse_tint": [1., .13, .13]}}}
    assert render.render_task(tmp_path / "scene_ep1.json", diagnostics, tmp_path / "preview",
                              source_profile=selection) == diagnostics
