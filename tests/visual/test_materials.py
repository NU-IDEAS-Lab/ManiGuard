"""USD-only checks, also runnable inside the simulator's Python environment."""

import pytest


def test_glass_override_preserves_authored_values_and_other_materials():
    pytest.importorskip("pxr.Usd")
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    from maniguard.visual.materials import GlassShadowOverride

    stage = Usd.Stage.CreateInMemory()
    glass = UsdShade.Material.Define(stage, "/Glass")
    shader = UsdShade.Shader.Define(stage, "/Glass/Shader")
    shader.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    glass.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    for name in ("Authored", "Unauthored", "Mixed"):
        mesh = UsdGeom.Mesh.Define(stage, "/" + name)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(glass)
    UsdGeom.Mesh.Define(stage, "/Opaque")
    mixed = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/Mixed"))
    mixed.CreateMaterialBindSubset("opaque_faces", [0])
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/Authored")).CreatePrimvar(
            "doNotCastShadows", Sdf.ValueTypeNames.Bool
        ).Set(False)
    root = stage.GetRootLayer().ExportToString()
    original_layers = list(stage.GetSessionLayer().subLayerPaths)
    override = GlassShadowOverride(stage)
    assert {row["mesh"] for row in override.report} == {"/Authored", "/Unauthored"}
    assert list(stage.GetSessionLayer().subLayerPaths) == original_layers
    assert stage.GetPrimAtPath("/Authored").GetAttribute("primvars:doNotCastShadows").Get() is True
    assert not stage.GetPrimAtPath("/Opaque").GetAttribute("primvars:doNotCastShadows")
    assert not stage.GetPrimAtPath("/Mixed").GetAttribute("primvars:doNotCastShadows")
    override.close()
    override.close()
    assert stage.GetPrimAtPath("/Authored").GetAttribute("primvars:doNotCastShadows").Get() is False
    assert not stage.GetPrimAtPath("/Unauthored").GetAttribute("primvars:doNotCastShadows")
    assert stage.GetRootLayer().ExportToString() == root
