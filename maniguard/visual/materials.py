"""Reversible appearance-only overrides for realtime RGB source capture."""


def is_omniglass_shader(asset_path, sub_identifier):
    """Use the MDL shader identity, never a material/object display name."""
    return str(asset_path).replace("\\", "/").split("/")[-1] == "OmniGlass.mdl" and sub_identifier == "OmniGlass"


class GlassShadowOverride:
    """Exclude pure OmniGlass meshes from shadow casting in the existing session layer.

    This improves visibility through glass in RTX realtime but approximates glass
    shadows. RTX segmentation can also change; this profile records RGB only.
    A mesh with mixed material subsets is left untouched.
    """

    def __init__(self, stage):
        from pxr import Sdf, Usd, UsdGeom, UsdShade

        self.stage = stage
        self.layer = stage.GetSessionLayer()
        self.backup = Sdf.Layer.CreateAnonymous("maniguard_glass_property_backup.usda")
        self.saved_properties = []
        self.report = []
        targets = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            api = UsdShade.MaterialBindingAPI(prim)
            # Do not suppress opaque subsets on a mixed glass/opaque mesh.
            if api.GetMaterialBindSubsets():
                continue
            material, _ = api.ComputeBoundMaterial()
            if not material:
                continue
            shader, _, _ = material.ComputeSurfaceSource("mdl")
            if not shader:
                continue
            asset = shader.GetSourceAsset("mdl")
            sub_id = shader.GetSourceAssetSubIdentifier("mdl")
            if not asset or not is_omniglass_shader(asset.path, sub_id):
                continue
            attr = prim.GetAttribute("primvars:doNotCastShadows")
            self.report.append(
                {
                    "mesh": str(prim.GetPath()),
                    "material": str(material.GetPath()),
                    "shader": str(shader.GetPath()),
                    "mdl_asset": asset.path,
                    "mdl_sub_identifier": sub_id,
                    "previous_value": attr.Get() if attr else None,
                    "value": True,
                }
            )
            targets.append(prim)
        # Do not insert/remove sublayers while PhysX tensor views are live: USD
        # resync can invalidate them. Edit only the visual primvar in place.
        for prim in targets:
            path = prim.GetPath().AppendProperty("primvars:doNotCastShadows")
            exists = bool(self.layer.GetPropertyAtPath(path))
            if exists:
                Sdf.CreatePrimInLayer(self.backup, prim.GetPath())
                Sdf.CopySpec(self.layer, path, self.backup, path)
            self.saved_properties.append((path, exists))
        try:
            with Usd.EditContext(stage, self.layer):
                for prim in targets:
                    UsdGeom.PrimvarsAPI(prim).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
            for prim in targets:
                if prim.GetAttribute("primvars:doNotCastShadows").Get() is not True:
                    raise RuntimeError(f"Glass shadow override was rejected: {prim.GetPath()}")
        except Exception:
            self.close()
            raise

    def close(self):
        """Reveal the original authored values without editing their layers."""
        from pxr import Sdf, Usd

        with Usd.EditContext(self.stage, self.layer):
            for path, existed in self.saved_properties:
                if existed:
                    Sdf.CopySpec(self.backup, path, self.layer, path)
                else:
                    self.stage.GetPrimAtPath(path.GetPrimPath()).RemoveProperty(path.name)
        self.saved_properties.clear()
