#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


PANDA_BUNDLE = "franka_panda"
MOUNTED_BUNDLE = "franka_mounted"
LONGFINGER_BUNDLE = "franka_panda_longfinger"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Panda-compatible long-finger Franka asset bundle.")
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=None,
        help="Root for omnigibson-robot-assets. Defaults to the active dataset path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the existing long-finger bundle if it already exists.",
    )
    return parser.parse_args()


def get_asset_root(explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        return explicit_root.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root.parent / "ManiGuard-data" / "datasets" / "omnigibson-robot-assets",
        repo_root / "datasets" / "omnigibson-robot-assets",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not infer omnigibson-robot-assets root. Pass --asset-root explicitly."
    )


def block_range(lines: list[str], header: str, start_idx: int = 0) -> tuple[int, int]:
    start = next(i for i in range(start_idx, len(lines)) if header in lines[i])
    depth = 0
    saw_open = False
    for end in range(start, len(lines)):
        depth += lines[end].count("{")
        if "{" in lines[end]:
            saw_open = True
        depth -= lines[end].count("}")
        if saw_open and depth == 0:
            return start, end
    raise ValueError(f"Unterminated block for header: {header}")


def replace_named_xml_element(dst_parent: ET.Element, src_parent: ET.Element, tag: str, name: str) -> None:
    dst_children = list(dst_parent)
    dst_index = next(i for i, elem in enumerate(dst_children) if elem.tag == tag and elem.attrib.get("name") == name)
    src_elem = next(elem for elem in src_parent if elem.tag == tag and elem.attrib.get("name") == name)
    dst_parent.remove(dst_children[dst_index])
    dst_parent.insert(dst_index, copy.deepcopy(src_elem))


def patch_urdf(dst_urdf: Path, mounted_urdf: Path) -> None:
    dst_tree = ET.parse(dst_urdf)
    mounted_tree = ET.parse(mounted_urdf)
    dst_root = dst_tree.getroot()
    mounted_root = mounted_tree.getroot()

    for finger in ("panda_leftfinger", "panda_rightfinger"):
        replace_named_xml_element(dst_root, mounted_root, "link", finger)

    ET.indent(dst_tree, space="    ")
    dst_tree.write(dst_urdf, encoding="utf-8", xml_declaration=False)


def patch_curobo(dst_yaml: Path, mounted_yaml: Path) -> None:
    with dst_yaml.open("r", encoding="utf-8") as f:
        dst_cfg = yaml.safe_load(f)
    with mounted_yaml.open("r", encoding="utf-8") as f:
        mounted_cfg = yaml.safe_load(f)

    dst_spheres = dst_cfg["robot_cfg"]["kinematics"]["collision_spheres"]
    mounted_spheres = mounted_cfg["robot_cfg"]["kinematics"]["collision_spheres"]
    for finger in ("panda_leftfinger", "panda_rightfinger"):
        dst_spheres[finger] = mounted_spheres[finger]

    with dst_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dst_cfg, f, sort_keys=False)


def replace_xform_child_meshes(target_block: list[str], donor_block: list[str]) -> list[str]:
    target_child_start = next(i for i, line in enumerate(target_block) if line.lstrip().startswith("def Mesh "))
    donor_child_start = next(i for i, line in enumerate(donor_block) if line.lstrip().startswith("def Mesh "))
    donor_children = [
        line for line in donor_block[donor_child_start:-1] if "material:binding" not in line
    ]
    return target_block[:target_child_start] + donor_children + [target_block[-1]]


def patch_usda(dst_usda: Path, mounted_usda: Path) -> None:
    dst_lines = dst_usda.read_text(encoding="utf-8").splitlines()
    mounted_lines = mounted_usda.read_text(encoding="utf-8").splitlines()

    for finger in ("panda_leftfinger", "panda_rightfinger"):
        dst_start, dst_end = block_range(dst_lines, f'def Xform "{finger}"')
        mounted_start, mounted_end = block_range(mounted_lines, f'def Xform "{finger}"')

        dst_block = dst_lines[dst_start : dst_end + 1]
        mounted_block = mounted_lines[mounted_start : mounted_end + 1]
        dst_block = replace_xform_child_meshes(dst_block, mounted_block)
        dst_lines = dst_lines[:dst_start] + dst_block + dst_lines[dst_end + 1 :]

    dst_usda.write_text("\n".join(dst_lines) + "\n", encoding="utf-8")


def ensure_expected_contract(dst_urdf: Path, dst_yaml: Path, dst_usda: Path) -> None:
    urdf_text = dst_urdf.read_text(encoding="utf-8")
    assert 'mesh filename="meshes/visual/tri_finger.obj"' in urdf_text
    assert 'mesh filename="meshes/collision/panda_leftfinger-col-0.obj"' in urdf_text
    assert 'mesh filename="meshes/collision/panda_rightfinger-col-0.obj"' in urdf_text
    assert '<joint name="panda_finger_joint1" type="prismatic">' in urdf_text
    assert '<joint name="panda_finger_joint2" type="prismatic">' in urdf_text

    with dst_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    spheres = cfg["robot_cfg"]["kinematics"]["collision_spheres"]
    assert len(spheres["panda_leftfinger"]) == 7
    assert len(spheres["panda_rightfinger"]) == 7
    assert "panda_base" not in spheres
    assert cfg["robot_cfg"]["kinematics"]["ee_link"] == "eef_link"

    usda_text = dst_usda.read_text(encoding="utf-8")
    assert 'point3f physics:localPos0 = (0, 0, 0.105)' in usda_text
    assert 'point3f physics:localPos0 = (0, 0, 0.185)' not in usda_text


def build_longfinger_bundle(asset_root: Path, force: bool) -> Path:
    franka_root = asset_root / "models" / "franka"
    src_dir = franka_root / PANDA_BUNDLE
    mounted_dir = franka_root / MOUNTED_BUNDLE
    dst_dir = franka_root / LONGFINGER_BUNDLE

    if not src_dir.is_dir():
        raise FileNotFoundError(f"Missing Panda asset bundle: {src_dir}")
    if not mounted_dir.is_dir():
        raise FileNotFoundError(f"Missing mounted asset bundle: {mounted_dir}")

    if dst_dir.exists():
        if not force:
            raise FileExistsError(f"Long-finger bundle already exists: {dst_dir}. Re-run with --force to rebuild.")
        shutil.rmtree(dst_dir)

    shutil.copytree(src_dir, dst_dir)

    dst_urdf = dst_dir / "urdf" / f"{LONGFINGER_BUNDLE}.urdf"
    dst_usda = dst_dir / "usd" / f"{LONGFINGER_BUNDLE}.usda"
    dst_yaml = dst_dir / "curobo" / f"{LONGFINGER_BUNDLE}_description_curobo_default.yaml"

    shutil.move(dst_dir / "urdf" / f"{PANDA_BUNDLE}.urdf", dst_urdf)
    shutil.move(dst_dir / "usd" / f"{PANDA_BUNDLE}.usda", dst_usda)
    shutil.move(dst_dir / "curobo" / f"{PANDA_BUNDLE}_description_curobo_default.yaml", dst_yaml)

    patch_urdf(dst_urdf, mounted_dir / "urdf" / f"{MOUNTED_BUNDLE}.urdf")
    patch_curobo(dst_yaml, mounted_dir / "curobo" / f"{MOUNTED_BUNDLE}_description_curobo_default.yaml")
    patch_usda(dst_usda, mounted_dir / "usd" / f"{MOUNTED_BUNDLE}.usda")
    ensure_expected_contract(dst_urdf, dst_yaml, dst_usda)
    return dst_dir


def main() -> None:
    args = parse_args()
    asset_root = get_asset_root(args.asset_root)
    dst_dir = build_longfinger_bundle(asset_root=asset_root, force=args.force)
    print(f"Built Panda long-finger asset bundle at: {dst_dir}")


if __name__ == "__main__":
    main()
