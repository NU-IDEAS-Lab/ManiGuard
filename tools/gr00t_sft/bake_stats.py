#!/usr/bin/env python
"""Compute GR00T stats for each family ONCE, offline (CPU-only, no model/GPU), from the
finalized LeRobot datasets, and copy them into the fork's gr00t_stats/<fam>/. Idempotent.

    python tools/gr00t_sft/bake_stats.py --data-root <root> [--family jar]

<root> may be either a HF-cache-style root (containing
``IDEAS-Lab-Northwestern/datagen-<fam>-v1-joint-5cam``) or the local master layout
(containing ``<fam>_pickup`` / ``<fam>_transport`` etc.). Only the parquet state/action
columns are read (identical across GOP versions), so any local copy of the data works.
"""
import argparse
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_CFG = _REPO / "maniguard" / "gr00t_sft" / "maniguard_embodiment.py"
FAM_DIR = {
    "clutter": "clutter_pickup", "cabinet": "cabinet_pickup", "stack": "stack_retrieve",
    "jar": "jar_transport", "lid": "lid_transport", "dusty": "dusty_transfer",
}


def _load_embodiment():
    spec = importlib.util.spec_from_file_location("maniguard_embodiment", _CFG)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load embodiment config from {_CFG}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # registers NEW_EMBODIMENT
    return m


def _resolve_src(fam: str, data_root: Path) -> Path:
    for cand in (
        data_root / f"IDEAS-Lab-Northwestern/datagen-{fam}-v1-joint-5cam",
        data_root / FAM_DIR[fam],
        data_root / f"datagen-{fam}-v1-joint-5cam",
    ):
        if (cand / "meta" / "info.json").is_file():
            return cand
    raise FileNotFoundError(f"no dataset for {fam} under {data_root}")


def bake(fam: str, data_root: Path, mod) -> None:
    src = _resolve_src(fam, data_root)
    # gr00t.data.stats writes into <dataset>/meta/. Work on a temp dir with a REAL meta/
    # (copy) + a symlinked data/ so the source's meta is never touched.
    with tempfile.TemporaryDirectory() as td:
        work = Path(td) / fam
        (work / "meta").mkdir(parents=True)
        for f in (src / "meta").glob("*"):
            if f.is_file():
                shutil.copy2(f, work / "meta" / f.name)
        (work / "data").symlink_to((src / "data").resolve(), target_is_directory=True)
        (work / "meta" / "modality.json").write_text(json.dumps(mod.MODALITY_JSON, indent=4) + "\n")

        from gr00t.data.stats import main as stats_main
        from gr00t.data.types import EmbodimentTag

        stats_main(str(work), EmbodimentTag.NEW_EMBODIMENT)  # CPU-only

        dst = _REPO / "gr00t_stats" / fam
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(work / "meta" / "stats.json", dst / "stats.json")
        rel = work / "meta" / "relative_stats.json"
        if not rel.is_file():
            raise RuntimeError(f"{fam}: relative_stats.json missing (RELATIVE arm requires it)")
        shutil.copy2(rel, dst / "relative_stats.json")
    print(f"[bake] {fam} -> gr00t_stats/{fam}/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--family", default=None, help="one family, else all 6")
    a = ap.parse_args()
    mod = _load_embodiment()
    fams = [a.family] if a.family else list(FAM_DIR)
    for fam in fams:
        bake(fam, Path(a.data_root), mod)


if __name__ == "__main__":
    main()
