"""Small immutable provenance for a simulator recording process."""

import hashlib
import subprocess
from functools import lru_cache
from importlib import metadata
from pathlib import Path


@lru_cache(maxsize=1)
def runtime_provenance():
    package = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    versions = {}
    for name in ("maniguard", "omnigibson", "torch", "numpy", "h5py", "Pillow", "isaacsim"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=package, capture_output=True, text=True, check=False)
    return {
        "maniguard_python_sha256": digest.hexdigest(),
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "packages": versions,
    }
