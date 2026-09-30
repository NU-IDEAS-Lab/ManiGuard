"""Select PyAV for supported LeRobot APIs when torchcodec's decoder import fails.

Probe torchcodec.decoders at import time. If it fails, replace
get_safe_default_codec in the supported lerobot.common.datasets modules with a
PyAV default. A successful decoder import leaves the default unchanged; it does
not verify that every dataset codec is supported or compare decoded pixels.

Raise when a supported LeRobot module imports but exposes no codec-selection
hook. Missing modules are skipped.
"""

from __future__ import annotations


def _torchcodec_loads() -> bool:
    """Return whether torchcodec.decoders imports successfully on this host. This checks library loading, not decoding support for a particular video.
    """
    import importlib

    try:
        importlib.import_module("torchcodec.decoders")
        return True
    except Exception:
        return False


def apply() -> None:
    """Force LeRobot's default video backend to ``pyav`` only if torchcodec can't load."""
    # Where torchcodec loads, leave LeRobot's default untouched -- this is what
    # keeps any host with a working torchcodec byte-for-byte unchanged.
    if _torchcodec_loads():
        return

    import importlib

    def _pyav_default() -> str:
        return "pyav"

    _pyav_default._maniguard_pyav = True  # type: ignore[attr-defined]

    saw_lerobot = False
    patched = False
    # Patch the name wherever LeRobot exposes it: video_utils defines it, and
    # lerobot_dataset imports it for LeRobotDataset.__init__'s default. Setting
    # both covers every binding without depending on which one a given call uses.
    for mod_name in (
        "lerobot.common.datasets.video_utils",
        "lerobot.common.datasets.lerobot_dataset",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        saw_lerobot = True
        fn = getattr(mod, "get_safe_default_codec", None)
        if fn is None:
            continue
        if not getattr(fn, "_maniguard_pyav", False):
            mod.get_safe_default_codec = _pyav_default
        patched = True

    # Tripwire: torchcodec is unusable AND LeRobot's hook is gone -> fail loud
    # rather than let the decode die later with a more confusing torchcodec error.
    if saw_lerobot and not patched:
        raise RuntimeError(
            "lerobot.get_safe_default_codec not found -- LeRobot video API changed; "
            "update maniguard.openpi_sft._lerobot_video_patch."
        )
