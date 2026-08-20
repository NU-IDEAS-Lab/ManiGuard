"""Fall back LeRobot's video decode to PyAV when torchcodec can't load.

LeRobot stores datasets as MP4 video and decodes frames via
``lerobot.common.datasets.video_utils.decode_video_frames``. With no backend
requested it calls ``get_safe_default_codec()``, which returns ``"torchcodec"``
whenever the ``torchcodec`` *package* is importable -- but it never checks that
torchcodec's native library actually loads. ``torchcodec`` does NOT bundle
FFmpeg; it dlopens the *system* libavcodec/libavutil. So on a host where the
system FFmpeg is missing or incompatible, the default stays ``"torchcodec"`` and
decoding then fails -- ``Could not load libtorchcodec`` (no FFmpeg present) or
``Could not push packet to decoder: Function not implemented`` (an FFmpeg build
that can't decode the stream; LeRobot v2.1 encodes video as AV1, which minimal
FFmpeg builds omit).

``pyav`` is a first-class LeRobot backend whose manylinux wheel **bundles its own
FFmpeg** (including AV1), so it decodes with no system FFmpeg dependency at all.

NON-INVASIVE BY DESIGN: this patch first checks whether torchcodec is *actually*
usable on the current host (by importing its decoder, which dlopens the shared
library). If torchcodec loads, the patch does **nothing** -- LeRobot's default is
left exactly as-is, so any host where torchcodec already works stays
byte-for-byte unchanged. Only when torchcodec cannot load does it override
``get_safe_default_codec`` to return ``"pyav"`` (in both the ``video_utils`` and
``lerobot_dataset`` namespaces, since the latter imports the name for
``LeRobotDataset.__init__``'s default). The decision is host-local and made at
import time, so it cannot affect another machine running the same codebase.

Behaviour-preserving: where it engages, pyav decodes the same RGB frames as
torchcodec; only the decoder implementation changes.

Mirrors ``_augmax_patch`` -- ManiGuard fixes stay out of the upstream trees.
"""

from __future__ import annotations


def _torchcodec_loads() -> bool:
    """True iff torchcodec's native library actually loads on this host.

    Importing ``torchcodec.decoders`` triggers ``load_torchcodec_shared_libraries``
    (the dlopen of libtorchcodec against the system FFmpeg). A clean import means
    torchcodec is genuinely usable here; any exception means it is not.
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
