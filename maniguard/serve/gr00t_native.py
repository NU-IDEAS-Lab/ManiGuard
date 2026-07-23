#!/usr/bin/env python3
"""Serve a ManiGuard GR00T N1.6 SFT checkpoint over the openpi-client websocket contract.

Runs in the Isaac-GR00T venv. Wraps gr00t's ``Gr00tPolicy`` and speaks the SAME
websocket / msgpack-numpy protocol as ``maniguard.serve.openpi_native`` and openpi's
``WebsocketPolicyServer`` -- so ``maniguard.eval.benchmark`` connects with NO client
change (same ``base_0``/wrist/state/prompt contract, same ``{"actions": (H, A)}`` reply).

Action reconstruction (verified against gr00t source):
    GR00T N1.6 reconstructs ABSOLUTE joint targets INTERNALLY. At inference,
    ``Gr00tPolicy`` -> ``processor.decode_action(..., state=state)`` ->
    ``StateActionProcessor.unapply_action`` -> ``to_absolute_chunking`` computes
    ``absolute = current_state + relative_pred`` for the state-relative arm
    (``use_relative_action`` is baked in the checkpoint's processor); the gripper is
    absolute throughout. So this server passes the action through AS-IS and MUST send the
    robot's TRUE current absolute arm joints in ``observation/state`` (that IS the
    reconstruction reference). It must NEVER re-add state -- that would double-add.

Obs contract (from ``benchmark._remap_obs_for_openpi``): the client sends
    observation/image_left, observation/wrist_image, observation/state (8-D joint), prompt.
We repack to GR00T's nested dict (B=1, T=1): video.{image_left,wrist}, state.{single_arm(7),
gripper(1)} (MUST be split), language.<task_key>=[[prompt]].

Usage (in the gr00t venv; needs ``pip install websockets msgpack``):
    <gr00t-venv>/bin/python -m maniguard.serve.gr00t_native \
        --checkpoint /path/to/gr00t-checkpoint --device cuda:0 --port 8000
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import shutil
import traceback
from pathlib import Path

import msgpack
import numpy as np
import websockets.asyncio.server as ws_server
import websockets.frames
from websockets.exceptions import ConnectionClosed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- msgpack-numpy, vendored inline from openpi_client.msgpack_numpy -----------------
# Inlined (not imported from maniguard.serve) so this server is self-contained in the
# Isaac-GR00T venv, whose `maniguard` is the FORK's gr00t_sft-only package (no serve/).
# Wire-compatible with the client's openpi_client.msgpack_numpy.
def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)
# ------------------------------------------------------------------------------------

# GR00T's single language modality key (annotation.human.action.task_description).
_TASK_KEY = "annotation.human.action.task_description"


def _to_np(x) -> np.ndarray:
    """Model outputs may be torch tensors (possibly on GPU); coerce to numpy."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


class Gr00tWebsocketServer:
    """openpi-contract websocket server wrapping a gr00t Gr00tPolicy."""

    def __init__(self, policy, task_key: str, host: str, port: int):
        self._policy = policy
        self._task_key = task_key
        self._host = host
        self._port = port
        self._last_seed = None

    def _maybe_reseed(self, obs: dict) -> None:
        """Re-seed torch's RNG (the sampling-noise source) when the client starts
        a rollout with a new ``episode_seed``; constant within a rollout, so the
        RNG advances normally between infers. Absent key = unseeded behavior."""
        seed = obs.pop("episode_seed", None)
        if seed is not None and seed != self._last_seed:
            import torch
            torch.manual_seed(int(seed))
            self._last_seed = seed
            logger.info(f"sampling RNG re-seeded: episode_seed={seed}")

    def _infer(self, obs: dict) -> dict:
        self._maybe_reseed(obs)
        # openpi contract -> GR00T nested dict (batch B=1, time T=1 = current frame only).
        img_left = np.asarray(obs["observation/image_left"], dtype=np.uint8)   # (H, W, 3)
        wrist = np.asarray(obs["observation/wrist_image"], dtype=np.uint8)     # (H, W, 3)
        state = np.asarray(obs["observation/state"], dtype=np.float32).reshape(-1)  # (8,)
        prompt = obs.get("prompt", "")
        if isinstance(prompt, (bytes, bytearray)):
            prompt = prompt.decode()

        gobs = {
            "video": {
                "image_left": img_left[None, None],   # (1, 1, H, W, 3) uint8
                "wrist": wrist[None, None],
            },
            "state": {
                # 8-D joint -> split; single_arm is the ABSOLUTE reconstruction reference.
                "single_arm": state[None, None, :7],   # (1, 1, 7) float32
                "gripper": state[None, None, 7:8],     # (1, 1, 1) float32
            },
            "language": {self._task_key: [[str(prompt)]]},  # (B=1, T=1)
        }

        result = self._policy.get_action(gobs)
        action = result[0] if isinstance(result, tuple) else result
        arm = _to_np(action["single_arm"]).astype(np.float32)   # (1, 16, 7)
        grip = _to_np(action["gripper"]).astype(np.float32)     # (1, 16, 1)
        # (16, 8) ABSOLUTE joint target, ordered [arm(7), gripper(1)] to match the datagen
        # action layout + the joint_position_raw controller. Passed through, NOT re-added.
        chunk = np.concatenate([arm[0], grip[0]], axis=-1)
        return {"actions": chunk}

    async def _handler(self, websocket):
        packer = Packer()
        await websocket.send(packer.pack({}))  # metadata handshake (openpi protocol)
        while True:
            try:
                obs = unpackb(await websocket.recv())
                action = self._infer(obs)
                await websocket.send(packer.pack(action))
            except ConnectionClosed:
                logger.info("client disconnected")
                break
            except Exception:  # noqa: BLE001 - report to client like openpi's server
                tb = traceback.format_exc()
                logger.error(tb)
                await websocket.send(tb)
                await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason="server error")
                break

    async def _run(self):
        async with ws_server.serve(self._handler, self._host, self._port,
                                   compression=None, max_size=None):
            logger.info(f"GR00T policy server listening on {self._host}:{self._port}")
            await asyncio.get_running_loop().create_future()  # run forever

    def serve_forever(self):
        asyncio.run(self._run())


def _ensure_processor_at_root(checkpoint: str) -> None:
    """Gr00tPolicy loads the processor via ``AutoProcessor.from_pretrained(<checkpoint root>)``, but
    our SFT checkpoints save it into a ``processor/`` subdir. If the root lacks
    ``processor_config.json`` while the subdir has it, copy the processor files up to the root — so
    HF-pulled checkpoints load without re-pushing. Idempotent (no-op if already at root)."""
    ckpt = Path(checkpoint)
    if (ckpt / "processor_config.json").exists():
        return
    sub = ckpt / "processor"
    if not (sub / "processor_config.json").exists():
        return
    logger.info(f"processor files are in {sub}/ -> copying to checkpoint root for AutoProcessor")
    for f in sub.iterdir():
        if f.is_file() and not (ckpt / f.name).exists():
            shutil.copy2(f, ckpt / f.name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve a ManiGuard GR00T N1.6 checkpoint (openpi ws contract).")
    ap.add_argument("--checkpoint", required=True, help="local GR00T N1.6 SFT checkpoint dir")
    ap.add_argument("--device", default="cuda:0", help="torch device (e.g. cuda:0, 0, cpu)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    _ensure_processor_at_root(args.checkpoint)  # handle the processor/ subdir layout (see fn doc)

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    # NEW_EMBODIMENT modality config is baked into the checkpoint's processor at SFT time
    # (AutoProcessor.from_pretrained restores it), so no runtime registration is needed.
    logger.info(f"Loading GR00T policy from {args.checkpoint} (device={args.device}) ...")
    policy = Gr00tPolicy(EmbodimentTag.NEW_EMBODIMENT, args.checkpoint, device=args.device)
    task_key = policy.language_key  # the single language modality key from the processor
    logger.info(f"GR00T policy loaded. language_key={task_key!r}")

    Gr00tWebsocketServer(policy, task_key, args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()
