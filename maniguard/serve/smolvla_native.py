#!/usr/bin/env python3
"""Serve a ManiGuard SmolVLA SFT checkpoint over the openpi-client websocket contract.

Runs in the lerobot-maniguard-sft venv. Wraps lerobot's ``SmolVLAPolicy`` behind the
SAME websocket / msgpack-numpy protocol as ``maniguard.serve.openpi_native`` and
``maniguard/serve/gr00t_native.py`` -- so ``maniguard.eval.benchmark`` connects with
NO client change (same image_left/wrist/state/prompt contract, same
``{"actions": (H, A)}`` reply).

Inference path (the fork's canonical stack, lerobot v0.5.x):
    prepare_observation_for_inference -> saved preprocessor pipeline
    (rename top/wrist->camera1/2, tokenize, to-device, normalize with the
    dataset stats baked at SFT time) -> ``policy.predict_action_chunk`` ->
    saved postprocessor (unnormalize action, to cpu).

Action contract: the checkpoint was trained on NATIVE FULLY-ABSOLUTE 8-D joint
targets [arm_q(7), gripper(1)] (no delta), so the chunk is passed through AS-IS.
``_get_action_chunk`` already slices the padded 32-D model output back to 8-D.

Obs contract (from ``benchmark._remap_obs_for_openpi``): the client sends
    observation/image_left, observation/wrist_image, observation/state (8-D), prompt.
We repack to the fork's SFT keys (maniguard_sft.embodiment): the overview ->
``observation.images.top`` and wrist -> ``observation.images.wrist``; the saved
rename step maps them onto the checkpoint's camera1/camera2 inputs. camera3 (a
base-model leftover in input_features) stays absent -- SmolVLA masks missing
cameras, exactly as during SFT.

Usage (in the lerobot fork venv; needs ``uv pip install websockets``):
    <fork>/.venv/bin/python maniguard/serve/smolvla_native.py \
        --checkpoint /path/to/smolvla-checkpoint --device cuda:0 --port 8000
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import traceback

import msgpack
import numpy as np
import websockets.asyncio.server as ws_server
import websockets.frames
from websockets.exceptions import ConnectionClosed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- msgpack-numpy, vendored inline from openpi_client.msgpack_numpy -----------------
# Inlined (not imported from maniguard.serve) so this server is self-contained in the
# lerobot venv, which has no ManiGuard `maniguard` package. Wire-compatible with the
# client's openpi_client.msgpack_numpy.
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

# The fork's SFT observation keys (maniguard_sft.embodiment OVERVIEW_KEY / WRIST_KEY /
# STATE_KEY); the checkpoint's saved rename step maps top/wrist -> camera1/camera2.
_OVERVIEW_KEY = "observation.images.top"
_WRIST_KEY = "observation.images.wrist"
_STATE_KEY = "observation.state"


class SmolVLAWebsocketServer:
    """openpi-contract websocket server wrapping a lerobot SmolVLAPolicy."""

    def __init__(self, policy, preprocessor, postprocessor, device, host: str, port: int):
        self._policy = policy
        self._pre = preprocessor
        self._post = postprocessor
        self._device = device
        self._host = host
        self._port = port
        self._last_seed = None

    def _maybe_reseed(self, obs: dict) -> None:
        """Re-seed torch's RNG (the flow-sampling noise source) when the client
        starts a rollout with a new ``episode_seed``; constant within a rollout,
        so the RNG advances normally between infers. Absent key = unseeded."""
        seed = obs.pop("episode_seed", None)
        if seed is not None and seed != self._last_seed:
            import torch
            torch.manual_seed(int(seed))
            self._last_seed = seed
            logger.info(f"sampling RNG re-seeded: episode_seed={seed}")

    def _infer(self, obs: dict) -> dict:
        self._maybe_reseed(obs)
        from lerobot.policies.utils import prepare_observation_for_inference

        img = np.ascontiguousarray(obs["observation/image_left"], dtype=np.uint8)   # (H, W, 3)
        wrist = np.ascontiguousarray(obs["observation/wrist_image"], dtype=np.uint8)
        state = np.asarray(obs["observation/state"], dtype=np.float32).reshape(-1)   # (8,)
        prompt = obs.get("prompt", "")
        if isinstance(prompt, (bytes, bytearray)):
            prompt = prompt.decode()

        obs_np = {_OVERVIEW_KEY: img, _WRIST_KEY: wrist, _STATE_KEY: state}
        batch = prepare_observation_for_inference(obs_np, self._device, task=str(prompt))
        batch = self._pre(batch)
        chunk = self._policy.predict_action_chunk(batch)   # (1, chunk_size, 8) absolute
        chunk = self._post(chunk)                          # unnormalize + cpu
        actions = np.asarray(chunk[0].numpy(), dtype=np.float32)   # (chunk_size, 8)
        return {"actions": actions}

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
            logger.info(f"SmolVLA policy server listening on {self._host}:{self._port}")
            await asyncio.get_running_loop().create_future()  # run forever

    def serve_forever(self):
        asyncio.run(self._run())


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve a ManiGuard SmolVLA checkpoint (openpi ws contract).")
    ap.add_argument("--checkpoint", required=True, help="local SmolVLA SFT checkpoint dir (HF snapshot)")
    ap.add_argument("--device", default="cuda:0", help="torch device (e.g. cuda:0, cpu)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    device = torch.device(args.device)
    logger.info(f"Loading SmolVLA policy from {args.checkpoint} (device={device}) ...")
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to(device).eval()
    policy.reset()
    # Saved pipelines carry the SFT-time rename + tokenizer + dataset norm stats;
    # only the device step is overridden to the requested device.
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    logger.info(f"SmolVLA policy loaded. chunk_size={cfg.chunk_size}, "
                f"action_dim={cfg.output_features['action'].shape[0]}")

    SmolVLAWebsocketServer(policy, pre, post, device, args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()
