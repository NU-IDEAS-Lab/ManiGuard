#!/usr/bin/env python3
"""
Serve the GR00T Stack-Cube policy over websocket.
Uses the RLinf GR00T model loader with isaaclab_stack_cube obs/action converters.

Usage (from RLinf .venv-gr00t):
    python3 tools/serve_gr00t_websocket.py --checkpoint-dir RLinf-Gr00t-SFT-Stack-cube
    python3 tools/serve_gr00t_websocket.py --checkpoint-dir RLinf-Gr00t-SFT-Stack-cube --check-only
"""
from __future__ import annotations

import asyncio
import argparse
import http
import logging
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from omegaconf import OmegaConf
from openpi_client import msgpack_numpy
import websockets
import websockets.asyncio.server as _server


REPO_ROOT = Path(__file__).resolve().parents[2]
RLINF_ROOT = REPO_ROOT / "RLinf"
if str(RLINF_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve GR00T policy over websocket."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(REPO_ROOT / "RLinf-Gr00t-SFT-Stack-cube"),
        help="GR00T checkpoint directory.",
    )
    parser.add_argument(
        "--embodiment-tag",
        default="isaaclab_franka",
        help="GR00T embodiment tag.",
    )
    parser.add_argument(
        "--obs-converter",
        default="isaaclab_stack_cube",
        help="Obs converter type (libero, isaaclab_stack_cube, maniskill).",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Websocket bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Websocket bind port.")
    parser.add_argument(
        "--device", default="cuda", help="Torch device for inference.",
    )
    parser.add_argument(
        "--num-action-chunks", type=int, default=1,
        help="Number of action chunks to output.",
    )
    parser.add_argument(
        "--denoising-steps", type=int, default=4,
        help="Number of denoising steps for GR00T.",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Load the model and exit without serving.",
    )
    return parser.parse_args()


class GR00TAdapter:
    def __init__(
        self,
        checkpoint_dir: str,
        embodiment_tag: str,
        obs_converter: str,
        device: str,
        num_action_chunks: int = 1,
        denoising_steps: int = 4,
    ) -> None:
        from rlinf.models.embodiment.gr00t import get_model
        from rlinf.models.embodiment.gr00t.simulation_io import (
            ACTION_CONVERSION,
            OBS_CONVERSION,
        )

        cfg = OmegaConf.create({
            "model_path": checkpoint_dir,
            "embodiment_tag": embodiment_tag,
            "denoising_steps": denoising_steps,
            "num_action_chunks": num_action_chunks,
            "obs_converter_type": obs_converter,
            "rl_head_config": {
                "add_value_head": False,
                "disable_dropout": False,
            },
        })
        self.model = get_model(cfg)
        self.device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()
        self.num_action_chunks = num_action_chunks
        self.action_convert_fn = ACTION_CONVERSION[obs_converter]

    def metadata(self) -> dict:
        return {
            "policy_type": "gr00t_stack_cube",
            "device": str(self.device),
            "action_dim": 7,
            "action_chunk": self.num_action_chunks,
            "returns_full_chunk": True,
        }

    def _ensure_batch(self, obs: dict) -> dict:
        """Ensure obs tensors have batch dimension."""
        batched = dict(obs)
        for key in ("main_images", "wrist_images", "states"):
            value = batched.get(key)
            if value is None:
                continue
            value = np.asarray(value)
            if key == "states" and value.ndim == 1:
                value = value[None, :]
            if key != "states" and value.ndim == 3:
                value = value[None, ...]
            batched[key] = torch.from_numpy(value)
        # states must be float
        if "states" in batched and isinstance(batched["states"], torch.Tensor):
            batched["states"] = batched["states"].float()

        prompts = batched.get("task_descriptions")
        if isinstance(prompts, str):
            prompts = [prompts]
        elif prompts is None:
            prompts = [""]
        batched["task_descriptions"] = prompts
        return batched

    @torch.no_grad()
    def act(self, obs: dict) -> torch.Tensor:
        batched_obs = self._ensure_batch(obs)
        action_chunk, _ = self.model.predict_action_batch(
            batched_obs, mode="eval"
        )
        # action_chunk is a dict like {"action.x": ..., "action.gripper": ...}
        # Convert to flat 7D array
        action_array = self.action_convert_fn(
            action_chunk, chunk_size=self.num_action_chunks
        )
        # action_array shape: [batch, chunk, 7]
        action = np.asarray(action_array, dtype=np.float32)
        if action.ndim == 3:
            action = action[0]  # remove batch dim
        return torch.from_numpy(action)

    def reset(self) -> None:
        return None


class CompatibleWebsocketPolicyServer:
    def __init__(
        self,
        policy: GR00TAdapter,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        logger.info("Starting websocket server on %s:%s", self._host, self._port)
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        while True:
            try:
                start_time = time.monotonic()
                result = msgpack_numpy.unpackb(await websocket.recv())
                if isinstance(result, dict) and result.get("reset"):
                    self._policy.reset()
                    continue
                infer_time = time.monotonic()
                action = self._policy.act(result)
                infer_time = time.monotonic() - infer_time
                payload = {
                    "action": action.cpu().numpy(),
                    "server_timing": {"infer_ms": infer_time * 1000},
                }
                await websocket.send(packer.pack(payload))
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                logger.error(
                    "Error in connection from %s:\n%s",
                    websocket.remote_address,
                    traceback.format_exc(),
                )
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error.",
                )
                raise


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    adapter = GR00TAdapter(
        checkpoint_dir=args.checkpoint_dir,
        embodiment_tag=args.embodiment_tag,
        obs_converter=args.obs_converter,
        device=args.device,
        num_action_chunks=args.num_action_chunks,
        denoising_steps=args.denoising_steps,
    )
    if args.check_only:
        print(adapter.metadata())
        return

    server = CompatibleWebsocketPolicyServer(
        policy=adapter,
        host=args.host,
        port=args.port,
        metadata=adapter.metadata(),
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
