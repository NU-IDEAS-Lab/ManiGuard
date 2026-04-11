#!/usr/bin/env python3
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


REPO_ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = REPO_ROOT / "RLinf"
if str(RLINF_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.models.embodiment.openpi import get_model


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the local Pi0.5 Franka-tabletop policy over websocket."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(REPO_ROOT / "checkpoints" / "RLinf-Pi05-LIBERO-SFT"),
        help="Converted local Pi0.5 checkpoint root.",
    )
    parser.add_argument(
        "--config-name",
        default="pi05_libero",
        help="RLinf openpi config name.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Websocket bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Websocket bind port.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device used for inference. Defaults to cuda when available.",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=7,
        help="Action dimension. 7 for LIBERO-style delta EEF, 8 for absolute joint.",
    )
    parser.add_argument(
        "--action-index",
        type=int,
        default=None,
        help="Optional chunk index to return from the model output. Defaults to returning the full chunk.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Load the model and exit without serving.",
    )
    return parser.parse_args()


class RLinfOpenPiPolicyAdapter:
    def __init__(
        self,
        checkpoint_dir: str,
        config_name: str,
        device: str,
        action_index: int,
        action_dim: int = 7,
    ) -> None:
        cfg = OmegaConf.create(
            {
                "model_path": checkpoint_dir,
                "openpi": {
                    "config_name": config_name,
                    "num_images_in_input": 2,
                    "noise_level": 0.5,
                    "action_chunk": 10,
                    "num_steps": 5,
                    "train_expert_only": True,
                    "action_env_dim": action_dim,
                    "noise_method": "flow_sde",
                    "add_value_head": False,
                    "value_after_vlm": False,
                    "value_vlm_mode": "mean_token",
                    "detach_critic_input": None,
                },
            }
        )
        self.model = get_model(cfg)
        self.device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()
        self.action_index = action_index

    def metadata(self) -> dict:
        return {
            "policy_type": "pi05_franka_tabletop",
            "device": str(self.device),
            "action_dim": int(self.model.config.action_env_dim),
            "action_chunk": int(self.model.config.action_chunk),
            "returns_full_chunk": self.action_index is None,
        }

    def _ensure_batch(self, obs: dict) -> dict:
        # Keep env keys as-is — obs_processor in openpi_action_model handles the mapping
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
            batched[key] = value

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
        actions, _ = self.model.predict_action_batch(
            batched_obs, mode="eval", compute_values=False
        )
        actions = np.asarray(actions, dtype=np.float32)
        if self.action_index is None:
            chunk_actions = actions[0] if actions.shape[0] == 1 else actions
            return torch.from_numpy(chunk_actions)

        step_actions = actions[:, self.action_index]
        return torch.from_numpy(step_actions[0] if step_actions.shape[0] == 1 else step_actions)

    def reset(self) -> None:
        return None


class CompatibleWebsocketPolicyServer:
    def __init__(
        self,
        policy: RLinfOpenPiPolicyAdapter,
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
        prev_total_time = None
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
                if prev_total_time is not None:
                    payload["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(payload))
                prev_total_time = time.monotonic() - start_time

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
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def main() -> None:
    args = parse_args()
    adapter = RLinfOpenPiPolicyAdapter(
        checkpoint_dir=args.checkpoint_dir,
        config_name=args.config_name,
        device=args.device,
        action_index=args.action_index,
        action_dim=args.action_dim,
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
