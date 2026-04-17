#!/usr/bin/env python3
"""Serve GR00T-N1.6 (native gr00t.policy.Gr00tPolicy) over websocket.

Unlike sentinel/serve/gr00t_server.py (which uses RLinf's N1.5 wrapper),
this server is written against the upstream Isaac-GR00T repo's API and
supports N1.6 checkpoints out of the box. Run from Isaac-GR00T/.venv:

    cd Isaac-GR00T
    source .venv/bin/activate
    python /path/to/sentinel/serve/gr00t_n16_server.py \
        --checkpoint-dir /path/to/GR00T-N1.6-DROID \
        --embodiment-tag oxe_droid \
        --host 0.0.0.0 --port 8000

Obs/action wire format (msgpack_numpy) is identical to other sentinel servers
so the benchmark client sees a uniform interface.
"""
from __future__ import annotations

import argparse
import asyncio
import http
import logging
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import websockets
import websockets.asyncio.server as _server


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sentinel.serve import _msgpack_numpy as msgpack_numpy  # noqa: E402


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GR00T-N1.6 websocket server.")
    parser.add_argument(
        "--checkpoint-dir", required=True,
        help="Path to a local GR00T N1.6 checkpoint (e.g. GR00T-N1.6-DROID).",
    )
    parser.add_argument(
        "--embodiment-tag", default="oxe_droid",
        help="Embodiment tag (e.g. oxe_droid, gr1). Must match checkpoint's "
             "processor_config.modality_configs keys.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--check-only", action="store_true",
        help="Load the model + metadata and exit without serving.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class N16Adapter:
    """Adapts benchmark obs → DROID modality dict; exposes .act() + .metadata()."""

    def __init__(self, checkpoint_dir: str, embodiment_tag: str, device: str) -> None:
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        tag_enum = EmbodimentTag(embodiment_tag)
        self.embodiment_tag = embodiment_tag
        self.policy = Gr00tPolicy(
            embodiment_tag=tag_enum,
            model_path=checkpoint_dir,
            device=device if (device != "cuda" or torch.cuda.is_available()) else "cpu",
        )
        self.device = torch.device(
            device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        )

        # Introspect modality config so we know which keys to construct.
        mcfg = self.policy.get_modality_config()
        self._video_keys = list(mcfg["video"].modality_keys)        # e.g. ["exterior_image_1_left","wrist_image_left"]
        self._state_keys = list(mcfg["state"].modality_keys)        # e.g. ["joint_position","gripper_position"]
        self._language_keys = list(mcfg["language"].modality_keys)  # e.g. ["annotation.language.language_instruction"]
        self._action_keys = list(mcfg["action"].modality_keys)      # e.g. ["joint_position","gripper_position"]
        logger.info("Modality — video=%s state=%s language=%s action=%s",
                    self._video_keys, self._state_keys, self._language_keys, self._action_keys)

    def metadata(self) -> dict:
        return {
            "policy_type": f"gr00t_n16_{self.embodiment_tag}",
            "device": str(self.device),
            "action_dim": 8,       # 7 joints + 1 gripper; set by benchmark profile
            "action_chunk": 32,    # DROID default horizon; client truncates via execute_horizon
            "returns_full_chunk": True,
        }

    # ----- obs construction -----

    @staticmethod
    def _resize_video_to_256(video: np.ndarray, size: int = 256) -> np.ndarray:
        """[B, H, W, C] uint8 → [B, size, size, C] uint8 via bilinear resize."""
        if video.ndim != 4:
            return video
        _, h, w, _ = video.shape
        if h == size and w == size:
            return video
        t = torch.from_numpy(np.array(video, copy=True)).permute(0, 3, 1, 2).float()
        t = torch.nn.functional.interpolate(
            t, size=(size, size), mode="bilinear", align_corners=False
        )
        return t.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).numpy()

    def _build_gr00t_obs(self, env_obs: dict) -> dict:
        """benchmark.py obs → DROID modality dict.

        benchmark.py sends:
          main_images [H,W,3] uint8
          wrist_images [H,W,3] uint8
          states (numpy, 8D for state_mode='joint': 7 arm + 1 gripper)
          task_descriptions (str)
        """
        out: dict = {}

        # Video: add a time axis (T=1). gr00t expects [T, H, W, C] uint8.
        main = np.asarray(env_obs["main_images"])
        if main.ndim == 3:
            main = main[None, ...]     # [T=1, H, W, 3]
        main = self._resize_video_to_256(main)
        wrist = np.asarray(env_obs.get("wrist_images"))
        if wrist is None or wrist.size == 0:
            wrist = np.zeros_like(main)
        elif wrist.ndim == 3:
            wrist = wrist[None, ...]
        wrist = self._resize_video_to_256(wrist)

        # Map to whatever the checkpoint's modality_config.video.modality_keys are.
        # DROID expects ["exterior_image_1_left", "wrist_image_left"]; if only 1
        # camera key exists, use main only.
        if len(self._video_keys) >= 1:
            out[f"video.{self._video_keys[0]}"] = main
        if len(self._video_keys) >= 2:
            out[f"video.{self._video_keys[1]}"] = wrist

        # State: slice 8D into 7 arm + 1 gripper based on modality_keys order.
        # DROID: state.joint_position (7), state.gripper_position (1).
        state = np.asarray(env_obs["states"], dtype=np.float32)
        if state.ndim == 1:
            state = state[None, :]   # [T=1, D]
        # Heuristic split: 7D arm, 1D gripper scalar.
        if len(self._state_keys) == 2:
            out[f"state.{self._state_keys[0]}"] = state[:, :7]
            out[f"state.{self._state_keys[1]}"] = state[:, 7:8]
        else:
            # Fallback: if the embodiment uses a single state key, send full state.
            out[f"state.{self._state_keys[0]}"] = state

        # Language: expand str -> [T=1] array of str (gr00t handles str arrays).
        prompt = env_obs.get("task_descriptions") or ""
        if isinstance(prompt, (list, tuple)):
            prompt = prompt[0] if prompt else ""
        for lk in self._language_keys:
            out[lk] = np.array([str(prompt)])

        return out

    # ----- action extraction -----

    def _flatten_action_chunk(self, action: dict) -> np.ndarray:
        """Action dict from gr00t → [chunk, action_dim] np.float32.

        For DROID: concat action.joint_position [chunk, 7] + action.gripper_position [chunk, 1] = [chunk, 8].
        """
        parts = []
        for key in self._action_keys:
            arr = np.asarray(action[f"action.{key}"], dtype=np.float32)
            # Expected shape [chunk, D]; some keys return [chunk] for 1-D scalars.
            if arr.ndim == 1:
                arr = arr[:, None]
            parts.append(arr)
        return np.concatenate(parts, axis=-1)

    # ----- main entry -----

    @torch.no_grad()
    def act(self, env_obs: dict) -> torch.Tensor:
        gr00t_obs = self._build_gr00t_obs(env_obs)
        action = self.policy.get_action(gr00t_obs)
        chunk = self._flatten_action_chunk(action)
        return torch.from_numpy(chunk)

    def reset(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Websocket server (identical to sentinel/serve/gr00t_server.py)
# ---------------------------------------------------------------------------

class CompatibleWebsocketPolicyServer:
    def __init__(self, policy: N16Adapter, host: str, port: int,
                 metadata: dict | None = None) -> None:
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
            self._handler, self._host, self._port,
            compression=None, max_size=None, process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        while True:
            try:
                result = msgpack_numpy.unpackb(await websocket.recv())
                if isinstance(result, dict) and result.get("reset"):
                    self._policy.reset()
                    continue
                t0 = time.monotonic()
                action = self._policy.act(result)
                infer_ms = (time.monotonic() - t0) * 1000
                payload = {
                    "action": action.cpu().numpy(),
                    "server_timing": {"infer_ms": infer_ms},
                }
                await websocket.send(packer.pack(payload))
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                logger.error(
                    "Error in connection from %s:\n%s",
                    websocket.remote_address, traceback.format_exc(),
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
    adapter = N16Adapter(
        checkpoint_dir=args.checkpoint_dir,
        embodiment_tag=args.embodiment_tag,
        device=args.device,
    )
    if args.check_only:
        print(adapter.metadata())
        return

    server = CompatibleWebsocketPolicyServer(
        policy=adapter, host=args.host, port=args.port,
        metadata=adapter.metadata(),
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
