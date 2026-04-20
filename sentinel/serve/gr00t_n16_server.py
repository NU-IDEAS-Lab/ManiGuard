#!/usr/bin/env python3
"""Serve GR00T-N1.6 (native gr00t.policy.Gr00tPolicy) over websocket.

Unlike sentinel/serve/gr00t_server.py (which uses RLinf's N1.5 wrapper),
this server is written against the upstream Isaac-GR00T repo's API and
supports N1.6 checkpoints out of the box. Run from vla_models/Isaac-GR00T/.venv:

    cd vla_models/Isaac-GR00T
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
    def _resize_video_to_256_5d(video: np.ndarray, size: int = 256) -> np.ndarray:
        """[B, T, H, W, C] uint8 → [B, T, size, size, C] uint8 via bilinear resize."""
        if video.ndim != 5:
            return video
        b, t_dim, h, w, c = video.shape
        if h == size and w == size:
            return video
        # Fold B and T together for interpolation, then restore.
        flat = torch.from_numpy(np.array(video, copy=True)).reshape(b * t_dim, h, w, c)
        flat = flat.permute(0, 3, 1, 2).float()
        flat = torch.nn.functional.interpolate(
            flat, size=(size, size), mode="bilinear", align_corners=False
        )
        out = flat.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)
        return out.reshape(b, t_dim, size, size, c).numpy()

    def _build_gr00t_obs(self, env_obs: dict) -> dict:
        """benchmark.py obs -> nested DROID modality dict.

        Gr00tPolicy.check_observation expects (see gr00t/policy/gr00t_policy.py
        :check_observation docstring):
          {
            "video":    {<key>: np.uint8  shape (B, T, H, W, C)},
            "state":    {<key>: np.float32 shape (B, T, D)},
            "language": {<key>: list[list[str]] shape (B, T)},
          }

        benchmark.py sends per step (B=T=1):
          main_images  [H, W, 3] uint8
          wrist_images [H, W, 3] uint8
          states       8D float32 (7 arm + 1 gripper for state_mode='joint')
          task_descriptions str
        """
        # ---------- Video ----------
        def _as_5d(img: np.ndarray) -> np.ndarray:
            img = np.asarray(img)
            if img.ndim == 3:
                img = img[None, None, ...]          # [B=1, T=1, H, W, C]
            elif img.ndim == 4:
                img = img[None, ...]                # add batch dim
            return self._resize_video_to_256_5d(img).astype(np.uint8, copy=False)

        main = _as_5d(env_obs["main_images"])
        wrist_raw = env_obs.get("wrist_images")
        wrist = _as_5d(wrist_raw) if (wrist_raw is not None and np.asarray(wrist_raw).size) else np.zeros_like(main)

        video_dict: dict = {}
        if len(self._video_keys) >= 1:
            video_dict[self._video_keys[0]] = main
        if len(self._video_keys) >= 2:
            video_dict[self._video_keys[1]] = wrist

        # ---------- State ----------
        state = np.asarray(env_obs["states"], dtype=np.float32)
        if state.ndim == 1:
            state = state[None, None, :]            # [B=1, T=1, D]
        elif state.ndim == 2:
            state = state[None, ...]                # add batch dim

        state_dict: dict = {}
        if len(self._state_keys) == 2:
            # DROID state: joint_position (7, radians) + gripper_position (1, [0,1]).
            # benchmark sends gripper as averaged finger qpos in meters ([0, 0.04]
            # for Franka). Scale to DROID's [0, 1] convention so the state lands
            # in the training distribution (mean=0.406, std=0.4).
            state_dict[self._state_keys[0]] = state[..., :7]
            grip = state[..., 7:8] / 0.04
            state_dict[self._state_keys[1]] = np.clip(grip, 0.0, 1.0).astype(np.float32)
        else:
            state_dict[self._state_keys[0]] = state

        # ---------- Language ----------
        prompt = env_obs.get("task_descriptions") or ""
        if isinstance(prompt, (list, tuple)):
            prompt = prompt[0] if prompt else ""
        lang_dict = {lk: [[str(prompt)]] for lk in self._language_keys}  # [B=1, T=1]

        return {"video": video_dict, "state": state_dict, "language": lang_dict}

    # ----- action extraction -----

    def _flatten_action_chunk(self, action: dict) -> np.ndarray:
        """Action dict from Gr00tPolicy.get_action -> [chunk, action_dim] float32.

        Raw output keys are bare (no 'action.' prefix) with shape [B, chunk, D];
        see scripts/deployment/standalone_inference_script.py:parse_action_gr00t.
        For DROID: joint_position [1, 32, 7] + gripper_position [1, 32, 1]
        -> concat -> [32, 8].
        """
        parts = []
        for key in self._action_keys:
            arr = np.asarray(action[key], dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[0]                # drop batch -> [chunk, D]
            if arr.ndim == 1:
                arr = arr[:, None]          # scalar key -> [chunk, 1]
            parts.append(arr)
        return np.concatenate(parts, axis=-1)

    # ----- main entry -----

    @torch.no_grad()
    def act(self, env_obs: dict) -> torch.Tensor:
        gr00t_obs = self._build_gr00t_obs(env_obs)
        # Gr00tPolicy.get_action returns (action_dict, info). We only need action.
        action, _ = self.policy.get_action(gr00t_obs)
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
