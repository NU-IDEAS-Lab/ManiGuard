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

Obs contract -- TWO client schemas, selected by ``--real``:
    sim (default, ``benchmark._remap_obs_for_openpi``):
        observation/image_left, observation/wrist_image, observation/state (8-D joint)
    real (``--real``, the DROID-schema real-robot client):
        observation/exterior_image_1_left, observation/wrist_image_left,
        observation/joint_position (7,), observation/gripper_position (1,)
Both are repacked to the SAME GR00T nested dict (B=1, T=1): video.{image_left,wrist},
state.{single_arm(7), gripper(1)} (MUST be split), language.<task_key>=[[prompt]] -- the
real embodiment config deliberately keeps sim's modality KEY names and differs only in
``original_key``, so nothing downstream of this unpacking branches on the mode.

``--real`` changes ONLY which observation keys are read. It does NOT change how actions are
interpreted: the action representation is baked into the checkpoint's processor at SFT time
(sim = state-relative arm, reconstructed to absolute internally; real = absolute, i.e. the
stored joint VELOCITY passed through). Serving a real checkpoint therefore also means the
returned chunk is joint velocity rad/s, which the real client applies as ``delta = action/15``
with NO clip.

⚠️ ``--real`` NEVER falls back to the sim keys. A silent fallback would produce a
plausible-looking rollout built from the wrong inputs, which is unrecoverable after the fact.
A missing DROID key raises, and the first assembled state is range-checked against the Franka
joint limits (a swapped concat order is otherwise invisible).

Usage (in the gr00t venv; needs ``pip install websockets msgpack``):
    <gr00t-venv>/bin/python -m maniguard.serve.gr00t_native \
        --checkpoint /path/to/gr00t-checkpoint --device cuda:0 --port 8000 [--real]
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

# Franka Panda joint position limits (rad), q1..q7. Used ONLY as a sanity gate on the state
# the client sends: it is the reconstruction reference for a sim checkpoint and the model's
# proprioception for both, so a mis-assembled state is worth failing on. q4 is the useful one
# -- it is strictly NEGATIVE, so a reversed or shifted concat almost always trips it.
_FRANKA_Q_LIMITS = (
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
)


def _to_np(x) -> np.ndarray:
    """Model outputs may be torch tensors (possibly on GPU); coerce to numpy."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _require(obs: dict, key: str, mode: str) -> np.ndarray:
    """Fetch a required observation key, or fail loudly naming the expected schema."""
    if key not in obs:
        raise KeyError(
            f"{mode} mode: missing observation key {key!r}. Present keys: "
            f"{sorted(k for k in obs if isinstance(k, str))}. "
            f"This server does NOT fall back to the other schema -- start it with the "
            f"{'--real' if mode == 'sim' else 'sim (no --real)'} setting instead if the "
            f"client is the other one."
        )
    return np.asarray(obs[key])


def _unpack_obs(obs: dict, real: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Client observation -> (image_left, wrist, state8), for either client schema."""
    if real:
        img_left = _require(obs, "observation/exterior_image_1_left", "real").astype(np.uint8)
        wrist = _require(obs, "observation/wrist_image_left", "real").astype(np.uint8)
        # Real stores proprioception in TWO columns; assemble the 8-D state in the same
        # [arm(7), gripper(1)] order the embodiment config slices it back out with.
        joints = _require(obs, "observation/joint_position", "real").astype(np.float32).reshape(-1)
        gripper = _require(obs, "observation/gripper_position", "real").astype(np.float32).reshape(-1)
        if joints.shape != (7,) or gripper.shape != (1,):
            raise ValueError(
                f"real mode: expected joint_position (7,) and gripper_position (1,), "
                f"got {joints.shape} and {gripper.shape}"
            )
        state = np.concatenate([joints, gripper])
    else:
        img_left = _require(obs, "observation/image_left", "sim").astype(np.uint8)
        wrist = _require(obs, "observation/wrist_image", "sim").astype(np.uint8)
        state = _require(obs, "observation/state", "sim").astype(np.float32).reshape(-1)
        if state.shape != (8,):
            raise ValueError(f"sim mode: expected observation/state (8,), got {state.shape}")
    return img_left, wrist, state


def _check_state_is_absolute(state: np.ndarray) -> None:
    """Range-check the assembled state once, at the first inference.

    The arm entries must be absolute Franka joint positions -- for a sim checkpoint they are
    the reference GR00T re-adds to its relative prediction, so a wrong state silently biases
    every action. Cheap, and it catches the failure this cannot otherwise see: a swapped
    concat order (gripper first) puts a [0,1] value where q4 must be negative.
    """
    bad = [
        f"q{i + 1}={state[i]:.4f} outside [{lo:.4f}, {hi:.4f}]"
        for i, (lo, hi) in enumerate(_FRANKA_Q_LIMITS)
        if not (lo <= float(state[i]) <= hi)
    ]
    if not 0.0 <= float(state[7]) <= 1.0:
        bad.append(f"gripper={state[7]:.4f} outside [0, 1]")
    if bad:
        raise ValueError(
            "first observation/state is not a valid absolute Franka joint vector -- "
            "check the client's key mapping and concat order. Offending entries: "
            + "; ".join(bad)
        )
    logger.info(f"state sanity OK (absolute Franka joints): {np.array2string(state, precision=4)}")


class Gr00tWebsocketServer:
    """openpi-contract websocket server wrapping a gr00t Gr00tPolicy."""

    def __init__(self, policy, task_key: str, host: str, port: int, *,
                 real: bool = False, metadata: dict | None = None):
        self._policy = policy
        self._task_key = task_key
        self._host = host
        self._port = port
        self._real = real
        self._metadata = metadata or {}
        self._last_seed = None
        self._state_checked = False

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
        img_left, wrist, state = _unpack_obs(obs, self._real)   # (H,W,3), (H,W,3), (8,)
        if not self._state_checked:
            _check_state_is_absolute(state)
            self._state_checked = True
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
        # (16, 8) ordered [arm(7), gripper(1)]. Passed through, NEVER re-added to state --
        # what the entries MEAN is set by the checkpoint, not by this server:
        #   sim  checkpoint -> absolute joint targets (GR00T already re-added the state
        #                      internally), consumed by the joint_position_raw controller
        #   real checkpoint -> joint VELOCITY rad/s; the client applies delta = action/15
        chunk = np.concatenate([arm[0], grip[0]], axis=-1)
        return {"actions": chunk}

    async def _handler(self, websocket):
        packer = Packer()
        # Metadata handshake (openpi protocol). Mirrors openpi_native.py: the client asserts
        # on this BEFORE the arm is initialised, so a wrong checkpoint or a sim/real mode
        # mismatch cannot go silent -- an empty dict here would leave it undetectable.
        await websocket.send(packer.pack(self._metadata))
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
    ap.add_argument("--real", action="store_true",
                    help="read the DROID-schema real-robot observation keys "
                         "(exterior_image_1_left / wrist_image_left / joint_position + "
                         "gripper_position) instead of the sim benchmark's. Must match the "
                         "checkpoint: a real checkpoint returns joint VELOCITY.")
    ap.add_argument("--task", default=None,
                    help="family this checkpoint was trained on (clutter / jar / "
                         "cab_higher_firsthalf). Only used to build the handshake's "
                         "serve_config so the client can assert per family. Required with "
                         "--real: one string per MODE would advertise the same value for all "
                         "three checkpoints and defeat the assertion it exists for.")
    args = ap.parse_args()

    if args.real and not args.task:
        ap.error("--real requires --task: the client asserts serve_config against a per-family "
                 "map, so serving the jar checkpoint while the client thinks it is running "
                 "clutter must be detectable at connect.")

    _ensure_processor_at_root(args.checkpoint)  # handle the processor/ subdir layout (see fn doc)

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    # NEW_EMBODIMENT modality config is baked into the checkpoint's processor at SFT time
    # (AutoProcessor.from_pretrained restores it), so no runtime registration is needed.
    logger.info(f"Loading GR00T policy from {args.checkpoint} (device={args.device}) ...")
    policy = Gr00tPolicy(EmbodimentTag.NEW_EMBODIMENT, args.checkpoint, device=args.device)
    task_key = policy.language_key  # the single language modality key from the processor
    mode = "real" if args.real else "sim"
    logger.info(f"GR00T policy loaded. language_key={task_key!r} obs_schema={mode}")

    # GR00T has no openpi-style train-config name, so the handshake label is synthesised. It
    # must be TASK-specific: the pi-series client asserts serve_config against a per-family map
    # precisely to catch "wrong checkpoint for the family the operator selected", and one label
    # per mode would make that mismatch invisible again. NOTE `task_key` above is the language
    # MODALITY key (identical across all six checkpoints) -- not the family; the family is only
    # known from --task.
    serve_config = f"gr00t_n16_{mode}" + (f"_{args.task}" if args.task else "")

    metadata = {
        "serve_config": serve_config,
        "checkpoint": str(args.checkpoint),
        "embodiment_tag": "NEW_EMBODIMENT",
        "obs_schema": mode,
    }
    Gr00tWebsocketServer(policy, task_key, args.host, args.port,
                         real=args.real, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
