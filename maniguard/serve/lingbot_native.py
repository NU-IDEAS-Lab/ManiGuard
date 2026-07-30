#!/usr/bin/env python3
"""Serve a ManiGuard LingBot-VLA 2.0 SFT checkpoint over the openpi-client websocket contract.

Runs in the ``lingbotvla`` conda env (the fork's own ``tools/create_train_env.sh`` recipe).
Wraps LingBot's ``LingbotVlaV2Policy`` behind the SAME websocket / msgpack-numpy protocol as
``maniguard.serve.openpi_native``, ``gr00t_native.py`` and ``smolvla_native.py`` -- so
``maniguard.eval.benchmark`` connects with NO client change.

Why not the fork's own ``deploy/lingbot_vla_v2_policy.py``: its loader assumes the checkpoint
still sits inside a training output tree. It reads the training config from
``<ckpt>/../../../lingbotvla_cli.yaml``, reads the robot config as ``configs/robot_configs/
<name>.yaml`` *relative to CWD*, and resolves the VLM through ``QWEN3VL_PATH``. An HF snapshot
has none of that layout. This shim resolves **everything from the checkpoint directory** plus
the in-repo training config, and logs each resolved source so a run's provenance is auditable
from the server log alone (the pattern the earlier eval waves rely on for unit-level checks --
``eval_config.json``'s ``serve_config_name`` is a static annotation and proves nothing).

Action contract: the checkpoint was trained with ``subtract_state: False`` on both action
features, i.e. it predicts **absolute** joint targets. ``FeatureTransform.unapply`` therefore
only unnormalises -- it does not add the state back -- so the chunk is forwarded AS-IS to the
JointController. Same contract as SmolVLA; pi0.5 / pi0 / GR00T all add state back instead.
Getting this backwards produces plausible-looking garbage without raising.

Norm stats: passed **explicitly** as ``<ckpt>/maniguard/norm_stats.json``. LingBot's
``FeatureTransform`` pops ``robot_config['norm_stats']`` when an explicit path is given, so the
per-family statistics always win over the yaml's default field (which is a stale pointer to the
clutter file in both published repos).

Usage (in the lingbotvla conda env):
    python maniguard/serve/lingbot_native.py \
        --checkpoint /path/to/lingbot-checkpoint --qwen-config /path/to/qwen3vl-config-dir \
        --device cuda:0 --port 8000
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import msgpack
import numpy as np
import websockets.asyncio.server as ws_server
import websockets.frames
from websockets.exceptions import ConnectionClosed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Observation keys the client sends (benchmark._remap_obs_for_openpi) and the flat datagen
# keys the checkpoint's robot_config maps from. The mapping is fixed by
# maniguard/lingbot_sft/robot_config.yaml: image_left -> camera_top, wrist_image ->
# camera_wrist_left, state[0:7] -> arm.position, state[7:8] -> effector.position.
_CLIENT_OVERVIEW = "observation/image_left"
_CLIENT_WRIST = "observation/wrist_image"
_CLIENT_STATE = "observation/state"
_ORIGIN_OVERVIEW = "image_left"
_ORIGIN_WRIST = "wrist_image"
_ORIGIN_STATE = "state"
# The single ORIGIN action key both action features write back into. `unapply` returns the
# origin key space, and the robot config routes action.arm.position -> actions[0:7] and
# action.effector.position -> actions[7:8], so this one array already carries the 8-D
# [arm_q(7), gripper(1)] layout in the right order.
_ORIGIN_ACTION = "actions"


# --- msgpack-numpy, vendored inline from openpi_client.msgpack_numpy -----------------------
# Inlined so this server is self-contained in the lingbotvla env, which has no openpi.
# Wire-compatible with the client's openpi_client.msgpack_numpy.
def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str,
                b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def Packer():  # noqa: N802 - mirrors openpi_client's factory name
    return msgpack.Packer(default=_pack_array, use_bin_type=True)


def unpackb(data):
    return msgpack.unpackb(data, object_hook=_unpack_array, raw=False)


class LingBotServer:
    """Websocket policy server around a LingBot-VLA 2.0 checkpoint."""

    def __init__(self, policy, feature_transform, action_dim, img_size, host, port):
        self._policy = policy
        self._ft = feature_transform
        self._action_dim = action_dim
        self._img_size = img_size
        self._host = host
        self._port = port
        self._last_seed = None

    # -- inference ------------------------------------------------------------------------
    def _maybe_reseed(self, obs: dict) -> None:
        """Re-seed the flow-sampling noise source when the client starts a rollout with a new
        ``episode_seed``. Constant within a rollout, so the RNG advances normally between
        infers -- the same semantics the other three servers implement."""
        seed = obs.pop("episode_seed", None)
        if seed is not None and seed != self._last_seed:
            import torch
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
            np.random.seed(int(seed) % (2 ** 32))
            self._last_seed = seed
            logger.info(f"sampling RNG re-seeded: episode_seed={seed}")

    def _infer(self, obs: dict) -> dict:
        import torch
        from torchvision.transforms.v2 import Resize

        self._maybe_reseed(obs)
        prompt = obs.get("prompt", "")
        if isinstance(prompt, (bytes, bytearray)):
            prompt = prompt.decode()

        # Build the item in the FLAT datagen key space the checkpoint's robot_config maps
        # from, then let LingBot's own FeatureTransform do the unified-vector packing,
        # normalisation and tokenisation -- byte-identical to the training path.
        item = {
            _ORIGIN_STATE: torch.as_tensor(
                np.asarray(obs[_CLIENT_STATE], dtype=np.float32).reshape(-1)),
            "task": str(prompt),
        }
        resize = Resize((self._img_size, self._img_size))
        for origin_key, client_key in ((_ORIGIN_OVERVIEW, _CLIENT_OVERVIEW),
                                       (_ORIGIN_WRIST, _CLIENT_WRIST)):
            img = np.ascontiguousarray(obs[client_key], dtype=np.uint8)   # (H, W, 3)
            t = torch.as_tensor(img).permute(2, 0, 1).contiguous().to(dtype=torch.float32)
            item[origin_key] = resize(t)

        transformed = self._ft.apply(item, policy_eval=True)
        # The fork's canonical single-observation path: select_action samples the chunk AND
        # runs FeatureTransform.unapply itself, returning the ORIGIN key space. So the result
        # is already unnormalised; because both action features are subtract_state False it is
        # also already absolute (no state added). Do NOT unapply again -- a second pass trips
        # reverse_pad_and_concat's mask/width assert.
        out = self._policy.select_action(transformed, use_bf16=True)
        chunk = np.asarray(out[_ORIGIN_ACTION], dtype=np.float32)  # (chunk, 8) absolute targets
        if chunk.ndim != 2 or chunk.shape[-1] != self._action_dim:
            raise ValueError(f"action chunk {chunk.shape}, expected (chunk, {self._action_dim})")
        return {"actions": np.ascontiguousarray(chunk, dtype=np.float32)}

    # -- websocket ------------------------------------------------------------------------
    async def _handler(self, websocket):
        packer = Packer()
        await websocket.send(packer.pack({}))  # metadata handshake (openpi protocol)
        while True:
            try:
                obs = unpackb(await websocket.recv())
                await websocket.send(packer.pack(self._infer(obs)))
            except ConnectionClosed:
                logger.info("client disconnected")
                break
            except Exception:  # noqa: BLE001 - report to the client like openpi's server
                tb = traceback.format_exc()
                logger.error(tb)
                await websocket.send(tb)
                await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR,
                                      reason="server error")
                break

    async def _run(self):
        async with ws_server.serve(self._handler, self._host, self._port,
                                   compression=None, max_size=None):
            logger.info(f"LingBot policy server listening on {self._host}:{self._port}")
            await asyncio.get_running_loop().create_future()

    def serve_forever(self):
        asyncio.run(self._run())


_PROCESSOR_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                    "added_tokens.json", "special_tokens_map.json", "chat_template.jinja",
                    "preprocessor_config.json", "video_preprocessor_config.json")


def _stage_vlm_assets(ckpt: Path, qwen_config_dir: Path, workdir: Path) -> Path:
    """Stage the one directory LingBot expects as its VLM asset root.

    LingBot resolves the VLM through a single ``config.tokenizer_path`` that must serve
    ``AutoConfig``, ``AutoTokenizer`` and ``build_processor`` at once -- and it keys behaviour
    off SUBSTRINGS of that path (``'qwen' in tokenizer_path`` in ``data/dataset.py``; the VLM
    patch key in ``models/auto.py``). The published checkpoint ships the full processor set but
    its ``config.json`` is LingBot's own, not the VLM's, so neither path alone works. Stage a
    dir that holds the VLM config plus the checkpoint's processor files, named after the VLM so
    the substring heuristics resolve. Processor files are hard-linked (falling back to copy) so
    they stay byte-identical to what the checkpoint shipped.
    """
    stage = workdir / "Qwen3-VL-4B-Instruct"
    stage.mkdir(parents=True, exist_ok=True)
    src_cfg = qwen_config_dir / "config.json"
    if not src_cfg.exists():
        raise SystemExit(f"--qwen-config must contain config.json: {src_cfg}")
    shutil.copy2(src_cfg, stage / "config.json")
    staged = []
    for name in _PROCESSOR_FILES:
        src = ckpt / name
        if not src.exists():
            continue
        dst = stage / name
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        staged.append(name)
    logger.info(f"VLM asset dir : {stage}  (VLM config from {qwen_config_dir}, "
                f"{len(staged)} processor files from the checkpoint)")
    return stage


def _build_data_config(train_config_path: Path) -> SimpleNamespace:
    """FeatureTransform needs the training `data` block (joints / cameras / norm_type /
    img_size). Read it from the in-repo SFT config so eval sees exactly the feature space
    training used."""
    import yaml
    with train_config_path.open() as f:
        cfg = yaml.safe_load(f)
    data = dict(cfg["data"])
    # Reproduce the training path's own coercion exactly. LingBot's yaml->args layer
    # (lingbotvla/utils/arguments.py: `cmd_args.extend([str(item) for item in arg_value])`)
    # stringifies every element of a list field, and the consumers then parse them back with
    # ast.literal_eval (FeatureInfo.update_info for `joints`, FeatureTransform for `norm_type`).
    # Handing FeatureTransform the raw yaml dicts instead raises
    # "malformed node or string: {'arm.position': 14}", so apply the same str() the trainer did
    # -- eval must see the byte-identical feature space training used.
    for key in ("joints", "norm_type", "cameras"):
        if isinstance(data.get(key), list):
            data[key] = [str(item) for item in data[key]]
    logger.info(f"data block from {train_config_path}: joints={data.get('joints')} "
                f"cameras={data.get('cameras')} norm_type={data.get('norm_type')} "
                f"img_size={data.get('img_size', 256)}")
    return SimpleNamespace(**data)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Serve a ManiGuard LingBot-VLA 2.0 checkpoint (openpi ws contract).")
    ap.add_argument("--checkpoint", required=True,
                    help="local LingBot SFT checkpoint dir (HF snapshot); must contain "
                         "config.json, the safetensors shards, the processor files and "
                         "maniguard/{norm_stats.json,robot_config.yaml}")
    ap.add_argument("--qwen-config", required=True,
                    help="dir holding Qwen3-VL-4B-Instruct's config.json (the VLM skeleton "
                         "LingBot merges before loading our weights)")
    ap.add_argument("--train-config", default=None,
                    help="SFT train_config.yaml providing the data block "
                         "(default: maniguard/lingbot_sft/train_config.yaml next to this repo)")
    ap.add_argument("--num-steps", type=int, default=None,
                    help="Euler steps for the flow sampler (default: the checkpoint's own "
                         "config.num_steps). Inference-time only -- raising it costs latency "
                         "and cannot change what the model learned, but it does change how far "
                         "the sample is integrated, so record whatever is used.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    robot_config = ckpt / "maniguard" / "robot_config.yaml"
    norm_stats = ckpt / "maniguard" / "norm_stats.json"
    for p, what in ((ckpt / "config.json", "config.json"), (robot_config, "robot_config.yaml"),
                    (norm_stats, "norm_stats.json")):
        if not p.exists():
            raise SystemExit(f"checkpoint is missing {what}: {p}")
    train_config = Path(args.train_config) if args.train_config else (
        Path(__file__).resolve().parents[1] / "lingbot_sft" / "train_config.yaml")

    # Provenance, logged so a unit's server log alone proves what was served.
    logger.info(f"checkpoint    : {ckpt}")
    logger.info(f"robot_config  : {robot_config}")
    logger.info(f"norm_stats    : {norm_stats}   (explicit -> overrides the yaml's field)")
    logger.info(f"qwen config   : {Path(args.qwen_config).resolve()}")

    import torch
    import yaml
    from transformers import AutoConfig
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    from lingbotvla.data.vla_data.utils import FeatureTransform
    from lingbotvla.models import build_processor
    # Both packages come from the fork, installed editable by tools/create_train_env.sh,
    # so no sys.path surgery is needed in the lingbotvla env.
    from deploy.lingbot_vla_v2_policy import LingBotVlaV2InferencePolicy

    device = torch.device(args.device)
    data_config = _build_data_config(train_config)
    workdir = Path(tempfile.mkdtemp(prefix="lingbot_vlm_"))
    vlm_dir = _stage_vlm_assets(ckpt, Path(args.qwen_config).resolve(), workdir)

    # 1. model config: the checkpoint's own LingBot config, merged with the VLM skeleton.
    with (ckpt / "config.json").open() as f:
        ckpt_cfg = yaml.safe_load(f)
    ckpt_cfg.pop("architectures", None)
    config = LingbotVLAV2Config(**ckpt_cfg)
    for k, v in ckpt_cfg.items():
        if not hasattr(config, k):
            setattr(config, k, v)
    config.attention_implementation = "eager"   # the fork's own eval choice
    config.use_cache = True                     # required at inference
    if args.num_steps is not None and int(args.num_steps) != int(config.num_steps):
        logger.info(f"flow sampler steps overridden: {config.num_steps} -> {args.num_steps}")
        config.num_steps = int(args.num_steps)
    config.tokenizer_path = str(vlm_dir)
    qwen_cfg = AutoConfig.from_pretrained(str(vlm_dir))
    qwen_dict = qwen_cfg.to_dict() if hasattr(qwen_cfg, "to_dict") else dict(qwen_cfg)
    for k, v in qwen_dict.items():
        if not hasattr(config, k):
            setattr(config, k, v)
    logger.info(f"model config  : chunk_size={config.chunk_size} num_steps={config.num_steps} "
                f"max_action_dim={config.max_action_dim} vlm_family={config.vlm_family}")

    # 2. processor / tokenizer straight from the checkpoint (byte-identical to SFT).
    processor = build_processor(str(vlm_dir))

    # 3. VLM monkey-patches. LingBot swaps in its own Qwen3-VL blocks and ADDS methods the
    # model then calls (e.g. Qwen3VLVisionModel.preprcess_grid_thw). The fork applies them in
    # its own server __init__ / models/auto.py; constructing the policy directly skips that and
    # fails later with a bare Module.__getattr__ on the missing method. Select the family the
    # same way models/auto.py does, off the VLM key, rather than hardcoding.
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import apply_lingbot_qwen2_patch
    vlm_key = f"{getattr(config, 'vlm_repo_id', '') or ''} {config.tokenizer_path or ''}".lower()
    if "qwen3" in vlm_key and "vl" in vlm_key:
        from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
        apply_lingbot_qwen3_vl_patch()
        logger.info("applied LingBot Qwen3-VL patch")
    else:
        from lingbotvla.models.vla.lingbot_vla.qwenvl_in_vla import apply_lingbot_qwen25_vl_patch
        apply_lingbot_qwen25_vl_patch()
        logger.info("applied LingBot Qwen2.5-VL patch")
    apply_lingbot_qwen2_patch()

    # 4. weights: all shards merged, strict -- a missing/extra key must fail loudly.
    from safetensors import safe_open
    policy = LingBotVlaV2InferencePolicy(config, eval=True)
    merged = {}
    for shard in sorted(ckpt.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                merged[key] = f.get_tensor(key)
    logger.info(f"loading {len(merged)} tensors from {len(list(ckpt.glob('*.safetensors')))} shards")
    policy.load_state_dict(merged, strict=True)
    del merged
    policy = policy.to(torch.bfloat16).to(device).eval()

    # 5. feature transform with the per-family stats passed EXPLICITLY.
    ft = FeatureTransform(str(robot_config), data_config, config, processor,
                          chunk_size=config.chunk_size, norm_stats_path=str(norm_stats))
    policy.feature_transform = ft
    # `org_features` values are SETS, so their iteration order is neither the robot config's
    # nor stable across processes (string hash randomisation). Concatenating several origin
    # arrays off that order would silently permute the action vector, so require the single
    # origin key this robot config declares and fail loudly if a future config adds another.
    origin = set(ft.org_features["actions"])
    if origin != {_ORIGIN_ACTION}:
        raise SystemExit(f"expected exactly one origin action key {_ORIGIN_ACTION!r}, "
                         f"got {sorted(origin)} -- the chunk layout would be ambiguous")
    # Make the ABSOLUTE contract self-verifying: this server adds no state back, so a
    # checkpoint trained with subtract_state True must not be served here.
    relative = [k for k, v in getattr(ft, "action_subtract_state", {}).items() if v]
    if relative:
        raise SystemExit(f"{relative} were trained with subtract_state True (state-relative); "
                         "this server forwards absolute targets and would corrupt them")
    action_dim = max(int(oi["end"]) for oi in ft.key_reverse_mapping[_ORIGIN_ACTION])
    logger.info(f"action key    : {_ORIGIN_ACTION} -> {action_dim}-D absolute joint targets "
                f"(subtract_state False on {sorted(ft.action_subtract_state)}, no state added)")
    logger.info("Model loaded successfully.")

    LingBotServer(policy, ft, action_dim, int(getattr(data_config, "img_size", 256)),
                  args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()
