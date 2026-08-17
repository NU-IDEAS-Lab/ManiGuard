#!/usr/bin/env python3
"""
Serve an openpi policy natively over websocket.
Uses openpi's own model loading and serving infrastructure.

Usage:
    # CPU (for single-GPU machines)
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
        <path-to-openpi-venv>/bin/python3 -m maniguard.serve.openpi_native --config <train-config>

    # GPU (needs a separate GPU from the sim)
    <path-to-openpi-venv>/bin/python3 -m maniguard.serve.openpi_native --config <train-config>

    # Custom checkpoint path
    <path-to-openpi-venv>/bin/python3 -m maniguard.serve.openpi_native \
        --config <train-config> --checkpoint /path/to/checkpoint
"""
from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve openpi policy natively.")
    parser.add_argument(
        "--config",
        default="pi05_droid",
        help="openpi config name (e.g., pi05_droid, pi0_droid, pi05_base).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. If not provided, downloads from GCS automatically.",
    )
    parser.add_argument(
        "--gcs-uri",
        default=None,
        help="GCS URI to download checkpoint from (e.g., gs://openpi-assets/checkpoints/pi05_droid).",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


# Default GCS URIs for known configs
DEFAULT_GCS_URIS = {
    "pi05_droid": "gs://openpi-assets/checkpoints/pi05_droid",
    "pi0_droid": "gs://openpi-assets/checkpoints/pi0_droid",
    "pi05_base": "gs://openpi-assets/checkpoints/pi05_base",
    "pi0_base": "gs://openpi-assets/checkpoints/pi0_base",
}


class _SeededPolicy:
    """Delegating wrapper that re-seeds the openpi policy's JAX sampling key
    whenever the client starts a rollout with a new ``episode_seed`` (sent in
    every request by ``maniguard.eval.benchmark`` when ``--seed`` is set). The
    key is STRIPPED before delegating so openpi's input transforms never see
    it. Within a rollout the seed value is constant, so the key is set once and
    then advances normally (openpi splits it per infer). Without episode_seed
    this is a transparent pass-through (openpi's default key(0) behavior)."""

    def __init__(self, policy):
        self._policy = policy
        self._last_seed = None

    @property
    def metadata(self):
        return self._policy.metadata

    def infer(self, obs: dict) -> dict:
        seed = obs.pop("episode_seed", None)
        if seed is not None and seed != self._last_seed:
            import jax
            self._policy._rng = jax.random.key(int(seed))
            self._last_seed = seed
            logger.info(f"sampling RNG re-seeded: episode_seed={seed}")
        return self._policy.infer(obs)


def main() -> None:
    args = parse_args()

    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as train_config

    # Register ManiGuard's pi0.5 SFT TrainConfigs into openpi's registry so
    # get_config() resolves the joint-controller names (e.g.
    # pi05_base_dusty_transfer_joint_2cam_lora). openpi_native runs in the openpi
    # venv where maniguard is installed; the configs attach at import time via
    # register() (a no-op if already registered). Harmless for stock openpi
    # config names (they stay resolvable).
    try:
        import maniguard.openpi_sft.train_configs as _mg_train_configs
        _mg_train_configs.register()
    except Exception as _mg_exc:  # noqa: BLE001 - stock openpi configs still work
        logger.warning(f"ManiGuard config registration skipped: {_mg_exc}")

    cfg = train_config.get_config(args.config)
    logger.info(f"Loaded config: {args.config}")

    if args.checkpoint:
        checkpoint = args.checkpoint
    else:
        from openpi.shared import download
        gcs_uri = args.gcs_uri or DEFAULT_GCS_URIS.get(args.config)
        if gcs_uri is None:
            raise ValueError(
                f"No default GCS URI for config '{args.config}'. "
                f"Provide --checkpoint or --gcs-uri."
            )
        logger.info(f"Downloading checkpoint from {gcs_uri}...")
        checkpoint = download.maybe_download(gcs_uri)

    logger.info(f"Loading model from {checkpoint}...")
    policy = policy_config.create_trained_policy(cfg, checkpoint)
    logger.info("Model loaded successfully.")

    # Advertise what is actually being served in the connect handshake, so a client can ASSERT
    # it is talking to the policy it thinks it is instead of trusting an operator to keep two
    # machines in step. A config/checkpoint mismatch is otherwise completely silent: the wrong
    # policy answers every request without erring, and only the results look wrong.
    server = websocket_policy_server.WebsocketPolicyServer(
        _SeededPolicy(policy),
        host=args.host,
        port=args.port,
        metadata={"serve_config": args.config, "checkpoint": str(checkpoint)},
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
