#!/usr/bin/env python3
"""
Serve an openpi policy natively over websocket.
Uses openpi's own model loading and serving infrastructure.

Usage:
    # DROID (CPU, for single-GPU machines)
    sudo CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
        RLinf/.venv/bin/python3 tools/serve_openpi_native.py --config pi05_droid

    # DROID (GPU, needs separate GPU from sim)
    sudo RLinf/.venv/bin/python3 tools/serve_openpi_native.py --config pi05_droid

    # Custom checkpoint path
    sudo RLinf/.venv/bin/python3 tools/serve_openpi_native.py \
        --config pi05_droid --checkpoint /path/to/checkpoint
"""
from __future__ import annotations

import argparse
import logging
import sys

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


def main() -> None:
    args = parse_args()

    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as train_config

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

    server = websocket_policy_server.WebsocketPolicyServer(
        policy, host=args.host, port=args.port
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
