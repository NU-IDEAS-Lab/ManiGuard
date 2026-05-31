"""ManiGuard pi0.5 LoRA SFT TrainConfigs, registered into pristine openpi.

Each task is a fully inline ``TrainConfig`` (model + freeze_filter written out
in full, no shared builders) so the entire recipe is readable at a glance.
``register()`` inserts them into openpi's ``_CONFIGS_DICT`` at import time, so
openpi's ``scripts/train.py`` / ``scripts/compute_norm_stats.py`` can resolve
them by name when launched via the wrappers in ``tools/openpi_sft/`` — openpi
itself is never edited.

JointController pipeline: every config uses ``Sim2CamLiberoDataConfig`` with
``use_delta_joint_actions=True`` (absolute-joint datasets; 7 arm joints ->
per-step delta, gripper absolute). All warm-start from ``pi05_base``.
"""

from __future__ import annotations

import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
from openpi.training.config import DataConfig, TrainConfig

from maniguard.openpi_sft.data_configs import Sim2CamLiberoDataConfig

_PI05_BASE = "gs://openpi-assets/checkpoints/pi05_base/params"


def _build_configs() -> list[TrainConfig]:
    return [
        # Sim food-dusty-transfer (wipe a dusty destination clean with a sponge,
        # then transfer food into it), LIBERO 2-cam, JOINT controller.
        #
        # Dataset: IDEAS-Lab-Northwestern/sim-dusty-transfer-30-joint-3cam
        #   3-cam rendered (image_left/image_right/wrist) but consumed 2-cam
        #   (image_left + wrist; image_right dropped, third slot black) per
        #   Sim2CamLiberoDataConfig. 8-D joint state + 8-D absolute-joint action.
        # use_delta_joint_actions=True is REQUIRED here: this is the
        #   JointController pipeline, so the absolute-joint actions are converted
        #   to per-step arm deltas (gripper absolute) for the model and
        #   reconstructed to absolute joint targets at eval (fed straight to a
        #   JointController, no eef->joint IK).
        # warm-start = pi05_base.
        #
        # Training scale below is a PLACEHOLDER sized to ~2 epochs at batch 4
        # with 5 evenly-spaced checkpoints; the user retunes for the actual
        # compute box. Keep these in lockstep when rescaling:
        #   ~2 epochs: 20_265 frames * 2 / batch 4 ~= 10_134 -> 10_000 steps
        #   decay_steps == num_train_steps; keep_period = steps // 5 (5 ckpts);
        #   peak_lr 2.5e-5 @ batch 4 (sqrt-scale up if you raise batch_size).
        TrainConfig(
            name="pi05_base_dusty_transfer_joint_2cam_lora",
            project_name="maniguard-sft",  # wandb project for all ManiGuard SFT
            # Handoff metadata: openpi never interprets policy_metadata, so we use
            # it to carry the run's HF target + default experiment name. run_sft.sh
            # reads these (via _config_meta.py) as defaults, so launching only
            # needs --config -- the HF repo, visibility, and exp/run name all come
            # from here. CLI flags still override. default_exp also becomes the
            # wandb run name and the outputs/sft_runs/<exp>/ folder name.
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-dusty-transfer-joint-2cam-lora",
                "hf_private": False,  # public model repo (datasets stay private)
                "default_exp": "dusty_transfer_joint_2cam",
            },
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/sim-dusty-transfer-30-joint-3cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,  # JointController: MUST be True
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=2.5e-5,
                decay_steps=10_000,  # == num_train_steps
                decay_lr=2.5e-6,
            ),
            num_train_steps=10_000,  # placeholder: ~2 epochs @ batch 4
            batch_size=4,
            keep_period=2_000,  # steps // 5 -> 5 evenly-spaced checkpoints
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
    ]


def register() -> None:
    """Insert the ManiGuard TrainConfigs into openpi's ``_CONFIGS_DICT``.

    Idempotent: re-registering overwrites by name. Must run before openpi's
    ``config.cli()`` / ``get_config()`` are called (the wrappers in
    ``tools/openpi_sft/`` import this package first, which triggers it).
    """
    from openpi.training.config import _CONFIGS_DICT

    for cfg in _build_configs():
        _CONFIGS_DICT[cfg.name] = cfg
