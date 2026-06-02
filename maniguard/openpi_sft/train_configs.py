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
        # Training scale: ~2 epochs over the 20,265-frame set at batch 12.
        # Keep in lockstep when rescaling: decay_steps == num_train_steps;
        # keep_period = steps // 5; peak_lr sqrt-scaled from 2.5e-5 @ batch 8.
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
                # float32 params/compute -- required for numerical stability when
                # training under FSDP parameter sharding (fsdp_devices > 1): in bf16
                # the sharded gather/scatter can overflow and diverge to NaN early in
                # training. On large-memory GPUs in full data-parallel (fsdp_devices=1,
                # no sharding), bf16 is stable and faster.
                dtype="float32",
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/sim-dusty-transfer-30-joint-3cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,  # JointController: MUST be True
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            # ~2 epochs @ batch 12; sqrt-LR recipe (base 2.5e-5 @ batch 8):
            #   2 epochs = 20_265 frames * 2 / batch 12 = 3377.5 -> 3380 steps
            #   peak_lr = 2.5e-5 * sqrt(12/8) ~= 3e-5; decay_lr = peak/10; warmup ~10%
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=340,  # ~10% of 3380
                peak_lr=3e-5,  # 2.5e-5 * sqrt(12/8)
                decay_steps=3_380,  # == num_train_steps
                decay_lr=3e-6,  # peak/10
            ),
            num_train_steps=3_380,  # ~2 epochs @ batch 12
            batch_size=12,
            num_workers=8,  # CPU dataloader prefetch workers
            log_interval=25,  # loss logging cadence
            # Shard the model across GPUs with FSDP: pi0.5 params are fp32
            # (~10-12 GB), so the full model doesn't fit replicated on a
            # memory-constrained GPU. Set to 1 for single-GPU / full data-parallel
            # on large-memory GPUs. (batch_size % num_devices == 0 and
            # num_devices % fsdp_devices == 0.)
            fsdp_devices=4,
            keep_period=676,  # steps // 5 -> 5 evenly-spaced checkpoints
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # Sim jar-transport (close a hinged jar's lid, then carry the closed jar
        # into the green goal sphere on the table), LIBERO 2-cam, JOINT controller.
        #
        # Dataset: IDEAS-Lab-Northwestern/sim-jar-transport-30-joint-3cam
        #   3-cam rendered (image_left/image_right/wrist) but consumed 2-cam
        #   (image_left + wrist; image_right dropped, third slot black). 8-D joint
        #   state + 8-D absolute-joint action; use_delta_joint_actions=True.
        # warm-start = pi05_base.
        #
        # Training scale: ~2 epochs over the 12,967-frame set at batch 12
        #   (12_967 * 2 / 12 = 2161 -> 2160 steps; keep_period = steps // 5).
        # dtype=float32 + fsdp_devices=4: see the dusty-transfer config above for
        # the rationale (FSDP-sharded bf16 diverges to NaN; float32 is stable).
        TrainConfig(
            name="pi05_base_jar_transport_joint_2cam_lora",
            project_name="maniguard-sft",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-jar-transport-joint-2cam-lora",
                "hf_private": False,
                "default_exp": "jar_transport_joint_2cam",
            },
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="float32",
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/sim-jar-transport-30-joint-3cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,  # JointController: MUST be True
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            # ~2 epochs @ batch 12; sqrt-LR recipe (base 2.5e-5 @ batch 8):
            #   peak_lr = 2.5e-5 * sqrt(12/8) ~= 3e-5; decay_lr = peak/10; warmup ~10%
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=216,  # ~10% of 2160
                peak_lr=3e-5,  # 2.5e-5 * sqrt(12/8)
                decay_steps=2_160,  # == num_train_steps
                decay_lr=3e-6,  # peak/10
            ),
            num_train_steps=2_160,  # ~2 epochs @ batch 12
            batch_size=12,
            num_workers=8,  # CPU dataloader prefetch workers
            log_interval=25,  # loss logging cadence
            fsdp_devices=4,  # shard across 4 GPUs (see dusty-transfer config note)
            keep_period=432,  # steps // 5 -> 5 evenly-spaced checkpoints
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # Sim cabinet-pickup (place the paper towel holder into the cabinet's open
        # drawer and close it, without knocking anything over), LIBERO 2-cam,
        # JOINT controller.
        #
        # Dataset: IDEAS-Lab-Northwestern/sim-cabinet-pickup-30-joint-3cam
        #   3-cam rendered (image_left/image_right/wrist) but consumed 2-cam
        #   (image_left + wrist; image_right dropped, third slot black). 8-D joint
        #   state + 8-D absolute-joint action; use_delta_joint_actions=True.
        # warm-start = pi05_base.
        #
        # Training scale: ~2 epochs over the 22,684-frame set at batch 12, rounded
        #   up to a clean 4000 steps. keep_period = 1000 matches the 1000-step
        #   save_interval, so checkpoints land at 1000/2000/3000/4000.
        # dtype=float32 + fsdp_devices=4: see the dusty-transfer config above for
        # the rationale (FSDP-sharded bf16 diverges to NaN; float32 is stable).
        TrainConfig(
            name="pi05_base_cabinet_pickup_joint_2cam_lora",
            project_name="maniguard-sft",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-cabinet-pickup-joint-2cam-lora",
                "hf_private": False,
                "default_exp": "cabinet_pickup_joint_2cam",
            },
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="float32",
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/sim-cabinet-pickup-30-joint-3cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,  # JointController: MUST be True
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            # ~2 epochs @ batch 12 rounded up to a clean 4000 steps; sqrt-LR recipe
            # (base 2.5e-5 @ batch 8): peak_lr = 2.5e-5 * sqrt(12/8) ~= 3e-5;
            # decay_lr = peak/10; warmup ~10%.
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=400,  # ~10% of 4000
                peak_lr=3e-5,  # 2.5e-5 * sqrt(12/8)
                decay_steps=4_000,  # == num_train_steps
                decay_lr=3e-6,  # peak/10
            ),
            num_train_steps=4_000,  # ~2 epochs @ batch 12 (22,684 frames, rounded up)
            batch_size=12,
            num_workers=8,  # CPU dataloader prefetch workers
            log_interval=25,  # loss logging cadence
            fsdp_devices=4,  # shard across 4 GPUs (see dusty-transfer config note)
            keep_period=1_000,  # matches the 1000-step save cadence -> ckpts at 1000/2000/3000/4000
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
