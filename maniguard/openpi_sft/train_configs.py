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

Scope: the DATAGEN v1 families we have collected + published (clutter, cabinet,
stack, jar). Each config is the executable train spec; ``num_train_steps`` = 2
epochs of that family's dataset (rounded UP). Tune params to the run's platform.
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
        # Sim cabinet-pickup (open the table-top cabinet drawer, put the target
        # object inside, and close it without knocking anything over), LIBERO 2-cam,
        # JOINT controller. DATAGEN v1 dataset (35 base tasks x 40 = 1400 demos).
        #
        # Dataset: IDEAS-Lab-Northwestern/datagen-cabinet-v1-joint-5cam
        #   5-cam rendered (image_opposite/left/right/left_shoulder + wrist_image)
        #   but consumed 2-cam: external_cam="left" overview + wrist_image; the
        #   other views dropped, pi0.5's third image slot zero-filled + masked.
        #   8-D joint state + 8-D absolute-joint action; use_delta_joint_actions=True.
        # warm-start = pi05_base. discrete_state_input=True (pi0.5: the 8-D robot
        #   state is discretized + tokenized into the language prefix alongside the
        #   prompt) -- set explicitly (Pi0Config would resolve None->pi05 anyway).
        #
        # Training scale: 2 epochs over the 4,172,962-frame set at batch 32
        #   (4_172_962 * 2 / 32 = 260,811 -> rounded UP to 261,000 to guarantee a
        #   full 2 epochs). decay_steps == num_train_steps; keep_period = steps // 5
        #   (5 checkpoints); warmup 3%; peak_lr sqrt-scaled from 2.5e-5 @ batch 8.
        TrainConfig(
            name="pi05-base_datagen_v1_cabinet_joint_2cam_lora",
            project_name="maniguard-sft",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-cabinet-joint-2cam-lora",
                "hf_private": False,
                "default_exp": "datagen_v1_cabinet_joint_2cam",
            },
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
                discrete_state_input=True,  # pi0.5: 8-D robot state -> discrete prompt tokens
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-cabinet-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,  # JointController: MUST be True
                external_cam="left",  # reviewed: datagen cabinet left overview is good
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            # sqrt-LR (base 2.5e-5 @ batch 8): peak = 2.5e-5*sqrt(32/8) = 5e-5;
            # decay_lr = peak/10; warmup 3% of steps; decay_steps == num_train_steps.
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=7_830,  # 3% of 261,000
                peak_lr=5e-5,  # 2.5e-5 * sqrt(32/8)
                decay_steps=261_000,  # == num_train_steps
                decay_lr=5e-6,  # peak/10
            ),
            num_train_steps=261_000,  # 2 epochs @ batch 32 (4,172,962 frames), rounded up
            batch_size=32,
            num_workers=16,  # CPU dataloader prefetch workers (pyav decode is CPU-bound)
            log_interval=5,  # loss logging cadence
            fsdp_devices=1,  # full data-parallel, model replicated, no sharding
            keep_period=52_200,  # steps // 5 -> 5 evenly-spaced checkpoints
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # Sim pnp-clutter (pick the target object out of a cluttered tabletop and
        # move it into the green goal sphere), LIBERO 2-cam, JOINT controller.
        # DATAGEN v1 dataset (42 base tasks, 2200 demos / 901,520 frames).
        #
        # Dataset: IDEAS-Lab-Northwestern/datagen-clutter-v1-joint-5cam
        #   5-cam rendered (image_opposite/left/right/left_shoulder + wrist_image)
        #   but consumed 2-cam: external_cam="left" overview + wrist_image; the
        #   other views dropped, pi0.5's third image slot zero-filled + masked.
        #   8-D joint state + 8-D absolute-joint action; use_delta_joint_actions=True.
        # warm-start = pi05_base. discrete_state_input=True (pi0.5: the 8-D robot
        #   state is discretized + tokenized into the language prefix alongside the
        #   prompt) -- set explicitly (Pi0Config would resolve None->pi05 anyway).
        #
        # Training scale: 2 epochs over the 901,520-frame set at batch 32
        #   (901_520 * 2 / 32 = 56,345 -> rounded UP to 57,000 to guarantee a full
        #   2 epochs). decay_steps == num_train_steps; keep_period = steps // 5
        #   (5 checkpoints); warmup 3%; peak_lr sqrt-scaled from 2.5e-5 @ batch 8.
        TrainConfig(
            name="pi05-base_datagen_v1_clutter_joint_2cam_lora",
            project_name="maniguard-sft",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-clutter-joint-2cam-lora",
                "hf_private": False,
                "default_exp": "datagen_v1_clutter_joint_2cam",
            },
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
                discrete_state_input=True,  # pi0.5: 8-D robot state -> discrete prompt tokens
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-clutter-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,  # JointController: MUST be True
                external_cam="left",  # reviewed: datagen clutter left overview is good
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            # sqrt-LR (base 2.5e-5 @ batch 8): peak = 2.5e-5*sqrt(32/8) = 5e-5;
            # decay_lr = peak/10; warmup 3% of steps; decay_steps == num_train_steps.
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_710,  # 3% of 57,000
                peak_lr=5e-5,  # 2.5e-5 * sqrt(32/8)
                decay_steps=57_000,  # == num_train_steps
                decay_lr=5e-6,  # peak/10
            ),
            num_train_steps=57_000,  # 2 epochs @ batch 32 (901,520 frames), rounded up
            batch_size=32,
            num_workers=16,  # CPU dataloader prefetch workers (pyav decode is CPU-bound)
            log_interval=5,  # loss logging cadence
            fsdp_devices=1,  # full data-parallel, model replicated, no sharding
            keep_period=11_400,  # steps // 5 -> 5 evenly-spaced checkpoints
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # Sim stack-retrieve (unstack the 3 same-object top pile onto a re-stack
        # pile aside, then retrieve the exposed bottom target into the green goal
        # sphere), LIBERO 2-cam, JOINT controller. DATAGEN v1 dataset (28 base
        # tasks x 40 = 1120 demos / 2,652,083 frames).
        #
        # Dataset: IDEAS-Lab-Northwestern/datagen-stack-v1-joint-5cam
        #   5-cam rendered (image_opposite/left/right/left_shoulder + wrist_image)
        #   but consumed 2-cam: external_cam="left" overview + wrist_image; the
        #   other views dropped, pi0.5's third image slot zero-filled + masked.
        #   8-D joint state + 8-D absolute-joint action; use_delta_joint_actions=True.
        # warm-start = pi05_base. discrete_state_input=True (pi0.5: the 8-D robot
        #   state is discretized + tokenized into the language prefix alongside the
        #   prompt) -- set explicitly (Pi0Config would resolve None->pi05 anyway).
        #
        # Training scale: 2 epochs over the 2,652,083-frame set at batch 32
        #   (2_652_083 * 2 / 32 = 165,755 -> rounded UP to 166,000 to guarantee a
        #   full 2 epochs). decay_steps == num_train_steps; keep_period = steps // 5
        #   (5 checkpoints); warmup 3%; peak_lr sqrt-scaled from 2.5e-5 @ batch 8.
        TrainConfig(
            name="pi05-base_datagen_v1_stack_joint_2cam_lora",
            project_name="maniguard-sft",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-stack-joint-2cam-lora",
                "hf_private": False,
                "default_exp": "datagen_v1_stack_joint_2cam",
            },
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
                discrete_state_input=True,  # pi0.5: 8-D robot state -> discrete prompt tokens
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-stack-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,  # JointController: MUST be True
                external_cam="left",  # reviewed (video): datagen stack left overview is good
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            # sqrt-LR (base 2.5e-5 @ batch 8): peak = 2.5e-5*sqrt(32/8) = 5e-5;
            # decay_lr = peak/10; warmup 3% of steps; decay_steps == num_train_steps.
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=4_980,  # 3% of 166,000
                peak_lr=5e-5,  # 2.5e-5 * sqrt(32/8)
                decay_steps=166_000,  # == num_train_steps
                decay_lr=5e-6,  # peak/10
            ),
            num_train_steps=166_000,  # 2 epochs @ batch 32 (2,652,083 frames), rounded up
            batch_size=32,
            num_workers=16,  # CPU dataloader prefetch workers (pyav decode is CPU-bound)
            log_interval=5,  # loss logging cadence
            fsdp_devices=1,  # full data-parallel, model replicated, no sharding
            keep_period=33_200,  # steps // 5 -> 5 evenly-spaced checkpoints
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
        # DATAGEN v1 dataset (26 base tasks x 40 = 1040 demos / 946,870 frames).
        #
        # Dataset: IDEAS-Lab-Northwestern/datagen-jar-v1-joint-5cam
        #   5-cam rendered (image_opposite/left/right/left_shoulder + wrist_image)
        #   but consumed 2-cam: external_cam="left" overview + wrist_image; the
        #   other views dropped, pi0.5's third image slot zero-filled + masked.
        #   8-D joint state + 8-D absolute-joint action; use_delta_joint_actions=True.
        # warm-start = pi05_base. discrete_state_input=True (pi0.5: the 8-D robot
        #   state is discretized + tokenized into the language prefix alongside the
        #   prompt) -- set explicitly (Pi0Config would resolve None->pi05 anyway).
        #
        # Training scale: 2 epochs over the 946,870-frame set at batch 32
        #   (946_870 * 2 / 32 = 59,179 -> rounded UP to 60,000 to guarantee a full
        #   2 epochs). decay_steps == num_train_steps; keep_period = steps // 5
        #   (5 checkpoints); warmup 3%; peak_lr sqrt-scaled from 2.5e-5 @ batch 8.
        TrainConfig(
            name="pi05-base_datagen_v1_jar_joint_2cam_lora",
            project_name="maniguard-sft",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-jar-joint-2cam-lora",
                "hf_private": False,
                "default_exp": "datagen_v1_jar_joint_2cam",
            },
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
                discrete_state_input=True,  # pi0.5: 8-D robot state -> discrete prompt tokens
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-jar-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,  # JointController: MUST be True
                external_cam="left",  # datagen jar left overview (default, confirm vs review)
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            # sqrt-LR (base 2.5e-5 @ batch 8): peak = 2.5e-5*sqrt(32/8) = 5e-5;
            # decay_lr = peak/10; warmup 3% of steps; decay_steps == num_train_steps.
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_800,  # 3% of 60,000
                peak_lr=5e-5,  # 2.5e-5 * sqrt(32/8)
                decay_steps=60_000,  # == num_train_steps
                decay_lr=5e-6,  # peak/10
            ),
            num_train_steps=60_000,  # 2 epochs @ batch 32 (946,870 frames), rounded up
            batch_size=32,
            num_workers=16,  # CPU dataloader prefetch workers (pyav decode is CPU-bound)
            log_interval=5,  # loss logging cadence
            fsdp_devices=1,  # full data-parallel, model replicated, no sharding
            keep_period=12_000,  # steps // 5 -> 5 evenly-spaced checkpoints
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
