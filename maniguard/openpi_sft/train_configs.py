"""ManiGuard pi0.5 LoRA SFT TrainConfigs, registered into pristine openpi.

Six task families (clutter, cabinet, stack, jar, lid, dusty), each a fully inline
``TrainConfig`` (model + freeze_filter written out in full, no shared builders)
so the entire recipe is readable at a glance. ``register()`` inserts them into
openpi's ``_CONFIGS_DICT`` at import time, so openpi's ``scripts/train.py`` /
``scripts/compute_norm_stats.py`` resolve them by name when launched via the
wrappers in ``tools/openpi_sft/`` — openpi itself is never edited.

JointController pipeline: every config uses ``Sim2CamLiberoDataConfig`` with
``use_delta_joint_actions=True`` (absolute-joint datasets; 7 arm joints ->
per-step delta, gripper absolute), and warm-starts from ``pi05_base``.

``num_train_steps`` covers ~2 epochs of each family's DATAGEN v1 dataset;
``decay_steps == num_train_steps``; ``keep_period = num_train_steps // 5``.
"""

from __future__ import annotations

import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
from openpi.training.config import DataConfig, TrainConfig

from maniguard.openpi_sft.data_configs import Sim2CamLiberoDataConfig

_PI05_BASE = "gs://openpi-assets/checkpoints/pi05_base/params"
_PI0_BASE = "gs://openpi-assets/checkpoints/pi0_base/params"


def _build_configs() -> list[TrainConfig]:
    return [
        # clutter — pick a named target out of a cluttered tabletop into the goal
        # region. LIBERO 2-cam, JOINT controller. 5-cam dataset consumed 2-cam
        # (external_cam overview + wrist); 8-D joint state/action, delta arm joints.
        # ~2 epochs over 901,520 frames at batch 128 -> 14,100 steps.
        TrainConfig(
            name="pi05-base_datagen_v1_clutter_joint_2cam_lora",
            project_name="maniguard-sft",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-clutter-joint-2cam-lora",
                "hf_private": False,
                "default_exp": "pi05-base_datagen_v1_clutter_joint_2cam_lora",
            },
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
                discrete_state_input=True,
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-clutter-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500,
                peak_lr=7e-5,
                decay_steps=14_100,
                decay_lr=7e-6,
            ),
            num_train_steps=14_100,
            batch_size=128,
            num_workers=48,
            log_interval=100,
            fsdp_devices=1,
            keep_period=2_820,  # steps // 5
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # cabinet — open the drawer, put the target inside, and close it. LIBERO
        # 2-cam, JOINT controller. Same 5-cam->2-cam consumption + joint semantics.
        # ~2 epochs over 4,172,962 frames at batch 128 -> 65,250 steps.
        TrainConfig(
            name="pi05-base_datagen_v1_cabinet_joint_2cam_lora",
            project_name="maniguard-sft",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-cabinet-joint-2cam-lora",
                "hf_private": False,
                "default_exp": "pi05-base_datagen_v1_cabinet_joint_2cam_lora",
            },
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
                discrete_state_input=True,
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-cabinet-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=2_000,
                peak_lr=7e-5,
                decay_steps=65_250,
                decay_lr=7e-6,
            ),
            num_train_steps=65_250,
            batch_size=128,
            num_workers=48,
            log_interval=100,
            fsdp_devices=1,
            keep_period=13_050,  # steps // 5
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # stack — pull the bottom object out from under a stack into the goal
        # region. LIBERO 2-cam, JOINT controller. Same 5-cam->2-cam + joint.
        # ~2 epochs over 2,652,083 frames at batch 128 -> 41,500 steps.
        TrainConfig(
            name="pi05-base_datagen_v1_stack_joint_2cam_lora",
            project_name="maniguard-sft",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-stack-joint-2cam-lora",
                "hf_private": False,
                "default_exp": "pi05-base_datagen_v1_stack_joint_2cam_lora",
            },
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
                discrete_state_input=True,
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-stack-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_250,
                peak_lr=7e-5,
                decay_steps=41_500,
                decay_lr=7e-6,
            ),
            num_train_steps=41_500,
            batch_size=128,
            num_workers=48,
            log_interval=100,
            fsdp_devices=1,
            keep_period=8_300,  # steps // 5
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # jar — close a hinged jar's lid, then carry the closed jar into the goal
        # region. LIBERO 2-cam, JOINT controller. Same 5-cam->2-cam + joint.
        # ~2 epochs over 946,870 frames at batch 128 -> 14,800 steps.
        TrainConfig(
            name="pi05-base_datagen_v1_jar_joint_2cam_lora",
            project_name="maniguard-sft",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-jar-joint-2cam-lora",
                "hf_private": False,
                "default_exp": "pi05-base_datagen_v1_jar_joint_2cam_lora",
            },
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
                discrete_state_input=True,
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-jar-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=450,
                peak_lr=7e-5,
                decay_steps=14_800,
                decay_lr=7e-6,
            ),
            num_train_steps=14_800,
            batch_size=128,
            num_workers=48,
            log_interval=100,
            fsdp_devices=1,
            keep_period=2_960,  # steps // 5
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # lid — place the lid on the container, then carry the closed container
        # into the goal region. LIBERO 2-cam, JOINT controller. Same 5-cam->2-cam.
        # ~2 epochs over 1,055,142 frames at batch 128 -> 16,500 steps.
        TrainConfig(
            name="pi05-base_datagen_v1_lid_joint_2cam_lora",
            project_name="maniguard-sft",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-lid-joint-2cam-lora",
                "hf_private": False,
                "default_exp": "pi05-base_datagen_v1_lid_joint_2cam_lora",
            },
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
                discrete_state_input=True,
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-lid-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500,
                peak_lr=7e-5,
                decay_steps=16_500,
                decay_lr=7e-6,
            ),
            num_train_steps=16_500,
            batch_size=128,
            num_workers=48,
            log_interval=100,
            fsdp_devices=1,
            keep_period=3_300,  # steps // 5
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # dusty — wipe the dusty pot clean with the sponge, then transfer food
        # into it. LIBERO 2-cam, JOINT controller. Same 5-cam->2-cam + joint.
        # ~2 epochs over 1,879,498 frames at batch 128 -> 29,400 steps.
        TrainConfig(
            name="pi05-base_datagen_v1_dusty_joint_2cam_lora",
            project_name="maniguard-sft",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-dusty-joint-2cam-lora",
                "hf_private": False,
                "default_exp": "pi05-base_datagen_v1_dusty_joint_2cam_lora",
            },
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
                discrete_state_input=True,
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-dusty-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=900,
                peak_lr=7e-5,
                decay_steps=29_400,
                decay_lr=7e-6,
            ),
            num_train_steps=29_400,
            batch_size=128,
            num_workers=48,
            log_interval=100,
            fsdp_devices=1,
            keep_period=5_880,  # steps // 5
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # ================= pi0 (base pi0, NOT pi0.5) -- same six families =================
        # Identical data pipeline + 8-GPU scale as the pi05 blocks above (same
        # Sim2CamLiberoDataConfig, delta-joint actions, external_cam, steps, LR,
        # batch, checkpoint cadence). The diffs are exactly the model generation:
        #   * warm-start pi0_base (not pi05_base);
        #   * Pi0Config default pi05=False -> continuous state input
        #     (discrete_state_input auto-resolves False, max_token_len 48);
        #   * action_horizon=50 (pi0's native chunk; the pi05 blocks use 16).
        # Norm stats are computed FRESH under each pi0 config name: the stats pass
        # chunks actions by action_horizon, so the pi05 stats are NOT reused.
        TrainConfig(
            name="pi0-base_datagen_v1_clutter_joint_2cam_lora",
            project_name="maniguard-sft",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi0-base-datagen-v1-clutter-joint-2cam-lora",
                "hf_private": False,
                "default_exp": "datagen_v1_clutter_joint_2cam",
            },
            model=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-clutter-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI0_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=200,
                peak_lr=7e-5,
                decay_steps=7_100,
                decay_lr=7e-6,
            ),
            num_train_steps=7_100,
            batch_size=256,
            num_workers=48,
            log_interval=100,
            fsdp_devices=1,
            save_interval=1_775,
            keep_period=1_775,
            freeze_filter=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        TrainConfig(
            name="pi0-base_datagen_v1_cabinet_joint_2cam_lora",
            project_name="maniguard-sft",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi0-base-datagen-v1-cabinet-joint-2cam-lora",
                "hf_private": False,
                "default_exp": "datagen_v1_cabinet_joint_2cam",
            },
            model=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-cabinet-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI0_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=7e-5,
                decay_steps=32_650,
                decay_lr=7e-6,
            ),
            num_train_steps=32_650,
            batch_size=256,
            num_workers=48,
            log_interval=100,
            fsdp_devices=1,
            save_interval=8_163,
            keep_period=8_163,
            freeze_filter=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        TrainConfig(
            name="pi0-base_datagen_v1_stack_joint_2cam_lora",
            project_name="maniguard-sft",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi0-base-datagen-v1-stack-joint-2cam-lora",
                "hf_private": False,
                "default_exp": "datagen_v1_stack_joint_2cam",
            },
            model=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-stack-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI0_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=650,
                peak_lr=7e-5,
                decay_steps=20_750,
                decay_lr=7e-6,
            ),
            num_train_steps=20_750,
            batch_size=256,
            num_workers=48,
            log_interval=100,
            fsdp_devices=1,
            save_interval=5_188,
            keep_period=5_188,
            freeze_filter=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        TrainConfig(
            name="pi0-base_datagen_v1_jar_joint_2cam_lora",
            project_name="maniguard-sft",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi0-base-datagen-v1-jar-joint-2cam-lora",
                "hf_private": False,
                "default_exp": "datagen_v1_jar_joint_2cam",
            },
            model=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-jar-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI0_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=250,
                peak_lr=7e-5,
                decay_steps=7_400,
                decay_lr=7e-6,
            ),
            num_train_steps=7_400,
            batch_size=256,
            num_workers=48,
            log_interval=100,
            fsdp_devices=1,
            save_interval=1_850,
            keep_period=1_850,
            freeze_filter=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        TrainConfig(
            name="pi0-base_datagen_v1_lid_joint_2cam_lora",
            project_name="maniguard-sft",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi0-base-datagen-v1-lid-joint-2cam-lora",
                "hf_private": False,
                "default_exp": "datagen_v1_lid_joint_2cam",
            },
            model=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-lid-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI0_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=250,
                peak_lr=7e-5,
                decay_steps=8_250,
                decay_lr=7e-6,
            ),
            num_train_steps=8_250,
            batch_size=256,
            num_workers=48,
            log_interval=100,
            fsdp_devices=1,
            save_interval=2_063,
            keep_period=2_063,
            freeze_filter=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        TrainConfig(
            name="pi0-base_datagen_v1_dusty_joint_2cam_lora",
            project_name="maniguard-sft",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi0-base-datagen-v1-dusty-joint-2cam-lora",
                "hf_private": False,
                "default_exp": "datagen_v1_dusty_joint_2cam",
            },
            model=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-dusty-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI0_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=450,
                peak_lr=7e-5,
                decay_steps=14_700,
                decay_lr=7e-6,
            ),
            num_train_steps=14_700,
            batch_size=256,
            num_workers=48,
            log_interval=100,
            fsdp_devices=1,
            save_interval=3_675,
            keep_period=3_675,
            freeze_filter=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
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
