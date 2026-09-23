"""Register ManiGuard training and inference configurations with openpi.

Configurations cover pi0 and pi0.5 simulation fine-tuning, zero-shot serving,
data-fraction and prompt variants, single-task simulation studies, and pi0 DROID
real-robot fine-tuning. Each entry specifies its model, dataset, action transform,
initial weights, and training schedule. Simulation recipes use joint-position
actions; the DROID recipes consume joint velocities without a delta transform.

The wrappers in tools/openpi_sft import this package before calling openpi.
register checks that each configuration's cosine decay length equals its training
step budget. CLI overrides and externally provided datasets/checkpoints require
matching normalization assets; this module does not train or publish models.
"""

from __future__ import annotations

import openpi.training.optimizer as _optimizer
from openpi.models import pi0_config
from openpi.training import weight_loaders
from openpi.training.config import AssetsConfig, DataConfig, LeRobotDROIDDataConfig, TrainConfig

from maniguard.openpi_sft.data_configs import Sim2CamLiberoDataConfig

_PI05_BASE = "gs://openpi-assets/checkpoints/pi05_base/params"
_PI0_BASE = "gs://openpi-assets/checkpoints/pi0_base/params"
_PI0_DROID = "gs://openpi-assets/checkpoints/pi0_droid/params"
_PI0_DROID_ASSETS = "gs://openpi-assets/checkpoints/pi0_droid/assets"


def _build_configs() -> list[TrainConfig]:
    return [
        # Clutter simulation: joint-position data, left overview and wrist inputs.
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
                warmup_steps=200,
                peak_lr=7e-5,
                decay_steps=7_100,
                decay_lr=7e-6,
            ),
            num_train_steps=7_100,
            batch_size=256,
            num_workers=48,  # dataloader workers feeding all 8 cards (the thread caps
            #                  in run_sft.sh make each worker a single thread). Sized
            #                  with ample headroom over what the GPUs consume, but it
            #                  MUST stay below the host's physical core count -- verify
            #                  against the actual machine before a long run. Pure perf
            #                  knob (no training-dynamics effect); tune with --num-workers.
            log_interval=100,
            fsdp_devices=1,  # no FSDP sharding: the model fits one card
            save_interval=1_775,  # Save interval in training steps; retention uses keep_period.
            keep_period=1_775,
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # Cabinet simulation: relocate blockers, open, place the target, and close.
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
                warmup_steps=1_000,
                peak_lr=7e-5,
                decay_steps=32_650,
                decay_lr=7e-6,
            ),
            num_train_steps=32_650,
            batch_size=256,
            num_workers=48,  # dataloader workers feeding all 8 cards (the thread caps
            #                  in run_sft.sh make each worker a single thread). Sized
            #                  with ample headroom over what the GPUs consume, but it
            #                  MUST stay below the host's physical core count -- verify
            #                  against the actual machine before a long run. Pure perf
            #                  knob (no training-dynamics effect); tune with --num-workers.
            log_interval=100,
            fsdp_devices=1,  # no FSDP sharding: the model fits one card
            save_interval=8_163,  # Save interval in training steps; retention uses keep_period.
            keep_period=8_163,
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # Stack simulation: relocate upper objects and retrieve the bottom target.
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
                warmup_steps=650,
                peak_lr=7e-5,
                decay_steps=20_750,
                decay_lr=7e-6,
            ),
            num_train_steps=20_750,
            batch_size=256,
            num_workers=48,  # dataloader workers feeding all 8 cards (the thread caps
            #                  in run_sft.sh make each worker a single thread). Sized
            #                  with ample headroom over what the GPUs consume, but it
            #                  MUST stay below the host's physical core count -- verify
            #                  against the actual machine before a long run. Pure perf
            #                  knob (no training-dynamics effect); tune with --num-workers.
            log_interval=100,
            fsdp_devices=1,  # no FSDP sharding: the model fits one card
            save_interval=5_188,  # Save interval in training steps; retention uses keep_period.
            keep_period=5_188,
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # Jar simulation: close the lid and transport the jar to the goal.
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
                warmup_steps=250,
                peak_lr=7e-5,
                decay_steps=7_400,
                decay_lr=7e-6,
            ),
            num_train_steps=7_400,
            batch_size=256,
            num_workers=48,  # dataloader workers feeding all 8 cards (the thread caps
            #                  in run_sft.sh make each worker a single thread). Sized
            #                  with ample headroom over what the GPUs consume, but it
            #                  MUST stay below the host's physical core count -- verify
            #                  against the actual machine before a long run. Pure perf
            #                  knob (no training-dynamics effect); tune with --num-workers.
            log_interval=100,
            fsdp_devices=1,  # no FSDP sharding: the model fits one card
            save_interval=1_850,  # Save interval in training steps; retention uses keep_period.
            keep_period=1_850,
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # Lid simulation: attach the lid and transport the container assembly.
        TrainConfig(
            name="pi05-base_datagen_v1_lid_joint_2cam_lora",
            project_name="maniguard-sft",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-lid-joint-2cam-lora",
                "hf_private": False,
                "default_exp": "datagen_v1_lid_joint_2cam",
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
                warmup_steps=250,
                peak_lr=7e-5,
                decay_steps=8_250,
                decay_lr=7e-6,
            ),
            num_train_steps=8_250,
            batch_size=256,
            num_workers=48,  # dataloader workers feeding all 8 cards (the thread caps
            #                  in run_sft.sh make each worker a single thread). Sized
            #                  with ample headroom over what the GPUs consume, but it
            #                  MUST stay below the host's physical core count -- verify
            #                  against the actual machine before a long run. Pure perf
            #                  knob (no training-dynamics effect); tune with --num-workers.
            log_interval=100,
            fsdp_devices=1,  # no FSDP sharding: the model fits one card
            save_interval=2_063,  # Save interval in training steps; retention uses keep_period.
            keep_period=2_063,
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # Dusty simulation: wipe the destination and transfer food from the source.
        TrainConfig(
            name="pi05-base_datagen_v1_dusty_joint_2cam_lora",
            project_name="maniguard-sft",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-dusty-joint-2cam-lora",
                "hf_private": False,
                "default_exp": "datagen_v1_dusty_joint_2cam",
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
                warmup_steps=450,
                peak_lr=7e-5,
                decay_steps=14_700,
                decay_lr=7e-6,
            ),
            num_train_steps=14_700,
            batch_size=256,
            num_workers=48,  # dataloader workers feeding all 8 cards (the thread caps
            #                  in run_sft.sh make each worker a single thread). Sized
            #                  with ample headroom over what the GPUs consume, but it
            #                  MUST stay below the host's physical core count -- verify
            #                  against the actual machine before a long run. Pure perf
            #                  knob (no training-dynamics effect); tune with --num-workers.
            log_interval=100,
            fsdp_devices=1,  # no FSDP sharding: the model fits one card
            save_interval=3_675,  # Save interval in training steps; retention uses keep_period.
            keep_period=3_675,
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # pi0 simulation fine-tuning: continuous state input and 50-step action chunks.
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
        # pi0.5 inference configurations using base-model weights and task-domain
        # normalization assets. These entries are intended for serving, not SFT.
        TrainConfig(
            name="pi05-zeroshot_datagen_v1_clutter_joint_2cam",
            project_name="maniguard-eval",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
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
        ),
        TrainConfig(
            name="pi05-zeroshot_datagen_v1_cabinet_joint_2cam",
            project_name="maniguard-eval",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
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
        ),
        TrainConfig(
            name="pi05-zeroshot_datagen_v1_stack_joint_2cam",
            project_name="maniguard-eval",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
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
        ),
        TrainConfig(
            name="pi05-zeroshot_datagen_v1_jar_joint_2cam",
            project_name="maniguard-eval",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
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
        ),
        TrainConfig(
            name="pi05-zeroshot_datagen_v1_lid_joint_2cam",
            project_name="maniguard-eval",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
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
        ),
        TrainConfig(
            name="pi05-zeroshot_datagen_v1_dusty_joint_2cam",
            project_name="maniguard-eval",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
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
        ),
        # pi0 inference configurations using base-model weights and task-domain
        # normalization assets. Their action horizon is 50.
        TrainConfig(
            name="pi0-zeroshot_datagen_v1_clutter_joint_2cam",
            project_name="maniguard-eval",
            model=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                dtype="bfloat16",
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-clutter-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI0_BASE),
        ),
        TrainConfig(
            name="pi0-zeroshot_datagen_v1_cabinet_joint_2cam",
            project_name="maniguard-eval",
            model=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                dtype="bfloat16",
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-cabinet-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI0_BASE),
        ),
        TrainConfig(
            name="pi0-zeroshot_datagen_v1_stack_joint_2cam",
            project_name="maniguard-eval",
            model=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                dtype="bfloat16",
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-stack-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI0_BASE),
        ),
        TrainConfig(
            name="pi0-zeroshot_datagen_v1_jar_joint_2cam",
            project_name="maniguard-eval",
            model=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                dtype="bfloat16",
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-jar-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI0_BASE),
        ),
        TrainConfig(
            name="pi0-zeroshot_datagen_v1_lid_joint_2cam",
            project_name="maniguard-eval",
            model=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                dtype="bfloat16",
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-lid-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI0_BASE),
        ),
        TrainConfig(
            name="pi0-zeroshot_datagen_v1_dusty_joint_2cam",
            project_name="maniguard-eval",
            model=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                dtype="bfloat16",
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-dusty-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI0_BASE),
        ),
        # ============ pi0.5 DATA-SCALING ablation (clutter + cabinet x 20/50/80%) ============
        # Same recipe as the corresponding pi05 100% blocks above in EVERY field; the
        # only experimental variable is the data budget:
        #   * data.episode_fraction: per-BASE-TASK subset — each base task stores 40
        #     consecutive episodes; a fraction keeps the FIRST ceil(40*f) of every
        #     40-block (0.2->8, 0.5->20, 0.8->32). Task coverage unchanged; applied
        #     read-only at load time (_episode_subset_patch), norm stats computed on
        #     each config's own subset under its own config name.
        #   * scale: fixed 2 EPOCHS of the subset -> steps shrink with data;
        #     warmup ~3%, save=keep=ceil(steps/4), decay==steps (guard in register()).
        # hf_repo (incl. -yanZ) and the scaling wandb project are BAKED IN so a run
        # with no --push-repo/--project flags still lands in the right places.
        # Subset sizes (exact, from episodes.jsonl): clutter 179,598 / 451,730 /
        # 721,590 frames; cabinet 829,803 / 2,084,892 / 3,339,452 frames.
        TrainConfig(
            name="pi05-base_datagen_v1_clutter_joint_2cam_lora_p20",
            project_name="maniguard-sft-scaling-yanZ",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-clutter-joint-2cam-lora-p20-yanZ",
                "hf_private": False,
                "default_exp": "datagen_v1_clutter_joint_2cam_p20",
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
                episode_fraction=0.2,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=50,
                peak_lr=7e-5,
                decay_steps=1_410,
                decay_lr=7e-6,
            ),
            num_train_steps=1_410,
            batch_size=256,
            num_workers=48,
            log_interval=100,
            fsdp_devices=1,
            save_interval=353,
            keep_period=353,
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        TrainConfig(
            name="pi05-base_datagen_v1_clutter_joint_2cam_lora_p50",
            project_name="maniguard-sft-scaling-yanZ",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-clutter-joint-2cam-lora-p50-yanZ",
                "hf_private": False,
                "default_exp": "datagen_v1_clutter_joint_2cam_p50",
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
                episode_fraction=0.5,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=110,
                peak_lr=7e-5,
                decay_steps=3_530,
                decay_lr=7e-6,
            ),
            num_train_steps=3_530,
            batch_size=256,
            num_workers=48,
            log_interval=100,
            fsdp_devices=1,
            save_interval=883,
            keep_period=883,
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        TrainConfig(
            name="pi05-base_datagen_v1_clutter_joint_2cam_lora_p80",
            project_name="maniguard-sft-scaling-yanZ",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-clutter-joint-2cam-lora-p80-yanZ",
                "hf_private": False,
                "default_exp": "datagen_v1_clutter_joint_2cam_p80",
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
                episode_fraction=0.8,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=170,
                peak_lr=7e-5,
                decay_steps=5_640,
                decay_lr=7e-6,
            ),
            num_train_steps=5_640,
            batch_size=256,
            num_workers=48,
            log_interval=100,
            fsdp_devices=1,
            save_interval=1_410,
            keep_period=1_410,
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        TrainConfig(
            name="pi05-base_datagen_v1_cabinet_joint_2cam_lora_p20",
            project_name="maniguard-sft-scaling-yanZ",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-cabinet-joint-2cam-lora-p20-yanZ",
                "hf_private": False,
                "default_exp": "datagen_v1_cabinet_joint_2cam_p20",
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
                episode_fraction=0.2,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=200,
                peak_lr=7e-5,
                decay_steps=6_490,
                decay_lr=7e-6,
            ),
            num_train_steps=6_490,
            batch_size=256,
            num_workers=48,
            log_interval=100,
            fsdp_devices=1,
            save_interval=1_623,
            keep_period=1_623,
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        TrainConfig(
            name="pi05-base_datagen_v1_cabinet_joint_2cam_lora_p50",
            project_name="maniguard-sft-scaling-yanZ",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-cabinet-joint-2cam-lora-p50-yanZ",
                "hf_private": False,
                "default_exp": "datagen_v1_cabinet_joint_2cam_p50",
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
                episode_fraction=0.5,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=490,
                peak_lr=7e-5,
                decay_steps=16_290,
                decay_lr=7e-6,
            ),
            num_train_steps=16_290,
            batch_size=256,
            num_workers=48,
            log_interval=100,
            fsdp_devices=1,
            save_interval=4_073,
            keep_period=4_073,
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        TrainConfig(
            name="pi05-base_datagen_v1_cabinet_joint_2cam_lora_p80",
            project_name="maniguard-sft-scaling-yanZ",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-cabinet-joint-2cam-lora-p80-yanZ",
                "hf_private": False,
                "default_exp": "datagen_v1_cabinet_joint_2cam_p80",
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
                episode_fraction=0.8,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=790,
                peak_lr=7e-5,
                decay_steps=26_090,
                decay_lr=7e-6,
            ),
            num_train_steps=26_090,
            batch_size=256,
            num_workers=48,
            log_interval=100,
            fsdp_devices=1,
            save_interval=6_523,
            keep_period=6_523,
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # Prompt-format variants use the family training recipe with rewritten prompts.
        # Dataset preparation links the same trajectory/video files and rewrites the
        # task table. Use the matching family prompt map for training and evaluation.
        TrainConfig(
            name="pi05-base_datagen_v1_clutter_joint_2cam_lora_promptnl",
            project_name="maniguard-sft-promptablation-yanZ",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-clutter-joint-2cam-lora-promptnl-yanZ",
                "hf_private": False,
                "default_exp": "datagen_v1_clutter_joint_2cam_promptnl",
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
                repo_id="IDEAS-Lab-Northwestern/datagen-clutter-v1-joint-5cam-promptnl",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
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
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        TrainConfig(
            name="pi05-base_datagen_v1_clutter_joint_2cam_lora_promptltl",
            project_name="maniguard-sft-promptablation-yanZ",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-clutter-joint-2cam-lora-promptltl-yanZ",
                "hf_private": False,
                "default_exp": "datagen_v1_clutter_joint_2cam_promptltl",
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
                repo_id="IDEAS-Lab-Northwestern/datagen-clutter-v1-joint-5cam-promptltl",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
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
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        TrainConfig(
            # jar / natural_language -- identical to the mainline jar block in every field except
            # the rewritten dataset and the run identity, so the only variable is the prompt.
            name="pi05-base_datagen_v1_jar_joint_2cam_lora_promptnl",
            project_name="maniguard-sft-promptablation-yanZ",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-jar-joint-2cam-lora-promptnl-yanZ",
                "hf_private": False,
                "default_exp": "datagen_v1_jar_joint_2cam_promptnl",
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
                repo_id="IDEAS-Lab-Northwestern/datagen-jar-v1-joint-5cam-promptnl",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
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
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        TrainConfig(
            # jar / ltl -- identical to the mainline jar block in every field except
            # the rewritten dataset and the run identity, so the only variable is the prompt.
            name="pi05-base_datagen_v1_jar_joint_2cam_lora_promptltl",
            project_name="maniguard-sft-promptablation-yanZ",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-jar-joint-2cam-lora-promptltl-yanZ",
                "hf_private": False,
                "default_exp": "datagen_v1_jar_joint_2cam_promptltl",
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
                repo_id="IDEAS-Lab-Northwestern/datagen-jar-v1-joint-5cam-promptltl",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
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
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        TrainConfig(
            # stack / natural_language -- identical to the mainline stack block in every field except
            # the rewritten dataset and the run identity, so the only variable is the prompt.
            name="pi05-base_datagen_v1_stack_joint_2cam_lora_promptnl",
            project_name="maniguard-sft-promptablation-yanZ",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-stack-joint-2cam-lora-promptnl-yanZ",
                "hf_private": False,
                "default_exp": "datagen_v1_stack_joint_2cam_promptnl",
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
                repo_id="IDEAS-Lab-Northwestern/datagen-stack-v1-joint-5cam-promptnl",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
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
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        TrainConfig(
            # stack / ltl -- identical to the mainline stack block in every field except
            # the rewritten dataset and the run identity, so the only variable is the prompt.
            name="pi05-base_datagen_v1_stack_joint_2cam_lora_promptltl",
            project_name="maniguard-sft-promptablation-yanZ",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-stack-joint-2cam-lora-promptltl-yanZ",
                "hf_private": False,
                "default_exp": "datagen_v1_stack_joint_2cam_promptltl",
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
                repo_id="IDEAS-Lab-Northwestern/datagen-stack-v1-joint-5cam-promptltl",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
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
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # Single-task pi0.5 simulation configurations. Each uses a 50,000-step budget,
        # task-specific batch size, and checkpoints every 10,000 steps. Cabinet has
        # full-task and drawer-opening variants with corresponding instructions.
        TrainConfig(
            # clutter task_0048 -- the sim scene aligned with the real clutter setup.
            # 23,904 frames / batch 4 -> 8.4 epochs over 50,000 steps.
            name="pi05-base_sim2real_clutter_task0048_sim_lora",
            project_name="maniguard-sft-sim2real-yanZ",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-sim2real-clutter-task0048-sim-lora",
                "hf_private": False,
                "default_exp": "sim2real_clutter_task0048_sim",
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
                repo_id="IDEAS-Lab-Northwestern/sim2real-clutter-task0048-sim-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=2.5e-5,
                decay_steps=50_000,
                decay_lr=2.5e-6,
            ),
            num_train_steps=50_000,
            batch_size=4,
            num_workers=8,
            log_interval=100,
            fsdp_devices=1,
            save_interval=10_000,
            keep_period=10_000,
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        TrainConfig(
            # jar task_0016 -- the sim scene aligned with the real jar setup.
            # 55,404 frames / batch 8 -> 7.2 epochs over 50,000 steps.
            name="pi05-base_sim2real_jar_task0016_sim_lora",
            project_name="maniguard-sft-sim2real-yanZ",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-sim2real-jar-task0016-sim-lora",
                "hf_private": False,
                "default_exp": "sim2real_jar_task0016_sim",
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
                repo_id="IDEAS-Lab-Northwestern/sim2real-jar-task0016-sim-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=3.5e-5,
                decay_steps=50_000,
                decay_lr=3.5e-6,
            ),
            num_train_steps=50_000,
            batch_size=8,
            num_workers=8,
            log_interval=100,
            fsdp_devices=1,
            save_interval=10_000,
            keep_period=10_000,
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        TrainConfig(
            # cabinet task_0019, FULL horizon (open -> place -> close) <-> real `higherZ`.
            # 190,701 frames / batch 32 -> 8.4 epochs over 50,000 steps.
            name="pi05-base_sim2real_cabinet_task0019_sim_lora",
            project_name="maniguard-sft-sim2real-yanZ",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-sim2real-cabinet-task0019-sim-lora",
                "hf_private": False,
                "default_exp": "sim2real_cabinet_task0019_sim",
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
                repo_id="IDEAS-Lab-Northwestern/sim2real-cabinet-task0019-sim-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=7e-5,
                decay_steps=50_000,
                decay_lr=7e-6,
            ),
            num_train_steps=50_000,
            batch_size=32,
            num_workers=8,
            log_interval=100,
            fsdp_devices=1,
            save_interval=10_000,
            keep_period=10_000,
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        TrainConfig(
            # cabinet task_0019, FIRSTHALF (blocker aside + drawer open) <-> real `higher-firsthalf`.
            # 126,982 frames / batch 16 -> 6.3 epochs over 50,000 steps.
            name="pi05-base_sim2real_cabinet_task0019_firsthalf_sim_lora",
            project_name="maniguard-sft-sim2real-yanZ",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-sim2real-cabinet-task0019-firsthalf-sim-lora",
                "hf_private": False,
                "default_exp": "sim2real_cabinet_task0019_firsthalf_sim",
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
                repo_id="IDEAS-Lab-Northwestern/sim2real-cabinet-task0019-firsthalf-sim-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=5e-5,
                decay_steps=50_000,
                decay_lr=5e-6,
            ),
            num_train_steps=50_000,
            batch_size=16,
            num_workers=8,
            log_interval=100,
            fsdp_devices=1,
            save_interval=10_000,
            keep_period=10_000,
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # Single-task pi0 simulation configurations using the same datasets and schedules
        # as their pi0.5 counterparts, with pi0 base weights and a 50-step horizon.
        # Their action horizon differs from the real DROID recipes below.
        TrainConfig(
            # clutter task_0048 -- the sim scene aligned with the real clutter setup.
            # 23,904 frames / batch 4 -> 8.4 epochs over 50,000 steps.
            name="pi0-base_sim2real_clutter_task0048_sim_lora",
            project_name="maniguard-sft-sim2real-yanZ",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi0-sim2real-clutter-task0048-sim-lora",
                "hf_private": False,
                "default_exp": "sim2real_clutter_task0048_sim_pi0",
            },
            model=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/sim2real-clutter-task0048-sim-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI0_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=2.5e-5,
                decay_steps=50_000,
                decay_lr=2.5e-6,
            ),
            num_train_steps=50_000,
            batch_size=4,
            num_workers=8,
            log_interval=100,
            fsdp_devices=1,
            save_interval=10_000,
            keep_period=10_000,
            freeze_filter=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        TrainConfig(
            # jar task_0016 -- the sim scene aligned with the real jar setup.
            # 55,404 frames / batch 8 -> 7.2 epochs over 50,000 steps.
            name="pi0-base_sim2real_jar_task0016_sim_lora",
            project_name="maniguard-sft-sim2real-yanZ",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi0-sim2real-jar-task0016-sim-lora",
                "hf_private": False,
                "default_exp": "sim2real_jar_task0016_sim_pi0",
            },
            model=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/sim2real-jar-task0016-sim-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI0_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=3.5e-5,
                decay_steps=50_000,
                decay_lr=3.5e-6,
            ),
            num_train_steps=50_000,
            batch_size=8,
            num_workers=8,
            log_interval=100,
            fsdp_devices=1,
            save_interval=10_000,
            keep_period=10_000,
            freeze_filter=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        TrainConfig(
            # cabinet task_0019, FULL horizon (open -> place -> close) <-> real `higherZ`.
            # 190,701 frames / batch 32 -> 8.4 epochs over 50,000 steps.
            name="pi0-base_sim2real_cabinet_task0019_sim_lora",
            project_name="maniguard-sft-sim2real-yanZ",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi0-sim2real-cabinet-task0019-sim-lora",
                "hf_private": False,
                "default_exp": "sim2real_cabinet_task0019_sim_pi0",
            },
            model=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/sim2real-cabinet-task0019-sim-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI0_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=7e-5,
                decay_steps=50_000,
                decay_lr=7e-6,
            ),
            num_train_steps=50_000,
            batch_size=32,
            num_workers=8,
            log_interval=100,
            fsdp_devices=1,
            save_interval=10_000,
            keep_period=10_000,
            freeze_filter=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        TrainConfig(
            # cabinet task_0019, FIRSTHALF (blocker aside + drawer open) <-> real `higher-firsthalf`.
            # 126,982 frames / batch 16 -> 6.3 epochs over 50,000 steps.
            name="pi0-base_sim2real_cabinet_task0019_firsthalf_sim_lora",
            project_name="maniguard-sft-sim2real-yanZ",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi0-sim2real-cabinet-task0019-firsthalf-sim-lora",
                "hf_private": False,
                "default_exp": "sim2real_cabinet_task0019_firsthalf_sim_pi0",
            },
            model=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/sim2real-cabinet-task0019-firsthalf-sim-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI0_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=5e-5,
                decay_steps=50_000,
                decay_lr=5e-6,
            ),
            num_train_steps=50_000,
            batch_size=16,
            num_workers=8,
            log_interval=100,
            fsdp_devices=1,
            save_interval=10_000,
            keep_period=10_000,
            freeze_filter=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=50,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # Real-robot DROID recipes use joint velocities, not joint-position deltas.
        # Reuse the configured pi0_droid normalization assets without recomputing
        # local statistics. These recipes use a 10-step horizon, batch size 4,
        # 50,000 updates, and a 10,000-step checkpoint interval.
        TrainConfig(
            name="pi0-droid_real_cab_higher_firsthalf_60_refined_lora",
            project_name="maniguard-sft-real-yanZ",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi0-real-cab-higher-firsthalf-60-droid-refined-lora",
                "hf_private": False,
                "default_exp": "real_cab_higher_firsthalf_60_refined",
            },
            model=pi0_config.Pi0Config(
                pi05=False,
                action_dim=32,
                action_horizon=10,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
            ),
            data=LeRobotDROIDDataConfig(
                repo_id="IDEAS-Lab-Northwestern/real-cab-higher-firsthalf-60-droid-refined",
                base_config=DataConfig(prompt_from_task=True),
                assets=AssetsConfig(assets_dir=_PI0_DROID_ASSETS, asset_id="droid"),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI0_DROID),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=2.5e-5,
                decay_steps=50_000,
                decay_lr=2.5e-6,
            ),
            num_train_steps=50_000,
            batch_size=4,
            num_workers=8,
            log_interval=100,
            fsdp_devices=1,
            save_interval=10_000,
            keep_period=10_000,
            freeze_filter=pi0_config.Pi0Config(
                pi05=False,
                action_dim=32,
                action_horizon=10,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        TrainConfig(
            name="pi0-droid_real_jar_60_refined_lora",
            project_name="maniguard-sft-real-yanZ",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi0-real-jar-60-droid-refined-lora",
                "hf_private": False,
                "default_exp": "real_jar_60_refined",
            },
            model=pi0_config.Pi0Config(
                pi05=False,
                action_dim=32,
                action_horizon=10,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
            ),
            data=LeRobotDROIDDataConfig(
                repo_id="IDEAS-Lab-Northwestern/real-jar-60-droid-refined",
                base_config=DataConfig(prompt_from_task=True),
                assets=AssetsConfig(assets_dir=_PI0_DROID_ASSETS, asset_id="droid"),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI0_DROID),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=2.5e-5,
                decay_steps=50_000,
                decay_lr=2.5e-6,
            ),
            num_train_steps=50_000,
            batch_size=4,
            num_workers=8,
            log_interval=100,
            fsdp_devices=1,
            save_interval=10_000,
            keep_period=10_000,
            freeze_filter=pi0_config.Pi0Config(
                pi05=False,
                action_dim=32,
                action_horizon=10,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        TrainConfig(
            name="pi0-droid_real_clutter_60_refined_lora",
            project_name="maniguard-sft-real-yanZ",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi0-real-clutter-60-droid-refined-lora",
                "hf_private": False,
                "default_exp": "real_clutter_60_refined",
            },
            model=pi0_config.Pi0Config(
                pi05=False,
                action_dim=32,
                action_horizon=10,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
            ),
            data=LeRobotDROIDDataConfig(
                repo_id="IDEAS-Lab-Northwestern/real-clutter-60-droid-refined",
                base_config=DataConfig(prompt_from_task=True),
                assets=AssetsConfig(assets_dir=_PI0_DROID_ASSETS, asset_id="droid"),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI0_DROID),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=2.5e-5,
                decay_steps=50_000,
                decay_lr=2.5e-6,
            ),
            num_train_steps=50_000,
            batch_size=4,
            num_workers=8,
            log_interval=100,
            fsdp_devices=1,
            save_interval=10_000,
            keep_period=10_000,
            freeze_filter=pi0_config.Pi0Config(
                pi05=False,
                action_dim=32,
                action_horizon=10,
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
        # The cosine schedule must span exactly the run: a decay_steps that outlives
        # num_train_steps silently stops training mid-anneal at a far-too-high LR.
        if cfg.lr_schedule.decay_steps != cfg.num_train_steps:
            raise ValueError(
                f"{cfg.name}: decay_steps ({cfg.lr_schedule.decay_steps}) must equal "
                f"num_train_steps ({cfg.num_train_steps}); the LR would not finish decaying."
            )
        _CONFIGS_DICT[cfg.name] = cfg
