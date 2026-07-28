"""ManiGuard openpi LoRA SFT TrainConfigs (pi0.5 AND pi0), registered into pristine openpi.

Six task families (clutter, cabinet, stack, jar, lid, dusty) x two model
generations: ``pi05-base_*`` (warm-start pi05_base) and ``pi0-base_*``
(warm-start pi0_base). Each is a fully inline ``TrainConfig`` (model +
freeze_filter written out in full, no shared builders) so the whole recipe is
readable at a glance. ``register()`` inserts them into openpi's
``_CONFIGS_DICT`` at import time, so openpi's ``scripts/train.py`` /
``scripts/compute_norm_stats.py`` resolve them by name when launched via the
wrappers in ``tools/openpi_sft/``; openpi itself is never edited.

JointController pipeline: all configs use ``Sim2CamLiberoDataConfig`` with
``use_delta_joint_actions=True`` (absolute-joint datasets; 7 arm joints ->
per-step delta, gripper absolute).

Scale: **one run owns all 8 GPUs** (pure data parallelism) -- GLOBAL
``batch_size=256`` = 32 samples/card, the measured per-card GPU-saturation
point (larger per-card batches add step time but no throughput).
``fsdp_devices=1``: params replicated per card -- the model fits comfortably,
so parameter sharding (FSDP) would solve a non-problem and pay per-layer
collectives for it. ``dtype=bfloat16`` throughout; the only cross-card traffic
is one trainable-grad all-reduce per step. No XLA memory env needed: JAX's
default preallocation is sufficient. Every config trains 2 epochs.
Steps cover ~2 epochs of each dataset; ``decay_steps == num_train_steps``
(enforced in ``register()``); ``warmup_steps`` ~3%; ``save_interval = keep_period = ceil(steps/4)`` -- a
checkpoint lands every half epoch and every one is a keeper, so exactly 4
checkpoints reach HF per 2-epoch run and no transient save is ever pushed.
``peak_lr = 7e-5`` (proven healthy at global batch 256); ``decay_lr = peak/10``.
Changing ``batch`` requires recomputing steps AND the LR -- prefer the shipped
values over ``--batch``.
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
        # Sim pnp-clutter (pick the target object out of a cluttered tabletop and
        # move it into the green goal sphere), LIBERO 2-cam, JOINT controller.
        # Dataset: IDEAS-Lab-Northwestern/datagen-clutter-v1-joint-5cam
        #   5-cam rendered (image_opposite/left/right/left_shoulder + wrist_image)
        #   but consumed 2-cam: external_cam="left" overview + wrist_image; the
        #   other views dropped, pi0.5's third image slot zero-filled + masked.
        #   8-D joint state + 8-D absolute-joint action; use_delta_joint_actions=True.
        # warm-start = pi05_base. discrete_state_input=True (pi0.5: the 8-D robot
        #   state is discretized + tokenized into the language prefix).
        # Scale: 2 epochs over the 901,520-frame set at GLOBAL batch 256
        #   (901_520 * 2 / 256 = 7,043 -> rounded up to 7,100).
        #   8-GPU pure data parallel: one run owns all 8 cards, 32 samples/card
        #   (the measured per-card sweet spot; larger per-card batches add no
        #   throughput, the GPU is already saturated).
        #   peak_lr 7e-5 = the value proven healthy at global batch 256.
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
            save_interval=1_775,  # checkpoint every half epoch -- with keep_period
            #                  equal, every save is a keeper: exactly 4 checkpoints
            #                  reach HF per 2-epoch run (0.5/1.0/1.5/2.0 epochs)
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
        # Sim cabinet-pickup (open the table-top cabinet drawer, put the target
        # object inside, and close it without knocking anything over), LIBERO 2-cam,
        # JOINT controller. Dataset: IDEAS-Lab-Northwestern/datagen-cabinet-v1-joint-5cam
        #   (same 5-cam->2-cam consumption + joint action semantics as clutter above).
        # Scale: 2 epochs over the 4,172,962-frame set at GLOBAL batch 256
        #   (4_172_962 * 2 / 256 = 32,601 -> rounded up to 32,650).
        #   8-GPU pure data parallel, 32 samples/card; peak_lr 7e-5 = the value
        #   proven healthy at global batch 256.
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
            save_interval=8_163,  # checkpoint every half epoch -- with keep_period
            #                  equal, every save is a keeper: exactly 4 checkpoints
            #                  reach HF per 2-epoch run (0.5/1.0/1.5/2.0 epochs)
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
        # Sim stack-retrieve (unstack the 3 same-object top pile onto a re-stack
        # pile aside, then retrieve the exposed bottom target into the green goal
        # sphere), LIBERO 2-cam, JOINT controller.
        # Dataset: IDEAS-Lab-Northwestern/datagen-stack-v1-joint-5cam (28 base
        #   tasks x 40 = 1120 demos; same 5-cam->2-cam + joint semantics as above).
        # Scale: 2 epochs over the 2,652,083-frame set at GLOBAL batch 256
        #   (2_652_083 * 2 / 256 = 20,719 -> rounded up to 20,750).
        #   8-GPU pure data parallel, 32 samples/card; peak_lr 7e-5 = the value
        #   proven healthy at global batch 256.
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
            save_interval=5_188,  # checkpoint every half epoch -- with keep_period
            #                  equal, every save is a keeper: exactly 4 checkpoints
            #                  reach HF per 2-epoch run (0.5/1.0/1.5/2.0 epochs)
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
        # Sim jar-transport (close a hinged jar's lid, then carry the closed jar
        # into the green goal sphere on the table), LIBERO 2-cam, JOINT controller.
        # Dataset: IDEAS-Lab-Northwestern/datagen-jar-v1-joint-5cam (26 base
        #   tasks x 40 = 1040 demos; same 5-cam->2-cam + joint semantics as above).
        # Scale: 2 epochs over the 946,870-frame set at GLOBAL batch 256
        #   (946_870 * 2 / 256 = 7,397 -> rounded up to 7,400).
        #   8-GPU pure data parallel, 32 samples/card; peak_lr 7e-5 = the value
        #   proven healthy at global batch 256.
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
            save_interval=1_850,  # checkpoint every half epoch -- with keep_period
            #                  equal, every save is a keeper: exactly 4 checkpoints
            #                  reach HF per 2-epoch run (0.5/1.0/1.5/2.0 epochs)
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
        # lid_transport: pick the lid, place it on the container mouth (it auto-snaps),
        #   grasp the now-lidded container, transport it into the goal region. 1200 demos.
        # Config-only (no baked norm-stats) — run_sft.sh computes them on the first run.
        # Scale: 2 epochs over the 1,055,142-frame set at GLOBAL batch 256
        #   (1_055_142 * 2 / 256 = 8,243 -> rounded up to 8,250).
        #   8-GPU pure data parallel, 32 samples/card; peak_lr 7e-5 = the value
        #   proven healthy at global batch 256.
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
            save_interval=2_063,  # checkpoint every half epoch -- with keep_period
            #                  equal, every save is a keeper: exactly 4 checkpoints
            #                  reach HF per 2-epoch run (0.5/1.0/1.5/2.0 epochs)
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
        # dusty_transfer: wipe the dust out of the container with the sponge, return the
        #   sponge, pick the source carrier with the food riding upright, tilt-pour the
        #   food into the container. 1040 demos.
        # Config-only (no baked norm-stats) — run_sft.sh computes them on the first run.
        # Scale: 2 epochs over the 1,879,498-frame set at GLOBAL batch 256
        #   (1_879_498 * 2 / 256 = 14,683 -> rounded up to 14,700).
        #   8-GPU pure data parallel, 32 samples/card; peak_lr 7e-5 = the value
        #   proven healthy at global batch 256.
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
            save_interval=3_675,  # checkpoint every half epoch -- with keep_period
            #                  equal, every save is a keeper: exactly 4 checkpoints
            #                  reach HF per 2-epoch run (0.5/1.0/1.5/2.0 epochs)
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
        # ====== pi0.5 PROMPT ablation (Q2: how the safety constraint is conveyed) ======
        # clutter base only, three conditions that hold the task instruction AND the LTL
        # automaton fixed, varying only how the constraint reaches the policy:
        #   no_instruction    -- the constraint is never stated  = the SHIPPED clutter
        #                        config/checkpoint above; nothing to train here.
        #   natural_language  -- the bench's own `description` clauses, appended.
        #   ltl               -- the bench's own LTL formulas, appended.
        # The two blocks below differ from the 100% clutter block in NOTHING but the
        # dataset (a prompt-rewritten variant) and the run's identity; the trajectories,
        # videos, batch, LR, and 7,100 steps are the same, so any difference in the
        # resulting policy is attributable to the prompt alone.
        # The variant datasets are built by tools/ablation_prompt/build_dataset_variant.py
        # (ManiGuard repo): meta/tasks.jsonl rewritten from
        # configs/ablation_prompt/clutter_base_prompts.json, with data/ + videos/
        # symlinked back to this same source dataset -- no trajectory is duplicated and
        # the source stays read-only. Eval reads its prompts from that same table.
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
