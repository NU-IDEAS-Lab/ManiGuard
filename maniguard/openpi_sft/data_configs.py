"""ManiGuard-owned openpi ``DataConfig`` factories for SFT.

Kept in ManiGuard (not appended to openpi's vendored ``config.py``) so openpi
stays a pristine, parallel clone. ``train_configs.register`` attaches the
TrainConfigs that use these into openpi's registry at import time.
"""

from __future__ import annotations

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from maniguard.openpi_sft.policies import sim_2cam_policy


@dataclasses.dataclass(frozen=True)
class Sim2CamLiberoDataConfig(DataConfigFactory):
    """LeRobot config for sim datasets under the LIBERO 2-cam convention.

    The dataset is rendered with three image streams (image_left / image_right /
    wrist_image), but only ``image_left`` (external overview) + ``wrist_image``
    are fed to the policy; ``image_right`` is dropped and pi0.5's third image
    slot is zero-filled and masked off (see :mod:`sim_2cam_policy`). This keeps
    the policy input identical to openpi's stock ``LeRobotLiberoDataConfig`` for
    any sim task — only the dataset (repo_id) and action semantics vary.

    ``use_delta_joint_actions`` selects the action representation:
      * True (default) — JOINT datasets: 8-D joint state + 8-D absolute-joint
        actions; the 7 arm joints are converted to per-step deltas (gripper kept
        absolute) before the model, and reconstructed to absolute at inference.
        Mirrors openpi's RLDSDroidDataConfig JOINT_POSITION handling so eval can
        feed the reconstructed absolute joint target straight to a
        JointController (no eef->joint IK).
      * False           — EEF-delta datasets: 8-D eef state + 7-D EEF-delta
        actions, no extra action transform.

    ManiGuard runs a **JointController end-to-end** (collection -> render ->
    SFT -> eval), so the default is ``True`` and every ManiGuard task config is
    expected to keep it ``True``. The ``False`` (eef) branch is retained only so
    the class stays general; do not use it for the joint-controller pipeline.
    """

    # Convert absolute joint-position actions to per-step deltas for the 7 arm
    # joints (gripper kept absolute) before the model. MUST stay True for the
    # JointController pipeline (our datasets are absolute-joint); False would
    # mis-interpret them as 7-D EEF-delta.
    use_delta_joint_actions: bool = True

    # Which third-person overview to feed the policy. The datagen dataset ships ALL
    # FOUR bench third-person views (image_opposite / image_left / image_right /
    # image_left_shoulder); exactly one is consumed as the policy's single overview
    # (the rest dropped, third pi0.5 slot stays zero+masked — see sim_2cam_policy).
    # Per task/family one view may be higher quality, so this picks the good one at
    # train time; eval must read the same choice back from the checkpoint's train
    # config to stay in distribution. The policy's input key is ALWAYS
    # ``observation/image_left`` (a fixed contract); this only changes WHICH dataset
    # stream feeds that key: ``"<cam>" -> image_<cam> -> observation/image_left`` for
    # cam in {opposite, left, right, left_shoulder}.
    # NOTE: legacy datasets ship only image_left/image_right — use one of those there.
    external_cam: str = "left"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.external_cam not in ("opposite", "left", "right", "left_shoulder"):
            raise ValueError(
                "external_cam must be one of opposite/left/right/left_shoulder, "
                f"got {self.external_cam!r}"
            )
        # Route the chosen dataset overview into the fixed policy key. The key
        # name stays observation/image_left regardless, so the policy + server
        # (Sim2CamInputs) are unchanged; only the source stream differs. The four
        # bench views are named ``image_<cam>``, so the mapping is uniform.
        overview_stream = f"image_{self.external_cam}"
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image_left": overview_stream,
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        action_dim = 8 if self.use_delta_joint_actions else 7
        data_transforms = _transforms.Group(
            inputs=[sim_2cam_policy.Sim2CamInputs(model_type=model_config.model_type)],
            outputs=[sim_2cam_policy.Sim2CamOutputs(action_dim=action_dim)],
        )

        if self.use_delta_joint_actions:
            # Absolute joint-position actions -> per-step delta for the 7 arm
            # joints (gripper absolute). Reconstructed to absolute at inference.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
