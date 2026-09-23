"""DataConfig factories and observation/action transforms for ManiGuard openpi recipes.

Select an overview stream and wrist stream, map them to the policy input keys,
and configure joint-position action conversion. SubsetDataConfig carries an
optional episode_fraction marker; a compatible data loader must implement the
selection because this factory does not select episodes itself.
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
    """Map one external view and the wrist view into the openpi two-camera layout.

    external_cam selects image_<name>; wrist_image supplies the wrist view. Extra
    dataset cameras are not passed to the policy. For joint-position datasets,
    use_delta_joint_actions enables DeltaActions for seven arm joints and leaves
    the gripper absolute; AbsoluteActions reconstructs inference outputs. The
    alternate branch returns seven action dimensions without this conversion.

    The episode_fraction field is propagated as metadata. Episode selection requires
    a compatible loader; it is not performed by create.
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
    # For datasets with only left/right overviews, select one of those available streams.
    external_cam: str = "left"

    # Optional fraction marker. A compatible data loader must implement episode
    # selection; this factory only validates and propagates the value.
    episode_fraction: float | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.episode_fraction is not None and not (0.0 < self.episode_fraction < 1.0):
            raise ValueError(
                f"episode_fraction must be in (0, 1) or None (full dataset), got {self.episode_fraction}"
            )
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

        cfg = dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
        if self.episode_fraction is None:
            return cfg
        # Preserve the marker on a DataConfig subclass for compatible loaders.
        return SubsetDataConfig(
            **{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)},
            episode_fraction=self.episode_fraction,
        )


@dataclasses.dataclass(frozen=True)
class SubsetDataConfig(DataConfig):
    """DataConfig carrying an episode_fraction marker for a compatible subset-aware loader.

    This dataclass stores the marker and does not filter episodes itself.
    """

    episode_fraction: float | None = None
