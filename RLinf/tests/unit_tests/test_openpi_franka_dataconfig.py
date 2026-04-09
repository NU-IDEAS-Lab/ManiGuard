import openpi.transforms as _transforms

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


def test_pi05_franka_tabletop_uses_joint_only_delta_mask():
    train_config = get_openpi_config("pi05_franka_tabletop")
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    delta_transforms = [
        transform
        for transform in data_config.data_transforms.inputs
        if isinstance(transform, _transforms.DeltaActions)
    ]
    absolute_transforms = [
        transform
        for transform in data_config.data_transforms.outputs
        if isinstance(transform, _transforms.AbsoluteActions)
    ]

    assert len(delta_transforms) == 1
    assert len(absolute_transforms) == 1
    assert delta_transforms[0].mask == _transforms.make_bool_mask(7, -1)
    assert absolute_transforms[0].mask == _transforms.make_bool_mask(7, -1)
