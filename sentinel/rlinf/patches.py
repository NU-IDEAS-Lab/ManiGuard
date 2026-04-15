"""Runtime patches that teach upstream RLinf about Sentinel.

We deliberately keep RLinf as an unmodified vendor / dependency. To still
have RLinf dispatch on ``env_type: sentinel`` and bring up our
``SentinelEnv``, we mutate three RLinf surfaces at import time:

1. ``rlinf.envs.SupportedEnvType`` — extend the Enum with ``SENTINEL``.
2. ``rlinf.envs.get_env_cls`` — wrap the factory to return our
   ``SentinelEnv`` for the sentinel env_type.
3. ``rlinf.config.validate_embodied_cfg`` — wrap to inject the
   ``omnigibson_cfg`` block when env_type is sentinel (mirrors what the
   in-tree BEHAVIOR branch does for r1pro_behavior.yaml).

All patches are idempotent. Importing this module multiple times is
safe; importing ``sentinel`` triggers it.

Why monkey-patch instead of upstreaming a register hook? RLinf doesn't
expose one for env types yet, only for SupportedModel. If/when upstream
adds a ``SupportedEnvType.register()`` API, we can drop the enum hack.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_SENTINEL_ENV_VALUE = "sentinel"
_SENTINEL_ENUM_NAME = "SENTINEL"


def _patch_supported_env_type() -> None:
    """Add a ``SENTINEL`` member to RLinf's ``SupportedEnvType`` enum.

    Python Enums aren't formally extensible. We inject via the same
    private machinery the Enum metaclass uses, which works on CPython
    3.10+. After this returns, both ``SupportedEnvType("sentinel")`` and
    ``SupportedEnvType.SENTINEL`` resolve to the new member.
    """
    from rlinf.envs import SupportedEnvType

    if _SENTINEL_ENUM_NAME in SupportedEnvType.__members__:
        return  # already patched
    new_member = object.__new__(SupportedEnvType)
    new_member._name_ = _SENTINEL_ENUM_NAME
    new_member._value_ = _SENTINEL_ENV_VALUE
    SupportedEnvType._member_map_[_SENTINEL_ENUM_NAME] = new_member
    SupportedEnvType._value2member_map_[_SENTINEL_ENV_VALUE] = new_member
    if hasattr(SupportedEnvType, "_member_names_"):
        SupportedEnvType._member_names_.append(_SENTINEL_ENUM_NAME)
    # NOTE: no setattr(cls, name, member) -- EnumMeta.__setattr__ guards
    # against reassigning anything already in _member_map_. Attribute
    # access `SupportedEnvType.SENTINEL` still works because EnumMeta's
    # __getattr__ falls back to _member_map_ when normal lookup fails.


def _patch_get_env_cls() -> None:
    """Wrap ``rlinf.envs.get_env_cls`` so ``env_type='sentinel'`` works.

    The wrapper short-circuits before delegating to upstream, so the
    upstream factory's giant if/elif (which has no SENTINEL branch)
    never has to be modified.
    """
    import rlinf.envs as _rlinf_envs

    if getattr(_rlinf_envs.get_env_cls, "_sentinel_patched", False):
        return  # already patched

    _orig = _rlinf_envs.get_env_cls

    def _patched(env_type, env_cfg=None):
        if isinstance(env_type, str) and env_type == _SENTINEL_ENV_VALUE:
            from sentinel.envs.sentinel_env import SentinelEnv
            return SentinelEnv
        from rlinf.envs import SupportedEnvType
        if isinstance(env_type, SupportedEnvType) and env_type.value == _SENTINEL_ENV_VALUE:
            from sentinel.envs.sentinel_env import SentinelEnv
            return SentinelEnv
        return _orig(env_type, env_cfg)

    _patched._sentinel_patched = True
    _rlinf_envs.get_env_cls = _patched


def _patch_validate_embodied_cfg() -> None:
    """Inject ``omnigibson_cfg`` for sentinel envs before upstream validate.

    Mirrors the BEHAVIOR / r1pro_behavior branch RLinf already has, but
    uses ``franka_mounted_sentinel.yaml`` and only sets ``rgb`` + ``proprio``
    obs modalities (Sentinel's runtime doesn't need depth).
    """
    import rlinf.config as _rlinf_config

    if getattr(_rlinf_config.validate_embodied_cfg, "_sentinel_patched", False):
        return

    _orig = _rlinf_config.validate_embodied_cfg

    def _patched(cfg):
        env_train_type = cfg.env.train.env_type
        env_eval_type = cfg.env.eval.env_type
        is_sentinel = (
            env_train_type == _SENTINEL_ENV_VALUE
            or env_eval_type == _SENTINEL_ENV_VALUE
        )
        if is_sentinel:
            import omnigibson as og
            import yaml
            from omegaconf import OmegaConf, open_dict

            base_name = cfg.env.train.get("base_config_name", "franka_mounted_sentinel")
            assert base_name == "franka_mounted_sentinel", (
                "Only franka_mounted_sentinel is supported for sentinel envs, "
                f"got {base_name}"
            )
            config_filename = os.path.join(
                og.example_config_path, "franka_mounted_sentinel.yaml"
            )
            omnigibson_cfg = yaml.load(
                open(config_filename, "r"), Loader=yaml.FullLoader
            )
            omnigibson_cfg = OmegaConf.create(omnigibson_cfg)
            with open_dict(omnigibson_cfg):
                omnigibson_cfg.robots[0].obs_modalities = ["rgb", "proprio"]
            cfg.env.train.omnigibson_cfg = omnigibson_cfg
            cfg.env.eval.omnigibson_cfg = omnigibson_cfg
        return _orig(cfg)

    _patched._sentinel_patched = True
    _rlinf_config.validate_embodied_cfg = _patched


def apply_all_patches() -> None:
    """Apply every Sentinel patch to RLinf. Idempotent."""
    _patch_supported_env_type()
    _patch_get_env_cls()
    _patch_validate_embodied_cfg()
    logger.debug("Sentinel patches applied to RLinf.")


# Apply on import so `import sentinel` is enough.
apply_all_patches()
