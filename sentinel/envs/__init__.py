from sentinel.envs.embodiment_profile import (
    FRANKA_TABLETOP_SINGLE_ARM_V1,
    SentinelEmbodimentProfile,
    get_sentinel_embodiment_profile,
)

# SentinelEnv pulls in rlinf, which only lives in the RLinf uv venv. Keep the
# import optional so consumers in the behavior conda env (eval, RL training,
# scene utilities) can still use this package.
try:
    from sentinel.envs.sentinel_env import SentinelEnv  # noqa: F401
except ModuleNotFoundError:
    SentinelEnv = None  # type: ignore[assignment]

__all__ = [
    "FRANKA_TABLETOP_SINGLE_ARM_V1",
    "SentinelEmbodimentProfile",
    "SentinelEnv",
    "get_sentinel_embodiment_profile",
]
