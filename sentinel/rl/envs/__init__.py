"""OmniGibson environment config + SB3 wrappers.

  - ``grasp_reset_scene.build_config`` — produces the OG ``Environment``
    config dict for the grasp-reset PickAndLiftTask setup (both scene-file
    and runtime-spawn modes, + JointController scene-patch side-effect).
  - ``wrappers.SentinelSB3VectorEnvironment`` — low-level OG-vec → SB3-vec
    adapter with numpy/torch boundary fixes vs upstream.
  - ``wrappers.build_vec_env`` — high-level constructor that turns parsed
    CLI args into a ready-to-train SB3 VecEnv. Every algorithm entry point
    calls this.
"""
