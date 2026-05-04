"""GraspGen-based per-object grasp evaluation + visualisation.

End-to-end pipeline: query the GraspGen ZMQ server for 6-DoF candidates,
run each in OmniGibson + cuRobo, dump diagnostic PNGs / success MP4s.

Modules:

  - ``mesh``               — OG ``BaseObject`` → trimesh (visual or collision).
  - ``graspgen_sampler``   — ZMQ client; ``sample_graspgen_grasps(mesh, ...)``
    returns ``(N, 4, 4)`` poses + per-grasp confidence.
  - ``collector``          — shared sim helpers (pose↔matrix conversions,
    action assembly, controller-goal reset, mimic-tolerant cuRobo motion plan).
  - ``render_grasps``      — main driver; reads a CSV of (category, model)
    rows, runs the eval, writes ``{stem}.mp4`` on success and
    ``{stem}_pcd_*.png`` on failure, plus ``{stem}_grasps_*.png`` always.
  - ``inspect_mesh``       — standalone script: render mesh + optional
    point-cloud PNGs without running the full eval.
  - ``visualize_grasps``   — standalone script: top-K GraspGen grasps
    overlaid on the point cloud (for pre-flight inspection).
  - ``_viz_helpers``       — matplotlib backend for the three scripts above.
  - ``reset``              — ``GraspDatasetResetter`` used by the RL training
    envs (``sentinel.rl.tasks.pick_and_lift`` etc.) — independent of the
    rest of this module, kept here for historical co-location.
"""
