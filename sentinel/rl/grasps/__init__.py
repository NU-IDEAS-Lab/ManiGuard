"""Grasp-dataset lifecycle: sampling, physical validation, and reset use.

Organises the end-to-end port of UW Lab OmniReset (ICLR 2026) to OmniGibson:

  - ``sampler``          — mesh-based antipodal grasp candidate generation
    (offline, trimesh-only; no sim).
  - ``collector``        — physics-validated grasp dataset builder: cuRobo IK
    → teleport → close → gravity-lift → shake → record.
  - ``mesh``             — OG ``BaseObject`` → trimesh + per-robot gripper
    params (``franka_mounted_gripper_params`` / ``franka_panda_gripper_params``
    / ``gripper_params_for_robot``).
  - ``reset``            — ``GraspDatasetResetter``: per-episode reset that
    teleports arm + gripper into a saved grasp pose via cuRobo IK (``"ik"``
    mode) or cached joint angles (``"cached"`` mode, multi-env-safe), with
    post-gravity Touching verification + retry.
  - ``collect_batch``    — CLI entry point for dataset collection across a
    curated ``(category, model)`` list. Supports ``--scene-file`` so grasps
    are sampled against the same scene RL will train on.
  - ``measure_gripper``  — one-shot utility to record ``finger_offset`` /
    ``max_aperture`` from a running robot USD (used to seed the sampler's
    gripper defaults; kept for when we add new robot embodiments).

The philosophy mirrors OmniReset's: the sampler is permissive, the collector
is strict (physics is the final gate), and at training time the resetter
seeds each episode from the validated dataset so PPO sees positive reward
signal early.
"""
