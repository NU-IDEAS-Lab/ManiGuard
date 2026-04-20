"""Grasp-dataset lifecycle: sampling, physical validation, and reset use.

Organizes the end-to-end port of UW Lab OmniReset (ICLR 2026) to OmniGibson:

  - ``sampler``    — mesh-based antipodal grasp candidate generation (offline,
    trimesh-only; no sim).
  - ``collector``  — physics-validated grasp dataset builder: cuRobo IK →
    teleport → close → gravity-lift → shake → record.
  - ``mesh``       — OG ``BaseObject`` → trimesh + per-robot gripper params
    (``franka_mounted_gripper_params`` / ``franka_panda_gripper_params`` /
    ``gripper_params_for_robot``).
  - ``reset``      — ``GraspDatasetResetter``: per-episode reset that teleports
    the arm+gripper into a saved grasp pose via cuRobo IK + per-body gravity
    management + post-gravity Touching verification with retry.
  - ``collect_scene`` / ``collect_batch`` — CLI entry points for dataset
    collection on benchmark scenes or curated ``(category, model)`` lists.
  - ``measure_gripper`` — one-shot utility to record ``finger_offset`` /
    ``max_aperture`` from a running robot USD.
  - ``smoke_test_reset`` — diagnostic: N calls to ``env.reset()`` with the
    resetter enabled, reports Touching rate + eef↔target distance.

The philosophy mirrors OmniReset's: the sampler is permissive, the collector
is strict (physics is the final gate), and at training time the resetter
seeds each episode from the validated dataset so PPO sees positive reward
signal early.
"""
