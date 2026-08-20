"""ManiGuard LingBot-VLA 2.0 embodiment contract: how the 5-cam datagen LeRobot export
maps into the 2-cam, 8-D joint interface LingBot-VLA 2.0 post-training expects.

This is the single source of truth for the LingBot data mapping. The contract is
**pure YAML, not Python**: LingBot registers a new robot declaratively, so there is
nothing to import at runtime.

The LingBot-VLA training code is consumed as a pristine, separate clone of upstream
``robbyant/lingbot-vla-v2``. The two files here plug into that clone at:

    robot_config.yaml  ->  configs/robot_configs/maniguard.yaml
    train_config.yaml  ->  configs/vla/maniguard/maniguard.yaml

Driver scripts live in ``tools/lingbot_sft/`` and run inside that clone, where
upstream's ``train.sh`` / ``scripts/compute_norm_stats.py`` exist.

Why no conversion step: LingBot loads **LeRobot v2.1 directly** — the
format our datagen datasets already ship — and its ``origin_keys`` are free-form lookups
into the LeRobot item, so our flat keys (``state`` / ``actions`` / ``image_left`` /
``wrist_image``) map straight through. The six datasets stay read-only and byte-identical
to what every other base model trains on.

Contract summary (see the YAML files for the full rationale):

* **State / action:** 8-D ``[arm_q(7), gripper(1)]`` -> ``arm.position`` ``[0:7)`` +
  ``effector.position`` ``[7:8)`` of LingBot's 55-D unified vector (the rest padded/masked).
* **Actions are ABSOLUTE** (``subtract_state: false`` on both features), following
  LingBot's own simulation recipe. ⚠️ At eval, feed the policy output straight to the
  JointController with **no** delta/un-relative step — same contract as SmolVLA, unlike
  pi0.5 / pi0 / GR00T which add the current state back.
* **Cameras:** 2 views — ``camera_top`` <- ``image_left`` overview, ``camera_wrist_left``
  <- ``wrist_image`` — identical to the other base models (benchmark parity).
* **Warm start:** ``robbyant/lingbot-vla-v2-6b`` (the PRETRAIN release), never
  ``…-6b-robotwin`` (already post-trained 50k steps on RoboTwin).
* **Action chunk:** 50 (LingBot's native ``chunk_size``). Eval must size
  ``execute_horizon`` against a 50-step chunk.
* **Scale:** upstream's own 8-GPU shape (micro 32 / global 256, lr 5e-5 cosine), 2 epochs;
  per-family step counts equal the pi0.5 / pi0 tracks'.
"""
