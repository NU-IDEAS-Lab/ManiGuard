"""SmolVLA SFT support for ManiGuard.

Thin layer around LeRobot's native SmolVLA policy (fine-tuned via the upstream
``lerobot-train`` CLI, from a pristine, separate ``lerobot`` clone). SmolVLA has
no config registry or
embodiment registration to hook into: ``lerobot-train`` derives the policy's
input/output features straight from the dataset's ``observation.*`` / ``action``
keys, and the architecture is agnostic to camera count and to the (padded)
state/action dimension. So the only ManiGuard-side artifact is the embodiment
*contract*:

- ``embodiment``: how the 5-cam datagen LeRobot export (flat keys ``image_*`` /
  ``state`` / ``actions``) maps into the 2-cam, standard-keyed LeRobot dataset
  SmolVLA expects (``observation.images.*`` / ``observation.state`` / ``action``).
  Pure constants + a rename map, no heavy deps, so ``tools/smolvla_sft`` scripts
  can import it without pulling in ``lerobot``.

The runnable tooling lives in ``tools/smolvla_sft/`` (``prepare_dataset.py``,
``run_sft.sh``, ``push_to_hf.py``, ``run_all.sh``). See ``docs/sft`` for the
end-to-end recipe.
"""
