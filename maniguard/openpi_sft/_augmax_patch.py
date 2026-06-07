"""Runtime guard for openpi's training-time image augmentation (augmax).

openpi's ``model.preprocess_observation`` applies random geometric image
augmentation (RandomCrop / Resize / Rotate via ``augmax``) when ``train=True``.
The augmax geometric path occasionally produces a **non-finite transform
matrix** -- a NaN surfaces in ``augmax.utils.apply_perspective``'s
``dot_general`` -- which turns the whole augmented image NaN and then poisons the
entire training batch (loss -> NaN, or a corrupted/diverging run). It is
stochastic and LR-independent; diagnosed by running training under
``JAX_DEBUG_NANS=True``, which halted exactly at that op.

openpi is consumed PRISTINE (we never edit it) and exposes no flag to disable or
guard the augmentation, so we fix it at import time with a minimal monkey-patch:
wrap augmax's transform entry point (``augmax.base.Transformation.__call__`` --
``Chain`` inherits it; nested transforms are dispatched via ``.apply`` so this
wraps only the outer call once) and **sanitize the output** -- any non-finite
leaf falls back element-wise to the original (un-augmented) input.

Behaviour-preserving by construction: for finite outputs
``where(isfinite(out), out, input) == out``, so normal augmentation (and the
already-trained dusty/jar runs) is bit-identical; only the rare pathological,
non-finite augmentation is neutralized (that image is left un-augmented that
step). We intentionally do NOT clamp the pixel range here -- ``ColorJitter`` can
legitimately push values slightly outside [0, 1], and clamping would alter
normal augmentation.

Mirrors how ``maniguard/_omnigibson_patches.py`` keeps ManiGuard fixes out of the
OmniGibson tree -- here we keep them out of openpi (and out of augmax's source).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def apply() -> None:
    """Idempotently wrap ``augmax.base.Transformation.__call__`` with a finite-guard."""
    import augmax.base as _ab

    # Guard against augmax API drift: fail loudly rather than silently no-op if the
    # hook we depend on disappears (augmax is pinned in the venv, so this is a
    # tripwire for an unexpected upgrade).
    if not hasattr(_ab, "Transformation") or not callable(
        getattr(_ab.Transformation, "__call__", None)
    ):
        raise RuntimeError(
            "augmax.base.Transformation.__call__ not found -- augmax API changed; "
            "update maniguard.openpi_sft._augmax_patch."
        )

    # Root-cause guard: augmax's geometric path projects coordinates with
    # ``utils.apply_perspective``, whose final step divides by the homogeneous
    # ``z`` (``yx / z``). A degenerate random perspective matrix can drive ``z``
    # to ~0, yielding inf/NaN that not only poisons the augmented image but also
    # backprops NaN gradients (the outer __call__ guard below sanitizes the
    # forward value, but ``jnp.where``'s unselected branch still passes NaN grads
    # -> grad_norm=nan -> divergence). Patch the division at the source so the
    # NaN never forms. geometric.py calls it as ``utils.apply_perspective`` (a
    # module-attribute lookup), so replacing the module attribute covers every
    # call site. Behaviour-preserving: ``z`` is ~1 for normal transforms, so the
    # 1e-6 floor only engages on the pathological near-singular case.
    import augmax.utils as _au

    if not getattr(_au.apply_perspective, "_maniguard_safe", False):
        def _safe_apply_perspective(xy, M):
            xyz = jnp.concatenate([xy, jnp.ones([1, *xy.shape[1:]])])
            xyz = jnp.tensordot(M, xyz, axes=1)
            yx, z = jnp.split(xyz, [2])
            # Floor |z| away from 0 to avoid div-by-zero -> inf/NaN. Preserve
            # z's sign (z is ~1 normally; the floor only bites near-singular z).
            safe_z = jnp.where(jnp.abs(z) < 1e-6, jnp.where(z < 0, -1e-6, 1e-6), z)
            return yx / safe_z

        _safe_apply_perspective._maniguard_safe = True
        _au.apply_perspective = _safe_apply_perspective

    # Idempotent: never double-wrap (re-import / re-register is safe).
    if getattr(_ab.Transformation.__call__, "_maniguard_guarded", False):
        return

    _orig_call = _ab.Transformation.__call__

    def _guarded_call(self, rng, inputs, input_types=None):
        out = _orig_call(self, rng, inputs, input_types)
        # Non-finite (NaN/inf) leaves fall back to the original input, element-wise.
        # Finite outputs are returned unchanged -> normal augmentation is identical.
        return jax.tree_util.tree_map(
            lambda o, i: jnp.where(jnp.isfinite(o), o, i), out, inputs
        )

    _guarded_call._maniguard_guarded = True
    _ab.Transformation.__call__ = _guarded_call
