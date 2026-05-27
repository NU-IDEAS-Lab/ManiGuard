"""TRPO / CPO mathematical utilities.

Ported from omnisafe's math / tools utilities (Apache-2.0):
  omnisafe/utils/math.py  — conjugate_gradients
  omnisafe/utils/tools.py — flat-param helpers

These are the only pieces of second-order optimisation not available in SB3.
Everything else (rollout, value function update, logging) stays in SB3.

Public API
----------
flat_params(params)            -> 1-D tensor (concatenated, contiguous)
set_flat_params(params, flat)  -> writes flat tensor back into param list
flat_grad(loss, params, ...)   -> 1-D gradient tensor
conjugate_gradients(Avp, b, …) -> approximate F^{-1} b via CG
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch as th


# -----------------------------------------------------------------------
# Flat-parameter helpers
# -----------------------------------------------------------------------

def flat_params(params: Iterable[th.Tensor]) -> th.Tensor:
    """Concatenate a list of tensors into a single 1-D contiguous tensor."""
    return th.cat([p.data.view(-1) for p in params])


def set_flat_params(
    params: Iterable[th.Tensor], flat: th.Tensor
) -> None:
    """Write a flat tensor back into the parameter list in-place."""
    offset = 0
    for p in params:
        numel = p.numel()
        p.data.copy_(flat[offset : offset + numel].view_as(p))
        offset += numel


def flat_grad(
    output: th.Tensor,
    params: list[th.Tensor],
    retain_graph: bool = False,
    create_graph: bool = False,
) -> th.Tensor:
    """Compute gradient of ``output`` w.r.t. ``params`` and return flat."""
    grads = th.autograd.grad(
        output,
        params,
        retain_graph=retain_graph or create_graph,
        create_graph=create_graph,
        allow_unused=True,
    )
    return th.cat(
        [
            g.contiguous().view(-1) if g is not None else th.zeros_like(p).view(-1)
            for g, p in zip(grads, params)
        ]
    )


# -----------------------------------------------------------------------
# Conjugate-gradient solver
# -----------------------------------------------------------------------

def conjugate_gradients(
    Avp: Callable[[th.Tensor], th.Tensor],
    b: th.Tensor,
    n_steps: int = 10,
    tol: float = 1e-10,
) -> th.Tensor:
    """Solve Ax = b approximately using conjugate gradients.

    ``Avp`` is a callable that computes the matrix-vector product A·v for a
    given vector v without forming A explicitly (the Pearlmutter trick for
    the Fisher information matrix).

    Parameters
    ----------
    Avp:
        Callable accepting a 1-D tensor v and returning A·v (1-D tensor).
    b:
        Right-hand side, 1-D tensor.
    n_steps:
        Maximum CG iterations (default 10, matches omnisafe).
    tol:
        Residual tolerance for early stopping.

    Returns
    -------
    x: 1-D tensor approximating A^{-1} b.
    """
    x = th.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdot = th.dot(r, r)

    for _ in range(n_steps):
        Avp_val = Avp(p)
        alpha = rdot / (th.dot(p, Avp_val) + 1e-8)
        x = x + alpha * p
        r = r - alpha * Avp_val
        rdot_new = th.dot(r, r)
        if rdot_new < tol:
            break
        beta = rdot_new / (rdot + 1e-8)
        p = r + beta * p
        rdot = rdot_new

    return x


__all__ = [
    "flat_params",
    "set_flat_params",
    "flat_grad",
    "conjugate_gradients",
]
