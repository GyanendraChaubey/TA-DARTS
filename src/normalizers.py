"""
Normalisation functions for architecture weights (α).

  sparsemax()          — projects logits onto the probability simplex,
                         yielding *exact zeros* for suppressed operations.
  annealed_sparsemax() — temperature-scaled version; tau → 0 converges
                         to argmax.

Reference: Martins & Astudillo, "From Softmax to Sparsemax", ICML 2016.
"""
from __future__ import annotations

import torch
from torch import Tensor


def sparsemax(z: Tensor, dim: int = -1) -> Tensor:
    """
    Sparsemax projection onto the probability simplex Δ^{K-1}.

    Solves  argmin_{p ∈ Δ^{K-1}} ‖p − z‖²  in O(K log K) time.

    Args:
        z   : Input tensor (arbitrary shape).
        dim : Dimension to normalise along (default: last).

    Returns:
        Probability tensor of same shape as ``z``; values ∈ [0, 1], sum = 1.
    """
    # Move target dim to last for uniform treatment
    z = z.transpose(dim, -1)
    orig_shape = z.shape
    z_2d = z.reshape(-1, z.shape[-1])      # (N, K)

    K = z_2d.shape[-1]
    z_sorted, _ = torch.sort(z_2d, dim=-1, descending=True)
    z_cumsum    = torch.cumsum(z_sorted, dim=-1)                    # (N, K)
    k_range     = torch.arange(1, K + 1, dtype=z.dtype,
                                device=z.device).unsqueeze(0)       # (1, K)
    # k* = max{k : 1 + k·z_k > cumsum_k}
    test   = 1 + k_range * z_sorted > z_cumsum                     # (N, K)
    k_star = test.sum(dim=-1, keepdim=True).float()                 # (N, 1)
    # τ(z) = (cumsum at k* − 1) / k*
    tau = (z_cumsum.gather(1, (k_star - 1).long()) - 1) / k_star   # (N, 1)
    p   = torch.clamp(z_2d - tau, min=0.0)

    return p.reshape(orig_shape).transpose(dim, -1)


def annealed_sparsemax(
    z:   Tensor,
    tau: float = 1.0,
    dim: int   = -1,
) -> Tensor:
    """
    Temperature-scaled sparsemax (contrib. [A][B]).

    Divides ``z`` by ``tau`` before projecting.
    As tau → 0 the output approaches a one-hot argmax.
    ``tau`` is managed externally by :class:`~src.controller.SearchController`.

    Args:
        z   : Raw architecture logits α.
        tau : Current temperature (positive float, decreasing over epochs).
        dim : Normalisation dimension.
    """
    return sparsemax(z / tau, dim=dim)


def annealed_softmax(
    z:   Tensor,
    tau: float = 1.0,
    dim: int   = -1,
) -> Tensor:
    """
    Temperature-scaled softmax — the non-sparse counterpart to
    :func:`annealed_sparsemax`, used for the sparsemax-vs-softmax ablation.

    Unlike sparsemax, never produces exact zeros, so every operation always
    receives some gradient signal. Divides ``z`` by ``tau`` before the
    softmax, matching ``annealed_sparsemax``'s temperature semantics so call
    sites can swap between the two via a single flag with no other changes.
    """
    return torch.softmax(z / tau, dim=dim)
