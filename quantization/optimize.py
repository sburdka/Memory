"""
Weight optimization routines that minimize quantization error:
    min ||W - dequantize(quantize(W))||_p^p

All functions operate in inference mode (no gradients on model weights).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from functools import partial


def shrink_lp_op(x: Tensor, thresh: float, p: float) -> Tensor:
    """Soft-thresholding (proximal operator) for Lp norm with p <= 1."""
    if p == 1:
        return x.sign() * torch.clamp(x.abs() - thresh, min=0)
    # Generalised for 0 < p < 1 via iterative approximation
    return x.sign() * torch.clamp(x.abs() - thresh * x.abs().pow(p - 1), min=0)


@torch.no_grad()
def update_scale_inverse_median(
    W: Tensor, W_q: Tensor, meta: dict, axis: int
) -> Tensor:
    """
    Recalibrate scale using the ratio of median absolute deviations.
    More robust than min/max for heavy-tailed distributions.
    """
    scale = meta["scale"]
    zero = meta["zero"]
    W_dq = (W_q.float() - zero) * scale
    ratio = (W.float().abs().median(dim=axis, keepdim=True).values + 1e-9) / (
        W_dq.abs().median(dim=axis, keepdim=True).values + 1e-9
    )
    return (scale * ratio).to(scale.dtype)


@torch.no_grad()
def update_scale_grid_search(
    W: Tensor,
    W_q: Tensor,
    meta: dict,
    axis: int,
    n_candidates: int = 128,
) -> Tensor:
    """
    Greedy per-group scale search over a grid of candidate values.
    Picks the scale that minimises ||W - dequant(W_q)||_2^2.
    """
    scale = meta["scale"].clone().float()
    zero = meta["zero"].float()
    W_f = W.float()
    n_levels = 2 ** meta["nbits"]

    best_scale = scale.clone()
    best_loss = torch.full_like(scale, float("inf"))

    for alpha in torch.linspace(0.7, 1.3, n_candidates, device=W.device):
        s_cand = scale * alpha
        W_dq = (W_f / s_cand + zero).round().clamp(0, n_levels - 1)
        W_dq = (W_dq - zero) * s_cand
        loss = (W_f - W_dq).pow(2).mean(dim=axis, keepdim=True)
        improved = loss < best_loss
        best_scale = torch.where(improved, s_cand, best_scale)
        best_loss = torch.where(improved, loss, best_loss)

    return best_scale.to(meta["scale"].dtype)


@torch.no_grad()
def optimize_weights_proximal(
    W: Tensor,
    scale: Tensor,
    zero: Tensor,
    axis: int,
    nbits: int,
    n_iter: int = 20,
    lp_norm: float = 0.5,
    tol: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Proximal optimisation: iteratively update scale/zero to minimise
        ||W - dequantize(quantize(W))||_p^p
    Returns (W_q, scale, zero).
    """
    n_levels = 2 ** nbits
    W_f = W.float()
    scale = scale.clone().float()
    zero = zero.clone().float()

    best_error = float("inf")
    best_scale, best_zero = scale.clone(), zero.clone()

    for _ in range(n_iter):
        W_q = (W_f / scale + zero).round().clamp(0, n_levels - 1)
        W_dq = (W_q - zero) * scale
        err = (W_f - W_dq)

        # Lp proximal update on scale
        grad_s = -(err * (W_q - zero)).mean(dim=axis, keepdim=True)
        scale = scale - 1e-2 * grad_s
        scale = scale.clamp(min=1e-9)

        mse = err.pow(2).mean().item()
        if mse < best_error:
            best_error = mse
            best_scale, best_zero = scale.clone(), zero.clone()

        if mse < tol:
            break

    W_q = (W_f / best_scale + best_zero).round().clamp(0, n_levels - 1).to(torch.uint8)
    return W_q, best_scale.to(W.dtype), best_zero.to(W.dtype)


@torch.no_grad()
def optimize_weights_autograd(
    W: Tensor,
    scale: Tensor,
    zero: Tensor,
    axis: int,
    nbits: int,
    n_iter: int = 100,
    lr: float = 1e-3,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    AdamW-based scale optimisation using straight-through estimator for the
    round() operation.
    """
    n_levels = 2 ** nbits
    W_f = W.float()
    log_scale = torch.log(scale.float().clamp(min=1e-9)).requires_grad_(True)
    optimizer = torch.optim.AdamW([log_scale], lr=lr)

    for _ in range(n_iter):
        optimizer.zero_grad()
        s = torch.exp(log_scale)
        W_q = (W_f / s + zero.float()).clamp(0, n_levels - 1)
        # Straight-through: gradient flows through round
        W_dq = (W_q.detach().round() - W_q.detach() + W_q - zero.float()) * s
        loss = F.mse_loss(W_dq, W_f)
        loss.backward()
        optimizer.step()

    best_scale = torch.exp(log_scale.detach()).to(W.dtype)
    W_q = (W_f / best_scale.float() + zero.float()).round().clamp(0, n_levels - 1).to(torch.uint8)
    return W_q, best_scale, zero


# Convenient presets
optimize_weights_proximal_fast = partial(optimize_weights_proximal, n_iter=10)
optimize_weights_proximal_full = partial(optimize_weights_proximal, n_iter=100)
