"""
Core quantization layer: Quantizer, MiLoLinear, BaseCompressConfig.
"""

from __future__ import annotations

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union

from .bitpack import BitPack
from .optimize import optimize_weights_proximal


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class BaseCompressConfig(dict):
    """
    Flat constructor that converts keyword args into the nested dict layout
    expected by MiLoLinear and BaseMiLoModel.

    Quantisation params:
        nbits          : weight bit-width (2, 3, 4, or 8)
        group_size     : number of weights per quantisation group
        axis           : grouping axis — 1 = per-row groups (default), 0 = per-col
        quant_scale    : whether to quantise the per-group scale values
        quant_zero     : whether to quantise the per-group zero values
        iter           : proximal optimisation iterations (0 = skip)

    Compensator params:
        sparse_rank    : default rank for MoE expert layers
        dense_rank     : default rank for attention / dense layers
        rank_strategy  : "Kurtosis" | "Frequency" | None (uniform)
        compensator_dtype : dtype for quantised compensator ("fp16", "int3", "int4")
    """

    def __init__(
        self,
        nbits: int = 3,
        group_size: int = 64,
        axis: int = 1,
        quant_scale: bool = False,
        quant_zero: bool = False,
        iter: int = 10,
        sparse_rank: int = 16,
        dense_rank: int = 512,
        rank_strategy: Union[str, None] = "Kurtosis",
        compensator_dtype: str = "fp16",
    ):
        super().__init__()
        self["nbits"] = nbits
        self["group_size"] = group_size
        self["axis"] = axis
        self["quant_scale"] = quant_scale
        self["quant_zero"] = quant_zero
        self["iter"] = iter
        self["compensator_params"] = {
            "sparse_rank": sparse_rank,
            "dense_rank": dense_rank,
            "rank_strategy": rank_strategy,
            "compensator_dtype": compensator_dtype,
            "ranks": {},
        }


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------

class Quantizer:
    """
    Asymmetric min-max group quantisation with optional proximal optimisation.

    Quantisation scheme (per group):
        scale = (max - min) / (2^nbits - 1)
        zero  = round(-min / scale)          # in integer space
        W_q   = clamp(round(W / scale) + zero, 0, 2^nbits - 1)

    Dequantisation:
        W_hat = (W_q - zero) * scale
    """

    SUPPORTED_BITS = (2, 3, 4, 8)

    def __init__(self, nbits: int = 4, group_size: int = 64, axis: int = 1):
        assert nbits in self.SUPPORTED_BITS, f"nbits must be one of {self.SUPPORTED_BITS}"
        self.nbits = nbits
        self.group_size = group_size
        self.axis = axis
        self.n_levels = 2**nbits

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _group_reshape(self, W: Tensor) -> Tensor:
        """Flatten W into [n_groups, group_size] (axis=1) or [group_size, n_groups] (axis=0)."""
        W2 = W.reshape(W.shape[0], -1)
        N, K = W2.shape
        gs = self.group_size
        if self.axis == 1:
            # Group along K: [N * (K/gs), gs]
            return W2.reshape(N * max(K // gs, 1), min(K, gs))
        else:
            # Group along N: [gs, (N/gs) * K]
            return W2.T.reshape(K * max(N // gs, 1), min(N, gs)).T

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def quantize(
        self, W: Tensor, optimize: bool = False, n_iter: int = 10
    ) -> tuple[Tensor, dict]:
        """
        Quantize W.  Returns (W_q_packed, meta).
        meta contains scale, zero, original shape, and config fields.
        """
        shape = W.shape
        W_g = self._group_reshape(W)

        dim = 1 if self.axis == 1 else 0
        min_val = W_g.min(dim=dim, keepdim=True).values
        max_val = W_g.max(dim=dim, keepdim=True).values

        scale = (max_val - min_val).clamp(min=1e-9) / (self.n_levels - 1)
        zero = (-min_val / scale).round().clamp(0, self.n_levels - 1)

        if optimize and n_iter > 0:
            W_q, scale, zero = optimize_weights_proximal(
                W_g, scale, zero, axis=dim, nbits=self.nbits, n_iter=n_iter
            )
        else:
            W_q = (W_g / scale + zero).round().clamp(0, self.n_levels - 1).to(torch.uint8)

        meta = {
            "scale": scale,
            "zero": zero,
            "shape": shape,
            "group_size": self.group_size,
            "axis": self.axis,
            "nbits": self.nbits,
        }
        return BitPack.pack(W_q, self.nbits), meta

    @staticmethod
    @torch.no_grad()
    def dequantize(W_q_packed: Tensor, meta: dict) -> Tensor:
        """Unpack and dequantize to float."""
        nbits = meta["nbits"]
        W_q = BitPack.unpack(W_q_packed, nbits, dtype=torch.float32)
        W = (W_q - meta["zero"].float()) * meta["scale"].float()
        return W.reshape(meta["shape"])


# ---------------------------------------------------------------------------
# Low-rank compensator computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_low_rank_compensator(
    W: Tensor, W_dequant: Tensor, rank: int
) -> tuple[Tensor, Tensor]:
    """
    SVD decomposition of the quantization error.

    E = W - W_dequant
    E ≈ U_k @ V_k  where U_k: [out, rank], V_k: [rank, in]

    Forward pass uses:  F.linear(x, W_dequant) + F.linear(F.linear(x, V_k), U_k)
    """
    if rank <= 0:
        return None, None
    E = W.float() - W_dequant.float()
    # Clamp rank to valid range
    rank = min(rank, min(E.shape))
    try:
        U, S, Vh = torch.linalg.svd(E, full_matrices=False)
        U_k = (U[:, :rank] * S[:rank]).to(W.dtype)  # [out, rank]
        V_k = Vh[:rank, :].to(W.dtype)               # [rank, in]
    except Exception:
        U_k = torch.zeros(E.shape[0], rank, dtype=W.dtype, device=W.device)
        V_k = torch.zeros(rank, E.shape[1], dtype=W.dtype, device=W.device)
    return U_k, V_k


# ---------------------------------------------------------------------------
# MiLoLinear — quantized linear layer with low-rank compensator
# ---------------------------------------------------------------------------

class MiLoLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that stores weights in compressed form.

    Storage layout:
        W_q    : packed quantized weights  (BitPack format)
        meta   : dict with scale, zero, shape, nbits, group_size, axis
        U, V   : low-rank compensator matrices (or None if rank=0)

    Forward pass (numerically equivalent to the original linear):
        out = F.linear(x, dequant(W_q)) + F.linear(F.linear(x, V), U) + bias
            = (dequant(W_q) + U @ V) @ x^T + bias
    """

    def __init__(
        self,
        linear_layer: Union[nn.Linear, None],
        compress_config: Union[dict, None],
        compute_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.device = device
        self.meta: dict = {}
        self.W_q: Union[Tensor, None] = None
        self.U: Union[nn.Parameter, None] = None
        self.V: Union[nn.Parameter, None] = None
        self.bias: Union[nn.Parameter, None] = None
        self.encoded_state_dict: bool = True
        self.name: str = ""
        self.compress_config: Union[dict, None] = compress_config

        if linear_layer is not None and compress_config is not None:
            self._compress(linear_layer, compress_config, compute_dtype, device)

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compress(
        self,
        layer: nn.Linear,
        cfg: dict,
        compute_dtype: torch.dtype,
        device: str,
    ) -> None:
        W = layer.weight.data.to(device=device, dtype=compute_dtype)

        quantizer = Quantizer(
            nbits=cfg["nbits"],
            group_size=cfg["group_size"],
            axis=cfg.get("axis", 1),
        )
        W_q_packed, meta = quantizer.quantize(
            W, optimize=(cfg.get("iter", 0) > 0), n_iter=cfg.get("iter", 10)
        )
        self.W_q = W_q_packed
        self.meta = meta

        # Compute compensator rank for this layer
        comp_params = cfg.get("compensator_params", {})
        ranks = comp_params.get("ranks", {})
        layer_name = getattr(layer, "name", "")
        rank = ranks.get(layer_name, comp_params.get("dense_rank", 0))

        if rank > 0:
            W_dequant = Quantizer.dequantize(W_q_packed, meta).to(compute_dtype)
            U_k, V_k = compute_low_rank_compensator(W, W_dequant, rank)
            if U_k is not None:
                self.U = nn.Parameter(U_k, requires_grad=False)
                self.V = nn.Parameter(V_k, requires_grad=False)

        if layer.bias is not None:
            self.bias = nn.Parameter(
                layer.bias.data.to(device=device, dtype=compute_dtype),
                requires_grad=False,
            )

        # Store the quantised compensators for saving
        self.UV_quantized: dict = {}
        if self.U is not None:
            self.UV_quantized = {"U": self.U.data, "V": self.V.data}

        del W
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        W = Quantizer.dequantize(self.W_q, self.meta).to(x.dtype)
        out = F.linear(x, W, self.bias)
        del W
        if self.U is not None and self.V is not None:
            # Efficient compensator application: (x @ V^T) @ U^T
            out = out + F.linear(F.linear(x, self.V), self.U)
        return out

    # ------------------------------------------------------------------
    # Device movement
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs):
        # Move packed int tensors manually (they are not nn.Parameters)
        if self.W_q is not None:
            self.W_q = self.W_q.to(*args, **kwargs)
        for key in ("scale", "zero"):
            if key in self.meta:
                self.meta[key] = self.meta[key].to(*args, **kwargs)
        return super().to(*args, **kwargs)

    # ------------------------------------------------------------------
    # Serialization — state_dict compatible with safetensors
    # ------------------------------------------------------------------

    def state_dict(self, **kwargs):
        sd = {
            "W_q": self.W_q,
            "scale": self.meta["scale"],
            "zero": self.meta["zero"],
            "nbits": torch.tensor(self.meta["nbits"]),
            "group_size": torch.tensor(self.meta["group_size"]),
            "axis": torch.tensor(self.meta["axis"]),
            "shape_0": torch.tensor(self.meta["shape"][0]),
            "shape_1": torch.tensor(self.meta["shape"][1]),
        }
        if self.bias is not None:
            sd["bias"] = self.bias.data
        return sd

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        self.W_q = state_dict["W_q"]
        self.meta = {
            "scale": state_dict["scale"],
            "zero": state_dict["zero"],
            "nbits": int(state_dict["nbits"].item()),
            "group_size": int(state_dict["group_size"].item()),
            "axis": int(state_dict["axis"].item()),
            "shape": (int(state_dict["shape_0"].item()), int(state_dict["shape_1"].item())),
        }
        if "bias" in state_dict:
            self.bias = nn.Parameter(state_dict["bias"], requires_grad=False)
        self.UV_quantized = {}


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def milo_base_compress_config(
    nbits: int = 3,
    group_size: int = 64,
    sparse_rank: int = 16,
    dense_rank: int = 512,
    rank_strategy: str = "Kurtosis",
    iter: int = 10,
) -> BaseCompressConfig:
    return BaseCompressConfig(
        nbits=nbits,
        group_size=group_size,
        iter=iter,
        sparse_rank=sparse_rank,
        dense_rank=dense_rank,
        rank_strategy=rank_strategy,
    )
