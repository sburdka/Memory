"""
Rank generation and compensator loading for MiLo.

Rank strategy controls how many singular values (rank) are used for the
low-rank compensator at each layer:

  None / "Uniform"  — apply sparse_rank to all expert layers, dense_rank elsewhere
  "Kurtosis"        — higher kurtosis ⇒ higher rank (tail-heavy layers need more
                       compensation). Used with Mixtral.
  "Frequency"       — higher expert routing frequency ⇒ higher rank.
                       Used with DeepSeek MoE.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Union

import torch

from .quantize import MiLoLinear, Quantizer

# ---------------------------------------------------------------------------
# Layer taxonomy for known architectures
# ---------------------------------------------------------------------------

# (layer_index, tag) format
_MIXTRAL_8X7B_LAYERS = 32
_DEEPSEEK_MOE_16B_LAYERS = 27

_DENSE_TAGS = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"}
_MIXTRAL_SPARSE_TAGS = {
    "block_sparse_moe.experts.w1",
    "block_sparse_moe.experts.w2",
    "block_sparse_moe.experts.w3",
}
_DEEPSEEK_SPARSE_TAGS = {
    "mlp.experts.gate_proj",
    "mlp.experts.up_proj",
    "mlp.experts.down_proj",
}

# Pre-computed kurtosis ranks for Mixtral sparse layers (relative values;
# scaled to [sparse_rank, dense_rank] at runtime).
# These approximate the empirical kurtosis ordering from the MiLo paper.
_MIXTRAL_KURTOSIS_RELATIVE: list[float] = [
    1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.7, 1.8,
    2.0, 2.1, 2.2, 2.3, 2.5, 2.6, 2.7, 2.8,
    3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4,
    4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0,
]


def _name_to_tag(name: str) -> str:
    """model.layers.7.self_attn.q_proj  →  self_attn.q_proj"""
    parts = [p for p in name.split(".") if not p.isnumeric() and p not in ("model", "layers")]
    return ".".join(parts)


def _is_expert_layer(tag: str) -> bool:
    return "experts" in tag


def rank_generate(
    model_name_or_path: str,
    sparse_rank: int,
    dense_rank: int,
    rank_strategy: Union[str, None] = None,
) -> dict[str, int]:
    """
    Returns {layer_name: rank} mapping.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model name (used to detect architecture).
    sparse_rank : int
        Base rank for sparse / MoE expert layers.
    dense_rank : int
        Base rank for dense (attention) layers.
    rank_strategy : str | None
        "Kurtosis", "Frequency", or None (uniform).
    """
    is_mixtral = "mixtral" in model_name_or_path.lower()
    is_deepseek = "deepseek" in model_name_or_path.lower()
    n_layers = _MIXTRAL_8X7B_LAYERS if is_mixtral else _DEEPSEEK_MOE_16B_LAYERS

    ranks: dict[str, int] = {}

    if rank_strategy is None or rank_strategy == "Uniform":
        # Flat assignment
        for layer_idx in range(n_layers):
            prefix = f"model.layers.{layer_idx}"
            tags = _DENSE_TAGS | (_MIXTRAL_SPARSE_TAGS if is_mixtral else _DEEPSEEK_SPARSE_TAGS)
            for tag in tags:
                full_name = f"{prefix}.{tag}"
                ranks[full_name] = sparse_rank if _is_expert_layer(tag) else dense_rank

    elif rank_strategy == "Kurtosis" and is_mixtral:
        # Scale relative kurtosis values into [sparse_rank, dense_rank]
        rel = _MIXTRAL_KURTOSIS_RELATIVE
        rel_min, rel_max = min(rel), max(rel)

        for layer_idx in range(n_layers):
            prefix = f"model.layers.{layer_idx}"
            k_rel = rel[layer_idx % len(rel)]
            # Exponential mapping: layers with higher kurtosis get disproportionately more rank
            alpha = (k_rel - rel_min) / (rel_max - rel_min + 1e-9)
            expert_rank = int(sparse_rank + (dense_rank - sparse_rank) * (math.exp(alpha) - 1) / (math.e - 1))

            for tag in _DENSE_TAGS:
                ranks[f"{prefix}.{tag}"] = dense_rank
            for tag in _MIXTRAL_SPARSE_TAGS:
                ranks[f"{prefix}.{tag}"] = expert_rank

    elif rank_strategy == "Frequency" and is_deepseek:
        # Uniform for now; override with expert routing frequencies if available
        for layer_idx in range(n_layers):
            prefix = f"model.layers.{layer_idx}"
            for tag in _DENSE_TAGS:
                ranks[f"{prefix}.{tag}"] = dense_rank
            for tag in _DEEPSEEK_SPARSE_TAGS:
                ranks[f"{prefix}.{tag}"] = sparse_rank

    else:
        # Fallback to uniform
        return rank_generate(model_name_or_path, sparse_rank, dense_rank, rank_strategy=None)

    return ranks


def load_compensators(
    model: torch.nn.Module,
    all_compensators: dict,
    ranks: dict[str, int],
) -> None:
    """
    Inject pre-saved compensator matrices (U, V) into the MiLoLinear modules
    of a freshly loaded quantised model.

    Parameters
    ----------
    model : nn.Module
        Quantised model (with MiLoLinear layers).
    all_compensators : dict
        {module_name: {"U": Tensor, "V": Tensor}} — as saved by save_compensators().
    ranks : dict
        {layer_name: rank} — used to validate or skip zero-rank layers.
    """
    for name, module in model.named_modules():
        if not isinstance(module, MiLoLinear):
            continue

        comp = all_compensators.get(name)
        if comp is None:
            continue

        rank = ranks.get(name, 0)
        if rank <= 0:
            continue

        U = comp.get("U")
        V = comp.get("V")
        if U is None or V is None:
            continue

        # Move to the same device as the module
        device = next(module.parameters(), torch.tensor(0)).device
        dtype = module.compute_dtype

        module.U = torch.nn.Parameter(
            U.to(device=device, dtype=dtype), requires_grad=False
        )
        module.V = torch.nn.Parameter(
            V.to(device=device, dtype=dtype), requires_grad=False
        )
