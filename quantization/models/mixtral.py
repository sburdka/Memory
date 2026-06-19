"""MiLo compression support for Mixtral-8x7B."""

from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm

from .base import BasePatch, BaseMiLoModel

try:
    import transformers
    from accelerate import init_empty_weights
except ImportError:
    transformers = None  # type: ignore


class MixtralPatch(BasePatch):
    """
    Architecture-specific patching for Mixtral-8x7B.

    Linear tags:
        Dense   : self_attn.{q,k,v,o}_proj
        Sparse  : block_sparse_moe.experts.{w1,w2,w3}
    """

    @classmethod
    def get_linear_tags(cls) -> list[str]:
        return [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "block_sparse_moe.experts.w1",
            "block_sparse_moe.experts.w2",
            "block_sparse_moe.experts.w3",
        ]

    @classmethod
    def patch_nonlinearlayers(cls, model: nn.Module, patch_fct, verbose: bool = True) -> None:
        base = model.model
        model.lm_head = patch_fct(model.lm_head)
        base.embed_tokens = patch_fct(base.embed_tokens)
        base.norm = patch_fct(base.norm)

        for i in tqdm(range(len(base.layers)), disable=not verbose, desc="Non-linear patching"):
            layer = base.layers[i]
            layer.self_attn.rotary_emb = patch_fct(layer.self_attn.rotary_emb)
            layer.input_layernorm = patch_fct(layer.input_layernorm)
            layer.post_attention_layernorm = patch_fct(layer.post_attention_layernorm)
            # Keep MoE gate in fp16 — it's small and routing-sensitive
            layer.block_sparse_moe.gate = patch_fct(layer.block_sparse_moe.gate)
            for expert in layer.block_sparse_moe.experts:
                expert.act_fn = patch_fct(expert.act_fn)

    @classmethod
    def patch_linearlayers(cls, model: nn.Module, patch_fct, patch_params: dict, verbose: bool = True) -> None:
        base = model.model
        for i in tqdm(range(len(base.layers)), disable=not verbose, desc="Linear patching"):
            layer = base.layers[i]
            attn = layer.self_attn
            attn.q_proj = patch_fct(attn.q_proj, patch_params.get("self_attn.q_proj"))
            attn.k_proj = patch_fct(attn.k_proj, patch_params.get("self_attn.k_proj"))
            attn.v_proj = patch_fct(attn.v_proj, patch_params.get("self_attn.v_proj"))
            attn.o_proj = patch_fct(attn.o_proj, patch_params.get("self_attn.o_proj"))

            for expert in layer.block_sparse_moe.experts:
                expert.w1 = patch_fct(expert.w1, patch_params.get("block_sparse_moe.experts.w1"))
                expert.w2 = patch_fct(expert.w2, patch_params.get("block_sparse_moe.experts.w2"))
                expert.w3 = patch_fct(expert.w3, patch_params.get("block_sparse_moe.experts.w3"))


class MixtralMiLoHF(BaseMiLoModel):
    """HuggingFace model lifecycle for Mixtral."""

    @classmethod
    def cache_model(cls, model, save_dir: str) -> None:
        model.config.save_pretrained(save_dir)

    @classmethod
    def create_model(cls, save_dir: str, kwargs: dict):
        assert transformers is not None, "transformers is required"
        config = transformers.AutoConfig.from_pretrained(save_dir, trust_remote_code=True)
        with init_empty_weights():
            model = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        return model


class MixtralMiLo(MixtralPatch, MixtralMiLoHF):
    """
    Unified entry point for Mixtral-8x7B compression and loading.

    Compress:
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", torch_dtype=torch.float16)
        cfg = BaseCompressConfig(nbits=3, group_size=64, sparse_rank=16, dense_rank=512)
        MixtralMiLo.compress_model(model, cfg, device="cuda")
        MixtralMiLo.save_compressed(model, "/path/to/output")

    Load:
        model = MixtralMiLo.from_compressed("/path/to/output")
    """
    pass
