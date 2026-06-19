"""MiLo compression support for DeepSeek-MoE-16B."""

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

_DEEPSEEK_MODEL_ID = "deepseek-ai/deepseek-moe-16b-base"


class DeepSeekPatch(BasePatch):
    """
    Architecture-specific patching for DeepSeek-MoE-16B.

    Linear tags:
        Dense  : self_attn.{q,k,v,o}_proj
        Sparse : mlp.experts.{gate,up,down}_proj
    """

    @classmethod
    def get_linear_tags(cls) -> list[str]:
        return [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.experts.gate_proj",
            "mlp.experts.up_proj",
            "mlp.experts.down_proj",
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
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                layer.mlp.gate = patch_fct(layer.mlp.gate)
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "act_fn"):
                layer.mlp.act_fn = patch_fct(layer.mlp.act_fn)
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "shared_experts"):
                se = layer.mlp.shared_experts
                if hasattr(se, "act_fn"):
                    se.act_fn = patch_fct(se.act_fn)

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

            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                for expert in layer.mlp.experts:
                    expert.gate_proj = patch_fct(expert.gate_proj, patch_params.get("mlp.experts.gate_proj"))
                    expert.up_proj = patch_fct(expert.up_proj, patch_params.get("mlp.experts.up_proj"))
                    expert.down_proj = patch_fct(expert.down_proj, patch_params.get("mlp.experts.down_proj"))

            # Shared expert (dense component of DeepSeek MoE)
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "shared_experts"):
                se = layer.mlp.shared_experts
                if hasattr(se, "gate_proj"):
                    se.gate_proj = patch_fct(se.gate_proj, patch_params.get("mlp.experts.gate_proj"))
                    se.up_proj = patch_fct(se.up_proj, patch_params.get("mlp.experts.up_proj"))
                    se.down_proj = patch_fct(se.down_proj, patch_params.get("mlp.experts.down_proj"))


class DeepSeekMiLoHF(BaseMiLoModel):
    @classmethod
    def cache_model(cls, model, save_dir: str) -> None:
        model.config.save_pretrained(save_dir)

    @classmethod
    def create_model(cls, save_dir: str, kwargs: dict):
        assert transformers is not None, "transformers is required"
        # DeepSeek config must always be loaded from the original model ID
        config = transformers.AutoConfig.from_pretrained(_DEEPSEEK_MODEL_ID, trust_remote_code=True)
        with init_empty_weights():
            model = transformers.AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, **{k: v for k, v in kwargs.items() if k == "attn_implementation"}
            )
        return model


class DeepSeekMiLo(DeepSeekPatch, DeepSeekMiLoHF):
    """
    Unified entry point for DeepSeek-MoE-16B compression and loading.

    Compress:
        model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-moe-16b-base",
                                                     torch_dtype=torch.float16, trust_remote_code=True)
        cfg = BaseCompressConfig(nbits=3, group_size=64, sparse_rank=16, dense_rank=512,
                                  rank_strategy="Frequency")
        DeepSeekMiLo.compress_model(model, cfg, device="cuda")
        DeepSeekMiLo.save_compressed(model, "/path/to/output")

    Load:
        model = DeepSeekMiLo.from_compressed("/path/to/output")
    """
    pass
