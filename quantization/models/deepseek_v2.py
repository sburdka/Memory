"""
MiLo compression support for DeepSeek-V2 / DeepSeek-V2-Lite.

Key architectural differences vs deepseek-moe-16b:
  - Multi-head Latent Attention (MLA): compressed KV cache via low-rank projection
    q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj
  - Routed MoE experts: gate_proj, up_proj, down_proj
  - Shared experts (always-active dense MoE component)

Target models:
  deepseek-ai/DeepSeek-V2-Lite   — 15.7B params, 2.4B active, fits 80GB VRAM in BF16
  deepseek-ai/DeepSeek-V2        — 236B params, 21B active, requires multi-GPU with quant
"""

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

_V2_LITE_ID = "deepseek-ai/DeepSeek-V2-Lite"
_V2_FULL_ID = "deepseek-ai/DeepSeek-V2"


class DeepSeekV2Patch(BasePatch):
    """
    Patching for DeepSeek-V2 / V2-Lite with Multi-head Latent Attention.

    Dense linear tags  (attention MLA projections):
        self_attn.q_a_proj           — query low-rank down-projection
        self_attn.q_b_proj           — query low-rank up-projection
        self_attn.kv_a_proj_with_mqa — key-value compressed projection
        self_attn.kv_b_proj          — key-value decompression
        self_attn.o_proj             — output projection

    Sparse linear tags  (MoE experts — routed + shared):
        mlp.experts.gate_proj
        mlp.experts.up_proj
        mlp.experts.down_proj
        mlp.shared_experts.gate_proj
        mlp.shared_experts.up_proj
        mlp.shared_experts.down_proj
    """

    @classmethod
    def get_linear_tags(cls) -> list[str]:
        return [
            # MLA attention
            "self_attn.q_a_proj",
            "self_attn.q_b_proj",
            "self_attn.kv_a_proj_with_mqa",
            "self_attn.kv_b_proj",
            "self_attn.o_proj",
            # Routed experts
            "mlp.experts.gate_proj",
            "mlp.experts.up_proj",
            "mlp.experts.down_proj",
            # Shared (always-active) experts
            "mlp.shared_experts.gate_proj",
            "mlp.shared_experts.up_proj",
            "mlp.shared_experts.down_proj",
        ]

    @classmethod
    def patch_nonlinearlayers(cls, model: nn.Module, patch_fct, verbose: bool = True) -> None:
        base = model.model
        model.lm_head = patch_fct(model.lm_head)
        base.embed_tokens = patch_fct(base.embed_tokens)
        base.norm = patch_fct(base.norm)

        for i in tqdm(range(len(base.layers)), disable=not verbose, desc="Non-linear patching"):
            layer = base.layers[i]
            layer.input_layernorm = patch_fct(layer.input_layernorm)
            layer.post_attention_layernorm = patch_fct(layer.post_attention_layernorm)

            attn = layer.self_attn
            if hasattr(attn, "rotary_emb"):
                attn.rotary_emb = patch_fct(attn.rotary_emb)
            if hasattr(attn, "q_a_layernorm"):
                attn.q_a_layernorm = patch_fct(attn.q_a_layernorm)
            if hasattr(attn, "kv_a_layernorm"):
                attn.kv_a_layernorm = patch_fct(attn.kv_a_layernorm)

            mlp = layer.mlp
            # Keep routing gate in fp16 — small and routing-critical
            if hasattr(mlp, "gate"):
                mlp.gate = patch_fct(mlp.gate)
            for expert in getattr(mlp, "experts", []):
                if hasattr(expert, "act_fn"):
                    expert.act_fn = patch_fct(expert.act_fn)
            shared = getattr(mlp, "shared_experts", None)
            if shared is not None and hasattr(shared, "act_fn"):
                shared.act_fn = patch_fct(shared.act_fn)

    @classmethod
    def patch_linearlayers(cls, model: nn.Module, patch_fct, patch_params: dict, verbose: bool = True) -> None:
        base = model.model

        for i in tqdm(range(len(base.layers)), disable=not verbose, desc="Linear patching"):
            layer = base.layers[i]
            attn = layer.self_attn

            # MLA projections
            for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
                if hasattr(attn, proj):
                    tag = f"self_attn.{proj}"
                    setattr(attn, proj, patch_fct(getattr(attn, proj), patch_params.get(tag)))

            # Routed experts
            mlp = layer.mlp
            for expert in getattr(mlp, "experts", []):
                expert.gate_proj = patch_fct(expert.gate_proj, patch_params.get("mlp.experts.gate_proj"))
                expert.up_proj   = patch_fct(expert.up_proj,   patch_params.get("mlp.experts.up_proj"))
                expert.down_proj = patch_fct(expert.down_proj, patch_params.get("mlp.experts.down_proj"))

            # Shared experts
            shared = getattr(mlp, "shared_experts", None)
            if shared is not None:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    if hasattr(shared, proj):
                        tag = f"mlp.shared_experts.{proj}"
                        setattr(shared, proj, patch_fct(getattr(shared, proj), patch_params.get(tag)))


class DeepSeekV2MiLoHF(BaseMiLoModel):
    @classmethod
    def cache_model(cls, model, save_dir: str) -> None:
        model.config.save_pretrained(save_dir)

    @classmethod
    def create_model(cls, save_dir: str, kwargs: dict):
        assert transformers is not None, "transformers is required"
        # Always load config from original model — handles architectures with
        # custom code not fully captured in the saved config.json
        model_id = kwargs.get("model_id", _V2_LITE_ID)
        config = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        with init_empty_weights():
            model = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        return model


class DeepSeekV2MiLo(DeepSeekV2Patch, DeepSeekV2MiLoHF):
    """
    MiLo for DeepSeek-V2-Lite / DeepSeek-V2.

    Compress (V2-Lite, single H100 80GB):
        from transformers import AutoModelForCausalLM
        from quantization.quantize import BaseCompressConfig
        from quantization.models.deepseek_v2 import DeepSeekV2MiLo

        model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        cfg = BaseCompressConfig(
            nbits=3, group_size=64,
            sparse_rank=64, dense_rank=2048,
            rank_strategy="Frequency",
        )
        DeepSeekV2MiLo.compress_model(model, cfg, device="cuda")
        DeepSeekV2MiLo.save_compressed(model, "/mnt/data/deepseek_v2_milo")

    Load:
        model = DeepSeekV2MiLo.from_compressed("/mnt/data/deepseek_v2_milo")
    """
    pass
