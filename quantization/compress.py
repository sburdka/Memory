"""
CLI entry point for MiLo compression.

Usage examples:

  # Compress Mixtral-8x7B to 3-bit
  python -m quantization.compress \\
      --model mistralai/Mixtral-8x7B-v0.1 \\
      --output /path/to/mixtral_milo \\
      --arch mixtral \\
      --nbits 3 \\
      --group-size 64 \\
      --sparse-rank 16 \\
      --dense-rank 512 \\
      --rank-strategy Kurtosis \\
      --iter 10

  # Compress DeepSeek-MoE-16B to 4-bit
  python -m quantization.compress \\
      --model deepseek-ai/deepseek-moe-16b-base \\
      --output /path/to/deepseek_milo \\
      --arch deepseek \\
      --nbits 4 \\
      --rank-strategy Frequency
"""

from __future__ import annotations

import argparse
import torch
from transformers import AutoModelForCausalLM

from .quantize import BaseCompressConfig
from .models.mixtral import MixtralMiLo
from .models.deepseek import DeepSeekMiLo

_ARCH_MAP = {
    "mixtral": MixtralMiLo,
    "deepseek": DeepSeekMiLo,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MiLo: Quantise MoE models with low-rank compensators")
    p.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    p.add_argument("--output", required=True, help="Directory to save the compressed model")
    p.add_argument("--arch", required=True, choices=list(_ARCH_MAP), help="Model architecture")
    p.add_argument("--nbits", type=int, default=3, choices=[2, 3, 4, 8])
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--axis", type=int, default=1, choices=[0, 1])
    p.add_argument("--iter", type=int, default=10, help="Proximal optimisation iterations")
    p.add_argument("--sparse-rank", type=int, default=16, help="Compensator rank for MoE expert layers")
    p.add_argument("--dense-rank", type=int, default=512, help="Compensator rank for attention layers")
    p.add_argument("--rank-strategy", default="Kurtosis", choices=["Kurtosis", "Frequency", "Uniform"])
    p.add_argument("--device", default="cuda", help="'cuda', 'cpu', or comma-separated list for multi-GPU")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    compute_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device: object = args.device
    if "," in args.device:
        device = [d.strip() for d in args.device.split(",")]

    print(f"Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )

    compress_config = BaseCompressConfig(
        nbits=args.nbits,
        group_size=args.group_size,
        axis=args.axis,
        iter=args.iter,
        sparse_rank=args.sparse_rank,
        dense_rank=args.dense_rank,
        rank_strategy=args.rank_strategy,
    )

    milo_cls = _ARCH_MAP[args.arch]
    print(f"Compressing with MiLo ({args.nbits}-bit, group_size={args.group_size}) ...")
    milo_cls.compress_model(model, compress_config, compute_dtype=compute_dtype, device=device)

    print(f"Saving to {args.output} ...")
    milo_cls.save_compressed(model, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
