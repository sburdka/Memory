"""
Config-driven DeepSeek compression runner.

Reads a YAML config file (deepseek_a100.yaml or deepseek_h100.yaml) and
runs the full MiLo compression pipeline end-to-end.

Usage:
    # On A100 VM
    conda activate milo
    python quantization/scripts/run_deepseek.py \
        --config quantization/configs/deepseek_a100.yaml

    # On H100 VM
    python quantization/scripts/run_deepseek.py \
        --config quantization/configs/deepseek_h100.yaml

    # Evaluate after compression
    python quantization/scripts/run_deepseek.py \
        --config quantization/configs/deepseek_a100.yaml \
        --eval-only
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# Register all model classes
_ARCH_MAP: dict = {}
try:
    from quantization.models.deepseek import DeepSeekMiLo
    _ARCH_MAP["deepseek"] = DeepSeekMiLo
except ImportError:
    pass
try:
    from quantization.models.deepseek_v2 import DeepSeekV2MiLo
    _ARCH_MAP["deepseek_v2"] = DeepSeekV2MiLo
except ImportError:
    pass

from quantization.quantize import BaseCompressConfig


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def print_gpu_stats(label: str = "") -> None:
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {label} — VRAM used: {used:.1f} GB / {total:.1f} GB")


def compress(cfg: dict) -> None:
    model_cfg = cfg["model"]
    comp_cfg = cfg["compress"]
    output_dir = cfg["output_dir"]
    device = cfg.get("device", "cuda")

    arch = model_cfg["arch"]
    assert arch in _ARCH_MAP, f"Unknown arch '{arch}'. Available: {list(_ARCH_MAP)}"
    milo_cls = _ARCH_MAP[arch]

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[model_cfg.get("torch_dtype", "bfloat16")]

    print(f"\n{'='*60}")
    print(f"  MiLo DeepSeek Compression")
    print(f"  Model   : {model_cfg['model_id']}")
    print(f"  Bits    : {comp_cfg['nbits']}-bit  group={comp_cfg['group_size']}")
    print(f"  Ranks   : sparse={comp_cfg['sparse_rank']}  dense={comp_cfg['dense_rank']}")
    print(f"  Strategy: {comp_cfg.get('rank_strategy', 'Uniform')}")
    print(f"  Output  : {output_dir}")
    print(f"{'='*60}\n")

    # ----------------------------------------------------------------
    # 1. Load base model
    # ----------------------------------------------------------------
    print("Loading base model ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_id"],
        torch_dtype=torch_dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        device_map="cpu",  # Load on CPU first, then move during compression
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print_gpu_stats("before compression")

    # ----------------------------------------------------------------
    # 2. Compress
    # ----------------------------------------------------------------
    compress_config = BaseCompressConfig(
        nbits=comp_cfg["nbits"],
        group_size=comp_cfg["group_size"],
        axis=comp_cfg.get("axis", 1),
        iter=comp_cfg.get("iter", 10),
        sparse_rank=comp_cfg["sparse_rank"],
        dense_rank=comp_cfg["dense_rank"],
        rank_strategy=comp_cfg.get("rank_strategy", "Frequency"),
        compensator_dtype=comp_cfg.get("compensator_dtype", "fp16"),
    )

    print("\nCompressing ...")
    t1 = time.time()
    milo_cls.compress_model(model, compress_config, compute_dtype=torch_dtype, device=device)
    print(f"  Compressed in {time.time() - t1:.1f}s")
    print_gpu_stats("after compression")

    # ----------------------------------------------------------------
    # 3. Save
    # ----------------------------------------------------------------
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {output_dir} ...")
    t2 = time.time()
    milo_cls.save_compressed(model, output_dir)
    print(f"  Saved in {time.time() - t2:.1f}s")

    # Write a run summary
    summary = {
        "model_id": model_cfg["model_id"],
        "arch": arch,
        "nbits": comp_cfg["nbits"],
        "group_size": comp_cfg["group_size"],
        "sparse_rank": comp_cfg["sparse_rank"],
        "dense_rank": comp_cfg["dense_rank"],
        "rank_strategy": comp_cfg.get("rank_strategy"),
        "torch_dtype": str(torch_dtype),
        "total_time_s": round(time.time() - t0, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    with open(os.path.join(output_dir, "compression_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nCompression summary:", json.dumps(summary, indent=2))


def evaluate(cfg: dict) -> None:
    from quantization.evaluate import eval_wikitext2_perplexity

    model_cfg = cfg["model"]
    output_dir = cfg["output_dir"]
    arch = model_cfg["arch"]
    device = cfg.get("device", "cuda")
    milo_cls = _ARCH_MAP[arch]

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map[model_cfg.get("torch_dtype", "bfloat16")]

    print(f"\nLoading compressed model from {output_dir} ...")
    model = milo_cls.from_compressed(output_dir, compute_dtype=torch_dtype, device=device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Evaluating WikiText-2 perplexity ...")
    ppl = eval_wikitext2_perplexity(model, tokenizer, device=device)
    print(f"  WikiText-2 PPL: {ppl:.4f}")

    results = {"wikitext2_ppl": ppl}
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Config-driven DeepSeek MiLo runner")
    p.add_argument("--config", required=True, help="Path to YAML config (deepseek_a100.yaml or deepseek_h100.yaml)")
    p.add_argument("--eval-only", action="store_true", help="Skip compression, only evaluate existing output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if not args.eval_only:
        compress(cfg)

    if args.eval_only or os.path.exists(cfg["output_dir"]):
        try:
            evaluate(cfg)
        except Exception as e:
            print(f"Evaluation skipped: {e}")


if __name__ == "__main__":
    main()
