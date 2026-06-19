"""
Evaluate a model (baseline or compressed) and save results to JSON.

Works for both:
  --mode baseline   → loads the original HuggingFace model in fp16/bf16
  --mode compressed → loads a MiLo-compressed model from disk

Usage:
    # Baseline
    python quantization/scripts/evaluate_model.py \
        --mode baseline \
        --model-id deepseek-ai/deepseek-moe-16b-base \
        --arch deepseek \
        --output results/baseline.json

    # Compressed
    python quantization/scripts/evaluate_model.py \
        --mode compressed \
        --model-id deepseek-ai/deepseek-moe-16b-base \
        --arch deepseek \
        --compressed-dir /mnt/data/deepseek_milo_a100 \
        --output results/compressed.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------
# Model loaders
# -----------------------------------------------------------------------

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
try:
    from quantization.models.mixtral import MixtralMiLo
    _ARCH_MAP["mixtral"] = MixtralMiLo
except ImportError:
    pass


def load_baseline(model_id: str, torch_dtype: torch.dtype, device: str):
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=device,
    )


def load_compressed(compressed_dir: str, arch: str, torch_dtype: torch.dtype, device: str):
    milo_cls = _ARCH_MAP[arch]
    return milo_cls.from_compressed(compressed_dir, compute_dtype=torch_dtype, device=device)


# -----------------------------------------------------------------------
# Memory measurement
# -----------------------------------------------------------------------

def get_vram_usage_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def get_peak_vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def get_model_disk_size_gb(path: str) -> float:
    if not os.path.isdir(path):
        return 0.0
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return total / 1e9


def count_parameters_gb(model) -> float:
    """Total parameter storage in GB (FP16 = 2 bytes per param)."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / 1e9


# -----------------------------------------------------------------------
# WikiText-2 perplexity
# -----------------------------------------------------------------------

@torch.no_grad()
def eval_perplexity(model, tokenizer, device: str, stride: int = 512, max_length: int = 2048) -> float:
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    except Exception as e:
        print(f"  [WARN] Could not load WikiText-2 dataset: {e}")
        return float("nan")

    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    seq_len = input_ids.shape[1]

    nlls = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end
        chunk = input_ids[:, begin:end]
        labels = chunk.clone()
        labels[:, :-target_len] = -100
        outputs = model(chunk, labels=labels)
        nlls.append(outputs.loss.item() * target_len)
        prev_end = end
        if end >= seq_len:
            break

    return math.exp(sum(nlls) / seq_len)


# -----------------------------------------------------------------------
# Throughput benchmark (tokens / second)
# -----------------------------------------------------------------------

@torch.no_grad()
def benchmark_throughput(
    model,
    tokenizer,
    device: str,
    prompt: str = "The future of artificial intelligence is",
    n_new_tokens: int = 128,
    n_runs: int = 3,
    warmup_runs: int = 1,
) -> dict:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = inputs.input_ids.shape[1]

    # Warmup
    for _ in range(warmup_runs):
        model.generate(**inputs, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latencies = []
    for _ in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(
            **inputs,
            max_new_tokens=n_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    avg_latency = sum(latencies) / len(latencies)
    tokens_per_sec = n_new_tokens / avg_latency
    time_to_first_token_ms = (avg_latency / n_new_tokens) * 1000  # rough approx

    return {
        "tokens_per_sec": round(tokens_per_sec, 1),
        "avg_latency_s": round(avg_latency, 3),
        "time_per_token_ms": round(avg_latency / n_new_tokens * 1000, 2),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": n_new_tokens,
    }


# -----------------------------------------------------------------------
# Zero-shot evaluation (optional — requires lm-eval)
# -----------------------------------------------------------------------

def eval_zeroshot(model, tokenizer, tasks: list[str], device: str) -> dict:
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        lm = HFLM(pretrained=model, tokenizer=tokenizer, device=device)
        results = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=0)
        out = {}
        for task, metrics in results["results"].items():
            acc = metrics.get("acc,none") or metrics.get("acc") or metrics.get("acc_norm,none")
            out[task] = round(float(acc), 4) if acc is not None else None
        return out
    except ImportError:
        print("  [WARN] lm-eval not installed. Skipping zero-shot. Run: pip install lm-eval")
        return {}
    except Exception as e:
        print(f"  [WARN] Zero-shot evaluation failed: {e}")
        return {}


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["baseline", "compressed"])
    p.add_argument("--model-id", required=True, help="HuggingFace model ID")
    p.add_argument("--arch", required=True, choices=list(_ARCH_MAP) + ["deepseek", "deepseek_v2", "mixtral"])
    p.add_argument("--compressed-dir", help="Path to MiLo output dir (compressed mode only)")
    p.add_argument("--output", required=True, help="JSON file to write results")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--tasks", nargs="*", default=[], help="Zero-shot tasks (e.g. arc_easy hellaswag)")
    p.add_argument("--skip-ppl", action="store_true", help="Skip perplexity evaluation")
    p.add_argument("--skip-throughput", action="store_true", help="Skip throughput benchmark")
    p.add_argument("--n-tokens", type=int, default=128, help="Tokens to generate for throughput test")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    results: dict = {
        "mode": args.mode,
        "model_id": args.model_id,
        "arch": args.arch,
        "dtype": args.dtype,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }

    print(f"\n{'='*60}")
    print(f"  Evaluating [{args.mode.upper()}]  {args.model_id}")
    print(f"{'='*60}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ---- Load model ----
    print("\n[1/4] Loading model ...")
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.mode == "baseline":
        model = load_baseline(args.model_id, dtype, args.device)
    else:
        assert args.compressed_dir, "--compressed-dir is required in compressed mode"
        model = load_compressed(args.compressed_dir, args.arch, dtype, args.device)

    model.eval()
    results["load_time_s"] = round(time.time() - t_load, 1)

    # ---- Memory ----
    print("[2/4] Measuring memory ...")
    results["vram_allocated_gb"] = round(get_vram_usage_gb(), 2)
    results["vram_peak_gb"] = round(get_peak_vram_gb(), 2)
    results["model_params_gb"] = round(count_parameters_gb(model), 2)
    if args.mode == "compressed" and args.compressed_dir:
        results["disk_size_gb"] = round(get_model_disk_size_gb(args.compressed_dir), 2)

    print(f"  VRAM allocated : {results['vram_allocated_gb']} GB")
    print(f"  VRAM peak      : {results['vram_peak_gb']} GB")
    print(f"  Param storage  : {results['model_params_gb']} GB")

    # ---- Perplexity ----
    if not args.skip_ppl:
        print("[3/4] WikiText-2 perplexity ...")
        ppl = eval_perplexity(model, tokenizer, args.device)
        results["wikitext2_ppl"] = round(ppl, 4)
        print(f"  PPL = {ppl:.4f}")
    else:
        print("[3/4] Perplexity skipped.")

    # ---- Throughput ----
    if not args.skip_throughput:
        print("[4/4] Throughput benchmark ...")
        tput = benchmark_throughput(model, tokenizer, args.device, n_new_tokens=args.n_tokens)
        results["throughput"] = tput
        print(f"  {tput['tokens_per_sec']} tok/s  |  {tput['time_per_token_ms']} ms/tok")
    else:
        print("[4/4] Throughput skipped.")

    # ---- Zero-shot (optional) ----
    if args.tasks:
        print(f"[+] Zero-shot tasks: {args.tasks}")
        zs = eval_zeroshot(model, tokenizer, args.tasks, args.device)
        results["zeroshot"] = zs
        for task, acc in zs.items():
            print(f"  {task}: {acc}")

    # ---- Save ----
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
