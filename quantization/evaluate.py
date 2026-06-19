"""
Evaluation utilities for compressed MiLo models.

Metrics:
  - WikiText-2 perplexity
  - Zero-shot accuracy (via lm-evaluation-harness)

Usage:
  python -m quantization.evaluate \\
      --model-dir /path/to/compressed \\
      --base-model mistralai/Mixtral-8x7B-v0.1 \\
      --arch mixtral \\
      --tasks wikitext2 arc_easy hellaswag winogrande
"""

from __future__ import annotations

import argparse
import math
import json
from pathlib import Path

import torch
from torch import Tensor
from transformers import AutoTokenizer

from .models.mixtral import MixtralMiLo
from .models.deepseek import DeepSeekMiLo

_ARCH_MAP = {
    "mixtral": MixtralMiLo,
    "deepseek": DeepSeekMiLo,
}


# ---------------------------------------------------------------------------
# WikiText-2 perplexity
# ---------------------------------------------------------------------------

def _tokenize_wikitext2(tokenizer, max_length: int = 2048) -> Tensor:
    """Load and tokenize the WikiText-2 test set."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    except Exception:
        # Fallback: look for local parquet shipped with MiLo
        parquet_path = Path(__file__).parent / "data" / "wikitext2_test.parquet"
        if parquet_path.exists():
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            text = "\n\n".join(df["text"].tolist())
        else:
            raise FileNotFoundError("WikiText-2 dataset not found. Install `datasets` or provide the parquet file.")

    enc = tokenizer(text, return_tensors="pt")
    return enc.input_ids


@torch.no_grad()
def eval_wikitext2_perplexity(
    model: torch.nn.Module,
    tokenizer,
    stride: int = 512,
    max_length: int = 2048,
    device: str = "cuda",
) -> float:
    """
    Compute sliding-window perplexity on WikiText-2.
    Lower is better.
    """
    model.eval()
    input_ids = _tokenize_wikitext2(tokenizer, max_length).to(device)
    seq_len = input_ids.shape[1]

    nlls: list[Tensor] = []
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end

        chunk = input_ids[:, begin:end]
        labels = chunk.clone()
        labels[:, :-target_len] = -100  # mask the prefix

        outputs = model(chunk, labels=labels)
        nlls.append(outputs.loss * target_len)

        prev_end = end
        if end == seq_len:
            break

    ppl = math.exp(torch.stack(nlls).sum().item() / seq_len)
    return ppl


# ---------------------------------------------------------------------------
# Zero-shot / few-shot via lm-evaluation-harness
# ---------------------------------------------------------------------------

def eval_zeroshot(
    model: torch.nn.Module,
    tokenizer,
    tasks: list[str] | None = None,
    num_fewshot: int = 0,
    device: str = "cuda",
) -> dict:
    """
    Run zero-shot benchmarks using lm-evaluation-harness.
    Requires: pip install lm-eval
    """
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        raise ImportError("lm-eval is required for zero-shot evaluation. Run: pip install lm-eval")

    if tasks is None:
        tasks = ["arc_easy", "arc_challenge", "hellaswag", "winogrande", "mmlu"]

    lm = HFLM(pretrained=model, tokenizer=tokenizer, device=device)
    results = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=num_fewshot)
    return results["results"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a MiLo compressed model")
    p.add_argument("--model-dir", required=True, help="Path to compressed model directory")
    p.add_argument("--base-model", required=True, help="Original HF model ID (for tokenizer)")
    p.add_argument("--arch", required=True, choices=list(_ARCH_MAP))
    p.add_argument("--tasks", nargs="*", default=["wikitext2"], help="Evaluation tasks")
    p.add_argument("--num-fewshot", type=int, default=0)
    p.add_argument("--stride", type=int, default=512, help="Sliding window stride for PPL")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None, help="Optional JSON file to write results")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    milo_cls = _ARCH_MAP[args.arch]
    print(f"Loading compressed model from {args.model_dir} ...")
    model = milo_cls.from_compressed(args.model_dir, device=args.device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results: dict = {}

    if "wikitext2" in args.tasks:
        print("Evaluating WikiText-2 perplexity ...")
        ppl = eval_wikitext2_perplexity(model, tokenizer, stride=args.stride, device=args.device)
        results["wikitext2_ppl"] = round(ppl, 4)
        print(f"  WikiText-2 PPL: {ppl:.4f}")

    zeroshot_tasks = [t for t in args.tasks if t != "wikitext2"]
    if zeroshot_tasks:
        print(f"Running zero-shot tasks: {zeroshot_tasks} ...")
        zs = eval_zeroshot(model, tokenizer, tasks=zeroshot_tasks, num_fewshot=args.num_fewshot, device=args.device)
        results.update(zs)
        for task, metrics in zs.items():
            acc = metrics.get("acc,none") or metrics.get("acc")
            print(f"  {task}: acc={acc:.4f}" if acc is not None else f"  {task}: {metrics}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
