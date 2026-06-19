"""
Compare baseline vs compressed evaluation JSON files and print a summary table.

Usage:
    python quantization/scripts/compare_results.py \
        --baseline results/baseline.json \
        --compressed results/compressed.json

Output example:
    ┌─────────────────────────┬──────────────┬──────────────┬──────────────────┐
    │ Metric                  │   Baseline   │  MiLo 3-bit  │  Delta / Gain    │
    ├─────────────────────────┼──────────────┼──────────────┼──────────────────┤
    │ VRAM allocated (GB)     │    33.2      │     9.8      │  -23.4 GB (3.4×) │
    │ WikiText-2 PPL          │    7.42      │    7.89      │  +0.47 (+6.3%)   │
    │ Throughput (tok/s)      │    18.3      │    54.7      │  +36.4 (3.0×)    │
    │ ms / token              │    54.6      │    18.3      │  -36.3 ms (3.0×) │
    │ arc_easy acc            │    0.742     │    0.731     │  -0.011 (-1.5%)  │
    └─────────────────────────┴──────────────┴──────────────┴──────────────────┘
"""

from __future__ import annotations

import argparse
import json
from typing import Any


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def pct(a: float, b: float) -> str:
    if b == 0:
        return "n/a"
    return f"{(a - b) / b * 100:+.1f}%"


def ratio(b: float, a: float) -> str:
    if a == 0:
        return "n/a"
    return f"{b / a:.1f}×"


def fmt(v: Any, decimals: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def print_row(label: str, base: Any, comp: Any, delta: str, w: tuple = (27, 14, 14, 20)) -> None:
    print(
        f"│ {label:<{w[0]}}│ {fmt(base):>{w[1]}} │ {fmt(comp):>{w[2]}} │ {delta:<{w[3]}}│"
    )


def sep(w: tuple = (27, 14, 14, 20)) -> None:
    print(f"├{'─'*(w[0]+1)}┼{'─'*(w[1]+2)}┼{'─'*(w[2]+2)}┼{'─'*(w[3]+1)}┤")


def header(w: tuple = (27, 14, 14, 20)) -> None:
    print(f"┌{'─'*(w[0]+1)}┬{'─'*(w[1]+2)}┬{'─'*(w[2]+2)}┬{'─'*(w[3]+1)}┐")
    print(f"│ {'Metric':<{w[0]}}│ {'Baseline':>{w[1]}} │ {'MiLo 3-bit':>{w[2]}} │ {'Delta / Gain':<{w[3]}}│")
    sep(w)


def footer(w: tuple = (27, 14, 14, 20)) -> None:
    print(f"└{'─'*(w[0]+1)}┴{'─'*(w[1]+2)}┴{'─'*(w[2]+2)}┴{'─'*(w[3]+1)}┘")


def compare(base: dict, comp: dict) -> None:
    print(f"\n  Baseline   : {base.get('model_id', '?')}  [{base.get('dtype', '?')}]")
    print(f"  Compressed : {comp.get('model_id', '?')}  [MiLo {comp.get('dtype', '?')}]")
    print(f"  GPU        : {base.get('gpu', '?')}\n")

    header()

    # -- Memory --
    bv = base.get("vram_allocated_gb")
    cv = comp.get("vram_allocated_gb")
    if bv and cv:
        delta = f"-{bv - cv:.1f} GB  ({ratio(bv, cv)} smaller)"
        print_row("VRAM allocated (GB)", bv, cv, delta)

    bp = base.get("model_params_gb")
    cp = comp.get("model_params_gb")
    if bp and cp:
        delta = f"-{bp - cp:.1f} GB  ({ratio(bp, cp)} smaller)"
        print_row("Param storage (GB)", bp, cp, delta)

    cd = comp.get("disk_size_gb")
    if cd:
        delta = f"{cd:.1f} GB on disk"
        print_row("Disk size (GB)", "—", cd, delta)

    sep()

    # -- Accuracy --
    bppl = base.get("wikitext2_ppl")
    cppl = comp.get("wikitext2_ppl")
    if bppl and cppl:
        delta = f"{pct(cppl, bppl)}  (+{cppl - bppl:.2f} absolute)"
        print_row("WikiText-2 PPL ↓", bppl, cppl, delta)

    sep()

    # -- Speed --
    bt = base.get("throughput", {})
    ct = comp.get("throughput", {})

    btps = bt.get("tokens_per_sec")
    ctps = ct.get("tokens_per_sec")
    if btps and ctps:
        delta = f"+{ctps - btps:.1f} tok/s  ({ratio(ctps, btps)} faster)"
        print_row("Throughput (tok/s) ↑", btps, ctps, delta)

    bms = bt.get("time_per_token_ms")
    cms = ct.get("time_per_token_ms")
    if bms and cms:
        delta = f"-{bms - cms:.1f} ms/tok  ({ratio(bms, cms)} faster)"
        print_row("Latency (ms/token) ↓", bms, cms, delta)

    # -- Zero-shot --
    bz = base.get("zeroshot", {})
    cz = comp.get("zeroshot", {})
    all_tasks = sorted(set(bz) | set(cz))
    if all_tasks:
        sep()
        for task in all_tasks:
            ba = bz.get(task)
            ca = cz.get(task)
            if ba is not None and ca is not None:
                sign = "+" if ca >= ba else ""
                delta = f"{sign}{ca - ba:.3f}  ({pct(ca, ba)})"
            else:
                delta = "—"
            print_row(f"{task} acc ↑", ba, ca, delta)

    footer()

    # -- Summary verdict --
    print()
    if bppl and cppl and btps and ctps:
        ppl_increase = (cppl - bppl) / bppl * 100
        speed_gain = ctps / btps
        mem_reduction = bv / cv if (bv and cv) else None

        print("  Summary:")
        print(f"    Accuracy cost : PPL increased by {ppl_increase:.1f}%")
        print(f"    Speed gain    : {speed_gain:.1f}× faster generation")
        if mem_reduction:
            print(f"    Memory saving : {mem_reduction:.1f}× less VRAM")
        if ppl_increase < 5 and speed_gain > 2:
            print("\n  ✓ Good trade-off: <5% accuracy loss with >2× speed gain")
        elif ppl_increase > 10:
            print("\n  ✗ High accuracy loss — consider increasing sparse_rank / dense_rank")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Compare baseline vs MiLo evaluation results")
    p.add_argument("--baseline", required=True, help="baseline.json from evaluate_model.py")
    p.add_argument("--compressed", required=True, help="compressed.json from evaluate_model.py")
    args = p.parse_args()

    base = load(args.baseline)
    comp = load(args.compressed)
    compare(base, comp)


if __name__ == "__main__":
    main()
