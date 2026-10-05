#!/usr/bin/env python3
"""Training compute Sequential 6+6 needs to reach the Transformer's pre-decay loss.

The Transformer (untied head) is evaluated at its pre-learning-rate-decay checkpoint
(the ladder checkpoint nearest 18.0B tokens, just before the decay over the final 2B
tokens). For each seed of Sequential 6+6, the token count at which its checkpoint ladder
first reaches that validation NLL is found by linear interpolation between the two
bracketing checkpoints. Both models have the same FLOPs per token, so the token ratio is
the training-FLOP ratio (the script asserts this).

Reads only the committed ledger snapshot (scripts/analysis/results/ledger.jsonl); CPU,
no arguments needed.

    python scripts/analysis/predecay_match.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parents[1] / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import paper_data as D  # noqa: E402
from plotting.paper_style import MANIFEST  # noqa: E402

PRE_DECAY_TOKENS = 18.0e9


def pre_decay_point(run: str) -> tuple[float, float]:
    tokens, nll = D.ladder_curve(run)
    i = int(np.argmin(np.abs(tokens - PRE_DECAY_TOKENS)))
    return float(tokens[i]), float(nll[i])


def tokens_to_reach(run: str, target: float) -> float | None:
    tokens, nll = D.ladder_curve(run)
    hit = np.flatnonzero(nll <= target)
    if hit.size == 0:
        return None
    k = int(hit[0])
    if k == 0:
        return float(tokens[0])
    return float(np.interp(target, [nll[k], nll[k - 1]], [tokens[k], tokens[k - 1]]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default=MANIFEST["body"]["transformer"],
                    help="arm whose pre-decay loss is the target (default: the body Transformer)")
    ap.add_argument("--model", default=MANIFEST["roles"]["sequential6"],
                    help="arm that must reach it (default: Sequential 6+6)")
    a = ap.parse_args()

    rows = D.ledger_rows()
    ratios, reaches, targets = [], [], []
    for ref in D.mean_runs(a.reference):
        ref_tokens, target = pre_decay_point(ref)
        targets.append(target)
        for run in D.mean_runs(a.model):
            fpt = rows[run]["fwd_flops_per_token"] / rows[ref]["fwd_flops_per_token"]
            assert abs(fpt - 1) < 1e-3, f"{run} vs {ref}: FLOPs/token ratio {fpt:.4f} != 1"
            reach = tokens_to_reach(run, target)
            if reach is None:
                print(f"{run}: never reaches {ref}'s pre-decay NLL {target:.4f}")
                continue
            ratios.append(ref_tokens / reach)
            reaches.append(reach)
            print(f"{run} reaches {ref}'s pre-decay NLL {target:.4f} "
                  f"(at {ref_tokens / 1e9:.1f}B tokens) after {reach / 1e9:.2f}B tokens: "
                  f"{ratios[-1]:.3f}x fewer training FLOPs")
    if targets:
        print(f"reference pre-decay NLL, {len(targets)}-seed mean: {np.mean(targets):.3f}")
    if ratios:
        print(f"tokens to reach it: {min(reaches) / 1e9:.2f}-{max(reaches) / 1e9:.2f}B")
        print(f"ratio range {min(ratios):.2f}-{max(ratios):.2f}x, mean {np.mean(ratios):.2f}x")


if __name__ == "__main__":
    main()
