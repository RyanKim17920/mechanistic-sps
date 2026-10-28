"""A3 -- where in the distance spectrum each arm wins or loses.  -> results table + appendix

No figure: it answers "where does the gain live", and nothing in the abstract rests on it.

Per predicted token, the bucket is the distance back to the MOST RECENT PRIOR OCCURRENCE
of that same target token inside the same document -- i.e. how far the model would have to
look to have seen the answer before.  Tokens whose target has never occurred are their own
bucket ("never"), and they are the majority of the loss, so a bucket mean that silently
folds them in says nothing.

Difficulty matching.  Raw per-bucket means are dominated by a frequency confound: rare
targets both sit in far buckets and are intrinsically harder.  So every bucket mean is
also reported CONDITIONED on the target token's frequency decile (deciles computed once
from the validation corpus itself and shared by every arm), and the cross-arm comparison
the table uses is the frequency-matched one.  The unmatched numbers are kept beside them
precisely so the size of the confound is visible rather than assumed away.

Scoring runs at batch 1 so the (T, V) logits never need more than one sequence's worth of
memory; the token mask is the model's own (padding and EOS-input positions dropped).

Usage:
  a3_distance_nll.py --run <run|role> [--n-seq 512] [--out <json>]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

import common as C
from modeling.models.model import GPT2_TOKENS

BUCKETS = [("never", None), ("1-16", (1, 16)), ("17-64", (17, 64)),
           ("65-256", (65, 256)), ("257-1024", (257, 1024)), ("1025+", (1025, 1 << 30))]
N_DECILE = 10


def token_frequencies(data, n_tokens=20_000_000):
    """Unigram counts over a fixed prefix of the val corpus -- identical for every arm."""
    arr = np.asarray(data[:n_tokens], dtype=np.int64)
    return np.bincount(arr, minlength=GPT2_TOKENS["vocab_size"]).astype(np.float64)


def decile_of_token(freqs):
    """-> (vocab,) int array giving each token id its frequency decile (0 = rarest)."""
    order = np.argsort(freqs)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    return (ranks * N_DECILE // len(order)).astype(np.int8)


def distance_to_prior(tokens, eos):
    """-> (T,) distance from position t+1's target back to its last prior occurrence.

    -1 means "never seen before in this document".  Document boundaries are EOS tokens,
    matching the model's own document segmentation.
    """
    T = len(tokens)
    dist = np.full(T, -1, dtype=np.int64)
    last = {}
    for t in range(T):
        tok = int(tokens[t])
        if tok == eos:
            last = {}
        prev = last.get(tok)
        if prev is not None:
            dist[t] = t - prev
        last[tok] = t
    return dist


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-seq", type=int, default=C.SUB_SWEEP_SEQS)
    args = ap.parse_args()

    run = C.resolve(args.run)
    out_path = args.out or C.result_path("a3_distance_nll", run)
    C.assert_node_local_triton()

    model, cfg, family, ckpath = C.load(run)
    eos = int(cfg.model.config.eos_token_id)
    pad = cfg.model.config.pad_token_id
    data = C.val_memmap(cfg)
    freqs = token_frequencies(data)
    dec = decile_of_token(freqs)
    starts = list(range(0, len(data) - C.BLOCK - 1, C.BLOCK))[:args.n_seq]
    print(f"run={run} family={family} ckpt={ckpath}\n  {len(starts)} sequences", flush=True)

    nll_all, bucket_all, dec_all = [], [], []
    t0 = time.time()
    for si, j in enumerate(starts):
        toks = data[j:j + C.BLOCK + 1].astype(np.int64)
        X = torch.from_numpy(toks[:C.BLOCK])[None].cuda()
        tgt = toks[1:C.BLOCK + 1]
        Yt = torch.from_numpy(tgt)[None].cuda()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # C.forward_logits, not model(X): the standard family's forward requires
            # targets and returns a tuple, the others do not.  One adapter, one path.
            logits = C.forward_logits(model, X, Yt)
        lp = F.cross_entropy(logits[0].float(), Yt[0],
                             reduction="none").cpu().numpy()
        # the model's own target mask: drop positions whose INPUT token is EOS (and pad)
        keep = (toks[:C.BLOCK] != eos)
        if pad is not None:
            keep &= (toks[:C.BLOCK] != int(pad))
        # distance from the TARGET token back to its own last prior occurrence
        d_full = distance_to_prior(toks, eos)          # index t = token at position t
        d_tgt = d_full[1:C.BLOCK + 1]
        nll_all.append(lp[keep])
        bucket_all.append(d_tgt[keep])
        dec_all.append(dec[tgt][keep])
        if (si + 1) % 64 == 0:
            print(f"  seq {si+1}/{len(starts)}  {time.time()-t0:.0f}s", flush=True)

    nll = np.concatenate(nll_all)
    dist = np.concatenate(bucket_all)
    deci = np.concatenate(dec_all).astype(np.int64)
    print(f"  {len(nll):,} scored tokens", flush=True)

    def sel_of(spec):
        if spec is None:
            return dist < 0
        lo, hi = spec
        return (dist >= lo) & (dist <= hi)

    buckets = {}
    for name, spec in BUCKETS:
        s = sel_of(spec)
        row = dict(n_tokens=int(s.sum()),
                   mean_nll=round(float(nll[s].mean()), 6) if s.any() else None,
                   share=round(float(s.mean()), 6))
        per_dec = {}
        for d in range(N_DECILE):
            sd = s & (deci == d)
            if sd.sum() >= 50:
                per_dec[str(d)] = dict(n=int(sd.sum()),
                                       mean_nll=round(float(nll[sd].mean()), 6))
        row["by_frequency_decile"] = per_dec
        buckets[name] = row

    res = dict(analysis="a3_distance_nll", run=run, model_id=C.mid_of(run),
               label=C.label_of(run), family=family, checkpoint=ckpath,
               scoring=dict(kind="sub-sweep", n_seq=len(starts), n_tokens=int(len(nll))),
               bucket_definition="distance from the target token to its last prior "
                                 "occurrence in the same document",
               overall_mean_nll=round(float(nll.mean()), 6),
               buckets=buckets,
               decile_weights={str(d): round(float((deci == d).mean()), 6)
                               for d in range(N_DECILE)})
    C.save_json(out_path, res)
    print("\nbucket           n_tokens    mean NLL", flush=True)
    for name, _ in BUCKETS:
        b = buckets[name]
        print(f"  {name:10s} {b['n_tokens']:12,}  {b['mean_nll']}", flush=True)


if __name__ == "__main__":
    main()
