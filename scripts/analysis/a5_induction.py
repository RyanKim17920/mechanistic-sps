"""A5 -- induction and previous-token structure, by query stream x key stream.  -> F3

The question it answers is mechanistic and cannot be read off the attention mask: an induction head needs
a previous-token head upstream of it, so the interesting statement is not "induction
exists" but WHERE the two halves of that circuit live -- which stream supplies the
prefix-matching query, which stream supplies the keys it matches against, and at which
depth the previous-token head that feeds it sits.

Three measurements, one pass each:

  synthetic   The standard induction probe (Olsson et al., 2022): a random token block
              repeated twice inside the full 4096-token context.  Attention from a query
              inside the SECOND copy to the successor of its match in the FIRST copy is
              induction and nothing else -- no frequency, no semantics, no document
              structure can produce it.  This is the clean number.
  natural     The same quantity on real validation text, with the successor-of-any-
              earlier-occurrence definition.
  prev_token  Mass at distance exactly 1, per block, per stream (no extra GPU).  A previous-token head in memory at depth d plus prefix matching
              in the readout at depth >= d is a CIRCUIT; two scatter plots are not.

Nulls, all three reported:
  * uniform-over-visible -- the lift denominator (a head that spreads its mass evenly
    over everything it can see has lift 1 by construction);
  * row-shuffled attention -- each query's probability vector permuted over its visible
    keys, which destroys structure while preserving the distribution's shape;
  * untrained init -- identical config, freshly initialised, same sampler (`--null`).
    "Verify, do not assume" that this sits at lift ~ 1.

Family handling.  The 2x2 (query stream x key stream) is complete for the joint family
(tied SPS): both streams supply queries and both supply keys.  For two-tower at
pred_window = 0 the readout tower has NO self keys, so the readout-query row has a single
key-stream column; that is the architecture, not a gap in the measurement, and the JSON
says so explicitly via `key_streams`.  A single-stream model has one cell.

Usage:
  a5_induction.py --run <run|role> [--out <json>] [--n-seq 32] [--null/--no-null]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import common as C

BLOCK_LEN = 256          # length of the repeated random block
COPY_A = 1024            # start of the first copy
COPY_B = 3072            # start of the second copy
N_SYNTH = 8              # synthetic sequences (they are noiseless; 8 is plenty)


# --------------------------------------------------------------------------------------
# accumulation
# --------------------------------------------------------------------------------------
class Acc:
    """Per (block, query stream, key stream, head) sums, pooled over queries."""

    def __init__(self):
        self.d = {}

    def add(self, key, name, arr):
        e = self.d.setdefault(key, {})
        e.setdefault(name, []).append(arr)

    def finish(self):
        out = {}
        for key, e in self.d.items():
            out[key] = {nm: np.concatenate(v, 0) for nm, v in e.items()}
        return out


def _per_stream_masses(v, ind_mask, k_stream_id):
    """(mass, uniform_baseline) on the induction keys of one key stream, per head."""
    sel = ind_mask & (v.k_stream == k_stream_id).view(1, -1) & v.vis
    mass = (v.p * sel.unsqueeze(1)).sum(-1)                       # (nq, nh)
    n_vis = v.vis.sum(-1).clamp(min=1).float()
    base = (sel.sum(-1).float() / n_vis).view(-1, 1).expand(-1, v.n_head)
    return mass.cpu().numpy(), base.cpu().numpy()


def _shuffled_null(v, ind_mask, k_stream_id, gen):
    """Row-shuffled attention null: each query's probability vector is randomly permuted
    AMONG ITS VISIBLE KEYS, then re-measured.  Structure is destroyed while the
    distribution's shape (its peakedness) is preserved -- which is the point: a very
    peaked head that happens to sit on an induction key must beat this, not just beat
    uniform.  Permuting the probabilities and the target set together would be a no-op
    (an inner product is permutation invariant), so ONLY the probabilities move.
    """
    sel = (ind_mask & (v.k_stream == k_stream_id).view(1, -1) & v.vis)
    nq, nh, K = v.p.shape
    big = torch.full((nq, K), 2.0, device=v.p.device)
    r1 = torch.where(v.vis, torch.rand(nq, K, device=v.p.device, generator=gen), big)
    r2 = torch.where(v.vis, torch.rand(nq, K, device=v.p.device, generator=gen), big)
    src = torch.argsort(r1, dim=-1)          # visible slots first, random order
    dst = torch.argsort(r2, dim=-1)          # a second random order of the same slots
    taken = torch.gather(v.p, 2, src.unsqueeze(1).expand(-1, nh, -1))
    p_sh = torch.zeros_like(v.p)
    p_sh.scatter_(2, dst.unsqueeze(1).expand(-1, nh, -1), taken)
    return (p_sh * sel.unsqueeze(1)).sum(-1).cpu().numpy()


def synthetic_batch(cfg, rng):
    """One 4096-token context with a 256-token random block repeated twice."""
    vocab = int(cfg.model.config.vocab_size)
    eos = int(cfg.model.config.eos_token_id)
    hi = min(vocab, eos)
    toks = rng.integers(0, hi, size=C.BLOCK, dtype=np.int64)
    toks[toks == eos] = (eos + 1) % hi
    block = rng.integers(0, hi, size=BLOCK_LEN, dtype=np.int64)
    block[block == eos] = (eos + 1) % hi
    toks[COPY_A:COPY_A + BLOCK_LEN] = block
    toks[COPY_B:COPY_B + BLOCK_LEN] = block
    return toks


def run_synthetic(model, cfg, family, n_seq, seed=7):
    """-> Acc over queries inside the second copy; induction key = successor of match."""
    rng = np.random.default_rng(seed)
    acc = Acc()
    gen = torch.Generator(device="cuda"); gen.manual_seed(seed)
    # queries: inside the second copy, excluding its first position (no match yet)
    qpos = np.arange(COPY_B + 1, COPY_B + BLOCK_LEN)
    qpos = qpos[:: max(1, len(qpos) // C.N_Q_ATTN)][:C.N_Q_ATTN]
    for si in range(n_seq):
        toks = synthetic_batch(cfg, rng)
        X = torch.from_numpy(toks)[None].cuda()
        Y = torch.from_numpy(np.concatenate([toks[1:], toks[:1]]))[None].cuda()
        # induction target in TOKEN space, mapped into key space per view below
        ind_tok = torch.zeros(len(qpos), C.BLOCK, dtype=torch.bool, device="cuda")
        for r, t in enumerate(qpos):
            ind_tok[r, COPY_A + (int(t) - COPY_B) + 1] = True    # successor of the match
        for v in C.attention_views(model, cfg, family, X, Y, qpos):
            ind = ind_tok[:, v.k_tok]
            for ks in (0, 1):
                if not bool((v.k_stream == ks).any()):
                    continue
                mass, base = _per_stream_masses(v, ind, ks)
                key = (v.block, v.stream, "state" if ks == 0 else "pred")
                acc.add(key, "induction_mass", mass)
                acc.add(key, "induction_uniform", base)
                acc.add(key, "induction_shuffled", _shuffled_null(v, ind, ks, gen))
        if si == 0:
            print(f"  synthetic: {len(qpos)} queries/seq", flush=True)
    return acc


def run_natural(model, cfg, family, data, starts, qpos):
    """Natural-text induction + previous-token + sink masses."""
    acc = Acc()
    t0 = time.time()
    for si, j in enumerate(starts):
        Xnp = data[j:j + C.BLOCK].astype(np.int64)
        X, Y = C.batch_of(data, j)
        # successor-of-earlier-occurrence sets
        last, succ = {}, {}
        for t in range(C.BLOCK):
            tid = int(Xnp[t])
            succ[t] = [u + 1 for u in last.get(tid, ()) if u + 1 < C.BLOCK]
            last.setdefault(tid, []).append(t)
        ind_tok = np.zeros((len(qpos), C.BLOCK), dtype=bool)
        for r, t in enumerate(qpos):
            u = succ[int(t)]
            if u:
                ind_tok[r, u] = True
        ind_tok_t = torch.from_numpy(ind_tok).cuda()
        for v in C.attention_views(model, cfg, family, X, Y, qpos):
            ind = ind_tok_t[:, v.k_tok]
            d = v.dist()
            for ks in (0, 1):
                ks_sel = (v.k_stream == ks).view(1, -1)
                if not bool(ks_sel.any()):
                    continue
                mass, base = _per_stream_masses(v, ind, ks)
                key = (v.block, v.stream, "state" if ks == 0 else "pred")
                acc.add(key, "induction_mass", mass)
                acc.add(key, "induction_uniform", base)
                sel1 = (d == 1) & v.vis & ks_sel
                acc.add(key, "prev_token_mass",
                        (v.p * sel1.unsqueeze(1)).sum(-1).cpu().numpy())
                sel0 = (d == 0) & v.vis & ks_sel
                acc.add(key, "self_mass", (v.p * sel0.unsqueeze(1)).sum(-1).cpu().numpy())
                selfirst = (v.k_tok == 0).view(1, -1) & v.vis & ks_sel
                acc.add(key, "sink_mass",
                        (v.p * selfirst.unsqueeze(1)).sum(-1).cpu().numpy())
        print(f"  natural seq {si+1}/{len(starts)}  {time.time()-t0:.0f}s", flush=True)
    return acc


def summarise(acc, family):
    """-> (per_head rows, 2x2 aggregate)."""
    rows = []
    for (blk, qs, ks), e in sorted(acc.d.items()):
        m = {nm: np.concatenate(v, 0) for nm, v in e.items()}
        ind = m.get("induction_mass")
        base = m.get("induction_uniform")
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.where(base > 0, ind / np.maximum(base, 1e-12), np.nan)
        row = dict(block=blk, query_stream=qs, key_stream=ks,
                   induction_mass=[round(float(z), 6) for z in ind.mean(0)],
                   induction_lift=[round(float(z), 4) for z in np.nanmean(lift, 0)])
        if "induction_shuffled" in m:
            sh = m["induction_shuffled"]
            with np.errstate(divide="ignore", invalid="ignore"):
                shl = np.where(base > 0, sh / np.maximum(base, 1e-12), np.nan)
            row["induction_lift_shuffled"] = [round(float(z), 4) for z in np.nanmean(shl, 0)]
        for nm in ("prev_token_mass", "self_mass", "sink_mass"):
            if nm in m:
                row[nm] = [round(float(z), 6) for z in m[nm].mean(0)]
        rows.append(row)

    agg = {}
    for qs in C.stream_names(family):
        for ks in C.key_stream_names(family):
            cells = [r for r in rows if r["query_stream"] == qs and r["key_stream"] == ks]
            if not cells:
                continue
            lifts = np.array([z for r in cells for z in r["induction_lift"]], dtype=float)
            lifts = lifts[np.isfinite(lifts)]
            if lifts.size == 0:
                continue
            best = max(cells, key=lambda r: max(r["induction_lift"]))
            agg[f"{qs}_query__{ks}_key"] = dict(
                max_lift=round(float(np.nanmax(lifts)), 4),
                mean_lift=round(float(np.nanmean(lifts)), 4),
                n_heads_lift_gt3=int((lifts > 3).sum()),
                n_heads=int(lifts.size),
                argmax_block=best["block"],
                argmax_head=int(np.argmax(best["induction_lift"])))
    return rows, agg


def prev_token_profile(rows):
    """Per (stream, block) max-over-heads previous-token mass -- the circuit's first half."""
    prof = {}
    for r in rows:
        if "prev_token_mass" not in r:
            continue
        key = f"{r['query_stream']}::{r['block']}"
        cur = prof.get(key)
        mx = max(r["prev_token_mass"])
        if cur is None or mx > cur["max_prev_token_mass"]:
            prof[key] = dict(stream=r["query_stream"], block=r["block"],
                             max_prev_token_mass=round(float(mx), 5),
                             head=int(np.argmax(r["prev_token_mass"])),
                             mean_prev_token_mass=round(float(np.mean(r["prev_token_mass"])), 5))
    return prof


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-seq", type=int, default=C.N_SEQ_ATTN)
    ap.add_argument("--n-synth", type=int, default=N_SYNTH)
    ap.add_argument("--null", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    run = C.resolve(args.run)
    out = args.out or C.result_path("a5_induction", run)
    C.assert_node_local_triton()

    model, cfg, family, ckpath = C.load(run)
    data = C.val_memmap(cfg)
    starts = C.seq_starts(data, args.n_seq)
    qpos = C.query_positions()
    print(f"run={run} family={family} ckpt={ckpath}\n"
          f"  {len(starts)} val sequences x {len(qpos)} queries, positions in "
          f"[{C.Q_MIN}, {C.BLOCK})", flush=True)

    syn = run_synthetic(model, cfg, family, args.n_synth)
    syn_rows, syn_agg = summarise(syn, family)
    nat = run_natural(model, cfg, family, data, starts, qpos)
    nat_rows, nat_agg = summarise(nat, family)

    res = dict(analysis="a5_induction", run=run, model_id=C.mid_of(run),
               label=C.label_of(run), family=family, checkpoint=ckpath,
               query_streams=list(C.stream_names(family)),
               key_streams=list(C.key_stream_names(family)),
               note=("two-tower readout blocks have no self keys at pred_window=0, so the "
                     "readout-query row has a single key-stream column by construction"
                     if family == "two_tower" else ""),
               sampler=dict(n_val_seq=len(starts), n_queries=len(qpos),
                            q_min=C.Q_MIN, seed=C.QPOS_SEED,
                            n_synthetic_seq=args.n_synth, synthetic_block_len=BLOCK_LEN,
                            synthetic_copies=[COPY_A, COPY_B]),
               synthetic=dict(per_head=syn_rows, aggregate_2x2=syn_agg),
               natural=dict(per_head=nat_rows, aggregate_2x2=nat_agg),
               prev_token_profile=prev_token_profile(nat_rows))

    del model
    torch.cuda.empty_cache()
    if args.null:
        print("--- untrained-init null", flush=True)
        um, ucfg, ufam, utag = C.load_untrained(run, C.UNTRAINED_SEEDS[0])
        usyn = run_synthetic(um, ucfg, ufam, max(2, args.n_synth // 2))
        urows, uagg = summarise(usyn, ufam)
        unat = run_natural(um, ucfg, ufam, data, starts[:4], qpos)
        unrows, unagg = summarise(unat, ufam)
        res["untrained_null"] = dict(tag=utag,
                                     synthetic=dict(per_head=urows, aggregate_2x2=uagg),
                                     natural=dict(per_head=unrows, aggregate_2x2=unagg),
                                     prev_token_profile=prev_token_profile(unrows))
        del um
        torch.cuda.empty_cache()

    C.save_json(out, res)
    print("\n2x2 induction lift (synthetic probe, max over heads/blocks):", flush=True)
    for k, v in syn_agg.items():
        print(f"  {k:28s} max={v['max_lift']:9.2f}  mean={v['mean_lift']:7.2f}  "
              f"heads>3: {v['n_heads_lift_gt3']}/{v['n_heads']}", flush=True)


if __name__ == "__main__":
    main()
