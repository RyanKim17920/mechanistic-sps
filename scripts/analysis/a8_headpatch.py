"""A8 -- causal head patch: does the READOUT matching head USE the MEMORY prev-token head?

A5 established the two halves of an induction circuit separately: previous-token heads
live in the MEMORY tower, prefix-matching (induction) heads live in the READOUT tower.
Co-existence is not a circuit: A5 never shows that the matching head USES the
prev-token head.  This script is the cheapest causal test of exactly that claim:

    zero the output of a candidate memory-stream previous-token head (its head slice of
    the attention output, before `StateBlock.c_proj`), re-run A5's OWN synthetic
    repeated-block probe, and measure whether the readout matching head's induction lift
    collapses.

Nothing is rebuilt.  The probe sequences, the induction-key definition, the attention
views, the lift computation and the model loader are `a5_induction` / `common` verbatim;
the only new thing is the ablation, which is a wrapper around `TwoTowerModel._attend`
(the ONE attention entry point of the two-tower forward, used both by the real forward
and by `common.two_tower_capture`, so the patch is on the model the analysis measures,
not on a copy of it).  Zeroing a head's `(b, h, t, d)` slice of the attention output is
exactly "that head contributes nothing to the residual stream", because `finish_attn`
flattens heads and applies `c_proj` to the concatenation.

Arms (all on the same 8 synthetic sequences, same seed, as A5):
  baseline            no ablation
  prev_k  (k=1..3)    zero one of the top-3 memory-stream previous-token heads
  prev_joint          zero all three at once
  ctrl_k  (k=1..3)    CONTROL: zero a random OTHER head in the SAME memory block, chosen
                      among the heads of that block that are NOT in the prev-token top-3
                      and sit below that block's median previous-token mass
  ctrl_joint          zero all three controls at once

Reported per arm: the readout matching head's induction lift (the headline), the
readout-wide max/mean lift, the memory-tower max lift, and the induction-probe NLL
(second repeat, and its core excluding the first 64 positions of the repeat, where a
match is guaranteed to exist).

Control DISTRIBUTION (a single draw is not a distribution):
  --ctrl-mode {same_block_lowprev (default), random_any, pattern_matched}
  --n-ctrl-draws N   N independent control draws (draw k: rng seed ctrl_seed + k)
  --ctrl-seed S      default CTRL_SEED, so the default run reproduces bit-for-bit
  --ctrl-exclude-layers L [L ...]  1-indexed memory layers whose heads may never be a
                     control (random_any / pattern_matched only); treated heads in those
                     layers stay treated and are reported; output gets '-exL<..>'
With a non-default mode / N>1 / seed, the result goes to
`a8_headpatch_<run>_ctrl-<mode>-n<N>[-s<seed>].json` (never the default file) and adds
`ctrl_draws` (per-draw joint-control cost), `ctrl_summary` (mean, sd, and the fraction of
draws costing >= prev_joint -- an empirical p-value) and `ctrl_mode`.  Selection rules
and the pattern_matched features are documented in `select_controls`.

Usage:
  a8_headpatch.py --run <run|role> [--out <json>] [--n-synth 8] [--no-joint] [--no-ctrl]
                  [--ctrl-mode M] [--n-ctrl-draws N] [--ctrl-seed S] [--ctrl-match-k K]
                  [--ctrl-exclude-layers L ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402

import common as C                                                     # noqa: E402
import a5_induction as A5                                              # noqa: E402

CTRL_SEED = 20260919
CTRL_MODES = ("same_block_lowprev", "random_any", "pattern_matched")
DEFAULT_CTRL_MODE = "same_block_lowprev"
MATCH_FEATURES = ("entropy", "log_distance", "sink_mass", "out_norm")
MATCH_K = 3            # pattern_matched: draw among the k nearest unused candidates
PM_PREV_FRAC = 0.5     # pattern_matched: candidates' prev mass < this x weakest treated
NLL_SKIP = 64          # positions of the second repeat with no usable match yet


# --------------------------------------------------------------------------------------
# the ablation
# --------------------------------------------------------------------------------------
class HeadZero:
    """Zero given heads' attention output in the MEMORY (state) tower.

    Wraps `model._attend`, the two-tower forward's only attention dispatch.  The state
    tower calls it once per state block in depth order (`_forward_towers`), so a call counter modulo the number of state blocks is the
    block index; `zero_by_block` maps that index to the head indices to zero.

    The returned tensor is `(b, n_head, t, head_dim)` for every backend, and
    `StateBlock.finish_attn` concatenates heads before `c_proj`, so setting a head's
    slice to zero removes exactly that head's contribution to the residual stream.
    """

    def __init__(self, model, zero_by_block: dict):
        self.model = model
        self.zero_by_block = {int(b): list(hs) for b, hs in zero_by_block.items() if hs}
        self.n_state = len(model.transformer.state_h)
        self._orig = None

    def __enter__(self):
        if not self.zero_by_block:
            return self
        self._orig = self.model._attend
        orig, n_state, zbb = self._orig, self.n_state, self.zero_by_block
        state = {"n": 0}

        def patched(q, k, v, masks, stream):
            y = orig(q, k, v, masks, stream)
            if stream != "state":
                return y
            blk = state["n"] % n_state
            state["n"] += 1
            hs = zbb.get(blk)
            if hs:
                y = y.clone()
                y[:, hs, :, :] = 0
            return y

        self.model._attend = patched
        return self

    def __exit__(self, *exc):
        if self._orig is not None:
            self.model._attend = self._orig
            self._orig = None
        return False


# --------------------------------------------------------------------------------------
# head selection, read off the committed A5 result
# --------------------------------------------------------------------------------------
def pick_heads(a5res: dict, n_prev: int = 3, ctrl_mode: str = None,
               ctrl_seed: int = CTRL_SEED, draw: int = 0, head_stats=None,
               match_k: int = None, exclude_layers=()):
    """-> (matching head dict, [top-n prev-token memory heads], [control heads]).

    Controls come from `select_controls` with rng `default_rng(draw_seed(ctrl_seed,
    draw))`; the defaults reproduce the historical single draw bit-for-bit.
    """
    ctrl_mode = ctrl_mode or DEFAULT_CTRL_MODE
    match_k = MATCH_K if match_k is None else match_k
    syn = a5res["synthetic"]["per_head"]
    nat = a5res["natural"]["per_head"]

    # strongest readout-query matching head on the SYNTHETIC probe
    best = None
    for r in syn:
        if r["query_stream"] != "pred":
            continue
        for h, lift in enumerate(r["induction_lift"]):
            if lift is None or not np.isfinite(lift):
                continue
            if best is None or lift > best["lift"]:
                best = dict(block=int(r["block"]), head=int(h), lift=float(lift),
                            key_stream=r["key_stream"])
    assert best is not None, "no readout-query induction lift in the A5 result"

    # memory-stream previous-token heads (natural text, as A5 measures prev_token_mass)
    prev_all = []
    per_block = {}
    for r in nat:
        if r["query_stream"] != "state" or "prev_token_mass" not in r:
            continue
        b = int(r["block"])
        per_block[b] = list(r["prev_token_mass"])
        for h, m in enumerate(r["prev_token_mass"]):
            prev_all.append(dict(block=b, head=int(h), prev_token_mass=float(m)))
    prev_all.sort(key=lambda d: -d["prev_token_mass"])
    top = prev_all[:n_prev]

    rng = np.random.default_rng(draw_seed(ctrl_seed, draw))
    ctrl = select_controls(top, per_block, mode=ctrl_mode, rng=rng,
                           head_stats=head_stats, match_k=match_k,
                           exclude_layers=exclude_layers)
    return best, top, ctrl


# --------------------------------------------------------------------------------------
# control-head selection (shared by A8 and A9 -- A9.pick_heads delegates here)
# --------------------------------------------------------------------------------------
def draw_seed(ctrl_seed: int = CTRL_SEED, draw: int = 0) -> int:
    """rng seed of control draw `draw`.  Draw 0 of the default seed IS the historical
    single draw (`np.random.default_rng(CTRL_SEED)`), so every committed result
    reproduces bit-for-bit; draws 1..N-1 use consecutive seeds."""
    return int(ctrl_seed) + int(draw)


def select_controls(top, per_block, mode=DEFAULT_CTRL_MODE, rng=None, head_stats=None,
                    exclude=(), match_k=MATCH_K, exclude_layers=()):
    """-> one control head per treated (previous-token) head, same count as `top`.

    top        the treated heads, [{block, head, prev_token_mass}, ...]
    per_block  {block: [prev_token_mass per head]} for the MEMORY stream
    rng        np.random.Generator (one per draw; see `draw_seed`)
    exclude    extra (block, head) keys a control may never be (used for the
               single-stream family, where the readout matching head lives in the same
               stream as the memory heads)
    exclude_layers  1-indexed memory layers (layer L = block L-1) whose heads are removed
               from the control CANDIDATE pool (random_any and pattern_matched; not
               allowed with same_block_lowprev).  Treated heads in those layers stay
               treated -- see `treated_in_excluded_layers`.  In pattern_matched the
               z-scoring still uses ALL heads, so the feature space is unchanged; only the
               candidate set shrinks.  Empty (default) = the unchanged selection.

    Modes
    -----
    same_block_lowprev  (default, the historical rule, code unchanged)  for each treated
        head, a head of the SAME block whose previous-token mass is <= that block's
        median and which is not itself a treated head.  `exclude` is NOT applied here so
        the historical draw is reproduced exactly.
    random_any  `len(top)` distinct heads drawn uniformly (without replacement) from
        every memory-stream head in ANY block, minus the treated heads and `exclude`.
    pattern_matched  for each treated head, a non-previous-token head whose attention
        statistics are closest, measured LIVE on the same eval batch the analysis scores
        (`mem_head_stats`).  Matching features (MATCH_FEATURES):
            entropy        mean attention entropy (nats) over the visible keys
            log_distance   log1p of the mean attended token distance sum_k p_k (q - k)
                           (log because raw distances are heavy-tailed and would
                           dominate the z-score)
            sink_mass      mean attention mass on token position 0
            out_norm       mean L2 norm of the head's output slice entering c_proj
        Each feature is z-scored over ALL memory-stream heads of the model; distance is
        the L2 norm in z-space.  Candidates exclude the treated heads, `exclude`, and
        every head whose previous-token mass is >= PM_PREV_FRAC x the weakest treated
        head's mass (i.e. heads that are themselves previous-token heads).  Treated heads
        are visited in an rng-permuted order; each takes one head uniformly from the
        `match_k` nearest still-unused candidates (sampling WITHOUT replacement, so the
        controls are distinct).  match_k=1 is the deterministic nearest neighbour; k>1
        is what gives N draws a distribution.
    """
    assert mode in CTRL_MODES, f"unknown ctrl mode {mode!r}; one of {CTRL_MODES}"
    if rng is None:
        rng = np.random.default_rng(CTRL_SEED)
    top_keys = {(d["block"], d["head"]) for d in top}
    xblocks = layers_to_blocks(exclude_layers)
    ctrl = []
    if mode == "same_block_lowprev":
        assert not xblocks, "exclude_layers is not defined for same_block_lowprev"
        for d in top:
            b = d["block"]
            masses = per_block[b]
            med = float(np.median(masses))
            elig = [h for h in range(len(masses))
                    if (b, h) not in top_keys and masses[h] <= med]
            if not elig:
                elig = [h for h in range(len(masses)) if (b, h) not in top_keys]
            h = int(rng.choice(elig))
            ctrl.append(dict(block=b, head=h, prev_token_mass=float(masses[h])))
        return ctrl

    bad = top_keys | {(int(b), int(h)) for b, h in exclude}
    if mode == "random_any":
        pool = [(int(b), int(h)) for b in sorted(per_block)
                for h in range(len(per_block[b]))
                if (int(b), int(h)) not in bad and int(b) not in xblocks]
        assert len(pool) >= len(top), "not enough non-treated memory heads to draw from"
        idx = rng.choice(len(pool), size=len(top), replace=False)
        for i in idx:
            b, h = pool[int(i)]
            ctrl.append(dict(block=b, head=h, prev_token_mass=float(per_block[b][h])))
        return ctrl

    # pattern_matched
    assert head_stats, "pattern_matched needs live head statistics (mem_head_stats)"
    keys = sorted(head_stats)
    F = np.array([[float(head_stats[k][f]) for f in MATCH_FEATURES] for k in keys])
    mu, sd = F.mean(0), F.std(0)
    Z = (F - mu) / np.where(sd > 1e-12, sd, 1.0)
    zi = {k: Z[i] for i, k in enumerate(keys)}
    thr = PM_PREV_FRAC * min(float(d["prev_token_mass"]) for d in top)
    cand = [k for k in keys if k not in bad and int(k[0]) not in xblocks
            and float(per_block[k[0]][k[1]]) < thr]
    assert len(cand) >= len(top), "not enough non-previous-token heads to match"
    taken = set()
    chosen = [None] * len(top)
    for ti in rng.permutation(len(top)):
        d = top[int(ti)]
        z0 = zi[(int(d["block"]), int(d["head"]))]
        avail = [k for k in cand if k not in taken]
        dist = np.array([float(np.linalg.norm(zi[k] - z0)) for k in avail])
        order = np.argsort(dist, kind="stable")[:max(1, int(match_k))]
        j = int(order[int(rng.integers(0, len(order)))])
        k = avail[j]
        taken.add(k)
        chosen[int(ti)] = dict(block=k[0], head=k[1],
                               prev_token_mass=float(per_block[k[0]][k[1]]),
                               matched_to=dict(block=int(d["block"]), head=int(d["head"])),
                               match_distance=round(float(dist[j]), 4))
    return chosen


def mem_head_stats(model, cfg, family, seqs, qpos=None):
    """Live per-head attention statistics of the MEMORY stream on `seqs` (the analysis'
    own eval batch).  -> {(block, head): {entropy, log_distance, sink_mass, out_norm,
    mean_distance}}.  See `select_controls` for the definitions.
    """
    import a9_patching as A9          # lazy: A9 imports this module
    mem_q = "single" if family == "standard" else "state"
    if qpos is None:
        qpos = C.query_positions()
    sites = A9.mem_attn_sites(model, family)
    acc = {}

    def add(k, f, v):
        acc.setdefault(k, {}).setdefault(f, []).append(float(v))

    for toks in seqs:
        toks = np.asarray(toks, dtype=np.int64)
        X = torch.from_numpy(toks)[None].cuda()
        Y = torch.from_numpy(np.concatenate([toks[1:], toks[:1]]))[None].cuda()
        # attention_views' standard path calls model(X, Y) un-cloned, and that forward
        # writes IGNORE_INDEX into its targets in place; hand it a copy
        for v in C.attention_views(model, cfg, family, X, Y.clone(), qpos):
            if v.stream != mem_q:
                continue
            p = v.p.float()                                         # (nq, nh, K)
            ent = -(p * torch.log(p.clamp_min(1e-30))).sum(-1).mean(0)
            dist = (p * v.dist().float().unsqueeze(1)).sum(-1).mean(0)
            sink = p[:, :, v.k_tok == 0].sum(-1).mean(0)
            for h in range(v.n_head):
                k = (int(v.block), h)
                add(k, "entropy", ent[h]); add(k, "mean_distance", dist[h])
                add(k, "sink_mass", sink[h])
        # output norm: the head's slice of the c_proj input, memory slots only
        T = X.shape[1]
        handles = []

        def mk(blk, nh, hd):
            def pre(mod, args):
                y = args[0]
                if y.dim() == 4:
                    y = y[:, :, 0]
                elif y.shape[1] == 2 * T:
                    y = y[:, 0::2]
                n = y.float().reshape(y.shape[0], y.shape[1], nh, hd).norm(dim=-1)
                for h, val in enumerate(n.mean(dim=(0, 1)).tolist()):
                    add((blk, h), "out_norm", val)
                return None
            return pre
        for blk, mod, nh, hd in sites:
            handles.append(mod.register_forward_pre_hook(mk(int(blk), int(nh), int(hd))))
        try:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lg = C.forward_logits(model, X, Y)
            del lg
        finally:
            for hd_ in handles:
                hd_.remove()
    out = {}
    for k, fs in acc.items():
        o = {f: float(np.mean(v)) for f, v in fs.items()}
        o["log_distance"] = float(np.log1p(o["mean_distance"]))
        out[k] = o
    return out


def layers_to_blocks(exclude_layers):
    """1-indexed layer numbers -> set of 0-indexed block indices."""
    out = set()
    for layer in exclude_layers or ():
        assert int(layer) >= 1, f"layers are 1-indexed, got {layer}"
        out.add(int(layer) - 1)
    return out


def treated_in_excluded_layers(top, exclude_layers):
    """Treated (previous-token) heads sitting in an excluded layer.  They are KEPT in the
    treated set (the claim is about them); this list is recorded so it is visible."""
    xb = layers_to_blocks(exclude_layers)
    return [dict(block=int(d["block"]), head=int(d["head"]), layer=int(d["block"]) + 1,
                 prev_token_mass=float(d["prev_token_mass"]))
            for d in top if int(d["block"]) in xb]


def exclude_layers_record(top, exclude_layers):
    """The result-JSON fields for a layer-excluded control run ({} when not excluded)."""
    if not exclude_layers:
        return {}
    kept = treated_in_excluded_layers(top, exclude_layers)
    if kept:
        print("  NOTE: treated prev-token head(s) in an excluded layer, kept as treated: "
              + ", ".join(f"L{d['layer']}H{d['head'] + 1}" for d in kept), flush=True)
    return dict(ctrl_exclude_layers=sorted(int(x) for x in exclude_layers),
                ctrl_exclude_layers_note=("layers are 1-indexed (layer L = block L-1); "
                                          "heads there are removed from the control pool "
                                          "only; treated heads there stay treated"),
                treated_in_excluded_layers=kept)


def stats_to_json(head_stats):
    return [dict(block=int(b), head=int(h), **{f: round(v, 6) for f, v in s.items()})
            for (b, h), s in sorted(head_stats.items())]


def ctrl_suffix(mode, n_draws, seed=CTRL_SEED, match_k=MATCH_K, exclude_layers=()):
    """'' for the historical default, else '_ctrl-<mode>-n<N>[-k<K>][-s<seed>]' so
    non-default control sets never overwrite the default result file or each other
    (-k only for pattern_matched with a non-default match_k; '-exL1' etc. when control
    layers are excluded)."""
    if (mode == DEFAULT_CTRL_MODE and int(n_draws) == 1 and int(seed) == CTRL_SEED
            and not exclude_layers):
        return ""
    s = f"_ctrl-{mode}-n{int(n_draws)}"
    if mode == "pattern_matched" and int(match_k) != MATCH_K:
        s += f"-k{int(match_k)}"
    if int(seed) != CTRL_SEED:
        s += f"-s{int(seed)}"
    if exclude_layers:
        s += "-ex" + "".join(f"L{int(x)}" for x in sorted(set(int(x) for x in exclude_layers)))
    return s


def summarise_draws(costs, prev_cost):
    """mean / sd / empirical p = fraction of control draws costing >= the treated set."""
    c = np.asarray(costs, dtype=float)
    return dict(n_draws=int(len(c)), prev_cost=float(prev_cost),
                ctrl_mean=float(c.mean()) if len(c) else None,
                ctrl_sd=float(c.std(ddof=1)) if len(c) > 1 else 0.0,
                ctrl_min=float(c.min()) if len(c) else None,
                ctrl_max=float(c.max()) if len(c) else None,
                frac_ctrl_ge_prev=float((c >= prev_cost).mean()) if len(c) else None,
                # permutation-style p with the +1 correction: (r + 1) / (N + 1), where r
                # = number of draws costing >= the treated set; never exactly 0
                p_plus1=(float(((c >= prev_cost).sum() + 1) / (len(c) + 1))
                         if len(c) else None),
                prev_over_ctrl_mean=(float(prev_cost / c.mean())
                                     if len(c) and abs(c.mean()) > 1e-12 else None))


# --------------------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------------------
def lift_summary(model, cfg, family, n_synth, match):
    """A5's synthetic probe + A5's summariser, read out at the matching head."""
    acc = A5.run_synthetic(model, cfg, family, n_synth)
    rows, agg = A5.summarise(acc, family)
    out = dict(matching_head_lift=None, readout_max_lift=None, readout_mean_lift=None,
               memory_max_lift=None)
    pred_lifts, state_lifts = [], []
    for r in rows:
        lifts = [z for z in r["induction_lift"] if z is not None and np.isfinite(z)]
        if r["query_stream"] == "pred":
            pred_lifts += lifts
            if int(r["block"]) == match["block"] and r["key_stream"] == match["key_stream"]:
                out["matching_head_lift"] = float(r["induction_lift"][match["head"]])
                out["matching_head_mass"] = float(r["induction_mass"][match["head"]])
        else:
            state_lifts += lifts
    if pred_lifts:
        out["readout_max_lift"] = float(np.max(pred_lifts))
        out["readout_mean_lift"] = float(np.mean(pred_lifts))
    if state_lifts:
        out["memory_max_lift"] = float(np.max(state_lifts))
    out["aggregate_2x2"] = agg
    return out


def probe_nll(model, cfg, n_synth, seed=7):
    """NLL on the SAME synthetic sequences A5 scores (same rng seed and draw order)."""
    rng = np.random.default_rng(seed)
    sums = dict(second_core=0.0, second_all=0.0, full=0.0)
    cnts = dict(second_core=0, second_all=0, full=0)
    b0, b1 = A5.COPY_B, A5.COPY_B + A5.BLOCK_LEN
    for _ in range(n_synth):
        toks = A5.synthetic_batch(cfg, rng)
        X = torch.from_numpy(toks)[None].cuda()
        Y = torch.from_numpy(np.concatenate([toks[1:], toks[:1]]))[None].cuda()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = C.forward_logits(model, X, Y)
        lp = torch.log_softmax(logits[0].float(), dim=-1)
        nll = -lp.gather(1, Y[0].view(-1, 1)).squeeze(1)      # nll[t] = -log p(tok t+1)
        nll = nll[: C.BLOCK - 1]                               # last target wraps around
        segs = dict(second_core=nll[b0 + NLL_SKIP: b1 - 1],
                    second_all=nll[b0: b1 - 1],
                    full=nll)
        for k, s in segs.items():
            sums[k] += float(s.sum())
            cnts[k] += int(s.numel())
        del logits, lp, nll
    return {k: round(sums[k] / max(1, cnts[k]), 5) for k in sums}


def run_arm(model, cfg, family, n_synth, match, name, zero_by_block, heads):
    t0 = time.time()
    print(f"\n=== arm {name}: zero {heads}", flush=True)
    with HeadZero(model, zero_by_block):
        res = lift_summary(model, cfg, family, n_synth, match)
        res["nll"] = probe_nll(model, cfg, n_synth)
    res["arm"] = name
    res["ablated_heads"] = heads
    res["seconds"] = round(time.time() - t0, 1)
    print(f"    matching-head lift={res['matching_head_lift']}  "
          f"readout_max={res['readout_max_lift']}  nll={res['nll']}  "
          f"({res['seconds']}s)", flush=True)
    return res


def add_ctrl_args(ap):
    """The control-distribution flags, shared by A8 and A9.  Defaults = historical run."""
    ap.add_argument("--ctrl-seed", type=int, default=CTRL_SEED,
                    help="seed of control draw 0; draw k uses ctrl_seed + k")
    ap.add_argument("--n-ctrl-draws", type=int, default=1,
                    help="independent control draws; >1 reports mean/sd/empirical p")
    ap.add_argument("--ctrl-mode", choices=CTRL_MODES, default=DEFAULT_CTRL_MODE)
    ap.add_argument("--ctrl-match-k", type=int, default=MATCH_K,
                    help="pattern_matched: sample among the k nearest unused candidates")
    ap.add_argument("--ctrl-exclude-layers", type=int, nargs="*", default=[],
                    help="1-indexed memory layers removed from the control candidate pool "
                         "(random_any / pattern_matched); default none")


def probe_seqs(cfg, n_synth, seed=7):
    """The synthetic sequences A5/probe_nll score (same rng seed and draw order)."""
    rng = np.random.default_rng(seed)
    return [A5.synthetic_batch(cfg, rng) for _ in range(n_synth)]


def _derive(base, r):
    r["d_matching_head_lift"] = round(r["matching_head_lift"] - base["matching_head_lift"], 4)
    r["frac_matching_head_lift_remaining"] = (
        round(r["matching_head_lift"] / base["matching_head_lift"], 4)
        if base["matching_head_lift"] else None)
    r["d_readout_max_lift"] = round(r["readout_max_lift"] - base["readout_max_lift"], 4)
    r["d_nll"] = {k: round(r["nll"][k] - base["nll"][k], 5) for k in r["nll"]}
    return r


def main_ctrl_draws(args, run, out, a5res):
    """Non-default control set: N independent control draws in `args.ctrl_mode`.

    The existing keys are filled exactly as in the default run, with draw 0 standing in
    for the single control (so `arms.ctrl_k` / `arms.ctrl_joint` / `control_heads` keep
    their meaning).  Every draw's JOINT control arm (all n_prev control heads zeroed at
    once, the comparator of `prev_joint`) goes to `ctrl_draws`; `ctrl_summary` compares
    the cost distribution with the previous-token heads' cost.
      cost (headline)  = baseline matching-head lift - arm lift      (lift destroyed)
      cost (secondary) = arm NLL - baseline NLL on the second repeat's core
    """
    model, cfg, family, ckpath = C.load(run)
    stats = None
    if args.ctrl_mode == "pattern_matched":
        t0 = time.time()
        stats = mem_head_stats(model, cfg, family, probe_seqs(cfg, args.n_synth))
        print(f"  head stats on {args.n_synth} probe seqs ({time.time()-t0:.1f}s)", flush=True)
    draws = []
    for k in range(args.n_ctrl_draws):
        match, top_prev, ctrl = pick_heads(a5res, args.n_prev, ctrl_mode=args.ctrl_mode,
                                           ctrl_seed=args.ctrl_seed, draw=k,
                                           head_stats=stats, match_k=args.ctrl_match_k,
                                           exclude_layers=args.ctrl_exclude_layers)
        draws.append(ctrl)
        print(f"  draw {k}: " + ", ".join(f"b{d['block']}h{d['head']}" for d in ctrl), flush=True)
    ctrl = draws[0]
    print(f"run={run} mode={args.ctrl_mode} draws={args.n_ctrl_draws}\n  matching head: "
          f"block={match['block']} head={match['head']} L0(a5)={match['lift']:.2f}", flush=True)
    for i, d in enumerate(top_prev, 1):
        print(f"  prev-token #{i}: block={d['block']} head={d['head']} "
              f"mass={d['prev_token_mass']:.4f}", flush=True)

    def zspec(heads):
        z = {}
        for d in heads:
            z.setdefault(d["block"], []).append(d["head"])
        return z

    arms = [("baseline", {}, [])]
    for i, d in enumerate(top_prev, 1):
        arms.append((f"prev_{i}", {d["block"]: [d["head"]]}, [d]))
    for i, d in enumerate(ctrl, 1):
        arms.append((f"ctrl_{i}", {d["block"]: [d["head"]]}, [d]))
    arms.append(("prev_joint", zspec(top_prev), list(top_prev)))
    arms.append(("ctrl_joint", zspec(ctrl), list(ctrl)))
    results = {}
    for name, zbb, heads in arms:
        results[name] = run_arm(model, cfg, family, args.n_synth, match, name, zbb, heads)
    base = results["baseline"]
    for name, r in results.items():
        if name != "baseline":
            _derive(base, r)

    def cost(r):
        return base["matching_head_lift"] - r["matching_head_lift"]

    def cost_nll(r):
        return r["nll"]["second_core"] - base["nll"]["second_core"]

    ctrl_draws = []
    for k, cd in enumerate(draws):
        r = results["ctrl_joint"] if k == 0 else _derive(base, run_arm(
            model, cfg, family, args.n_synth, match, f"ctrl_joint_draw{k}", zspec(cd), cd))
        ctrl_draws.append(dict(
            draw=k, seed=draw_seed(args.ctrl_seed, k), heads=cd,
            matching_head_lift=r["matching_head_lift"],
            frac_matching_head_lift_remaining=r["frac_matching_head_lift_remaining"],
            nll=r["nll"], d_nll=r["d_nll"],
            cost=round(cost(r), 4), cost_nll_second_core=round(cost_nll(r), 5)))
    pj = results["prev_joint"]
    summary = dict(
        metric="matching-head induction lift destroyed (baseline - arm), joint arms",
        comparator="prev_joint",
        **summarise_draws([d["cost"] for d in ctrl_draws], cost(pj)),
        nll_second_core=summarise_draws([d["cost_nll_second_core"] for d in ctrl_draws],
                                        cost_nll(pj)))
    res = dict(analysis="a8_headpatch", run=run, model_id=C.mid_of(run),
               label=C.label_of(run), family=family, checkpoint=ckpath,
               question=("does the readout matching head USE the memory prev-token head? "
                         "zero the memory head's output, re-measure the readout head's "
                         "induction lift on A5's synthetic probe"),
               matching_head=match, prev_token_heads=top_prev, control_heads=ctrl,
               probe=dict(source="a5_induction.run_synthetic (verbatim)",
                          n_synthetic_seq=args.n_synth,
                          synthetic_block_len=A5.BLOCK_LEN,
                          synthetic_copies=[A5.COPY_A, A5.COPY_B],
                          nll_skip_first=NLL_SKIP, ctrl_seed=args.ctrl_seed),
               arms=results,
               ctrl_mode=args.ctrl_mode, n_ctrl_draws=args.n_ctrl_draws,
               ctrl_seed=args.ctrl_seed, ctrl_match_k=args.ctrl_match_k,
               ctrl_draws=ctrl_draws, ctrl_summary=summary,
               **exclude_layers_record(top_prev, args.ctrl_exclude_layers))
    if stats is not None:
        res["ctrl_match_features"] = list(MATCH_FEATURES)
        res["ctrl_match_stats"] = stats_to_json(stats)
    C.save_json(out, res)
    print(f"\n[{args.ctrl_mode}] prev_joint cost={summary['prev_cost']:.2f}  ctrl "
          f"{summary['ctrl_mean']:.2f} +- {summary['ctrl_sd']:.2f}  "
          f"p(ctrl>=prev)={summary['frac_ctrl_ge_prev']:.3f}", flush=True)
    for d in ctrl_draws:
        print(f"  draw {d['draw']:2d} cost={d['cost']:9.2f}  "
              f"frac={d['frac_matching_head_lift_remaining']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-synth", type=int, default=A5.N_SYNTH)
    ap.add_argument("--n-prev", type=int, default=3)
    ap.add_argument("--joint", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ctrl", action=argparse.BooleanOptionalAction, default=True)
    add_ctrl_args(ap)
    args = ap.parse_args()
    sfx = ctrl_suffix(args.ctrl_mode, args.n_ctrl_draws, args.ctrl_seed, args.ctrl_match_k,
                      args.ctrl_exclude_layers)
    multi = bool(sfx)

    run = C.resolve(args.run)
    out = args.out or C.result_path("a8_headpatch", run + sfx)
    C.assert_node_local_triton()

    a5_path = C.result_path("a5_induction", run)
    assert os.path.exists(a5_path), f"A8 reads A5's committed result; missing {a5_path}"
    with open(a5_path) as f:
        a5res = json.load(f)
    assert a5res["family"] == "two_tower", "A8's ablation is the two-tower state tower"

    if multi:
        return main_ctrl_draws(args, run, out, a5res)

    match, top_prev, ctrl = pick_heads(a5res, args.n_prev)
    print(f"run={run}\n  matching head: block={match['block']} head={match['head']} "
          f"L0(a5)={match['lift']:.2f} key_stream={match['key_stream']}", flush=True)
    for i, d in enumerate(top_prev, 1):
        print(f"  prev-token #{i}: block={d['block']} head={d['head']} "
              f"mass={d['prev_token_mass']:.4f}", flush=True)
    for i, d in enumerate(ctrl, 1):
        print(f"  control   #{i}: block={d['block']} head={d['head']} "
              f"mass={d['prev_token_mass']:.4f}", flush=True)

    model, cfg, family, ckpath = C.load(run)
    arms = [("baseline", {}, [])]
    for i, d in enumerate(top_prev, 1):
        arms.append((f"prev_{i}", {d["block"]: [d["head"]]}, [d]))
    if args.ctrl:
        for i, d in enumerate(ctrl, 1):
            arms.append((f"ctrl_{i}", {d["block"]: [d["head"]]}, [d]))
    if args.joint:
        jz, cz = {}, {}
        for d in top_prev:
            jz.setdefault(d["block"], []).append(d["head"])
        arms.append(("prev_joint", jz, list(top_prev)))
        if args.ctrl:
            for d in ctrl:
                cz.setdefault(d["block"], []).append(d["head"])
            arms.append(("ctrl_joint", cz, list(ctrl)))

    results = {}
    for name, zbb, heads in arms:
        results[name] = run_arm(model, cfg, family, args.n_synth, match, name, zbb, heads)

    base = results["baseline"]
    for name, r in results.items():
        if name == "baseline":
            continue
        r["d_matching_head_lift"] = round(r["matching_head_lift"] - base["matching_head_lift"], 4)
        r["frac_matching_head_lift_remaining"] = (
            round(r["matching_head_lift"] / base["matching_head_lift"], 4)
            if base["matching_head_lift"] else None)
        r["d_readout_max_lift"] = round(r["readout_max_lift"] - base["readout_max_lift"], 4)
        r["d_nll"] = {k: round(r["nll"][k] - base["nll"][k], 5) for k in r["nll"]}

    res = dict(analysis="a8_headpatch", run=run, model_id=C.mid_of(run),
               label=C.label_of(run), family=family, checkpoint=ckpath,
               question=("does the readout matching head USE the memory prev-token head? "
                         "zero the memory head's output, re-measure the readout head's "
                         "induction lift on A5's synthetic probe"),
               matching_head=match, prev_token_heads=top_prev, control_heads=ctrl,
               probe=dict(source="a5_induction.run_synthetic (verbatim)",
                          n_synthetic_seq=args.n_synth,
                          synthetic_block_len=A5.BLOCK_LEN,
                          synthetic_copies=[A5.COPY_A, A5.COPY_B],
                          nll_skip_first=NLL_SKIP, ctrl_seed=CTRL_SEED),
               arms=results)
    C.save_json(out, res)

    print("\narm            match-lift   frac   readout-max   nll(second_core)", flush=True)
    for name, r in results.items():
        fr = r.get("frac_matching_head_lift_remaining")
        print(f"  {name:12s} {r['matching_head_lift']:10.2f}  "
              f"{('%.3f' % fr) if fr is not None else '  1.000'}  "
              f"{r['readout_max_lift']:11.2f}   {r['nll']['second_core']:.4f}", flush=True)


if __name__ == "__main__":
    main()
