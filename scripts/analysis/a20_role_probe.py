"""A20 -- per-block LINEAR probes on the residual stream: does the two-tower split the
two ROLES a single-stream residual has to carry at once?

THE HYPOTHESIS.  In a single-stream transformer every position's residual is doing two
jobs simultaneously:

    (i)  MEMORY -- it is the k/v every LATER position attends to, so it has to keep a
         legible record of what token was HERE (and, for induction, what token was just
         before here);
    (ii) PREDICTION -- the same vector, at the top, is the thing the LM head reads to
         name the NEXT token.

The two-tower architecture removes (ii) from the state tower by construction: the state
residual is never read by the LM head, only by the readout tower's cross-attention, and
the loss reaches it ONLY through those reads.  If role separation is real and not just a
story, the state tower's residual should become LESS next-token-predictive than a
single-stream residual at the same depth, while staying AT LEAST as informative about its
own token and its predecessor.  And the next-token predictiveness should not vanish -- it
should move into the PRED tower's blocks.

THE MEASUREMENT.  Three frozen linear probes on the same residual h_l (the post-block
residual, i.e. AFTER the residual add of block l, plus the final normed output):

    P1  next-token       target = token at t+1   ("prediction role")
    P2  current-token    target = token at t     ("memory role": is this token legible?)
    P3  previous-token   target = token at t-1   (the prev-token cue A5 found living in
                                                  the state stream)

Everything is reported as HELD-OUT NLL.  The probes are fit on N_FIT val sequences and
scored on a DISJOINT N_EVAL val sequences -- a probe fit and scored on the same tokens
measures the probe's capacity, not the residual's content, and with a 768 -> V readout on
~0.5 M samples that difference is large.

WHAT IS AND IS NOT COMPARABLE.
  * Across blocks / towers / runs: fully comparable.  Every probe has the SAME
    architecture, the same optimiser schedule, the same sample count, and is scored on the
    same token positions of the same val sequences.
  * Against the model's own NLL: comparable ONLY to `model_eval_nll_reduced`, which is the
    model's own next-token NLL pushed through the identical label reduction (below) on the
    identical eval positions.  `model_eval_nll_full` is the ordinary full-vocab number and
    is reported for provenance only.

LABEL REDUCTION (the cost control the task allows).  A full 768 -> 50257 readout is 38.6 M
parameters against ~0.5 M held-out-fit samples; every probe would then be dominated by its
own overfitting rather than by the residual's content, and 75 of them do not fit the
budget.  So the label space is the TOP-K token ids by frequency in the fit split (K=8192
by default) plus a single catch-all "other" class -- 768 x 8193 = 6.3 M parameters.  The
top-K set is chosen on the FIT split only and reused verbatim for the eval split, for
every probe of a run, and the coverage (fraction of eval targets that are not "other") is
recorded in the JSON.  `--vocab-mode full` runs the full 50257-way readout instead.

FEATURES.  Standardised per dimension using the FIT split's mean/std.  Residual norms grow
by ~an order of magnitude across depth in these models, and an unnormalised feature scale
turns a fixed Adam lr into a per-block confound -- the exact way a depth plot can be made
to say anything at all.

POSITIONS.  p in {1, 5, 9, ...} of each 4096-token sequence (stride 4 -> 1024 positions).
p >= 1 so that P3 exists; the next-token target is taken from the model's own Y shift so
p = T-1 is fine.  Positions whose input token is EOS are dropped, mirroring what the
model's own loss ignores.

RESIDUALS ARE THE MODEL'S OWN.  Two-tower residuals come from `common.two_tower_capture`,
which replays `TwoTowerModel._forward_towers` through the model's public per-block methods
and the model's OWN `_MaskSet` -- nothing here re-derives a mask (see
common.py, point 2).  `state_res[l+1]` / `pred_res[l+1]` are the post-block residuals of block l.  The
single-stream residuals come from forward hooks on `transformer.h`, so they too are the
tensors the model actually computes.

GATE.  The probe at the `final` site is a linear readout of exactly the vector the model's
own LM head reads, so its P1 held-out NLL must land in the neighbourhood of
`model_eval_nll_reduced` (a fresh 6.3 M-parameter head refit on 0.5 M tokens is expected to
be somewhat WORSE, never dramatically better).  A `final` P1 far below the model's own
number means the label reduction or the position/target alignment is wrong, and the script
says so loudly.

BASELINE FLOORS (opt-in; the default path is unchanged bit-for-bit).
  --untrained [--init-seed S]  the same measurement on a fresh random init of the SAME
      config (`common.load_untrained`, exactly as A16/A18 --untrained; the only control is
      the global torch seed set before `instantiate`).  Every block site is probed, plus
      the layer-0 input-embedding sites, plus the token-statistics floors.  Writes
      `a20_role_probe_<run>_init.json` (`_init_s<S>` for S != 1234).  Answers "is this
      role-legibility number a property of TRAINING or of the architecture at init?".
  --embed-only   probe ONLY the layer-0 residual(s) of the TRAINED model: the token
      embedding the first block reads (these models use RoPE, so there is no added
      positional vector -- the layer-0 residual is exactly wte(x), or the <predict>
      embedding on pred slots/tower).  Floor for "the stream still carries the token".
      Writes `a20_role_probe_<run>_embed.json`.
  --embed-site   add the layer-0 site(s) to an otherwise ordinary run (implied by both
      of the above).
  Floors (`token_floors`, implied by both modes): unigram and Jelinek-Mercer bigram
  cross-entropies on the SAME eval positions, in the SAME reduced label space, with
  counts from the FIT sequences only and the interpolation weight chosen on a split of
  the fit sequences (never on eval).  `bigram_next_given_cur` is what a probe that knows
  only the current token can reach for P1; `bigram_prev_given_cur` the same for P3.
  New-mode outputs refuse to overwrite an existing file unless --force.

Usage:
  a20_role_probe.py --run <run|role> [--n-fit 512] [--n-eval 512] [--pos-stride 4]
  a20_role_probe.py --run <run> --untrained [--init-seed 1234]
  a20_role_probe.py --run <run> --embed-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import numpy as np                                                     # noqa: E402

from modeling.models.model import GPT2_TOKENS                          # noqa: E402

ANALYSIS = "a20_role_probe"
N_TOKENIZER = GPT2_TOKENS["eos_token_id"] + 1   # GPT-2 tokenizer ids 0..EOS; the rest is padding

TARGETS = ("P1", "P2", "P3")


# ======================================================================================
# feature capture
# ======================================================================================
def _positions(block_size: int, stride: int):
    """Probe positions p: 1 <= p <= T-1, stride `stride`.  p >= 1 so P3 exists."""
    return np.arange(1, block_size, stride, dtype=np.int64)


def _joint_sites(model, X, Y, embed=False):
    """JOINT (tied-SPS) branch: one shared block stack over the interleaved 2T sequence.

    The post-block residual of block l is tapped once by a forward hook (the tensor the
    model actually computes) and split by slot parity through `common.joint_slot_masks`'
    convention: `state_l` = the STATE slots 2p (which embed the real token x_p) and
    `pred_l` = the PRED slots 2p+1 (the <predict> slot whose final output the LM head
    reads).  Both are indexed by the SAME token position p, so P1/P2/P3 target
    x_{p+1} / x_p / x_{p-1} for both slot types -- exactly the two-tower convention, where
    state_res and pred_res at index p also belong to token p.  `final` is the output of
    `output_norm` at the PRED slots only, i.e. literally the input of `lm_head` in
    `SPSModelBase.forward` (`lm_head(x[:, 1::2])`), so the a20 gate means the same thing
    here as for the other families.
    """
    import common as C

    blocks = C.joint_blocks(model)
    cap, handles = {}, []

    def mk(i):
        def hook(_m, _args, out):
            cap[i] = out
        return hook
    for i, blk in enumerate(blocks):
        handles.append(blk.register_forward_hook(mk(i)))
    if embed:
        # layer-0 residual = the input of block 0 (wte of the interleaved sequence)
        handles.append(blocks[0].register_forward_pre_hook(
            lambda _m, args: cap.__setitem__("emb", args[0])))
    fin = {}
    handles.append(model.transformer.output_norm.register_forward_hook(
        lambda _m, _a, out: fin.__setitem__("x", out)))
    try:
        model(X, Y)
    finally:
        for h in handles:
            h.remove()
    sites = {}
    if embed:
        e = cap["emb"]
        assert e.dim() == 3 and e.shape[1] == 2 * X.shape[1], e.shape
        sites["state_emb"] = e[:, 0::2]
        sites["pred_emb"] = e[:, 1::2]
    for i in range(len(blocks)):
        x = cap[i]
        assert x.dim() == 3 and x.shape[1] == 2 * X.shape[1], x.shape
        sites[f"state_{i}"] = x[:, 0::2]
        sites[f"pred_{i}"] = x[:, 1::2]
    return sites, fin["x"][:, 1::2]


def capture_sites(model, family, X, Y, pos_t, ref=None, embed=False, embed_only=False):
    """-> dict site -> (b, n_pos, d) fp16 residuals.

    Sites are the POST-block residuals (after the residual add) of every block of every
    tower, plus `final` = the model's final normed output -- the vector its LM head reads.

    `ref`, when given, is `(lut_cuda, topk_cuda, y_next, acc)`: the model's own LM head is
    applied to the final residual and the full-vocab / reduced-vocab NLL SUMS are
    accumulated into `acc` in place.  The logits themselves are never returned -- a
    (0.5 M, 50304) fp32 logit cache is 105 GB and was the first thing to blow this script up.

    `embed` adds the LAYER-0 residual(s) -- the tensor block 0 reads -- as `<tower>_emb`
    sites; `embed_only` returns only those.  Both default off, in which case nothing below
    differs from the original code path.
    """
    embed = embed or embed_only
    import torch
    import common as C

    sites = {}
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        if family == "two_tower":
            docs = model.generate_document_idx(X)
            capt = C.two_tower_capture(model, X, docs)
            # the q/k/v caches are ~1 GB per micro-batch and nothing here reads them
            capt.pop("state_qk", None)
            capt.pop("pred_qk", None)
            capt.pop("masks", None)
            for l in range(len(capt["state_res"]) - 1):
                sites[f"state_{l}"] = capt["state_res"][l + 1]
            for l in range(len(capt["pred_res"]) - 1):
                sites[f"pred_{l}"] = capt["pred_res"][l + 1]
            if embed:
                sites["state_emb"] = capt["state_res"][0]
                sites["pred_emb"] = capt["pred_res"][0]
            final = capt["out"]
        elif family == "standard":
            cap = {}
            handles = []

            def mk(i):
                def hook(_m, _args, out):
                    cap[i] = out
                return hook
            blocks = list(model.transformer.h)
            for i, blk in enumerate(blocks):
                handles.append(blk.register_forward_hook(mk(i)))
            if embed:
                handles.append(blocks[0].register_forward_pre_hook(
                    lambda _m, args: cap.__setitem__("emb", args[0])))
            fin = {}
            handles.append(model.transformer.output_norm.register_forward_hook(
                lambda _m, _a, out: fin.__setitem__("x", out)))
            try:
                model(X, Y)
            finally:
                for h in handles:
                    h.remove()
            if embed:
                sites["single_emb"] = cap["emb"]
            for i in range(len(blocks)):
                sites[f"single_{i}"] = cap[i]
            final = fin["x"]
        elif family == "sps":
            sites, final = _joint_sites(model, X, Y, embed=embed)
        else:
            raise SystemExit(f"a20 supports family 'two_tower', 'standard' and 'sps', "
                             f"got {family!r}")

        sites["final"] = final
        if embed_only:
            sites = {k: v for k, v in sites.items() if k.endswith("_emb")}
        out = {k: v.index_select(1, pos_t).to(torch.float16) for k, v in sites.items()}
        if ref is not None:
            import torch.nn.functional as F
            lut_c, topk_c, rest_c, y_next, acc, keep_b = ref
            lg = model.lm_head(final).index_select(1, pos_t)
            lg = lg.reshape(-1, lg.shape[-1])[:, :N_TOKENIZER].float()[keep_b]
            y = y_next[keep_b]
            acc["n"] += int(y.numel())
            acc["full"] += float(F.cross_entropy(lg, y, reduction="sum"))
            if topk_c is None:
                acc["red"] = acc["full"]
            else:
                # the reduced distribution is the model's OWN distribution marginalised
                # onto the probe's label space: p(other) = sum of p over every id outside
                # the top-K, i.e. a logsumexp of the remaining logits.  Nothing is
                # renormalised, so this is directly comparable to a probe's held-out NLL.
                kept = lg.index_select(1, topk_c)
                other = torch.logsumexp(lg.index_select(1, rest_c), dim=1, keepdim=True)
                acc["red"] += float(F.cross_entropy(torch.cat([kept, other], 1),
                                                    lut_c[y], reduction="sum"))
                del kept, other
            del lg
    return out


def build_cache(model, family, cfg, starts, pos, micro_batch, ref_tables=None,
                embed=False, embed_only=False):
    """Sweep `starts`, returning per-site fp16 CPU feature matrices + the token targets.

    Cost control: one sweep, every site cached at once.  At the defaults this is
    n_sites x n_tokens x 768 x 2 B ~= 40 GB of host RAM for a 12+12 two-tower, which is
    why the job needs ~220 GB of host RAM and why `--pos-stride` exists.
    """
    import torch
    import common as C

    data = C.val_memmap(cfg)
    pos_t = torch.from_numpy(pos).cuda()
    eos = int(model.config.eos_token_id)

    acc = dict(full=0.0, red=0.0, n=0)
    feats: dict[str, list] = {}
    tok_cur, tok_next, tok_prev, keep = [], [], [], []
    for i in range(0, len(starts), micro_batch):
        b = starts[i:i + micro_batch]
        X = torch.stack([torch.from_numpy(data[j:j + C.BLOCK].astype(np.int64))
                         for j in b]).cuda()
        Y = torch.stack([torch.from_numpy(data[j + 1:j + 1 + C.BLOCK].astype(np.int64))
                         for j in b]).cuda()
        xs = X.index_select(1, pos_t)
        xp = X.index_select(1, pos_t - 1)
        ys = Y.index_select(1, pos_t)
        keep_b = (xs != eos).reshape(-1)
        ref = None
        if ref_tables is not None:
            ref = (*ref_tables, ys.reshape(-1).clamp(max=N_TOKENIZER - 1), acc, keep_b)
        sites = capture_sites(model, family, X, Y, pos_t, ref=ref, embed=embed,
                              embed_only=embed_only)
        for k, v in sites.items():
            feats.setdefault(k, []).append(v.reshape(-1, v.shape[-1]).cpu())
        tok_cur.append(xs.reshape(-1).cpu())
        tok_prev.append(xp.reshape(-1).cpu())
        tok_next.append(ys.reshape(-1).cpu())
        keep.append(keep_b.cpu())
        del sites, X, Y, xs, xp, ys, keep_b, ref
    torch.cuda.empty_cache()
    return dict(
        feats={k: torch.cat(v) for k, v in feats.items()},
        cur=torch.cat(tok_cur), prev=torch.cat(tok_prev), next=torch.cat(tok_next),
        keep=torch.cat(keep), ref=acc,
    )


# ======================================================================================
# probe
# ======================================================================================
def train_probe(Xf, yf, Xe, ye, n_class, *, epochs, lr, batch, seed=0):
    """Linear softmax probe, Adam + cosine decay, held-out NLL after every epoch.

    Returns (best_eval_nll, per_epoch_eval_nll, final_train_nll).  The reported number is
    the MINIMUM over epochs -- early stopping on the held-out split is a mild optimism, but
    it is applied identically to every probe, so no block or tower is advantaged by it, and
    it protects the comparison from one probe happening to sit on the far side of its
    overfitting turn.
    """
    import torch
    import torch.nn.functional as F

    dev = "cuda"
    Xf = Xf.to(dev, non_blocking=True)
    Xe = Xe.to(dev, non_blocking=True)
    yf = yf.to(dev)
    ye = ye.to(dev)

    mu = Xf.float().mean(0)
    sd = Xf.float().std(0).clamp_min(1e-5)

    g = torch.Generator(device=dev).manual_seed(seed)
    W = torch.nn.Linear(Xf.shape[1], n_class).to(dev)
    torch.nn.init.zeros_(W.weight)
    torch.nn.init.zeros_(W.bias)
    opt = torch.optim.Adam(W.parameters(), lr=lr)
    n = Xf.shape[0]
    steps = epochs * ((n + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))

    def evaluate(W, Xe):
        s = 0.0
        with torch.no_grad():
            for i in range(0, Xe.shape[0], 16384):
                h = (Xe[i:i + 16384].float() - mu) / sd
                s += float(F.cross_entropy(W(h).float(), ye[i:i + 16384],
                                           reduction="sum"))
        return s / Xe.shape[0]

    per_epoch, train_nll = [], float("nan")
    with torch.enable_grad():
        for _ep in range(epochs):
            perm = torch.randperm(n, generator=g, device=dev)
            tot = 0.0
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                h = (Xf.index_select(0, idx).float() - mu) / sd
                loss = F.cross_entropy(W(h).float(), yf.index_select(0, idx))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                tot += float(loss) * idx.numel()
            train_nll = tot / n
            per_epoch.append(evaluate(W, Xe))
    del Xf, Xe, W, opt
    torch.cuda.empty_cache()
    return min(per_epoch), per_epoch, train_nll


# ======================================================================================
# token-statistics floors (opt-in)
# ======================================================================================
FLOOR_LAMBDAS = (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0)


def _pair_counts(data, starts, block, lut, n_class):
    """Dense (n_class, n_class) counts of adjacent label pairs (x_t, x_{t+1}) over the
    FULL sequences at `starts` (the pair crossing into the next sequence is included, as
    the model's own Y shift includes it)."""
    import torch
    V = lut.shape[0]
    idx = []
    for j in starts:
        seq = torch.from_numpy(np.asarray(data[j:j + block + 1]).astype(np.int64))
        seq = lut[seq.clamp(max=V - 1)]
        idx.append(seq[:-1] * n_class + seq[1:])
    return torch.bincount(torch.cat(idx), minlength=n_class * n_class).view(n_class, n_class)


def _jm_ce(N, cur, tgt, direction, lam):
    """Held-out CE of a Jelinek-Mercer bigram: lam * p_ML(tgt|cur) + (1-lam) * unigram.

    direction 'next': p(next=b | cur=a) = N[a, b] / rowsum[a];
    direction 'prev': p(prev=a | cur=b) = N[a, b] / colsum[b].
    The unigram is add-one smoothed over the label space so the CE is always finite.
    """
    import torch
    Nf = N.double()
    n_class = N.shape[0]
    uni = (Nf.sum(0) + 1.0) / (Nf.sum() + n_class)
    if direction == "next":
        cnt = Nf[cur, tgt]
        tot = Nf.sum(1)[cur]
    else:
        cnt = Nf[tgt, cur]
        tot = Nf.sum(0)[cur]
    p_ml = torch.where(tot > 0, cnt / tot.clamp_min(1.0), torch.zeros_like(cnt))
    p = lam * p_ml + (1.0 - lam) * uni[tgt]
    return float(-torch.log(p.clamp_min(1e-300)).mean())


def token_floors(data, fit_starts, block, pos, lut, n_class, eos, ev_cur, ev_next, ev_prev):
    """Unigram / bigram cross-entropy floors on the eval probe targets.

    Counts come from the FIT sequences only.  The JM weight is chosen per direction on a
    2-way split of the fit sequences (counts on the first half, CE on the second half's
    probe positions), then the counts are rebuilt on all fit sequences and scored on eval.
    `ev_*` are the kept eval targets in LABEL space (already through `lut`).
    """
    import torch
    V = lut.shape[0]
    half = len(fit_starts) // 2
    N_a = _pair_counts(data, fit_starts[:half], block, lut, n_class)
    # tuning positions: the probe positions of the second half, input token != EOS
    X = np.stack([np.asarray(data[j:j + block + 1]).astype(np.int64)
                  for j in fit_starts[half:]])
    xs, xn, xp = X[:, pos], X[:, pos + 1], X[:, pos - 1]
    keep = (xs != eos).reshape(-1)
    to_lab = lambda a: lut[torch.from_numpy(a.reshape(-1)[keep]).clamp(max=V - 1)]  # noqa
    t_cur, t_next, t_prev = to_lab(xs), to_lab(xn), to_lab(xp)
    lam = {}
    for d, tg in (("next", t_next), ("prev", t_prev)):
        ces = {l: _jm_ce(N_a, t_cur, tg, d, l) for l in FLOOR_LAMBDAS}
        lam[d] = min(ces, key=ces.get)
    del N_a
    N = _pair_counts(data, fit_starts, block, lut, n_class)
    Nf = N.double()
    uni = (Nf.sum(0) + 1.0) / (Nf.sum() + n_class)
    uce = lambda t: float(-torch.log(uni[t]).mean())  # noqa: E731
    return dict(
        label_space="same reduced label space as the probes",
        counts_from="fit sequences only (all adjacent pairs)",
        n_eval=int(ev_cur.numel()),
        unigram_P1=uce(ev_next), unigram_P2=uce(ev_cur), unigram_P3=uce(ev_prev),
        bigram_next_given_cur=_jm_ce(N, ev_cur, ev_next, "next", lam["next"]),
        bigram_prev_given_cur=_jm_ce(N, ev_cur, ev_prev, "prev", lam["prev"]),
        lambda_next=lam["next"], lambda_prev=lam["prev"],
        P2_lookup_floor=0.0,
        note=("unigram_* = CE of a probe with NO information; bigram_next_given_cur / "
              "bigram_prev_given_cur = CE reachable by a probe that knows ONLY the current "
              "token (P1 / P3). P2 is 0 for any stream that still identifies the token."),
    )


def _parse_site(site):
    """site name -> (tower, block).  `<tower>_emb` (layer-0 residual) is block -1."""
    if site == "final":
        return "final", -1
    tower, idx = site.rsplit("_", 1)
    return tower, (-1 if idx == "emb" else int(idx))


def _mode(args):
    """-> (suffix, extra JSON fields).  Both empty on the default path."""
    untrained = bool(getattr(args, "untrained", False))
    embed_only = bool(getattr(args, "embed_only", False))
    embed_site = bool(getattr(args, "embed_site", False))
    assert not (untrained and embed_only), "--untrained and --embed-only are exclusive"
    if not (untrained or embed_only or embed_site):
        return "", {}
    if untrained:
        s = int(args.init_seed)
        suffix = "_init" if s == 1234 else f"_init_s{s}"
    elif embed_only:
        suffix = "_embed"
    else:
        suffix = "_embsite"
    extra = dict(untrained=untrained, init_seed=int(args.init_seed) if untrained else None,
                 embed_site=True, embed_only=embed_only)
    return suffix, extra


def out_path(args, run):
    import common as C
    suffix, _ = _mode(args)
    return args.out or os.path.join(C.RESULTS_DIR, f"{ANALYSIS}_{run}{suffix}.json")


# ======================================================================================
# measurement
# ======================================================================================
def run_measurement(args):
    import torch
    import common as C

    run = C.resolve(args.run)
    t0 = time.time()
    suffix, extra = _mode(args)
    new_mode = bool(extra)
    embed_only = bool(extra.get("embed_only"))
    embed = bool(extra.get("embed_site"))
    p_out = out_path(args, run)
    if new_mode and os.path.exists(p_out) and not args.force:
        raise SystemExit(f"[a20] {p_out} exists; refusing to overwrite (use --force)")
    if extra.get("untrained"):
        model, cfg, family, ckpath = C.load_untrained(run, seed=int(args.init_seed))
    else:
        model, cfg, family, ckpath = C.load(run)
    print(f"[a20] run={run} family={family} ckpt={ckpath}", flush=True)

    data = C.val_memmap(cfg)
    all_starts = list(range(0, len(data) - C.BLOCK - 1, C.BLOCK))
    need = args.n_fit + args.n_eval
    assert len(all_starts) >= need, f"val set has only {len(all_starts)} sequences"
    fit_starts = all_starts[:args.n_fit]
    eval_starts = all_starts[args.n_fit:need]          # DISJOINT by construction
    pos = _positions(C.BLOCK, args.pos_stride)
    print(f"[a20] fit={len(fit_starts)} seqs  eval={len(eval_starts)} seqs  "
          f"pos/seq={len(pos)}", flush=True)

    # -- label space, decided BEFORE any forward pass ---------------------------------
    # The top-K set is a property of the FIT sequences' token stream only, so it is read
    # straight off the memmap.  Deciding it here (rather than after the fit sweep) is what
    # lets the eval sweep fold the model's own reduced-vocab reference NLL into the same
    # pass instead of caching (0.5 M, 50304) logits -- 105 GB.
    V = N_TOKENIZER
    fit_tokens = np.concatenate([np.asarray(data[j:j + C.BLOCK])[pos] for j in fit_starts])
    fit_tokens = torch.from_numpy(fit_tokens.astype(np.int64)).clamp(max=V - 1)
    if args.vocab_mode == "full":
        n_class, topk, rest = V, None, None
        lut = torch.arange(V)
    else:
        counts = torch.bincount(fit_tokens, minlength=V)
        topk = torch.topk(counts, k=min(args.topk, V)).indices.sort().values
        lut = torch.full((V,), len(topk), dtype=torch.long)
        lut[topk] = torch.arange(len(topk))
        n_class = len(topk) + 1
        keep_mask = torch.ones(V, dtype=torch.bool)
        keep_mask[topk] = False
        rest = torch.nonzero(keep_mask, as_tuple=False).flatten()
    del fit_tokens
    ref_tables = (lut.cuda(),
                  None if topk is None else topk.cuda(),
                  None if rest is None else rest.cuda())
    print(f"[a20] vocab_mode={args.vocab_mode} n_class={n_class}", flush=True)

    fit = build_cache(model, family, cfg, fit_starts, pos, args.micro_batch,
                      embed=embed, embed_only=embed_only)
    print(f"[a20] fit cache done ({time.time() - t0:.0f}s) "
          f"sites={len(fit['feats'])} tokens={int(fit['keep'].sum())}", flush=True)
    ev = build_cache(model, family, cfg, eval_starts, pos, args.micro_batch,
                     ref_tables=ref_tables, embed=embed, embed_only=embed_only)
    print(f"[a20] eval cache done ({time.time() - t0:.0f}s)", flush=True)
    del ref_tables
    torch.cuda.empty_cache()

    kf, ke = fit["keep"], ev["keep"]
    if args.vocab_mode == "full":
        coverage = 1.0
    else:
        cov_t = torch.cat([ev["next"][ke], ev["cur"][ke], ev["prev"][ke]]).clamp(max=V - 1)
        coverage = float((lut[cov_t] != len(topk)).float().mean())
        del cov_t

    # -- the model's own reference NLL on the SAME eval positions ---------------------
    acc = ev["ref"]
    assert acc["n"] == int(ke.sum()), (acc["n"], int(ke.sum()))
    model_full = acc["full"] / acc["n"]
    model_red = acc["red"] / acc["n"]
    print(f"[a20] eval_coverage={coverage:.4f}  model NLL on eval positions: "
          f"full={model_full:.4f} reduced={model_red:.4f}", flush=True)

    # -- targets ----------------------------------------------------------------------
    tgt_fit = {"P1": lut[fit["next"][kf].clamp(max=V - 1)],
               "P2": lut[fit["cur"][kf].clamp(max=V - 1)],
               "P3": lut[fit["prev"][kf].clamp(max=V - 1)]}
    tgt_ev = {"P1": lut[ev["next"][ke].clamp(max=V - 1)],
              "P2": lut[ev["cur"][ke].clamp(max=V - 1)],
              "P3": lut[ev["prev"][ke].clamp(max=V - 1)]}

    # -- probes -------------------------------------------------------------------------
    def site_order(names):
        def key(s):
            if s == "final":
                return (3, 0)
            tower, idx = _parse_site(s)
            return ({"single": 0, "state": 1, "pred": 2}[tower], idx)
        return sorted(names, key=key)

    probes = {}
    for site in site_order(fit["feats"].keys()):
        Xf = fit["feats"][site][kf]
        Xe = ev["feats"][site][ke]
        tower, blk = _parse_site(site)
        rec = dict(tower=tower, block=blk, d=int(Xf.shape[1]))
        for tg in TARGETS:
            best, per_ep, tr = train_probe(
                Xf, tgt_fit[tg], Xe, tgt_ev[tg], n_class,
                epochs=args.epochs, lr=args.lr, batch=args.batch, seed=args.seed)
            rec[tg] = dict(eval_nll=best, per_epoch=per_ep, train_nll=tr)
            print(f"[a20] {site:>10s} {tg} eval_nll={best:.4f} "
                  f"train_nll={tr:.4f}  t={time.time() - t0:.0f}s", flush=True)
        probes[site] = rec
        del Xf, Xe
        fit["feats"][site] = None
        ev["feats"][site] = None

    # -- GATE ---------------------------------------------------------------------------
    if new_mode and ("final" not in probes or extra.get("untrained")):
        # untrained: the model's own NLL is ~ln V, so "a refit head is no better" is
        # vacuous; embed-only: there is no final-site probe.  Recorded as not applicable.
        gate_final = probes["final"]["P1"]["eval_nll"] if "final" in probes else None
        gate_ok = None
        print(f"[a20] GATE N/A ({'untrained' if extra.get('untrained') else 'embed-only'})"
              f" final P1={gate_final} model_red={model_red:.4f}", flush=True)
    else:
        gate_final = probes["final"]["P1"]["eval_nll"]
        gate_ok = gate_final > model_red - 0.15
    msg = (f"final-site P1 held-out NLL "
           f"{gate_final if gate_final is None else format(gate_final, '.4f')} vs the "
           f"model's own {model_red:.4f} on the same positions")
    if gate_ok is None:
        pass
    elif gate_ok:
        print(f"[a20] GATE PASS: {msg} (a refit head is expected to be no better)",
              flush=True)
    else:
        print(f"[a20] GATE FAIL: {msg} -- a fresh linear head cannot beat the trained "
              f"one on held-out tokens; suspect target/position misalignment or a broken "
              f"label reduction. Numbers below are NOT trustworthy.", flush=True)

    mc = cfg.model.config
    out = dict(
        analysis=ANALYSIS, run=run, mid=C.mid_of(run), label=C.label_of(run),
        family=family, ckpt=ckpath,
        n_layer=int(getattr(mc, "n_layer", 0) or 0),
        state_n_layer=int(getattr(mc, "state_n_layer", 0) or 0),
        pred_n_layer=int(getattr(mc, "pred_n_layer", 0) or 0),
        read_map=str(getattr(mc, "read_map", "")),
        n_fit_seqs=len(fit_starts), n_eval_seqs=len(eval_starts),
        pos_stride=args.pos_stride, pos_per_seq=len(pos),
        n_fit_tokens=int(kf.sum()), n_eval_tokens=int(ke.sum()),
        vocab_mode=args.vocab_mode, n_class=int(n_class), topk=int(args.topk),
        eval_label_coverage=coverage,
        epochs=args.epochs, lr=args.lr, batch=args.batch, seed=args.seed,
        model_eval_nll_full=model_full, model_eval_nll_reduced=model_red,
        gate_pass=bool(gate_ok) if gate_ok is not None else None,
        probes=probes, seconds=time.time() - t0,
    )
    if new_mode:
        out.update(extra)
        out["gate_applicable"] = gate_ok is not None
        if args.vocab_mode == "topk":
            out["floors"] = token_floors(
                data, fit_starts, C.BLOCK, pos, lut, n_class,
                int(model.config.eos_token_id),
                tgt_ev["P2"], tgt_ev["P1"], tgt_ev["P3"])
            print(f"[a20] floors: {json.dumps({k: v for k, v in out['floors'].items() if isinstance(v, float)})}",
                  flush=True)
        out["seconds"] = time.time() - t0
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    p = p_out
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"WROTE {p}  ({out['seconds']:.0f}s)", flush=True)


# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-fit", type=int, default=512)
    ap.add_argument("--n-eval", type=int, default=512)
    ap.add_argument("--pos-stride", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--vocab-mode", choices=("topk", "full"), default="topk")
    ap.add_argument("--topk", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    # -- baseline floors (all default OFF; default path unchanged) ----------------------
    ap.add_argument("--untrained", action="store_true",
                    help="INIT CONTROL: same probes on a fresh random init of the config")
    ap.add_argument("--init-seed", type=int, default=1234,
                    help="torch seed for --untrained (C.load_untrained; 1234 = A16/A18)")
    ap.add_argument("--embed-only", action="store_true",
                    help="probe only the layer-0 (input-embedding) residual(s)")
    ap.add_argument("--embed-site", action="store_true",
                    help="add the layer-0 site(s) to the probed sites")
    ap.add_argument("--force", action="store_true",
                    help="allow a new-mode run to overwrite its existing output JSON")
    run_measurement(ap.parse_args())


if __name__ == "__main__":
    main()
