"""A21 -- does the learned prediction table diverge from the state table? (20B, valid test)

Why a7 is not enough. a7 reports the per-row cosine between the state table S
(`transformer.wte`) and the prediction-slot input table P (`predict_wte`): 0.00023 on the
20B equal arm. That number cannot distinguish "diverged" from "never related": the 20B runs
initialise P INDEPENDENTLY of S (`training.two_tower_emb_probe_init=false`), and the two
tables live in unrelated bases, so a raw same-row cosine of two independent 768-d tables is
~0 +- 1/sqrt(768) = 0.036 whether or not they learned the same structure.  (a7's measured
sd is 0.036.)  A basis-dependent statistic on independently initialised tables is
uninformative by construction.

What this measures instead (all over ACTIVE rows = token ids seen in val.bin, a7's filter):

1. Rotation-invariant similarity of P vs S, each against nulls:
     * linear CKA (centered; invariant to orthogonal transforms and isotropic scale),
     * per-row cosine after orthogonal Procrustes alignment, both in-sample and 2-fold
       cross-validated (the in-sample fit has 768^2 free parameters and is biased upward;
       the CV number has a ~0 null),
     * k=10 nearest-neighbour overlap per token (centered cosine neighbourhoods).
   Nulls: (a) S vs the untied LM head H -- also independently initialised, also trained,
   the "free control" for what two related trained tables look like; (b) a fresh random
   N(0, 0.02) pair of the same shape; (b') the reconstructed seed-1337 init pair (P_0, S_0),
   i.e. the actual starting point; (c) a row-permutation null (P rows shuffled WITHIN each
   frequency bin, which keeps each table's spectrum and norm-by-frequency profile but
   breaks token correspondence).  Everything is broken down by val-frequency rank bin.

2. Drift from init.  P_0/S_0/H_0 are rebuilt exactly as scripts/train.py builds them
   (compose the run's config, `torch.manual_seed(training.seed + 0)` on rank 0,
   `instantiate(cfg.model)` on CPU, no emb-probe re-draw).  Verified by (i) determinism
   (two builds are bitwise equal), (ii) the rows of ids 50257..50303, which never occur in
   the GPT-2-tokenised corpus: an embedding row that never receives gradient is only
   shrunk by AdamW's decoupled weight decay, so in the earliest ladder checkpoint it must
   be EXACTLY the init row times one common scalar (row cosine 1, identical norm ratio
   across rows and across P and S); (iii) a wrong-seed build (seed+1) must fail (ii).
   Then, along the ~1B-token checkpoint ladder, per frequency bin: relative Frobenius
   drift ||X_t - X_0|| / ||X_0|| and mean row-cosine to init for P, S, H; the P/S norm
   ratio; effective rank and top-1 variance fraction of P and S; CKA / CV-Procrustes /
   kNN of (P_t, S_t) and CKA of the control pair (S_t, H_t).

3. (--functional) Val NLL, full deterministic sweep (the ledger's metric of record,
   `eval_runs.full_sweep_nll`), with P replaced by: its mean active row, a row
   permutation, zeros, and the state table S.  The untouched baseline and an identity
   replacement AFTER all the others (proves the restore) must both reproduce the ledger.

Usage: a21_emb_divergence.py --run <run> [--functional] [--out <json>]

Inputs: every ckpt_tokens_*.pt of the run (the 1B ladder, pre_decay and final), or, when only
the final is on disk, the run's emb_ladder.pt (the three tables of each of those
checkpoints, from the ryankim17920/mechanistic-sps-extra Hugging Face repo:
`hf_export.py download --extra <run>`).  --functional needs the final checkpoint
(`hf_export.py download <run>`) and a GPU.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import numpy as np
import torch

from modeling.models.model import GPT2_TOKENS

K_NN = 10
N_PERM = 3
CV_SEED = 0
RAND_SEED = 0
PERM_SEED = 1
INIT_STD = 0.02
# Frequency-rank bins over ACTIVE tokens (rank 0 = most frequent in val).  Every bin has
# more rows than the table width (768) at the 48k active tokens of fineweb-edu val.
BIN_EDGES = ((0.00, 0.01), (0.01, 0.05), (0.05, 0.20), (0.20, 0.50), (0.50, 1.00))
# GPT-2 tokenizer ids end at the EOS id; the rest of the padded vocab never occurs
UNSEEN_IDS = range(GPT2_TOKENS["eos_token_id"] + 1, GPT2_TOKENS["vocab_size"])
ROUND = 5


# ======================================================================================
# pure metrics (torch; CPU or GPU).  Unit-tested in
# src/modeling/tests/analysis/test_a21_emb_divergence.py
# ======================================================================================
def _f64(x):
    return x.to(torch.float64)


def center(x):
    return x - x.mean(0, keepdim=True)


def linear_cka(x, y):
    """Linear CKA (Kornblith et al. 2019) on row-aligned x (n,d1), y (n,d2)."""
    x, y = center(_f64(x)), center(_f64(y))
    xy = (x.T @ y).pow(2).sum()
    xx = (x.T @ x).pow(2).sum().sqrt()
    yy = (y.T @ y).pow(2).sum().sqrt()
    return float(xy / (xx * yy))


def procrustes_fit(a, b):
    """Orthogonal R minimising ||a R - b||_F (a, b already centered)."""
    u, _, vh = torch.linalg.svd(_f64(a).T @ _f64(b), full_matrices=False)
    return u @ vh


def row_cosine(a, b):
    a, b = _f64(a), _f64(b)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-30)


def procrustes_cos(a, b, seed=CV_SEED):
    """-> (in-sample per-row cosine, 2-fold CV per-row cosine), each (n,).

    Both tables are centered on their own mean row (the common-mode vector of a trained
    table otherwise dominates the fit).  CV: rows split at random into two folds; each
    fold's rows are scored with the rotation fitted on the OTHER fold only."""
    a, b = center(_f64(a)), center(_f64(b))
    n = a.shape[0]
    ins = row_cosine(a @ procrustes_fit(a, b), b)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).to(a.device)
    f0, f1 = perm[: n // 2], perm[n // 2:]
    cv = torch.empty(n, dtype=torch.float64, device=a.device)
    for fit, ev in ((f0, f1), (f1, f0)):
        r = procrustes_fit(a[fit], b[fit])
        cv[ev] = row_cosine(a[ev] @ r, b[ev])
    return ins, cv


def knn_indices(x, k=K_NN, chunk=4096):
    """Top-k cosine neighbours (self excluded) of every row of centered x -> (n, k)."""
    x = center(x.float())
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    n = x.shape[0]
    out = torch.empty(n, k, dtype=torch.long, device=x.device)
    for i in range(0, n, chunk):
        s = x[i:i + chunk] @ x.T
        idx = torch.arange(i, min(i + chunk, n), device=x.device)
        s[torch.arange(len(idx), device=x.device), idx] = -float("inf")
        out[i:i + chunk] = s.topk(k, dim=1).indices
    return out


def knn_overlap(a, b, k=K_NN, chunk=4096):
    """Per-row fraction of the k nearest neighbours shared between tables a and b."""
    na, nb = knn_indices(a, k, chunk), knn_indices(b, k, chunk)
    # |A ∩ B| per row via sorted membership: k is small, so the (n,k,k) compare is cheap.
    hits = (na.unsqueeze(2) == nb.unsqueeze(1)).any(2).sum(1)
    return hits.double() / k


def spectrum_stats(x):
    """Effective rank (Roy & Vetterli: exp of the entropy of normalised singular values)
    and top-1 variance fraction, both of the CENTERED table; plus the share of the raw
    table's energy in its mean row (the common-mode component centering removes)."""
    x = _f64(x)
    xc = center(x)
    s = torch.linalg.svdvals(xc)
    p = s / s.sum()
    p = p[p > 0]
    erank = float(torch.exp(-(p * p.log()).sum()))
    top1 = float(s[0] ** 2 / (s ** 2).sum())
    mean_frac = float(x.shape[0] * x.mean(0).pow(2).sum() / x.pow(2).sum())
    return dict(eff_rank=erank, top1_var_frac=top1, mean_row_energy_frac=mean_frac)


def rel_drift(xt, x0):
    return float((_f64(xt) - _f64(x0)).norm() / _f64(x0).norm())


def freq_bins(counts, active_ids, edges=BIN_EDGES):
    """Positions (into the active-row arrays) of each val-frequency-rank bin.

    Ranks are over ACTIVE tokens by descending val count (ties broken by id, stable), so
    bins are fixed-size and identical for every run on the same val.bin."""
    c = np.asarray(counts)[active_ids]
    order = np.lexsort((active_ids, -c))           # descending count, then ascending id
    n = len(active_ids)
    bins = {}
    for lo, hi in edges:
        a, b = int(round(lo * n)), int(round(hi * n))
        bins[f"rank{int(lo * 100)}-{int(hi * 100)}pct"] = np.sort(order[a:b])
    bins["all"] = np.arange(n)
    return bins


def _r(x):
    return round(float(x), ROUND)


def similarity(a, b, bins, with_knn=True, seed=CV_SEED):
    """CKA (per bin, on that bin's rows), Procrustes cosine (rotation fitted on ALL active
    rows; per-bin mean of per-row cosine), kNN overlap (neighbours over all active rows;
    per-bin mean).  a, b are the active-row tables, row-aligned."""
    ins, cv = procrustes_cos(a, b, seed=seed)
    kn = knn_overlap(a, b) if with_knn else None
    out = {}
    for name, idx in bins.items():
        ti = torch.as_tensor(idx, device=a.device)
        d = dict(cka=_r(linear_cka(a[ti], b[ti])),
                 procrustes_cos_insample=_r(ins[ti].mean()),
                 procrustes_cos_cv=_r(cv[ti].mean()))
        if kn is not None:
            d["knn10_overlap"] = _r(kn[ti].mean())
        out[name] = d
    return out


def permute_within_bins(x, bins, gen):
    y = x.clone()
    for name, idx in bins.items():
        if name == "all":
            continue
        ti = torch.as_tensor(idx, device=x.device)
        p = torch.randperm(len(idx), generator=gen).to(x.device)
        y[ti] = x[ti[p]]
    return y


def table_stats(tables, init, bins):
    """Per-bin drift / norm / spectrum stats for a dict of active-row tables at step t."""
    out = {}
    for name, idx in bins.items():
        ti = torch.as_tensor(idx, device=tables["P"].device)
        d = {}
        for k in ("P", "S", "H"):
            x = tables[k][ti]
            if init is not None:
                d[f"drift_{k}"] = _r(rel_drift(x, init[k][ti]))
                d[f"rowcos_to_init_{k}"] = _r(row_cosine(x, init[k][ti]).mean())
            d[f"mean_row_norm_{k}"] = _r(_f64(x).norm(dim=1).mean())
        d["norm_ratio_P_over_S"] = _r(d["mean_row_norm_P"] / d["mean_row_norm_S"])
        for k in ("P", "S"):
            for sk, sv in spectrum_stats(tables[k][ti]).items():
                d[f"{sk}_{k}"] = _r(sv)
        out[name] = d
    return out


# ======================================================================================
# I/O
# ======================================================================================
TOK_RE = re.compile(r"ckpt_tokens_(\d+)(_pre_decay|_final)?\.pt$")
KEYS = dict(S="transformer.wte.weight", P="predict_wte.weight", H="lm_head.weight")


# The three tables of every ladder checkpoint, extracted into one file per run (the
# ryankim17920/mechanistic-sps-extra Hugging Face repo, `hf_export.py download --extra <run>`):
# {"checkpoints": {<checkpoint stem>: {file, tokens, tag, iter_num, seed, emb_probe_init}},
#  "tables": {<checkpoint stem>: {"S", "P", "H"}}}, the tensors exactly as stored in the
# checkpoint.  Used when the run directory holds no ladder checkpoint besides the final.
EMB_LADDER = "emb_ladder.pt"


def ladder(run_dir):
    """-> sorted [(tokens, tag, path)] of the run's ckpt_tokens_*.pt checkpoints.

    Without the checkpoints themselves (only the final, or nothing, on disk) but with
    `EMB_LADDER` present, the ladder is the one the extracted tables came from, and each
    `path` is that of the checkpoint they were extracted from (load_tables reads them from
    `EMB_LADDER` when that path is absent), so the result JSON is the same either way."""
    items = []
    for p in glob.glob(os.path.join(run_dir, "ckpt_tokens_*.pt")):
        m = TOK_RE.search(os.path.basename(p))
        if m:
            items.append((int(m.group(1)), m.group(2) or "", p))
    emb = os.path.join(run_dir, EMB_LADDER)
    if os.path.isfile(emb) and all(tag == "_final" for _, tag, _ in items):
        items = []
        for c in _emb_ladder(emb)["checkpoints"].values():
            m = TOK_RE.search(c["file"])
            items.append((int(m.group(1)), m.group(2) or "", os.path.join(run_dir, c["file"])))
    return sorted(items)


def _emb_ladder(emb):
    return torch.load(emb, map_location="cpu", mmap=True, weights_only=True)


def _load_tables_extracted(path):
    e = _emb_ladder(os.path.join(os.path.dirname(path), EMB_LADDER))
    stem = os.path.basename(path)[:-len(".pt")]
    meta, tab = e["checkpoints"][stem], e["tables"][stem]
    out = {k: tab[k].float().clone() for k in KEYS}
    del e, tab
    return out, dict(iter_num=meta["iter_num"], seed=meta["seed"],
                     emb_probe_init=meta["emb_probe_init"], provenance=None)


def load_tables(path):
    if not os.path.exists(path):
        return _load_tables_extracted(path)
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    sd = ck["model"]
    out = {}
    for k, name in KEYS.items():
        hit = [v for kk, v in sd.items()
               if kk.replace("_orig_mod.", "").replace("module.", "") == name]
        assert len(hit) == 1, f"{path}: {name} not found"
        out[k] = hit[0].float().clone()
    meta = dict(iter_num=int(ck.get("iter_num", -1)),
                seed=int(((ck.get("config") or {}).get("training") or {}).get("seed", -1)),
                emb_probe_init=bool(((ck.get("config") or {}).get("training") or {})
                                    .get("two_tower_emb_probe_init", False)),
                provenance=ck.get("provenance"))
    del ck, sd
    return out, meta


def rebuild_init(run, seed_offset=0):
    """scripts/train.py's scratch init, rank 0: manual_seed(seed), random.seed(seed),
    instantiate(cfg.model) on CPU.  Returns (tables, cfg, seed)."""
    import random
    import common as C
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    with initialize_config_dir(config_dir=C.CONF, version_base=None):
        cfg = compose("config", overrides=[f"+experiment={run}",
                                           f"system.data_root={C.repo_paths.data_root()}"])
    assert not bool(cfg.training.get("two_tower_emb_probe_init", False)), \
        "emb_probe_init run: init is re-drawn after construction, not handled here"
    seed = int(cfg.training.get("seed", 1337)) + seed_offset
    torch.manual_seed(seed)
    random.seed(seed)
    with torch.device("cpu"):
        m = instantiate(cfg.model)
    sd = m.state_dict()
    t = {k: sd[n].detach().float().clone() for k, n in KEYS.items()}
    del m
    return t, cfg, seed


def verify_init(init, init_again, wrong, first):
    """Bitwise determinism + the never-seen-row test against the earliest checkpoint."""
    res = dict(deterministic=all(torch.equal(init[k], init_again[k]) for k in KEYS))
    ids = torch.tensor(list(UNSEEN_IDS))
    for tag, ref in (("seed", init), ("wrong_seed", wrong)):
        d = {}
        for k in ("P", "S"):
            a, b = first[k][ids].double(), ref[k][ids].double()
            cos = row_cosine(a, b)
            ratio = a.norm(dim=1) / b.norm(dim=1)
            d[k] = dict(min_row_cos=_r(cos.min()), mean_row_cos=_r(cos.mean()),
                        norm_ratio_mean=round(float(ratio.mean()), 7),
                        norm_ratio_rel_sd=float(ratio.std() / ratio.mean()),
                        max_abs_resid_after_scale=float(
                            (a - ratio.mean() * b).abs().max()))
        res[tag] = d
    s = res["seed"]
    res["passed"] = bool(
        res["deterministic"]
        and min(s["P"]["min_row_cos"], s["S"]["min_row_cos"]) > 0.99999
        and max(s["P"]["norm_ratio_rel_sd"], s["S"]["norm_ratio_rel_sd"]) < 1e-5
        and abs(s["P"]["norm_ratio_mean"] - s["S"]["norm_ratio_mean"]) < 1e-5
        and max(res["wrong_seed"]["P"]["mean_row_cos"],
                res["wrong_seed"]["S"]["mean_row_cos"]) < 0.5)
    return res


# ======================================================================================
# main analysis
# ======================================================================================
def compute(run, functional=False, device="cuda"):
    import common as C
    run_dir = str(C.repo_paths.run_dir(run))
    lad = ladder(run_dir)
    assert lad and lad[-1][1] == "_final", f"{run}: no final checkpoint in {run_dir}"

    init, cfg, seed = rebuild_init(run)
    init_again, _, _ = rebuild_init(run)
    wrong, _, _ = rebuild_init(run, seed_offset=1)
    mc = cfg.model.config
    assert str(mc.predict_embedding) == "separate" and not bool(mc.tie_lm_head), \
        f"{run}: needs predict_embedding=separate and tie_lm_head=false"
    assert init["P"].shape == init["S"].shape, "P and S widths differ"

    first, first_meta = load_tables(lad[0][2])
    assert first_meta["seed"] == seed and not first_meta["emb_probe_init"], first_meta
    ver = verify_init(init, init_again, wrong, first)
    ver["earliest_checkpoint"] = lad[0][2]
    ver["note"] = ("no iter-0 checkpoint exists; the earliest ladder checkpoint is used. "
                   "ids 50257..50303 never occur in the corpus, so their P/S rows get no "
                   "gradient and must equal init x one common weight-decay scalar.")
    print("init verification:", ver, flush=True)
    del init_again, wrong

    counts = np.bincount(np.asarray(C.val_memmap(cfg), dtype=np.int64),
                         minlength=int(mc.vocab_size))
    active = np.nonzero(counts > 0)[0]
    assert not np.isin(active, np.array(list(UNSEEN_IDS))).any()
    bins = freq_bins(counts, active)
    act = torch.as_tensor(active)
    bin_info = {k: dict(n=int(len(v)),
                        val_count_min=int(counts[active[v]].min()),
                        val_count_max=int(counts[active[v]].max())) for k, v in bins.items()}

    def dev(t):
        return {k: v[act].to(device) for k, v in t.items()}

    init_a = dev(init)

    # ---------------- trajectory ----------------
    traj = []
    points = [(0, "init", None)] + lad
    final_tables = None
    for tok, tag, path in points:
        t0 = time.time()
        if path is None:
            ta = init_a
        else:
            tb, _ = load_tables(path)
            ta = dev(tb)
            del tb
        sim_ps = similarity(ta["P"], ta["S"], bins)
        cka_sh = {k: _r(linear_cka(ta["S"][torch.as_tensor(v, device=device)],
                                   ta["H"][torch.as_tensor(v, device=device)]))
                  for k, v in bins.items()}
        st = table_stats(ta, init_a, bins)
        traj.append(dict(tokens=int(tok), tag=tag.strip("_") or ("init" if path is None
                                                                  else "ladder"),
                         checkpoint=path, P_vs_S=sim_ps, cka_S_vs_H=cka_sh, stats=st))
        a = sim_ps["all"]
        print(f"[{tag or 'ladder'} {tok/1e9:6.2f}B] cka={a['cka']:.4f} "
              f"procCV={a['procrustes_cos_cv']:.4f} knn={a['knn10_overlap']:.4f} "
              f"ckaSH={cka_sh['all']:.4f} driftP={st['all']['drift_P']:.3f} "
              f"driftS={st['all']['drift_S']:.3f} ({time.time()-t0:.1f}s)", flush=True)
        if tag == "_final":
            final_tables = ta
        elif path is not None:
            del ta

    # ---------------- final: P vs S against every null ----------------
    fa = final_tables
    g = torch.Generator().manual_seed(RAND_SEED)
    shape = (int(mc.vocab_size), fa["P"].shape[1])
    r1 = (torch.randn(shape, generator=g) * INIT_STD)[act].to(device)
    r2 = (torch.randn(shape, generator=g) * INIT_STD)[act].to(device)
    gp = torch.Generator().manual_seed(PERM_SEED)
    perm_runs = [similarity(permute_within_bins(fa["P"], bins, gp), fa["S"], bins)
                 for _ in range(N_PERM)]
    perm_null = {}
    for b in bins:
        perm_null[b] = {m: dict(mean=_r(np.mean([p[b][m] for p in perm_runs])),
                                max=_r(np.max([p[b][m] for p in perm_runs])))
                        for m in perm_runs[0][b]}
    final = dict(
        P_vs_S=traj[-1]["P_vs_S"],
        null_a_S_vs_H=similarity(fa["S"], fa["H"], bins),
        also_P_vs_H=similarity(fa["P"], fa["H"], bins),
        null_b_random_pair=similarity(r1, r2, bins),
        null_b_init_pair_P0_vs_S0=traj[0]["P_vs_S"],
        null_c_rowperm_within_bin=perm_null,
    )
    del r1, r2

    res = dict(analysis="a21_emb_divergence", run=run, seed=seed,
               config=dict(state_n_layer=int(mc.state_n_layer),
                           pred_n_layer=int(mc.pred_n_layer),
                           read_map=str(mc.read_map),
                           predict_embedding=str(mc.predict_embedding),
                           tie_lm_head=bool(mc.tie_lm_head),
                           emb_probe_init=False),
               n_active=int(len(active)), bins=bin_info, k_nn=K_NN,
               init_verification=ver, final=final, trajectory=traj,
               summary=_summary(final, traj))

    if functional:
        res["functional"] = functional_check(run)
    return res


def _summary(final, traj):
    out = {}
    for b in final["P_vs_S"]:
        row = {}
        for m in ("cka", "procrustes_cos_cv", "knn10_overlap"):
            row[m] = dict(P_vs_S=final["P_vs_S"][b][m],
                          null_S_vs_H=final["null_a_S_vs_H"][b][m],
                          null_random=final["null_b_random_pair"][b][m],
                          null_init_pair=final["null_b_init_pair_P0_vs_S0"][b][m],
                          null_rowperm_max=final["null_c_rowperm_within_bin"][b][m]["max"])
        row["cka_traj_P_vs_S"] = [(t["tokens"], t["P_vs_S"][b]["cka"]) for t in traj]
        row["cka_traj_S_vs_H"] = [(t["tokens"], t["cka_S_vs_H"][b]) for t in traj]
        out[b] = row
    return out


# ======================================================================================
# functional check
# ======================================================================================
def ledger_val_nll(run):
    """The run's val_nll_full_sweep in the ledger snapshot (newest non-null record), or None."""
    import repo_paths
    from ledger import load
    v = [r["val_nll_full_sweep"] for r in load(repo_paths.LEDGER)
         if r.get("run") == run and r.get("val_nll_full_sweep") is not None]
    return v[-1] if v else None


def functional_check(run):
    import common as C
    model, cfg, _, ckpath = C.load(run)
    from eval_runs import val_path_of
    vp = val_path_of(cfg)
    w = model.predict_wte.weight
    orig = w.detach().clone()
    counts = np.bincount(np.asarray(C.val_memmap(cfg), dtype=np.int64),
                         minlength=w.shape[0])
    act = torch.as_tensor(np.nonzero(counts > 0)[0], device=w.device)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(w.shape[0], generator=g).to(w.device)
    variants = [
        ("baseline_untouched", None),
        ("mean_active_row", lambda: orig[act].mean(0, keepdim=True).expand_as(orig)),
        ("row_permutation", lambda: orig[perm]),
        ("zeros", lambda: torch.zeros_like(orig)),
        ("state_table", lambda: model.transformer.wte.weight.detach().clone()),
        ("identity_after_restore", lambda: orig.clone()),
    ]
    out = dict(checkpoint=ckpath, eval="eval_runs.full_sweep_nll (full deterministic "
               "val sweep, bf16 autocast) -- the ledger's metric of record",
               ledger_val_nll=ledger_val_nll(run), variants={})
    for name, fn in variants:
        if fn is not None:
            with torch.no_grad():
                w.copy_(fn())
        t0 = time.time()
        nll, ntok, nseq = C.sweep_nll(model, vp, None)
        out["variants"][name] = dict(val_nll=round(nll, 6), tokens=ntok, seqs=nseq)
        print(f"functional {name}: nll={nll:.6f} ({time.time()-t0:.0f}s)", flush=True)
        with torch.no_grad():
            w.copy_(orig)
    base = out["variants"]["baseline_untouched"]["val_nll"]
    for v in out["variants"].values():
        v["delta_vs_baseline"] = round(v["val_nll"] - base, 6)
    out["identity_exact"] = out["variants"]["identity_after_restore"]["val_nll"] == base
    if out["ledger_val_nll"] is not None:
        out["baseline_matches_ledger"] = abs(base - out["ledger_val_nll"]) < 5e-6
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--functional", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import common as C
    run = C.resolve(args.run)
    out = args.out or C.result_path("a21_emb_divergence", run)
    C.assert_node_local_triton()
    res = compute(run, functional=args.functional)
    C.save_json(out, res)
    s = res["summary"]["all"]
    print(f"\n{run} ALL-active: " + "  ".join(
        f"{m}: PS={v['P_vs_S']} SH={v['null_S_vs_H']} rand={v['null_random']} "
        f"perm={v['null_rowperm_max']}" for m, v in s.items() if isinstance(v, dict)),
        flush=True)


if __name__ == "__main__":
    main()
