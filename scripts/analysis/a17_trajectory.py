#!/usr/bin/env python3
"""A17 -- WHEN does the copying circuit form, and when does the sharing conflict appear?

Everything else in this directory measures FINAL checkpoints.  The sharing/access axis of
the paper is therefore an end-state statement: untied two towers
carry a 743.5x matching head, tying the attention weights costs some of it (121.9x), and
sharing attention AND the FFN destroys it (2.4x).  a16_grad_orthogonality.py
supplies the mechanism at that same end state: the two streams' gradient demands on a
shared tensor are near-orthogonal, and the ONE low-dimensional shared resource -- the
fully-shared arm's pooling gate `W_in` -- is actively anti-aligned (median -0.778).

Both statements are about t = 20B tokens.  Neither says WHEN.  This script walks the
committed checkpoint ladder of five arms and asks three questions of the trajectory:

  PART A (circuit formation).  At each ladder point, the SYNTHETIC induction probe of
  `a5_induction.py` -- a 256-token random block repeated twice inside a 4096-token
  context, so a head that scores above chance can only be doing in-context copying.  The
  reported number is `a5`'s own `max_lift` for the readout-query x state-key cell (the
  single-stream model has one stream, so its one cell is used).  A companion pass
  measures the other half of the circuit, the STATE-side previous-token mass, on the same
  synthetic contexts.  `run_synthetic`, `summarise` and `prev_token_profile` are IMPORTED
  from `a5_induction.py`, not re-implemented: the trajectory has to be measured by the
  same estimator that produced the end-state numbers or it cannot be compared to them.

  PART B (sharing conflict).  At each ladder point of the two arms that actually share a
  tensor, the gradient decomposition of `a16_grad_orthogonality.py` -- `split_mode`,
  `shared_specs`, `accumulate`, `disattenuated_cos` are IMPORTED from it for the same
  reason.  Reported per point: the overall noise-corrected cos(g_state, g_pred), the
  pooling gate's own cosine (fully-shared arm only), the batch-split SNR control, and the
  additivity gate.

  SAMPLE SIZE IS LOAD-BEARING.  On 32 sequences the batch-split control comes back at
  +0.004 / -0.016 -- the estimator saying its own gradient is minibatch noise, at which
  point NO cross-stream cosine measured on it means anything.  Part B therefore sweeps the ENTIRE val set by
  default, and every point reports its SNR control.  A point whose control falls below
  `--snr-floor` (0.3) is marked UNUSABLE and is dropped from the figure rather than drawn
  with a caveat.

Usage
-----
    scripts/run/eval_shim.sh scripts/analysis/a17_trajectory.py partA --arm untied [--null]
    scripts/run/eval_shim.sh scripts/analysis/a17_trajectory.py partB --arm afsps
    scripts/analysis/a17_trajectory.py ladder          # CPU only: which points exist
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np

import common as C  # noqa: E402  (puts src/ and scripts/dualsps on sys.path)

ANALYSIS = "a17_trajectory"
OUTDIR = os.path.join(C.RESULTS_DIR, "a17")

# The ladder.  `ckpt_tokens_*.pt` files are written at ~1B intervals but the exact token
# count is whatever the step boundary landed on, so a point is matched by nearest-B with a
# tolerance rather than by filename arithmetic.  A point with no file is REPORTED MISSING,
# never interpolated.
LADDER_B = (1, 2, 3, 4, 6, 8, 12, 16, 20)
TOL_B = 0.08

# arm key (the tag in the result file names) -> run name: paper_manifest.yaml `trajectory`
ARMS = C.MANIFEST["trajectory"]

SNR_FLOOR = 0.3          # below this the point carries no interpretable cosine
ADDITIVITY_MAX = 1e-2    # a16_grad_orthogonality's additivity gate; must stay under it


# ======================================================================================
# checkpoint ladder resolution
# ======================================================================================
def ladder_paths(run: str) -> dict:
    """-> {B: path} for every ladder point this run actually has on disk.

    `*_pre_decay.pt` is excluded: it is a branch point of the LR schedule, not a point on
    the training trajectory the other checkpoints trace.  Checkpoints are looked for in the
    run directory and in its `ladder/` subdirectory, where
    `hf_export.py download --extra <run>` puts the weights-only ladder of the
    ryankim17920/mechanistic-sps-extra Hugging Face repo.
    """
    from _paths import run_glob  # noqa: E402
    found = {}
    for p in run_glob(run, "ckpt_tokens_*.pt") + run_glob(run, "ladder/ckpt_tokens_*.pt"):
        base = os.path.basename(p)
        if "pre_decay" in base:
            continue
        m = re.search(r"ckpt_tokens_(\d+)", base)
        if not m:
            continue
        tb = int(m.group(1)) / 1e9
        for b in LADDER_B:
            if abs(tb - b) <= TOL_B:
                # prefer the *_final* file when two land on the same point
                if b not in found or "final" in base:
                    found[b] = p
    return found


def load_ckpt(run: str, ckpt_path: str):
    """`eval_runs.load_model`, but for a CHOSEN checkpoint instead of the final one.

    Same config path, same config/checkpoint agreement assert, same key rewriting.  The
    only reason this exists is that `load_model` hardcodes `*final*.pt`; the trajectory
    needs the ladder.  `strict=False` is what the loader uses, so the unexpected-key set
    is asserted empty here -- a silently-unloaded tensor would look like a checkpoint that
    simply had not learned anything yet, which is exactly the signal this script measures.
    """
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from eval_runs import assert_config_matches_checkpoint  # noqa: E402

    with initialize_config_dir(config_dir=C.CONF, version_base=None):
        cfg = compose("config", overrides=[f"+experiment={run}", f"system.data_root={C.repo_paths.data_root()}"])
    m = instantiate(cfg.model).cuda().eval()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert_config_matches_checkpoint(run, m, ck)
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ck["model"].items()}
    sd.pop("freqs_cis", None)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected, f"{ckpt_path}: checkpoint keys not in the model: {unexpected[:8]}"
    bad = [k for k in missing if "freqs_cis" not in k]
    assert not bad, f"{ckpt_path}: model parameters absent from the checkpoint: {bad[:8]}"
    return m, cfg, C.family_of(cfg), ck


def point_path(part: str, arm: str, tag) -> str:
    return os.path.join(OUTDIR, f"{ANALYSIS}_{part}_{arm}_{tag}.json")


# ======================================================================================
# PART A -- circuit formation
# ======================================================================================
def prev_token_synth(model, cfg, family, n_seq: int, seed: int = 7):
    """State-side previous-token mass on the SAME synthetic contexts a5 uses.

    `a5.run_synthetic` records the induction masses only; the previous-token half of the
    circuit is recorded by `a5.run_natural`, which this script does not run (the synthetic
    probe is the clean number and the natural one costs a val sweep per ladder point).
    So the mass at distance exactly 1 -- a5's own `prev_token_mass` definition -- is
    accumulated here over a second pass of the same contexts, into an `a5.Acc`, and handed
    to a5's own `summarise` / `prev_token_profile`.  The probe, the accumulator and the
    summary are a5's; only the extra pass is new.
    """
    import torch  # noqa: F401
    import a5_induction as A5
    import common as C

    rng = np.random.default_rng(seed)
    acc = A5.Acc()
    qpos = np.arange(A5.COPY_B + 1, A5.COPY_B + A5.BLOCK_LEN)
    qpos = qpos[:: max(1, len(qpos) // C.N_Q_ATTN)][:C.N_Q_ATTN]
    for _si in range(n_seq):
        toks = A5.synthetic_batch(cfg, rng)
        X = torch.from_numpy(toks)[None].cuda()
        Y = torch.from_numpy(np.concatenate([toks[1:], toks[:1]]))[None].cuda()
        for v in C.attention_views(model, cfg, family, X, Y, qpos):
            d = v.dist()
            for ks in (0, 1):
                ks_sel = (v.k_stream == ks).view(1, -1)
                if not bool(ks_sel.any()):
                    continue
                key = (v.block, v.stream, "state" if ks == 0 else "pred")
                sel1 = (d == 1) & ks_sel & v.vis
                acc.add(key, "prev_token_mass",
                        (v.p * sel1.unsqueeze(1)).sum(-1).cpu().numpy())
    return acc


def matching_cell(family: str) -> str:
    """The 2x2 cell the trajectory follows: the READOUT asking, the MEMORY answering."""
    return "single_query__state_key" if family == "standard" else "pred_query__state_key"


def part_a_point(arm: str, b, ckpt_path: str, n_synth: int, untrained: bool = False):
    import torch
    import a5_induction as A5
    import common as C

    run = ARMS[arm]
    t0 = time.time()
    if untrained:
        model, cfg, family, tag = C.load_untrained(run, C.UNTRAINED_SEEDS[0])
        ckpt = tag
    else:
        model, cfg, family, _ck = load_ckpt(run, ckpt_path)
        ckpt = ckpt_path

    syn = A5.run_synthetic(model, cfg, family, n_synth)
    # The previous-token pass re-draws the SAME synthetic contexts (same seed) and is
    # folded into the same accumulator, so a5's own `summarise` -- which expects the
    # induction masses and picks up `prev_token_mass` when it is present -- produces both
    # halves of the circuit in one set of rows.
    prev = prev_token_synth(model, cfg, family, n_synth)
    for k, e in prev.d.items():
        for nm, v in e.items():
            syn.d.setdefault(k, {}).setdefault(nm, []).extend(v)
    syn_rows, syn_agg = A5.summarise(syn, family)
    prof = A5.prev_token_profile(syn_rows)

    cell = matching_cell(family)
    a = syn_agg.get(cell) or {}
    state_prof = [v for v in prof.values()
                  if v["stream"] == ("single" if family == "standard" else "state")]
    best_prev = max(state_prof, key=lambda v: v["max_prev_token_mass"]) if state_prof else None

    res = dict(analysis=ANALYSIS, part="A", arm=arm, run=run, family=family,
               tokens_B=(None if untrained else b), untrained=bool(untrained),
               checkpoint=str(ckpt), n_synthetic_seq=n_synth, cell=cell,
               matching_lift=a.get("max_lift"), matching_mean_lift=a.get("mean_lift"),
               matching_block=a.get("argmax_block"), matching_head=a.get("argmax_head"),
               n_heads_lift_gt3=a.get("n_heads_lift_gt3"),
               prev_token_state=(None if best_prev is None else best_prev["max_prev_token_mass"]),
               prev_token_state_block=(None if best_prev is None else best_prev["block"]),
               prev_token_state_head=(None if best_prev is None else best_prev["head"]),
               aggregate_2x2=syn_agg, prev_token_profile=prof,
               synthetic_per_head=syn_rows, seconds=round(time.time() - t0, 1))
    del model
    torch.cuda.empty_cache()
    return res


# ======================================================================================
# PART B -- the sharing conflict, over training
# ======================================================================================
def part_b_point(arm: str, b, ckpt_path: str, n_seq: int, micro_batch: int):
    import torch
    import common as C
    import a16_grad_orthogonality as A16   # NB: re-enables autograd process-wide

    run = ARMS[arm]
    t0 = time.time()
    model, cfg, family, _ck = load_ckpt(run, ckpt_path)
    assert family == "two_tower", f"{run}: part B is a two-tower measurement, got {family}"
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    data = C.val_memmap(cfg)
    starts = C.seq_starts(data, n_seq)
    batches = []
    for i in range(0, len(starts), micro_batch):
        sl = starts[i:i + micro_batch]
        X = torch.stack([torch.from_numpy(data[j:j + C.BLOCK].astype(np.int64)) for j in sl]).cuda()
        Y = torch.stack([torch.from_numpy(data[j + 1:j + 1 + C.BLOCK].astype(np.int64)) for j in sl]).cuda()
        batches.append((X, Y))
    n_mb = len(batches)
    half_of = lambda bi: 0 if bi < n_mb // 2 else 1  # noqa: E731

    specs = A16.shared_specs(model)
    assert specs, f"{run}: no shared parameter tensors"
    pas = {name: (p, sl) for name, _k, _b, p, sl in specs}
    res, losses = {}, {}
    for mode in ("total", "state", "pred"):
        res[mode], losses[mode] = A16.accumulate(model, batches, pas, mode, half_of)
        print(f"  [{arm} {b}B] mode={mode} loss={np.mean(losses[mode]):.8f}", flush=True)
    lm = {m: float(np.mean(v)) for m, v in losses.items()}
    tot = {m: {k: res[m][0][k] + res[m][1][k] for k in res[m][0]} for m in res}

    n_block = len(model.transformer.state_h)
    rows, worst_rel, spread_max = [], 0.0, 0.0
    for name, kind, blk, _p, _sl in specs:
        gt = tot["total"].get(name)
        if gt is None:
            continue
        zero = torch.zeros_like(gt)
        gs = tot["state"].get(name, zero)
        gp = tot["pred"].get(name, zero)
        den = float(gt.norm())
        rel = float((gs + gp - gt).norm()) / den if den > 0 else 0.0
        worst_rel = max(worst_rel, rel)
        est = [v for v in (A16.cos(gs, gp), A16.cos(gt - gp, gp), A16.cos(gs, gt - gs))
               if v is not None]
        if len(est) == 3:
            spread_max = max(spread_max, max(est) - min(est))
        ns, np_ = float(gs.norm()), float(gp.norm())
        rows.append(dict(
            name=name, kind=kind, block=blk,
            cos_state_pred=A16.cos(gs, gp),
            cos_disattenuated=A16.disattenuated_cos(
                res["state"][0].get(name, zero), res["state"][1].get(name, zero),
                res["pred"][0].get(name, zero), res["pred"][1].get(name, zero)),
            control_batch_split_state=A16.cos(res["state"][0].get(name, zero),
                                              res["state"][1].get(name, zero)),
            control_batch_split_pred=A16.cos(res["pred"][0].get(name, zero),
                                             res["pred"][1].get(name, zero)),
            norm_state=ns, norm_pred=np_, rel_err=rel,
            stream_exclusive=(ns == 0.0 or np_ == 0.0)))

    # a16_grad_orthogonality excludes the last block from its headline: under read_map='pre'
    # the fully-shared arm's pred block 11 reads the state residual BEFORE state block 11
    # writes it, so that block is gradient-inert on the state side and its "cosine" is a
    # ratio of vanishing norms.  Same exclusion here, applied to both arms so the two
    # trajectories are the same statistic.
    live = [r for r in rows if not r["stream_exclusive"] and r["block"] != n_block - 1]

    def med(key, sel):
        v = [r[key] for r in sel if r.get(key) is not None]
        return float(np.median(v)) if v else None

    pool = [r for r in live if r["kind"] == "ffn_pool_gate"]
    snr_s, snr_p = med("control_batch_split_state", live), med("control_batch_split_pred", live)
    snr = None if (snr_s is None or snr_p is None) else min(snr_s, snr_p)

    out = dict(analysis=ANALYSIS, part="B", arm=arm, run=run, tokens_B=b,
               checkpoint=str(ckpt_path), n_seq=len(starts), micro_batch=micro_batch,
               tokens=len(starts) * C.BLOCK, loss_by_mode=lm,
               loss_max_spread=max(lm.values()) - min(lm.values()),
               n_tensors=len(live), last_block_excluded=n_block - 1,
               cos_overall=med("cos_disattenuated", live),
               cos_overall_raw=med("cos_state_pred", live),
               cos_pool_gate=med("cos_disattenuated", pool) if pool else None,
               snr_state=snr_s, snr_pred=snr_p, snr=snr,
               snr_floor=SNR_FLOOR, usable=bool(snr is not None and snr >= SNR_FLOOR),
               additivity_max_rel_err=worst_rel,
               additivity_gate_max=ADDITIVITY_MAX,
               additivity_passed=bool(worst_rel < ADDITIVITY_MAX),
               estimator_spread_max=spread_max,
               tensors=rows, seconds=round(time.time() - t0, 1))
    del model, batches, res, tot
    torch.cuda.empty_cache()
    return out


# ======================================================================================
# main
# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("partA", "partB", "ladder"))
    ap.add_argument("--arm", default=None)
    ap.add_argument("--points", default=None, help="comma-separated B values")
    ap.add_argument("--n-synth", type=int, default=8)
    ap.add_argument("--n-seq", type=int, default=0, help="part B val sequences; 0 = ALL")
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--null", action="store_true", help="part A: also the untrained init")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)

    if args.mode == "ladder":
        for arm, run in ARMS.items():
            have = ladder_paths(run)
            miss = [b for b in LADDER_B if b not in have]
            print(f"{arm:9s} {run:34s} have={sorted(have)} MISSING={miss}")
        return

    assert args.arm in ARMS, f"--arm must be one of {sorted(ARMS)}"
    C.assert_node_local_triton()
    run = ARMS[args.arm]
    have = ladder_paths(run)
    want = [int(x) for x in args.points.split(",")] if args.points else list(LADDER_B)
    print(f"{args.arm}: ladder on disk {sorted(have)}; "
          f"MISSING {[b for b in want if b not in have]}", flush=True)

    if args.mode == "partA" and args.null:
        p = point_path("partA", args.arm, "init")
        if args.force or not os.path.exists(p):
            C.save_json(p, part_a_point(args.arm, None, None, args.n_synth, untrained=True))

    for b in want:
        if b not in have:
            continue
        p = point_path(args.mode, args.arm, f"{b}B")
        if os.path.exists(p) and not args.force:
            print(f"  skip {args.arm} {b}B (exists)", flush=True)
            continue
        if args.mode == "partA":
            r = part_a_point(args.arm, b, have[b], args.n_synth)
            print(f"  {args.arm} {b}B lift={r['matching_lift']} "
                  f"prev={r['prev_token_state']} ({r['seconds']}s)", flush=True)
        else:
            # n_seq=0 means "the whole val set": C.seq_starts caps the request at the
            # number of non-overlapping sequences the val.bin actually holds (3,198), so
            # asking for more can never silently reuse data.
            n_seq = args.n_seq if args.n_seq else 10 ** 9
            r = part_b_point(args.arm, b, have[b], n_seq, args.micro_batch)
            print(f"  {args.arm} {b}B cos={r['cos_overall']} pool={r['cos_pool_gate']} "
                  f"snr={r['snr']} usable={r['usable']} ({r['seconds']}s)", flush=True)
        C.save_json(p, r)


if __name__ == "__main__":
    main()
