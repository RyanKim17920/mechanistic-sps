#!/usr/bin/env python3
"""numbers.tex: the numbers main.tex states in its prose, as \\newcommand macros computed from
the same data as the tables and figures (paper_data.py, make_tables.py).

The pre-registration bands are the one set of constants: they were fixed before the runs
they judge.

Usage:  python scripts/analysis/paper_numbers.py [--outdir paper/tables]
        (make_tables.py writes it too)
"""
from __future__ import annotations

import argparse
import itertools
import statistics
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

import make_tables as MT  # noqa: E402  (puts src/ and scripts/ on sys.path)
import paper_data as D  # noqa: E402
from plotting import paper_style as PS  # noqa: E402
from plotting.paper_style import ARMS, MANIFEST  # noqa: E402

# Pre-registered final-loss bands (nats/token), fixed before training: Sequential 6+6 seed 1
# against the Two-tower 12+12 seed-1 loss at registration (anchor + margin), and the Two-tower
# 12+6 targets.  Constants on purpose: re-deriving them from today's ledger would rewrite a
# registration after the fact.
PREREG = {"PreregAnchor": "2.732", "PreregMargin": "0.02", "PreregSuccess": "2.752",
          "PreregPartial": "2.782", "PreregFailure": "2.806",
          "PreregAsymSuccess": "2.737", "PreregAsymPartial": "2.752"}
TOLERANCE = 0.005     # the reporting tolerance (nats/token) the text compares gaps against
REPLICATION = 1.5     # an intervention magnitude replicates when the seeds agree within this factor


def rnd(x, nd):
    return f"{x:.{nd}f}"


def double_round(x, nd):
    """x rounded half-up to nd+1 decimals, then to nd.  This is the rule the two confidence
    bounds typed into the submitted text follow (seed SD lower bound 0.001446 -> 0.00145
    -> 0.0015; tied-attention gradient CI lower bound 0.04247 -> 0.0425 -> 0.043); rounding
    once gives 0.0014 and 0.042.  Kept so the text is reproduced exactly; used only there."""
    q = Decimal(repr(float(x)))
    step = Decimal(1).scaleb(-nd)
    return str(q.quantize(step / 10, ROUND_HALF_UP).quantize(step, ROUND_HALF_UP))


def millions(x, nd=1):
    return rnd(x / 1e6, nd) + "M"


def signed(x, nd):
    """a negative value in math mode, as the prose prints it ($-0.003$)"""
    s = rnd(x, nd)
    return f"${s}$" if s.startswith("-") else s


def span(vals, nd, scale=1):
    """'lo--hi' at nd decimals"""
    lo, hi = rnd(scale * min(vals), nd), rnd(scale * max(vals), nd)
    return lo if lo == hi else f"{lo}--{hi}"


def pvalue(p):
    """two significant digits from 0.01 up, else one digit times a power of ten"""
    if p >= 0.01:
        return f"{p:.2g}"
    m, e = f"{p:.0e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def mean(k):
    return statistics.fmean(D.nll_seeds(k))


def seed_sd():
    """Pooled within-arm seed SD of the final loss over every arm with several ledger runs,
    with its chi-square 95% CI, and the arms' sums of squares."""
    L = D.ledger()
    groups = {k: [L[s] for s in a["seeds"] if s in L] for k, a in ARMS.items()}
    ss = {k: sum((v - np.mean(vs)) ** 2 for v in vs) for k, vs in groups.items() if len(vs) > 1}
    df = sum(len(groups[k]) - 1 for k in ss)
    tot = sum(ss.values())
    sd = np.sqrt(tot / df)
    ci = [np.sqrt(tot / stats.chi2.ppf(q, df)) for q in (0.975, 0.025)]
    return sd, df, ci, ss, tot


def numbers() -> dict:
    N = dict(PREREG)
    N["Tolerance"] = rnd(TOLERANCE, 3)
    N["ReplicationFactor"] = rnd(REPLICATION, 1)
    T, S = PS.BODY_TRANSFORMER, PS.BODY_SPS
    two_by_two = MANIFEST["two_by_two"]
    (tt12, seq12), (tt6, seq6) = [(tt, seq) for _, _, tt, seq in two_by_two["rows"]]
    realloc, pause = two_by_two["two_tower_only"], two_by_two["sequential_only"]
    L = D.ledger()

    # ---- final losses, the read effect, efficiency ---------------------------------------
    N["SeqGainTwelve"] = rnd(mean(tt12) - mean(seq12), 3)
    N["SeqGainSix"] = rnd(mean(tt6) - mean(seq6), 3)
    for name, k in (("LossTransformer", T), ("LossSPS", S), ("LossTransformerTied", D.TRANSFORMER),
                    ("LossSPSTied", D.SPS), ("LossTwoTowerTwelve", tt12),
                    ("LossSeqTwelve", seq12), ("LossSeqSix", seq6), ("LossRealloc", realloc),
                    ("LossSeqSixPause", pause), ("LossAsym", D.ASYM)):
        N[name] = rnd(mean(k), 3)
    N["LossSeqSixSeedOne"] = rnd(L[ARMS[seq6]["seeds"][0]], 3)
    N["SharedGap"] = rnd(mean(D.SEQ_TIED) - mean(D.W0_SHARED), 3)
    gain = mean(T) - mean(S)
    N["RecoveryPct"] = rnd(100 * (mean(T) - mean(seq6)) / gain, 0)
    N["RecoveryPausePct"] = rnd(100 * (mean(T) - mean(pause)) / gain, 0)
    N["SeqSixAboveSPS"] = rnd(mean(seq6) - mean(S), 3)
    N["SeqSixSPSGap"] = "+" + rnd(mean(seq6) - mean(S), 4)
    N["ParamsTotalSeqSix"] = rnd(D.arch_stats(seq6)["params_total"] / 1e6, 0) + "M"
    N["ParamsTotalTransformer"] = rnd(D.arch_stats(T)["params_total"] / 1e6, 0) + "M"
    assert mean(pause) - mean(S) > TOLERANCE    # "more than 0.005 above SPS"

    # ---- compute frontier (fig:frontier) ---------------------------------------------------
    f = D.frontier()
    cur = f["cur"]
    N["ComputePct"] = rnd(100 * f["tT"] / cur[S][0][-1], 0)
    N["PFLOPsPreDecay"] = rnd(f["tc"] / 1e3, 1) + "k"
    N["PFLOPsShort"] = rnd(f["tT"] / 1e3, 1) + "k"
    N["PFLOPsSPS"] = rnd(cur[S][0][-1] / 1e3, 1) + "k"
    assert cur[tt12][0][-1] == cur[seq12][0][-1]
    N["PFLOPsTwelve"] = rnd(cur[seq12][0][-1] / 1e3, 1) + "k"
    final = {k: D.value_at(cur, k, None) for k in cur}
    N["SeedRangeMax"] = rnd(max(max(v[1]) - min(v[1]) for v in final.values() if v[1]), 3)
    pre = {k: D.value_at(cur, k, f["tc"])[0] for k in cur}
    N["PreDecayGap"] = rnd(pre[S] - pre[seq6], 3)
    N["PreDecaySeqSix"] = rnd(pre[seq6], 3)

    # ---- seed noise (App. training) --------------------------------------------------------
    sd, df, (_lo, hi), ss, tot = seed_sd()
    # the lower bound (0.001446) follows the text's double-rounding rule (double_round)
    N.update(SeedSD=rnd(sd, 4), SeedDF=str(df), SeedSDLo=double_round(_lo, 4), SeedSDHi=rnd(hi, 4),
             ToleranceSDs=rnd(TOLERANCE / sd, 1), SSShareTransformer=rnd(100 * ss[T] / tot, 0))
    d = mean(seq6) - mean(S)
    half = stats.t.ppf(0.975, df) * sd * np.sqrt(1 / len(D.nll_seeds(seq6)) + 1 / len(D.nll_seeds(S)))
    N["SeqSixSPSLo"], N["SeqSixSPSHi"] = rnd(d - half, 4), rnd(d + half, 4)
    # delta method for the recovery R = (T - Q) / (T - S), each mean over two seeds
    t_, s_, q_ = mean(T), mean(S), mean(seq6)
    grad = np.array([1 / (t_ - s_) - (t_ - q_) / (t_ - s_) ** 2, (t_ - q_) / (t_ - s_) ** 2,
                     -1 / (t_ - s_)])
    N["RecoveryDeltaPP"] = rnd(100 * stats.t.ppf(0.975, df) * sd / np.sqrt(2)
                               * np.linalg.norm(grad), 1)
    runs = D.ledger_rows()
    N["RunsTotal"] = str(len(runs))
    N["RunsTwentyB"] = str(sum("_20b" in r for r in runs))
    N["RunsShort"] = str(sum("_20b" not in r for r in runs))
    N["RunsListed"] = str(sum(s in L for a in ARMS.values() for s in a["seeds"]))
    vt = {runs[r]["val_tokens"] for r in runs if "_20b" in r}
    assert len(vt) == 1
    N["ValTokens"] = rnd(vt.pop() / 1e6, 1) + "M"

    # ---- circuit (tab:circuit), patching ----------------------------------------------------
    sep = [tt12, seq12, tt6, seq6]
    rows = MT.circuit_rows([(k, k, MT.circuit_runs(k)) for k in sep])

    def seed_mean(k, fn):
        return statistics.fmean(fn(r) for r in rows if r["key"] == k)
    N["PrevMassState"] = span([seed_mean(k, lambda r: r["sub"][0]["pm"][0]) for k in sep], 2)
    N["PrevMassPred"] = span([seed_mean(k, lambda r: r["sub"][0]["pm"][1]) for k in sep], 2)
    N["AblRatio"] = span([seed_mean(k, lambda r: r["abl"][0]) / seed_mean(k, lambda r: r["allc"][0])
                          for k in sep], 1)
    N["CtrlRatio"] = span([seed_mean(k, lambda r: r["abl"][1]) / seed_mean(k, lambda r: r["allc"][1])
                           for k in sep], 1)
    recs = MT.patching_recs()
    N["PatchRecovery"] = span([s["head"][0] for k in sep for s in recs[k]], 2)
    N["DonorMax"] = rnd(max(abs(s["donor"][0]) for k in sep for s in recs[k]), 2)

    # ---- read reach and read depth ----------------------------------------------------------
    i64 = D.A2_CAPS.index(64)
    N["KeepSixtyFour"] = span([statistics.fmean(c[i64] for c in D.a2_means(k, "mask"))
                               for k in sep], 2)
    N["KeepSixtyFourTransformer"] = rnd(
        statistics.fmean(c[i64] for _, c in D.a2_ctx_all_curves(T)), 3)
    caps = {K: statistics.fmean(v) for K, v in MT._a13_grid(tt12, "caps").items()}
    floors = {K: statistics.fmean(v) for K, v in MT._a13_grid(tt12, "floors").items()}
    n_levels = D.arch_stats(tt12)["state_n_layer"] + 1
    # a floor at K admits the m = n_levels - K deepest levels, paired with the cap at m + 1
    pairs = [(caps[n_levels + 1 - K], floors[K]) for K in floors if n_levels + 1 - K in caps]
    assert len(pairs) == 5, pairs
    N["FloorEight"], N["CapSix"] = rnd(floors[8], 2), rnd(caps[6], 2)
    N["DeepShallowRatio"] = span([c / fl for c, fl in pairs], 1)

    # ---- probes ------------------------------------------------------------------------------
    probe, bigram = MT.probe_floor_rows()
    N["BigramNext"], N["BigramPrev"] = rnd(bigram["P1"], 3), rnd(bigram["P3"], 3)
    N["BigramBoth"] = rnd(statistics.fmean(bigram.values()), 2)
    assert rnd(bigram["P1"], 2) == rnd(bigram["P3"], 2) == N["BigramBoth"]
    N["StatePrevProbe"] = span([v for k in [S] + sep for v in probe[k]["P3"][0]], 3)
    N["UntrainedProbe"] = rnd(statistics.fmean(v for c in probe.values() for v in c["P3"][1]), 1)

    # ---- gradient alignment (App. grad) -----------------------------------------------------
    from paper_appendix import app_gradcos
    names = MANIFEST["gradcos"]["macros"]
    grad = app_gradcos.summaries()
    for k, g in grad.items():
        a = "Grad" + names[k]
        N[a + "Mean"] = signed(g["mean_trained"], 3)
        lo, hi = g["ci_trained"]
        # the tied-attention lower bound (0.04247) follows the text's double-rounding rule
        N[a + "CI"] = "[" + ", ".join([double_round(lo, 3) if k == D.TIEDATTN else signed(lo, 3),
                                       signed(hi, 3)]) + "]"
        N[a + "Init"] = signed(g["mean_init"], 3)
        N[a + "P"] = pvalue(g["wilcoxon_p"])
    N["GradAFSPSInitCI"] = "[" + ", ".join(signed(v, 3) for v in grad[D.AFSPS]["ci_init"]) + "]"
    N["GradSharedTTPosInit"] = rnd(100 * grad[D.W0_SHARED]["frac_pos_init"], 0)
    N["GradSharedTTPosTrained"] = rnd(100 * grad[D.W0_SHARED]["frac_pos_trained"], 0)

    # ---- where Sequential's gain falls (tab:token-gain, tab:finalread) ---------------------
    rd, ind = MT.repeat_distance(), MT.induction_sweep()
    never = [rd[t]["buckets"]["never"] for t in rd]
    N["NeverShare"] = rnd(100 * never[0]["share"], 1)
    N["NeverGain"] = span([x["share_of_gain"] for x in never], 0, 100)
    for t, name in (("12+12", "Twelve"), ("6+6", "Six")):
        N["NeverGainPairs" + name] = span(
            [p["buckets"]["never"]["share_of_gain"] for p in rd[t]["per_pair"].values()], 0, 100)
        for seed, (_, g) in zip(("A", "B"), ind[t]):
            N[f"InductionGain{name}{seed}"] = rnd(100 * g, 1)
    N["InductionShare"] = rnd(100 * ind["12+12"][0][0], 1)
    N["InductionGain"] = span([g for pairs in ind.values() for _, g in pairs], 0, 100)
    a9 = D.result("a9_patching", D.mean_runs(tt12)[0])["natural"]
    N["InductionSeqs"] = str(a9["_definition"]["n_val_seq"])
    N["InductionTokens"] = f"{a9['all']['n']:,}"
    N["InductionTargets"] = f"{a9['induction']['n']:,}"
    a3 = D.result("a3_distance_nll", D.mean_runs(seq12)[0])["scoring"]
    assert a3["n_tokens"] == sum(b["n_tokens"] for b in rd["12+12"]["buckets"].values())
    N["SubSweepSeqs"] = str(a3["n_seq"])
    N["SubSweepTokens"] = f"{a3['n_tokens']:,}"
    N["SubSweepTokensM"] = rnd(a3["n_tokens"] / 1e6, 2) + "M"
    N.update(more_numbers(N, T, S, tt12, seq12, tt6, seq6, realloc, pause, rows, recs))
    return N


def more_numbers(N, T, S, tt12, seq12, tt6, seq6, realloc, pause, rows, recs) -> dict:
    """The rest of the numbers the text states: architecture facts, the control and variant
    arms, the appendix analyses."""
    M = {}
    L = D.ledger()
    st = D.arch_stats
    TT, ST = D.TRANSFORMER, D.SPS            # tied-head references
    slim, s39, s93 = (D.ROLES[r] for r in ("sequential6_slim", "sequential3p9", "sequential9p3"))

    def gflops(k):
        return st(k)["flops_fwd_per_token"] / 1e9

    def runs_nll(k):
        return [L[r] for r in D.mean_runs(k)]

    # ---- architecture: FLOPs and parameters ------------------------------------------------
    M["GFLOPsSPS"], M["GFLOPsTransformer"] = rnd(gflops(S), 3), rnd(gflops(T), 3)
    M["FLOPsRatioSPS"] = rnd(gflops(S) / gflops(T), 1)
    M["FLOPsRatioSPSFine"] = rnd(gflops(S) / gflops(T), 2)
    M["ParamsEmbedding"] = millions(st(TT)["params_total"] - st(TT)["params_nonemb"])
    M["ParamsTotalSeqSixFine"] = millions(st(seq6)["params_total"])
    M["ParamsTotalTransformerFine"] = millions(st(T)["params_total"])
    kv = st(D.SEQ_TIED)["params_nonemb"] - st(D.W0_SHARED)["params_nonemb"]
    M["ParamsSharedSeqKV"] = millions(kv)
    M["ParamsSharedSeqKVFine"] = millions(kv, 2)
    M["FLOPsSharedSeqKV"] = millions(st(D.SEQ_TIED)["flops_fwd_per_token"]
                                     - st(D.W0_SHARED)["flops_fwd_per_token"])
    M["ParamsNonembSeqSix"] = millions(st(seq6)["params_nonemb"])
    M["ParamsNonembSharedSeq"] = millions(st(D.SEQ_TIED)["params_nonemb"])
    M["GFLOPsSharedSeq"] = rnd(gflops(D.SEQ_TIED), 3)
    M["FLOPsFewerSeqSixSharedSeqPct"] = rnd(100 * (1 - gflops(seq6) / gflops(D.SEQ_TIED)), 0)
    M["FLOPsPctSeqSixSPS"] = rnd(100 * gflops(seq6) / gflops(S), 0)

    # ---- circuit (SPS key populations, ablation ratios) ------------------------------------
    sps_rows = MT.circuit_rows([(S, S, MT.circuit_runs(S))])
    kps = [s_["kp"] for s_ in sps_rows[0]["sub"]]
    assert sorted(kps) == ["pred", "state"], kps
    for kp, name in (("pred", "PredKeys"), ("state", "StateKeys")):
        M["SPSPrevMass" + name] = rnd(statistics.fmean(r["sub"][kps.index(kp)]["pm"][0]
                                                       for r in sps_rows), 2)
    rows = rows + sps_rows
    sep = [tt12, seq12, tt6, seq6]
    M["AblRatioRuns"] = span([r["abl"][0] / r["allc"][0] for r in rows if r["key"] in sep + [S]], 1)
    abl6 = [r["abl"][0] for r in rows if r["key"] == seq6]
    M["AblSeqSixSeeds"] = " / ".join(rnd(v, 3) for v in abl6)
    M["AblSeqSixSeedOne"], M["AblSeqSixSeedTwo"] = (rnd(v, 3) for v in abl6)

    # ---- patching (App. patching) ----------------------------------------------------------
    def seeds_of(k, t="head"):
        return " / ".join(rnd(s_[t][0], 2) for s_ in recs[k])
    for k, name in ((tt12, "TwoTowerTwelve"), (seq12, "SeqTwelve"), (tt6, "TwoTowerSix"),
                    (seq6, "SeqSix"), (T, "Transformer"), (S, "SPS")):
        M["Patch" + name] = seeds_of(k)
    M["PatchTransformerTied"] = rnd(recs[TT][0]["head"][0], 2)
    M["PatchSPSTied"] = span([s_["head"][0] for s_ in recs[ST]], 2)
    donors = {k: [s_["donor"][0] for s_ in recs[k]] for k in sep}
    M["DonorMin"] = signed(min(v for vs in donors.values() for v in vs), 2)

    # ---- read reach (App. readreach) --------------------------------------------------------
    i64 = D.A2_CAPS.index(64)
    keep = {k: [c[i64] for c in D.a2_means(k, "mask")] for k in sep + [S]}
    M["KeepSixtyFourSeeds"] = span([v for k in sep for v in keep[k]], 3)
    M["KeepSixtyFourTwoTowerSix"] = rnd(statistics.fmean(keep[tt6]), 3)
    M["KeepSixtyFourSeqSix"] = rnd(statistics.fmean(keep[seq6]), 3)
    M["KeepSixtyFourSPSSeeds"] = " / ".join(rnd(v, 3) for v in keep[S])
    ctx = {k: [c for _, c in D.a2_ctx_all_curves(k)] for k in (T, TT)}
    names = {16: "Sixteen", 256: "TwoFiftySix", 1024: "TenTwentyFour"}  # 64: KeepSixtyFourTransformer
    for i, n in enumerate(D.A2_CAPS):
        if n in names:
            M["TransformerReach" + names[n]] = rnd(statistics.fmean(c[i] for c in ctx[T]), 3)
    M["TransformerReachTiedShort"] = " / ".join(rnd(ctx[TT][0][i], 3) for i in (0, 1))
    M["TransformerReachTiedLong"] = " / ".join(rnd(ctx[TT][0][i], 3) for i in (2, 3))
    k64 = [c[i64] for c in ctx[T]]
    M["KeepSixtyFourTransformerPM"] = (rnd(statistics.fmean(k64), 3) + r"\pm"
                                       + rnd((max(k64) - min(k64)) / 2, 3))

    # ---- read depth (App. readdepth) ---------------------------------------------------------
    caps6 = {K: statistics.fmean(v) for K, v in MT._a13_grid(tt6, "caps").items()}
    floors6 = {K: statistics.fmean(v) for K, v in MT._a13_grid(tt6, "floors").items()}
    M["FloorFourSix"], M["CapTwoSix"], M["CapFourSix"] = (
        rnd(floors6[4], 2), rnd(caps6[2], 2), rnd(caps6[4], 2))
    imp12 = [d["caps"]["12"]["delta"] for _, d in D.result_seeds("a13_read_depth_cap", seq12)]
    M["ImposeTwoTowerOnSeqTwelve"] = " and ".join(rnd(v, 3) for v in imp12)
    n6 = str(st(tt6)["state_n_layer"])
    imp6 = [d["floors"][n6]["delta"] for _, d in D.result_seeds("a13_read_depth_cap", tt6)]
    M["ImposeSeqOnTwoTowerSixSeeds"] = " and ".join(rnd(v, 3) for v in imp6)
    M["ImposeSeqOnTwoTowerSix"] = rnd(statistics.fmean(imp6), 2)
    M["ImposeSeqOnTwoTowerSixPM"] = (rnd(statistics.fmean(imp6), 3) + r"\pm"
                                     + rnd((max(imp6) - min(imp6)) / 2, 3))

    # ---- lesions ------------------------------------------------------------------------------
    last = str(st(seq12)["state_n_layer"])
    fin = [dict(zip(*D.lesion_curve(d, "state")))[int(last)]
           for _, d in D.result_seeds("a12_depth_lesion", seq12)]
    M["SeqFinalStateLesion"] = span(fin, 2)

    # ---- losses of the control and variant arms ------------------------------------------------
    def m(k):
        return statistics.fmean(runs_nll(k))
    M["LossAFSPS"] = rnd(m(D.AFSPS), 3)
    M["MFLOPsAFSPS"] = millions(st(D.AFSPS)["flops_fwd_per_token"])
    M["FLOPsRatioAFSPS"] = rnd(gflops(D.AFSPS) / gflops(T), 2)
    M["AFSPSGainTied"] = rnd(m(TT) - m(D.AFSPS), 3)
    M["AFSPSBehindSPSTied"] = span([m(D.AFSPS) - L[r] for r in D.mean_runs(ST)], 3)
    M["FLOPsPctAFSPS"] = rnd(100 * gflops(D.AFSPS) / gflops(ST), 0)
    M["ParamsTotalAFSPS"] = millions(st(D.AFSPS)["params_total"], 0)
    M["LossSPSTiedSeedOne"] = rnd(L[ARMS[ST]["seeds"][0]], 3)
    M["LossSharedSeq"] = rnd(m(D.SEQ_TIED), 3)
    M["SharedSeqAboveSPSTied"] = rnd(m(D.SEQ_TIED) - m(ST), 3)
    M["SharedSeqBelowTransformerTied"] = rnd(m(TT) - m(D.SEQ_TIED), 3)
    gaps = [L[r] - m(D.W0_SHARED) for r in D.mean_runs(D.SEQ_TIED)]
    M["SharedGapPM"] = (rnd(statistics.fmean(gaps), 3) + r"\pm"
                        + rnd((max(gaps) - min(gaps)) / 2, 3))
    M["SeqSixBeatsSharedSeq"] = rnd(m(D.SEQ_TIED) - m(seq6), 3)
    M["LossTiedAttn"] = rnd(m(D.TIEDATTN), 3)
    M["ParamsNonembTiedAttn"] = millions(st(D.TIEDATTN)["params_nonemb"], 2)
    M["TiedAttnAboveTwoTower"] = "+" + rnd(m(D.TIEDATTN) - m(tt12), 3)
    M["LossSeqThreeNine"], M["LossSeqNineThree"] = rnd(m(s39), 3), rnd(m(s93), 3)
    M["SplitsBehindSeqSix"] = " / ".join(rnd(m(k) - m(seq6), 3) for k in (s39, s93))
    M["HalvingTwoTower"] = rnd(m(tt6) - m(tt12), 3)
    M["HalvingSeq"] = rnd(m(seq6) - m(seq12), 3)
    M["OwnTableVsPause"] = signed(m(seq6) - m(pause), 3)
    M["MFLOPsAsym"] = millions(st(D.ASYM)["flops_fwd_per_token"])
    # tied heads: Sequential 6+6 (tied head, <predict> input) against tied SPS / Transformer
    M["SlimBehindSPSTied"] = rnd(m(slim) - m(ST), 3)
    M["RecoveryTied"] = span([(m(TT) - L[r]) / (m(TT) - m(ST)) for r in D.mean_runs(slim)], 0, 100)
    # recovery over every combination of Transformer, SPS and Sequential 6+6 seeds
    rec = [(t - q) / (t - s_) for t, s_, q in itertools.product(runs_nll(T), runs_nll(S),
                                                                runs_nll(seq6))]
    M["RecoveryCombos"] = span(rec, 0, 100)
    M["SeqSixSPSGapPairs"] = "/".join("+" + rnd(q - s_, 4)
                                      for q, s_ in zip(runs_nll(seq6), runs_nll(S)))
    f = D.frontier()
    M["PreDecayTokensTwelve"] = rnd(f["tc"] * 1e15 / (3 * st(seq12)["flops_fwd_per_token"]) / 1e9,
                                    1) + "B"

    # ---- wall-clock ----------------------------------------------------------------------------
    W = MT.wallclock_by_arm()
    tok = {t: MT._median_metric(W.get(t, []), "train")[0] for t in W}
    M["TrainKtokSeqSix"] = rnd(tok["tt_seq6"] / 1e3, 0) + "k"
    M["TrainKtokTransformerFlex"] = rnd(tok[MT.FLEX_TAG] / 1e3, 0) + "k"
    M["SpeedSPS"] = rnd(tok["sps_tied"] / tok[MT.BASE_TAG], 2)
    spread = [(max(v) - min(v)) / statistics.median(v)
              for v in ([r["train"]["tokens_per_s"] for r in rs if isinstance(r.get("train"), dict)
                         and r["train"].get("tokens_per_s")] for rs in W.values()) if len(v) > 1]
    M["WallclockSpreadPct"] = rnd(100 * max(spread), 1)
    M.update(control_numbers(sep, S, rows))
    M.update(appendix_numbers(T, TT, S, ST, tt12, seq12, tt6, seq6))
    return M


def control_numbers(sep, S, rows) -> dict:
    """App. controls: Holm thresholds, the 100-draw control p values and counts."""
    M = {}
    tt12, seq12, tt6, seq6 = sep
    n_tests = len(MT.CTRL_FILES)
    M["HolmThresholds"] = ", ".join(rnd(MT.ALPHA / (n_tests - i), 4 if i < n_tests - 2 else 3)
                                    for i in range(n_tests - 1)) + " and " + rnd(MT.ALPHA, 2)
    body = MANIFEST["control_counts"]
    n_pm = [len(set(v["draws"])) for k in body for run in MT.circuit_runs(k)
            for m, v in MT.ctrl_p(run).items() if m.startswith("pattern_matched")]
    M["PatternTriples"] = span(n_pm, 0)
    by_run = {r["run"]: r for r in rows}

    def p(run, mode):
        return MT.ctrl_p(run)[mode]["p"]
    M["PMarginalTwoTowerTwelve"] = rnd(p(D.mean_runs(tt12)[0], "random_any"), 3)
    six = [r for k in (tt6, seq6) for r in MT.circuit_runs(k)]
    pp6 = [p(r, "pattern_matched") for r in six]
    M["PPatternSixSixLo"], M["PPatternSixSixHi"] = rnd(min(pp6), 2), rnd(max(pp6), 2)
    M["PPatternSPSOne"], M["PPatternSPSTwo"] = (rnd(p(r, "pattern_matched"), 2)
                                                for r in MT.circuit_runs(S))
    exc = {k: [p(r, "random_any_exL1") for r in MT.circuit_runs(k)] for k in (seq6, tt6)}
    M["PExclSeqSixSeedTwo"], M["PExclTwoTowerSixSeedTwo"] = (rnd(exc[k][1], 3) for k in (seq6, tt6))
    passing = sum(p(r, m) <= MT.ALPHA for r in six for m in MT.P_PAIRS[1])
    M["SixSixExclPass"] = f"{NUMBER_WORDS[passing]} of {NUMBER_WORDS[2 * len(six)]}"
    fr = {m: [MT.ctrl_p(r)[m]["k"] / MT.ctrl_p(r)[m]["n"] for r in MT.circuit_runs(S)]
          for m, _ in MT.CTRL_FILES}
    for m, name in (("random_any", "RandomAll"), ("random_any_exL1", "RandomExcl"),
                    ("pattern_matched", "PatternAll"), ("pattern_matched_exL1", "PatternExcl")):
        M["SPSExceedFrac" + name] = " / ".join(rnd(v, 2) for v in fr[m])
    M["PPatternSPSTiedSeedTwo"] = rnd(p(D.mean_runs(D.SPS)[0], "pattern_matched"), 3)
    M["PRandomSharedTT"] = rnd(p(D.W0_SHARED, "random_any"), 3)
    # the draws clipped at the right edge of Fig. app-ctrl-dist (seed 1 of each body model)
    from paper_appendix import app_ctrl_dist as CD

    def clipped(run, mode):
        d = MT.ctrl_files(run)[mode]
        prev = float(d["ctrl_summary"]["headline"]["prev_cost"])
        r = [-x["sets"][MT.P_METRIC]["d_acc_control"] / prev for x in d["ctrl_draws"]
             if x["sets"].get(MT.P_METRIC, {}).get("n")]
        return span([v for v in r if v > CD.XMAX], 2)
    M["ClippedSPSPattern"] = clipped(D.mean_runs(S)[0], "pattern_matched")
    M["ClippedTwoTowerSixRandom"] = clipped(D.mean_runs(tt6)[0], "random_any")
    M["ClippedTwoTowerSixPattern"] = clipped(D.mean_runs(tt6)[0], "pattern_matched")
    M["ClippedSeqSixPattern"] = clipped(D.mean_runs(seq6)[0], "pattern_matched")
    M["CtrlDistXMax"] = f"{CD.XMAX:g}"
    # App. seeds: per-seed head-ablation costs and SPS previous-token mass
    sps_runs = MT.circuit_runs(S)
    M["SPSAblAccSeeds"] = " / ".join(
        rnd(float(MT.ctrl_files(r)["random_any"]["ctrl_summary"]["headline"]["prev_cost"]), 3)
        for r in sps_runs)
    M["SPSAblSeeds"] = " / ".join(rnd(by_run[r]["abl"][0], 3) for r in sps_runs)
    kp = [s_["kp"] for s_ in by_run[sps_runs[0]]["sub"]].index("pred")
    M["SPSPrevMassRange"] = " to ".join(rnd(by_run[r]["sub"][kp]["pm"][0], 2) for r in sps_runs)
    for k, name in ((tt12, "TwoTowerTwelve"), (tt6, "TwoTowerSix")):
        M["Abl" + name + "Seeds"] = " / ".join(rnd(by_run[r]["abl"][0], 3)
                                              for r in MT.circuit_runs(k))
    tied = MT.circuit_rows([(D.SPS, D.SPS, D.mean_runs(D.SPS))])
    M["AblSPSTiedSeeds"] = " / ".join(rnd(r["abl"][0], 3) for r in tied)
    ks = [s_["kp"] for s_ in tied[0]["sub"]].index("state")
    M["SPSTiedPrevMassStateKeys"] = " / ".join(rnd(r["sub"][ks]["pm"][0], 2) for r in tied)
    return M


NUMBER_WORDS = {n: w for n, w in enumerate(
    "zero one two three four five six seven eight nine ten".split())}


def appendix_numbers(T, TT, S, ST, tt12, seq12, tt6, seq6) -> dict:
    """App. readreach (the Transformer's reach), probes, lesions, where the gain falls."""
    M = {}
    i64 = D.A2_CAPS.index(64)
    # the Transformer with some or all layers restricted to the nearest N tokens
    ctx = {r: d["sweeps"] for k in (T, TT) for r, d in D.result_seeds("a2_ctx_knockout", k)}

    def dl(run, key):
        sw = ctx[run]
        return sw[key]["val_nll"] - sw["control_uncapped"]["val_nll"]
    untied = D.mean_runs(T)
    M["TransformerMeanSixtyFourUntied"] = " / ".join(rnd(dl(r, "all_near64_mean"), 3) for r in untied)
    M["TransformerMeanSixtyFourTied"] = rnd(dl(D.mean_runs(TT)[0], "all_near64_mean"), 3)
    last = [n for n in (3, 6, 9)]
    for ab, name in (("mask", "Mask"), ("mean", "Mean")):
        M["TransformerLastLayers" + name] = span(
            [dl(r, f"last{n}_near64_{ab}") for r in ctx for n in last], 3)
    M["TransformerLastOne"] = span([dl(r, "last1_near64_mask") for r in ctx], 3)
    l1 = [dl(r, "last1_near64_mask") for r in untied]
    M["TransformerLastOnePM"] = rnd(statistics.fmean(l1), 3) + r"\pm" + rnd((max(l1) - min(l1)) / 2, 3)
    assert D.A2_CAPS[i64] == 64

    # role probes: the last layer of each stream
    def last_probe(k, tower, tg):
        return [D.probe_curve(d, tower, tg)[1][-1] for _, d in D.result_seeds("a20_role_probe", k)]
    M["TransformerPrevProbeFinal"] = rnd(statistics.fmean(last_probe(T, "single", "P3")), 3)
    M["TransformerNextProbeFinal"] = rnd(statistics.fmean(last_probe(T, "single", "P1")), 2)
    for k, name in ((S, "SPS"), (seq12, "SeqTwelve"), (tt12, "TwoTowerTwelve")):
        M["StateNextProbe" + name] = rnd(statistics.fmean(last_probe(k, "state", "P1")), 2)

    # zero vs mean whole-layer ablation
    zm = D.result("a12_depth_lesion", "zero_vs_mean")["models"]
    rho = [zm[r]["profiles"][f"{s_}_block"]["spearman"] for k in PS.body_arms()
           for r in D.pick_seeds(k, zm) for s_ in ("state", "pred")]
    M["LesionRhoMin"] = rnd(min(rho), 2)

    def prof(k, abl, layer):
        return statistics.fmean(zm[r]["profiles"]["state_block"][abl][layer - 1]
                                for r in D.pick_seeds(k, zm))
    M["MeanAblTwoTowerTwelveLayerOne"] = rnd(prof(tt12, "mean", 1), 2)
    # the text quotes the first seed only here (the second: 0.964)
    M["MeanAblTwoTowerSixLayerThreeSeedOne"] = rnd(
        zm[D.mean_runs(tt6)[0]]["profiles"]["state_block"]["mean"][2], 3)

    # per-layer lesions (a12) of the Sequential variants
    def lesion(k, kind, layer=None):
        """per seed-rule run: the state-tower `kind` lesion cost at `layer` (default: last)"""
        out = []
        for _, d in D.result_seeds("a12_depth_lesion", k):
            xs, ys = D.lesion_curve(d, "state", kind)
            out.append(ys[xs.index(layer)] if layer else ys[-1])
        return out
    mlp = lesion(seq12, "mlp")
    M["SeqTwelveLastStateMLPPM"] = (rnd(statistics.fmean(mlp), 3) + r"\pm"
                                    + rnd((max(mlp) - min(mlp)) / 2, 3))
    blk = lesion(D.SEQ_TIED, "block")
    M["SharedSeqLastStateRatio"] = rnd(max(blk) / min(blk), 1)
    splits = [seq6] + [D.ROLES[r] for r in ("sequential3p9", "sequential9p3")]
    M["SplitStateAttnLayerOne"] = span([v for k in splits for v in lesion(k, "attn", 1)], 3)
    # replacing Sequential 6+6's prediction table at inference (a21 functional variants)
    rep_ = [v["delta_vs_baseline"] for r in D.mean_runs(seq6)
            for n, v in D.result("a21_emb_divergence", r)["functional"]["variants"].items()
            if n not in ("baseline_untouched", "identity_after_restore")]
    M["TableReplaceSeqSix"] = span(rep_, 3)
    M["ZeroAblTwoTowerTwelveLayerOne"] = rnd(prof(tt12, "zero", 1), 2)

    # where the gain falls: the 12+12 next-token probe gap, Two-tower minus Sequential, over
    # the four seed pairs (the layer-4, layer-12 and 6+6 gap ranges the text prints differ
    # from these data in the third decimal: typed, see results/SOURCES.md)
    g3 = json.loads((D.RESULTS / "g3_final_read_mechanism.json").read_text())
    a20 = g3["pairs"]["12+12"]["a20"]
    gaps = np.array([p["gap_tt_minus_seq"] for p in a20["per_pair"].values()])
    M["ProbeGapTwelveFirst"], M["ProbeGapTwelveEleven"] = (span(gaps[:, i], 3) for i in (0, 10))

    def first_below(curve, thr):
        return next(i + 1 for i, v in enumerate(curve) if v < thr)
    for thr, name in ((5.0, "Five"), (4.5, "FourHalf")):
        M["ProbeThreshold" + name] = rnd(thr, 3)
        M["ProbeFirstBelow" + name + "Seq"] = str(first_below(a20["seq_mean"], thr))
        M["ProbeFirstBelow" + name + "TwoTower"] = str(first_below(a20["tt_mean"], thr))
    M["ProbeFirstBelowFiveTransformer"] = str(first_below(g3["transformer_reference"]["a20"]["curve"],
                                                          5.0))
    # the repeat-distance gain of the final read, per bucket and seed pair (App. gains)
    for t, name in (("12+12", "Twelve"), ("6+6", "Six")):
        M["SeqBucketGainPct" + name] = span(
            [-b["rel_diff_pct"] for p in g3["pairs"][t]["a3"]["per_pair"].values()
             for b in p["buckets"].values()], 0)
    M.update(mechanism_numbers(T, S, ST, tt12, seq12, tt6, seq6))
    return M


LESION_MLP_BOUND = 0.080   # the MLP-lesion bound App. lesions states; asserted below
LESION_TOTALS_WITHIN = 1.3  # the seed agreement App. lesions/seeds state for SPS's totals; asserted


def mechanism_numbers(T, S, ST, tt12, seq12, tt6, seq6) -> dict:
    """App. induction (synthetic lifts), path tracing, gains by repeat distance, lesions,
    gradient alignment and the seed-replication spreads."""
    M = {}
    sep = ((tt12, "TwoTowerTwelve"), (seq12, "SeqTwelve"), (tt6, "TwoTowerSix"), (seq6, "SeqSix"))

    # synthetic repeated random tokens (a5): max / mean lift over heads, per seed
    for k, name in sep:
        agg = [d["synthetic"]["aggregate_2x2"] for _, d in D.result_seeds("a5_induction", k)]
        M["SynthPredMax" + name] = span([a["pred_query__state_key"]["max_lift"] for a in agg], 0)
        M["SynthStateMax" + name] = span([a["state_query__state_key"]["max_lift"] for a in agg], 1)
        M["SynthPredMean" + name] = span([a["pred_query__state_key"]["mean_lift"] for a in agg], 1)

    # path tracing (a10): shares of the matching head's QK differential
    def qk(d):
        return d["tables"]["qk_differential"]
    a10 = dict(D.result_seeds("a10_pathtrace", tt12))
    for seed, d in zip(("A", "B"), a10.values()):
        M["PathTop" + seed] = rnd(100 * qk(d)["all_sources"][0]["frac_of_total"], 0)
        # a previous-token head above the matching head's read level has no row (None):
        # it is not upstream of the read and contributes nothing
        M["PathPrevHeads" + seed] = rnd(
            100 * sum(r["frac_of_total"] for r in d["prev_token_head_rows"]["qk_differential"]
                      if r is not None), 0)
        M["PathAttn" + seed] = rnd(
            100 * sum(r["frac_of_total"] for r in qk(d)["all_sources"] if r["source"] == "attn"), 0)
    M["PathTopOthers"] = span([qk(d)["all_sources"][0]["frac_of_total"] for k, _ in sep[1:]
                               for _, d in D.result_seeds("a10_pathtrace", k)], 0, 100)
    arms = D.result("a9_patching", next(iter(a10)))["patching"]["natural"]["arms"]
    M["PatchPrevKeySlots"] = rnd(100 * arms["prev_top3_keyslots"]["recovery_fraction"], 0)
    M["PatchAllKeySlots"] = rnd(100 * arms["ceiling_allheads_keyslots"]["recovery_fraction"], 0)

    # SPS against the Transformer on never-seen tokens (a3), each SPS seed against the
    # Transformer's seed mean
    a3 = {k: [d["buckets"]["never"]["mean_nll"] for _, d in D.result_seeds("a3_distance_nll", k)]
          for k in (T, S)}
    t_never = statistics.fmean(a3[T])
    for seed, v in zip(("A", "B"), a3[S]):
        M["SPSNeverGainPct" + seed] = rnd(100 * (t_never - v) / t_never, 2)

    # Shared Two-tower's next-token probe after its final norm (a20 tower "final")
    fin = [r for r in D.result("a20_role_probe", D.W0_SHARED)["probes"].values()
           if r["tower"] == "final"]
    M["SharedTTFinalNextProbe"] = rnd(fin[0]["P1"]["eval_nll"], 3)

    # lesions: SPS's unread final state layer, and the MLP bound under mean ablation
    M["SPSFinalStateLesion"] = span([D.lesion_curve(d, "state")[1][-1]
                                     for _, d in D.result_seeds("a12_depth_lesion", S)], 3)

    def mean_abl_mlp(k, stream):
        return [D.lesion_curve(D.result("a12_depth_lesion", f"{r}_mean"), stream, "mlp")[1]
                for r in D.mean_runs(k)]
    assert all(min(c[1:]) >= LESION_MLP_BOUND for c in mean_abl_mlp(T, "single"))
    assert all(max(c[3:]) <= LESION_MLP_BOUND for c in mean_abl_mlp(S, "state"))
    assert all(max(c[3:-1]) <= LESION_MLP_BOUND for c in mean_abl_mlp(seq12, "state"))
    M["LesionMLPBound"] = rnd(LESION_MLP_BOUND, 3)

    # SPS's whole-layer lesion totals per stream agree across its seeds within this factor
    for stream in ("state", "pred"):
        tot = [sum(D.lesion_curve(d, stream)[1]) for _, d in D.result_seeds("a12_depth_lesion", S)]
        assert max(tot) / min(tot) <= LESION_TOTALS_WITHIN, (stream, tot)
    M["LesionTotalsWithin"] = rnd(LESION_TOTALS_WITHIN, 1)

    # tied-head SPS's layer-1 state-MLP lesion: seed mean +- half-range
    l1 = [D.lesion_curve(d, "state", "mlp")[1][0] for _, d in D.result_seeds("a12_depth_lesion", D.SPS)]
    M["SPSTiedLayerOneStateMLPPM"] = (rnd(statistics.fmean(l1), 3) + r"\pm"
                                      + rnd((max(l1) - min(l1)) / 2, 3))

    # gradient alignment (a16): share of RMSNorm gains with a negative cosine, tied-head SPS
    # trained seeds and their untrained-init controls (Fig. app-gradcos)
    def neg_norm_share(run):
        c = [t["cos_state_pred"] for t in D.result("a16_grad_orthogonality", run)["tensors"]
             if t["kind"] in ("norm_attn", "norm_mlp") and t["cos_state_pred"] is not None]
        return sum(x < 0 for x in c) / len(c)
    inits = dict((a, i) for a, i, *_ in MANIFEST["gradcos"]["panels"])[D.SPS]
    M["NormNegSPSTied"] = span([neg_norm_share(r) for r in D.mean_runs(D.SPS)], 0, 100)
    M["NormNegSPSTiedInit"] = span([neg_norm_share(r) for r in inits], 0, 100)

    # seed replication (App. seeds): a value replicates when its seeds agree within REPLICATION
    def ratio(vs):
        return max(vs) / min(vs)
    probes = [ratio(v) for slot in ("state", "pred") for tg in ("P1", "P2", "P3")
              for v in zip(*(D.joint_probe_curve(d, slot, tg)[1]
                             for _, d in D.result_seeds("a20_role_probe", D.SPS)))]
    M["ProbeReplicatePctSPSTied"] = rnd(100 * statistics.fmean(r <= REPLICATION for r in probes), 0)

    def failing(grid):
        return [r for vs in grid.values() for r in [ratio(vs)] if r > REPLICATION]
    M["ReadDepthSpreadSeqTwelve"] = span(
        [r for w in ("caps", "floors") for r in failing(MT._a13_grid(seq12, w))], 1)
    M["FloorSpreadTwoTowerTwelve"] = span(failing(MT._a13_grid(tt12, "floors")), 1)
    reach = [ratio(vs) for vs in zip(*D.a2_means(D.SPS, "mask"))]
    # printed as "1.7--7": the upper end at whole-number precision
    M["ReachSpreadSPSTied"] = rnd(min(reach), 1) + "--" + rnd(max(reach), 0)
    return M


def write(out: Path):
    lines = [r"% AUTO-GENERATED by scripts/analysis/paper_numbers.py -- do not edit by hand."]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in numbers().items()]
    p = out / "numbers.tex"
    p.write_text("\n".join(lines) + "\n")
    print(f"WROTE {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(D.REPO / "paper" / "tables"))
    write(Path(ap.parse_args().outdir))


if __name__ == "__main__":
    main()
