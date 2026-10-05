"""G3 -- mechanism for the final-read gain (Sequential vs Two-tower), from existing data.

Eval-free: reads the committed A3 distance-bucket NLL and A20 role-probe JSONs.

  (1) A3.  Per repeat-distance bucket (distance from the target token to its last prior
      occurrence in the same document; "never" = not seen before in the document), the
      NLL difference Sequential - Two-tower, for every seed pair.  All arms score the SAME
      512 sub-sweep sequences, so each bucket holds IDENTICAL tokens in every arm and a
      per-bucket difference is paired (no frequency confound between arms).  The overall
      difference decomposes exactly as sum_b share_b * diff_b; we report each bucket's
      share of the gain and its per-token gain relative to the overall per-token gain
      ("concentration": 1.0 everywhere = perfectly broad).
  (2) A20.  Next-token (P1) linear-probe loss on the prediction stream after each layer,
      both reads, both seeds: how much earlier the final read's prediction stream
      becomes predictive (per-layer gap; first layer under fixed thresholds).

Pairs: 12+12 = Sequential {seed1, seed2} x Two-tower {seed1, seed2};
        6+6  = Sequential {seed1, seed2} x Two-tower {seed1, seed2}.
Transformer (untied head, the paper's reference) is carried alongside.

Usage: g3_final_read_mechanism.py  -> results/g3_final_read_mechanism.json
"""
from __future__ import annotations

import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_data as D  # noqa: E402  (puts src/ on sys.path)
from plotting.paper_style import MANIFEST  # noqa: E402

RES = str(D.RESULTS)

# the separate-weight rows of tab:2x2, each arm's seed-rule runs (paper_manifest.yaml)
PAIRS = {towers: dict(seq=D.mean_runs(seq), tt=D.mean_runs(tt))
         for _, towers, tt, seq in MANIFEST["two_by_two"]["rows"]}
TRANSFORMER = MANIFEST["body"]["transformer"]
BUCKETS = ["never", "1-16", "17-64", "65-256", "257-1024", "1025+"]
P1_THRESHOLDS = (5.0, 4.5, 4.0, 3.75)


def load(analysis, run):
    with open(os.path.join(RES, f"{analysis}_{run}.json")) as f:
        return json.load(f)


def r6(x):
    return None if x is None else round(float(x), 6)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


# ------------------------------------------------------------------------------ A3
def a3_block(seq_runs, tt_runs):
    A = {r: load("a3_distance_nll", r) for r in seq_runs + tt_runs + [TRANSFORMER]}
    ref = A[TRANSFORMER]
    n_ref = {b: A[seq_runs[0]]["buckets"][b]["n_tokens"] for b in BUCKETS}
    for r, d in A.items():                    # identical tokens in every arm
        assert d["scoring"]["n_tokens"] == ref["scoring"]["n_tokens"], r
        for b in BUCKETS:
            assert d["buckets"][b]["n_tokens"] == n_ref[b], (r, b)
    share = {b: A[seq_runs[0]]["buckets"][b]["share"] for b in BUCKETS}
    per_pair = {}
    for s, t in itertools.product(seq_runs, tt_runs):
        ds, dt = A[s], A[t]
        overall = ds["overall_mean_nll"] - dt["overall_mean_nll"]
        rows = {}
        for b in BUCKETS:
            diff = ds["buckets"][b]["mean_nll"] - dt["buckets"][b]["mean_nll"]
            rows[b] = dict(
                diff=r6(diff),
                rel_diff_pct=r6(100 * diff / dt["buckets"][b]["mean_nll"]),
                share_of_gain=r6(share[b] * diff / overall),
                concentration=r6(diff / overall),
            )
        rep = [b for b in BUCKETS if b != "never"]
        rep_share = sum(share[b] for b in rep)
        rep_diff = sum(share[b] * (ds["buckets"][b]["mean_nll"] - dt["buckets"][b]["mean_nll"])
                       for b in rep) / rep_share
        never_diff = ds["buckets"]["never"]["mean_nll"] - dt["buckets"]["never"]["mean_nll"]
        per_pair[f"{s}__vs__{t}"] = dict(
            overall_diff=r6(overall),
            overall_diff_recomposed=r6(sum(share[b] * rows[b]["diff"] for b in BUCKETS)),
            buckets=rows,
            repeated_vs_never=dict(
                repeated_share_tokens=r6(rep_share), never_share_tokens=r6(share["never"]),
                repeated_diff=r6(rep_diff), never_diff=r6(never_diff),
                never_over_repeated_per_token=r6(never_diff / rep_diff),
                never_share_of_gain=r6(share["never"] * never_diff / overall)),
        )
    # seed-mean summary per bucket
    summ = {}
    for b in BUCKETS:
        m_seq = mean(A[r]["buckets"][b]["mean_nll"] for r in seq_runs)
        m_tt = mean(A[r]["buckets"][b]["mean_nll"] for r in tt_runs)
        m_tr = ref["buckets"][b]["mean_nll"]
        diffs = [p["buckets"][b]["diff"] for p in per_pair.values()]
        summ[b] = dict(
            n_tokens=n_ref[b], share=r6(share[b]),
            seq_mean_nll=r6(m_seq), tt_mean_nll=r6(m_tt), transformer_nll=r6(m_tr),
            diff_seq_minus_tt=r6(m_seq - m_tt), diff_range=[r6(min(diffs)), r6(max(diffs))],
            rel_diff_pct=r6(100 * (m_seq - m_tt) / m_tt),
            share_of_gain=r6(mean(p["buckets"][b]["share_of_gain"]
                                  for p in per_pair.values())),
            concentration=r6(mean(p["buckets"][b]["concentration"]
                                  for p in per_pair.values())),
            seq_gain_vs_transformer_pct=r6(100 * (m_tr - m_seq) / m_tr),
            tt_gain_vs_transformer_pct=r6(100 * (m_tr - m_tt) / m_tr),
        )
    overall = mean(p["overall_diff"] for p in per_pair.values())
    conc = [summ[b]["concentration"] for b in BUCKETS]
    return dict(runs=dict(seq=seq_runs, tt=tt_runs, transformer=TRANSFORMER),
                overall_diff_seq_minus_tt=r6(overall),
                buckets=summ, per_pair=per_pair,
                concentration_max_over_min=r6(max(conc) / min(conc)),
                never_share_of_gain=r6(summ["never"]["share_of_gain"]),
                never_over_repeated_per_token=r6(mean(
                    p["repeated_vs_never"]["never_over_repeated_per_token"]
                    for p in per_pair.values())))


# ------------------------------------------------------------------------------ A20
def p1_curve(run):
    d = load("a20_role_probe", run)
    assert d["gate_pass"], run
    pr = d["probes"]
    stream = "single" if any(k.startswith("single_") for k in pr) else "pred"
    L = sum(1 for k in pr if k.startswith(stream + "_"))
    return dict(curve=[r6(pr[f"{stream}_{i}"]["P1"]["eval_nll"]) for i in range(L)],
                final=r6(pr["final"]["P1"]["eval_nll"]), read_map=d.get("read_map"),
                model_eval_nll_reduced=r6(d["model_eval_nll_reduced"]))


def first_layer_below(curve, thr):
    for i, v in enumerate(curve):
        if v <= thr:
            return i + 1                       # 1-indexed layer
    return None


def a20_block(seq_runs, tt_runs):
    C = {r: p1_curve(r) for r in seq_runs + tt_runs}
    L = len(C[seq_runs[0]]["curve"])
    assert all(len(c["curve"]) == L for c in C.values())
    per_pair = {}
    for s, t in itertools.product(seq_runs, tt_runs):
        gap = [r6(C[t]["curve"][i] - C[s]["curve"][i]) for i in range(L)]
        per_pair[f"{s}__vs__{t}"] = dict(
            gap_tt_minus_seq=gap,
            max_gap=r6(max(gap)), argmax_layer=1 + max(range(L), key=lambda i: gap[i]),
            mean_gap_all_layers=r6(mean(gap)),
            last_layer_gap=gap[-1],
            final_gap=r6(C[t]["final"] - C[s]["final"]))
    first = {}
    for thr in P1_THRESHOLDS:
        first[str(thr)] = {r: first_layer_below(C[r]["curve"], thr) for r in C}
    seq_mean = [r6(mean(C[r]["curve"][i] for r in seq_runs)) for i in range(L)]
    tt_mean = [r6(mean(C[r]["curve"][i] for r in tt_runs)) for i in range(L)]
    # Two-tower layer at which it first matches Sequential's layer-k value ("layers earlier")
    lead = []
    for k in range(L):
        tgt = seq_mean[k]
        j = next((i for i in range(L) if tt_mean[i] <= tgt), None)
        lead.append(None if j is None else j - k)
    return dict(curves={r: C[r] for r in C}, seq_mean=seq_mean, tt_mean=tt_mean,
                gap_mean_tt_minus_seq=[r6(t - s) for s, t in zip(seq_mean, tt_mean)],
                layers_lead_of_seq=lead,
                first_layer_below=first, per_pair=per_pair)


def main():
    out = dict(analysis="g3_final_read_mechanism",
               source="committed a3_distance_nll_*.json and a20_role_probe_*.json",
               bucket_definition="distance from the target token to its last prior "
                                 "occurrence in the same document; 'never' = no prior "
                                 "occurrence",
               sign="diff = Sequential - Two-tower (negative = final read better); "
                    "probe gap = Two-tower - Sequential (positive = Sequential more "
                    "predictive at that layer)",
               transformer_reference=dict(run=TRANSFORMER,
                                          a20=p1_curve(TRANSFORMER)),
               pairs={})
    for name, pr in PAIRS.items():
        out["pairs"][name] = dict(a3=a3_block(pr["seq"], pr["tt"]),
                                  a20=a20_block(pr["seq"], pr["tt"]))
    path = os.path.join(RES, "g3_final_read_mechanism.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"WROTE {path}")
    for name, blk in out["pairs"].items():
        a3 = blk["a3"]
        print(f"\n== {name}: overall Seq-TT {a3['overall_diff_seq_minus_tt']:+.4f}  "
              f"concentration max/min {a3['concentration_max_over_min']:.2f}  "
              f"never share of gain {a3['never_share_of_gain']:.3f}")
        print(f"   {'bucket':9s} {'share':>6s} {'diff':>8s} {'range':>18s} {'rel%':>7s} "
              f"{'gainShare':>9s} {'conc':>6s} {'seq/TF%':>8s} {'tt/TF%':>7s}")
        for b in BUCKETS:
            r = a3["buckets"][b]
            print(f"   {b:9s} {r['share']:6.3f} {r['diff_seq_minus_tt']:+8.4f} "
                  f"[{r['diff_range'][0]:+.4f},{r['diff_range'][1]:+.4f}] "
                  f"{r['rel_diff_pct']:+7.2f} {r['share_of_gain']:9.3f} "
                  f"{r['concentration']:6.2f} {r['seq_gain_vs_transformer_pct']:8.2f} "
                  f"{r['tt_gain_vs_transformer_pct']:7.2f}")
        a20 = blk["a20"]
        print("   P1 seq  :", a20["seq_mean"])
        print("   P1 tt   :", a20["tt_mean"])
        print("   gap tt-seq:", a20["gap_mean_tt_minus_seq"])
        print("   lead (layers):", a20["layers_lead_of_seq"])
        print("   first layer below:", json.dumps(a20["first_layer_below"]))
        for k, p in a20["per_pair"].items():
            print(f"   {k}: max gap {p['max_gap']:+.3f} @L{p['argmax_layer']}, "
                  f"mean {p['mean_gap_all_layers']:+.3f}, last {p['last_layer_gap']:+.3f}, "
                  f"final {p['final_gap']:+.3f}")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
