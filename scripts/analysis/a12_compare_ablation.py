"""A12 zero- vs mean-ablation comparison: do the depth-lesion conclusions survive?

Reads a12_depth_lesion_<run>.json (zero) and a12_depth_lesion_<run>_mean.json (mean) for
the body models and writes results/a12_depth_lesion_zero_vs_mean.json with, per model and
per (stream, kind) profile: Spearman rank correlation of the per-layer cost profiles,
the magnitude ratio sum(mean)/sum(zero), and the four qualitative claims evaluated
under each ablation with fixed, pre-stated operational tests:

  C1 separated (two-tower / sequential): state layer 1 is the costliest state block
     lesion AND the costliest block lesion of either stream.
  C2 SPS: state slots front-loaded -- state layer 1 is the argmax and every state layer
     4..L costs < 0.05 nats; pred slots matter at every depth -- every pred layer costs
     >= 0.05 nats.
  C3 final read vs same-layer read: last state block's cost is larger under the final
     read (seq) than under the same-layer read (w0_equal) at matched depth and seed.
  C4 intermediate layers cheap: the median block lesion over layers 2..L-1 (1-indexed;
     both streams) costs < 10% of the state-layer-1 block lesion (max also reported).

Layers are reported 1-indexed.
Usage: a12_compare_ablation.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_data as D  # noqa: E402  (puts src/ on sys.path)
from plotting.paper_style import MANIFEST  # noqa: E402

RES = str(D.RESULTS)

# the separate-weight rows of tab:2x2 (paper_manifest.yaml), each arm's seed-rule runs
ROWS = [(towers.split("+")[0], D.mean_runs(tt), D.mean_runs(seq))
        for _, towers, tt, seq in MANIFEST["two_by_two"]["rows"]]
SEPARATED = {f"{fam}{n}": runs for n, tt, seq in ROWS for fam, runs in (("tt", tt), ("seq", seq))}
SPS = MANIFEST["lesion_compare"]["sps"]
C3_PAIRS = [pair for _, tt, seq in ROWS for pair in zip(seq, tt)]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def load(run, abl):
    p = os.path.join(RES, f"a12_depth_lesion_{run}{'' if abl == 'zero' else '_' + abl}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if not d.get("control_gate_passed"):
        return None
    return d


def profiles(d):
    L = d["lesions"]
    out = {}
    for s in ("state", "pred"):
        for k in ("attn", "mlp", "block"):
            tags = sorted(t for t in L if L[t]["stream"] == s and L[t]["kind"] == k)
            out[f"{s}_{k}"] = [L[t]["delta"] for t in sorted(tags, key=lambda t: L[t]["block"])]
    return out


def c1(p):
    st, pr = p["state_block"], p["pred_block"]
    return dict(passed=bool(int(np.argmax(st)) == 0 and st[0] >= max(pr)),
                state1=st[0], state_argmax_layer=int(np.argmax(st)) + 1,
                max_other=max(st[1:] + pr))


def c2(p):
    st, pr = p["state_block"], p["pred_block"]
    front = int(np.argmax(st)) == 0 and max(st[3:]) < 0.05
    every = min(pr) >= 0.05
    return dict(passed=bool(front and every), state_front_loaded=bool(front),
                pred_every_depth=bool(every), state1=st[0], max_state_4_to_L=max(st[3:]),
                argmax_state_4_to_L=int(np.argmax(st[3:])) + 4, min_pred=min(pr),
                argmin_pred=int(np.argmin(pr)) + 1)


def c4(p):
    st, pr = p["state_block"], p["pred_block"]
    mid = st[1:-1] + pr[1:-1]
    ref = st[0]
    return dict(passed=bool(float(np.median(mid)) < 0.10 * ref),
                max_mid_over_state1=max(mid) / ref, median_mid_over_state1=float(np.median(mid)) / ref)


def main():
    out = dict(analysis="a12_compare_ablation", note=__doc__.split("Usage")[0].strip(),
               models={}, c3={})
    runs = [r for g in SEPARATED.values() for r in g] + SPS
    for run in runs:
        z, m = load(run, "zero"), load(run, "mean")
        if z is None or m is None:
            out["models"][run] = dict(missing=[a for a, d in (("zero", z), ("mean", m)) if d is None])
            continue
        pz, pm = profiles(z), profiles(m)
        rec = dict(native_nll=z["native_sub_sweep_nll"],
                   native_nll_mean_file=m["native_sub_sweep_nll"], profiles={}, claims={})
        for key in pz:
            a, b = pz[key], pm[key]
            rec["profiles"][key] = dict(
                zero=a, mean=b, spearman=spearman(a, b),
                sum_zero=float(sum(a)), sum_mean=float(sum(b)),
                ratio_sum=float(sum(b) / sum(a)) if sum(a) else float("nan"),
                argmax_zero=int(np.argmax(a)) + 1, argmax_mean=int(np.argmax(b)) + 1)
        if run in SPS:
            rec["claims"]["C2"] = dict(zero=c2(pz), mean=c2(pm))
            g = m.get("gates", {}).get("dead_end_last_state_block", {})
            rec["dead_end_last_state_block_mean"] = g.get("delta")
            rec["dead_end_last_state_block_zero"] = z.get("gates", {}).get(
                "dead_end_last_state_block", {}).get("delta")
        else:
            rec["claims"]["C1"] = dict(zero=c1(pz), mean=c1(pm))
        rec["claims"]["C4"] = dict(zero=c4(pz), mean=c4(pm))
        rec["mean_ablation"] = {k: v for k, v in m.get("mean_ablation", {}).items()
                                if k != "per_write"}
        out["models"][run] = rec
    for fin, same in C3_PAIRS:
        r = {}
        for abl in ("zero", "mean"):
            a, b = load(fin, abl), load(same, abl)
            if a is None or b is None:
                r[abl] = None
                continue
            la, lb = profiles(a)["state_block"][-1], profiles(b)["state_block"][-1]
            r[abl] = dict(final_read_last_state=la, same_layer_last_state=lb,
                          ratio=la / lb if lb else float("nan"), passed=bool(la > lb))
        out["c3"][f"{fin}__vs__{same}"] = r
    p = os.path.join(RES, "a12_depth_lesion_zero_vs_mean.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE", p)

    # console table
    for run, rec in out["models"].items():
        if "missing" in rec:
            print(f"{run}: missing {rec['missing']}")
            continue
        print(f"\n{run}")
        for key, v in rec["profiles"].items():
            print(f"  {key:12s} rho={v['spearman']:+.2f} ratio={v['ratio_sum']:.2f} "
                  f"argmax z/m={v['argmax_zero']}/{v['argmax_mean']}  "
                  f"Z[{' '.join(f'{x:.3f}' for x in v['zero'])}]  "
                  f"M[{' '.join(f'{x:.3f}' for x in v['mean'])}]")
        for c, v in rec["claims"].items():
            print(f"  {c}: zero={v['zero']}\n      mean={v['mean']}")
        if "dead_end_last_state_block_mean" in rec:
            print(f"  dead-end last state (mean mode) = {rec['dead_end_last_state_block_mean']}")
    print("\nC3")
    for k, v in out["c3"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
