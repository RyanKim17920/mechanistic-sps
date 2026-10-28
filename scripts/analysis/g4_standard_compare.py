"""G4 + read-reach baseline: compare the single-stream Transformer to the separated models.

Reads committed JSONs only:
  a12_depth_lesion_<run>[_mean].json   per-layer lesion profiles (zero / mean ablation)
  a2_ctx_knockout_<transformer>.json   Transformer context-window knockout (keep-W)
  a2_knockout_<separated>.json         separated models' prediction-read keep-W

Depth-profile statistics, per (model, stream, kind, ablation), on the per-layer deltas d:
  layer1          d[0]
  layer1_over_next_max   d[0] / max(d[1:])        (>1: layer 1 is the costliest)
  layer1_share    d[0] / sum(d)
  first_third_share  sum(d[:L/3]) / sum(d)
  last_half_share    sum(d[L/2:]) / sum(d)
  n_ge_0p1        layers costing >= 0.1
  n_lt_0p05       layers costing < 0.05 (among layers 2..L)

-> results/g4_standard_compare.json
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))
import repo_paths  # noqa: E402
import yaml  # noqa: E402

RES = str(repo_paths.RESULTS)

with open(os.path.join(HERE, "paper_manifest.yaml")) as _f:
    _RUNS = yaml.safe_load(_f)["standard_compare"]
TRANSFORMERS = _RUNS["transformers"]
SEPARATED = _RUNS["separated"]
KINDS = ("mlp", "attn", "block")
CAPS = (16, 64, 256, 1024)


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def r4(x):
    return None if x is None else round(float(x), 4)


def profile(d, stream, kind):
    L = 0
    while f"{stream}_{L:02d}_{kind}" in d["lesions"]:
        L += 1
    return [d["lesions"][f"{stream}_{i:02d}_{kind}"]["delta"] for i in range(L)]


def stats(ds):
    L = len(ds)
    tot = sum(ds)
    return dict(
        L=L, deltas=[r4(x) for x in ds], layer1=r4(ds[0]),
        layer1_over_next_max=r4(ds[0] / max(ds[1:])) if max(ds[1:]) > 0 else None,
        layer1_share=r4(ds[0] / tot), first_third_share=r4(sum(ds[:L // 3]) / tot),
        last_half_share=r4(sum(ds[L // 2:]) / tot), total=r4(tot),
        n_ge_0p1=sum(1 for x in ds if x >= 0.1),
        n_lt_0p05_after_layer1=sum(1 for x in ds[1:] if x < 0.05),
        argmax_layer=1 + max(range(L), key=lambda i: ds[i]))


def lesion_block():
    out = {}
    for ab, suf in (("zero", ""), ("mean", "_mean")):
        for label, run in TRANSFORMERS.items():
            d = load(f"a12_depth_lesion_{run}{suf}.json")
            if d is None or "summary" not in d:
                continue
            assert d.get("control_gate_passed"), run
            out.setdefault(ab, {})[label] = {
                "single": {k: stats(profile(d, "single", k)) for k in KINDS}}
        for label, runs in SEPARATED.items():
            per_seed = []
            for run in runs:
                d = load(f"a12_depth_lesion_{run}{suf}.json")
                if d is None or "summary" not in d:
                    continue
                per_seed.append({s: {k: stats(profile(d, s, k)) for k in KINDS}
                                 for s in ("state", "pred")})
            if per_seed:
                out.setdefault(ab, {})[label] = dict(
                    runs=runs, seeds=per_seed)
    return out


def knockout_block():
    out = {"transformer": {}, "separated": {}}
    for label, run in TRANSFORMERS.items():
        d = load(f"a2_ctx_knockout_{run}.json")
        if d is None:
            continue
        assert d["control_gate_passed"], run
        rows = {tag: r4(v["delta_nll"]) for tag, v in d["sweeps"].items()
                if "delta_nll" in v}
        out["transformer"][label] = dict(run=run, native=d["native_sub_sweep_nll"],
                                         control_delta=d["control_delta_vs_native"],
                                         last_n=d["last_n"], deltas=rows)
    for label, runs in SEPARATED.items():
        seeds = []
        for run in runs:
            d = load(f"a2_knockout_{run}.json")
            if d is None or "sweeps" not in d:
                continue
            seeds.append({f"keep{c}_{ab}": r4(d["sweeps"][f"pred_near{c}_{ab}"]["delta_nll"])
                          for c in CAPS for ab in ("mask", "mean")})
        out["separated"][label] = dict(runs=runs, seeds=seeds)
    return out


def main():
    res = dict(analysis="g4_standard_compare", lesions=lesion_block(),
               knockout=knockout_block())
    path = os.path.join(RES, "g4_standard_compare.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"WROTE {path}")
    for ab, blk in res["lesions"].items():
        print(f"\n== lesions ({ab}); cols: layer1, layer1/next-max, layer1 share, "
              f"first-third share, last-half share, n>=0.1, n<0.05 after L1, argmax")
        for label, v in blk.items():
            items = ([("single", v["single"])] if "single" in v else
                     [(f"{s}[s{i+1}]", seed[s]) for i, seed in enumerate(v["seeds"])
                      for s in ("state", "pred")])
            for name, st in items:
                for k in ("mlp", "block"):
                    x = st[k]
                    print(f"  {label:22s} {name:10s} {k:5s} {x['layer1']:7.3f} "
                          f"{x['layer1_over_next_max'] or 0:6.2f} {x['layer1_share']:6.3f} "
                          f"{x['first_third_share']:6.3f} {x['last_half_share']:6.3f} "
                          f"{x['n_ge_0p1']:3d} {x['n_lt_0p05_after_layer1']:3d} "
                          f"L{x['argmax_layer']}")
    print("\n== knockout keep-W, dNLL (mask / mean)")
    for label, v in res["knockout"]["transformer"].items():
        dd = v["deltas"]
        for scope in ["all"] + [f"last{n}" for n in v["last_n"]]:
            row = "  ".join(f"W{c}: {dd.get(f'{scope}_near{c}_mask', float('nan')):.4f}/"
                            f"{dd.get(f'{scope}_near{c}_mean', float('nan')):.4f}"
                            for c in CAPS)
            print(f"  {label:22s} {scope:6s} {row}")
    for label, v in res["knockout"]["separated"].items():
        for i, s in enumerate(v["seeds"]):
            row = "  ".join(f"W{c}: {s[f'keep{c}_mask']:.4f}/{s[f'keep{c}_mean']:.4f}"
                            for c in CAPS)
            print(f"  {label:22s} s{i+1}     {row}")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
