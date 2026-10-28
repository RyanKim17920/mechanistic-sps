#!/usr/bin/env python3
"""Data layer of the paper pipeline: every number the figures, tables and numbers.tex use
is read here, nothing is typed in.

  * validation loss   repo_paths.LEDGER (val_nll_full_sweep; the last non-null row per run)
  * analyses          repo_paths.RESULTS / <analysis>_<run>.json
  * params / FLOPs    repo_paths.RESULTS / arch_stats.json, derived from each arm's Hydra
                      config exactly as scripts/bench_wallclock.py builds it (never the
                      ledger's training-time FLOPs).  Regenerate after a config or FLOP
                      accounting change:
                          python scripts/analysis/paper_data.py --refresh-arch-stats

Seeds are discovered: an arm's candidate runs are in paper_manifest.yaml and a run is used
iff its ledger row / result JSON exists (the seed rule, pick_seeds).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "scripts", REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import repo_paths  # noqa: E402
from plotting import paper_style as PS  # noqa: E402
from plotting.paper_style import ARMS, MANIFEST  # noqa: E402

RESULTS = repo_paths.RESULTS
LEDGER = repo_paths.LEDGER
WALLCLOCK = repo_paths.WALLCLOCK
ARCH_STATS = RESULTS / "arch_stats.json"

# the arms the code refers to by role (paper_manifest.yaml `roles`)
ROLES = MANIFEST["roles"]
TRANSFORMER = ROLES["transformer"]        # tied head
SPS = ROLES["sps"]                        # tied head
INTER = ROLES["two_tower12"]
INTER6 = ROLES["two_tower6"]
TIEDATTN = ROLES["tied_attn"]
SEQ = ROLES["sequential12"]
SEQ_TIED = ROLES["shared_sequential"]
W0_SHARED = ROLES["shared_two_tower"]     # shared weights, Two-tower (l-1) read
SEQ6 = ROLES["sequential6"]
AFSPS = ROLES["afsps"]
ASYM = ROLES["asym"]

A2_CAPS = (16, 64, 256, 1024)      # read-reach sweep: keep the N most recent state positions
KS = (2, 4, 6, 8, 10, 12)          # read-depth sweep: caps / floors at level K


def warn(msg):
    print(f"WARNING: {msg}", flush=True)


# ---------------------------------------------------------------------------------------
# ledger and the seed rule
# ---------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def ledger_rows() -> dict:
    """run -> full ledger record (last record wins)."""
    out = {}
    with open(LEDGER) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("run"):
                    out[r["run"]] = r
    return out


@lru_cache(maxsize=None)
def ledger() -> dict:
    """run -> val_nll_full_sweep (last non-null record wins)."""
    out = {}
    with open(LEDGER) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            v = r.get("val_nll_full_sweep")
            if r.get("run") and v is not None:
                out[r["run"]] = float(v)
    return out


def pick_seeds(key: str, have) -> list[str]:
    """SEED RULE: the runs a reported mean uses -- the first PS.N_MEAN_SEEDS runs, in the
    arm's mean-seed order, for which `have(run)` holds (or that are `in have`)."""
    ok = have if callable(have) else (lambda r: r in have)
    return [r for r in PS.mean_seed_order(key) if ok(r)][:PS.N_MEAN_SEEDS]


def mean_runs(key: str) -> list[str]:
    """the runs whose ledger loss enters the arm's reported mean"""
    return pick_seeds(key, ledger())


def nll_seeds(key: str) -> list[float]:
    L = ledger()
    return [L[s] for s in mean_runs(key)]


# ---------------------------------------------------------------------------------------
# analysis result JSONs
# ---------------------------------------------------------------------------------------
def result(analysis: str, run: str):
    p = RESULTS / f"{analysis}_{run}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def result_seeds(analysis: str, key: str, runs=None) -> list[tuple[str, dict]]:
    """[(run, json)] for the seed-rule runs of `key` that have a result (or for every one of
    the explicit `runs` that has one)."""
    if runs is None:
        runs = pick_seeds(key, lambda r: (RESULTS / f"{analysis}_{r}.json").exists())
    out = []
    for r in runs:
        d = result(analysis, r)
        if d is not None:
            out.append((r, d))
    return out


# ---------------------------------------------------------------------------------------
# architecture stats (cached: building the 17 models on CPU takes ~30 s)
# ---------------------------------------------------------------------------------------
TWO_TOWER_FLAGS = ("tie_attn_across_towers", "tie_ffn_across_towers",
                   "share_ffn_across_towers", "tie_norms_across_towers")


def build_arch_stats(run: str) -> dict:
    """FLOPs/token, parameter counts and the config facts the tables report, from the run's
    own Hydra config, by the exact path scripts/bench_wallclock.py uses."""
    import bench_wallclock as B

    with contextlib.redirect_stdout(io.StringIO()):
        model, cfg = B.build(run)
    st = {"flops_fwd_per_token": float(B.flops_per_token(model, cfg)),
          "family": B.family(cfg), **B.param_counts(model)}
    mc = model.config          # the instantiated dataclass: carries every default
    st["tie_lm_head"] = bool(getattr(mc, "tie_lm_head", True))
    if st["family"] != "standard":
        st["predict_embedding"] = str(getattr(mc, "predict_embedding", "constant"))
    if st["family"] == "two_tower":
        st["state_n_layer"] = int(model.state_n_layer)
        st["pred_n_layer"] = int(model.pred_n_layer)
        st["read_map"] = str(mc.read_map)
        st.update({f: bool(getattr(mc, f, False)) for f in TWO_TOWER_FLAGS})
    return st


def refresh_arch_stats():
    stats = {k: build_arch_stats(k) for k in ARMS}
    ARCH_STATS.write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n")
    print(f"WROTE {ARCH_STATS}")


@lru_cache(maxsize=None)
def _arch_stats_all() -> dict:
    return json.loads(ARCH_STATS.read_text())


def arch_stats(key: str) -> dict:
    """Architecture stats of an arm (= those of its primary config; seeds share it)."""
    stats = _arch_stats_all()
    if key not in stats:
        raise KeyError(f"{key} not in {ARCH_STATS}; run paper_data.py --refresh-arch-stats")
    return stats[key]


def train_pflops(key, tokens):
    """Cumulative training compute = tokens x 3 x forward FLOPs/token, in PFLOPs."""
    return np.asarray(tokens, float) * 3.0 * arch_stats(key)["flops_fwd_per_token"] / 1e15


def tower_group(key) -> str:
    """'12+12' / '6+6' (state + prediction layers) for a two-tower model; 'ref' for the
    single-stream Transformer and SPS references."""
    st = arch_stats(key)
    if st["family"] != "two_tower":
        return "ref"
    return f"{st['state_n_layer']}+{st['pred_n_layer']}"


def ladder_curve(run):
    """(tokens, val NLL) of a run's checkpoint ladder (ledger `curve`: every saved
    checkpoint on the full validation sweep; the last point is the final loss), or None."""
    r = ledger_rows().get(run)
    c = (r or {}).get("curve")
    if not c:
        return None
    c = sorted(c, key=lambda p: p["tokens"])
    return np.array([p["tokens"] for p in c], float), np.array([p["val_nll"] for p in c], float)


def frontier_curves(k, missing):
    """(x PFLOPs, seed-mean loss, [(x, y) per seed]) on the checkpoints every seed-rule run
    has, or None (a run without a ladder is left out and reported in `missing`)."""
    curves = []
    for run in mean_runs(k):
        c = ladder_curve(run)
        if c is None:
            missing.append(run)
            continue
        curves.append(c)
    if not curves:
        return None
    common = sorted(set.intersection(*[set(np.round(c[0], -6)) for c in curves]))
    m = np.array([np.mean([c[1][np.round(c[0], -6) == t][0] for c in curves])
                  for t in common])
    return (train_pflops(k, common), m,
            [(train_pflops(k, tk), y) for tk, y in curves] if len(curves) >= 2 else [])


@lru_cache(maxsize=None)
def frontier() -> dict:
    """The body models' checkpoint-ladder curves in training PFLOPs (`cur`), their tower
    group (`grp`), the Transformer's final compute (`tT`) and its 18B pre-decay checkpoint
    compute (`tc`, the matched pre-decay point)."""
    T = PS.BODY_TRANSFORMER
    missing, cur = [], {}
    for k in [k for k in PS.body_arms() if nll_seeds(k)]:
        c = frontier_curves(k, missing)
        if c is None:
            missing.append(k)
            continue
        cur[k] = c
    grp = {k: tower_group(k) for k in cur}
    tT = train_pflops(T, [ladder_curve(mean_runs(T)[0])[0][-1]])[0]
    pre = [c for c in ledger_rows()[mean_runs(T)[0]]["curve_checkpoints"] if "_pre_decay" in c]
    tc = train_pflops(T, [int(re.search(r"ckpt_tokens_(\d+)", pre[0]).group(1))])[0]
    return dict(cur=cur, grp=grp, tT=tT, tc=tc, missing=missing)


def value_at(cur, k, at_x):
    """(seed-mean loss, per-seed losses, read at a checkpoint?) of arm k at compute at_x:
    the checkpoint there, else linear interpolation in compute; at_x=None -> final losses."""
    x, m, seeds = cur[k]
    if at_x is None:
        return float(m[-1]), [float(ys[-1]) for _, ys in seeds], True
    exact = float(np.min(np.abs(x - at_x))) / at_x < 1e-4

    def read(xx, yy):
        return (float(yy[int(np.argmin(np.abs(xx - at_x)))]) if exact
                else float(np.interp(at_x, xx, yy)))
    return read(x, m), [read(xs, ys) for xs, ys in seeds], exact


# ---------------------------------------------------------------------------------------
# joint-SPS results (a12 slot lesions / a20 slot probes): a record names its slot; a file
# without both slots over the full depth is skipped with a warning, never guessed at
# ---------------------------------------------------------------------------------------
_SLOT_ALIASES = {"state": "state", "state_slot": "state", "input": "state", "token": "state",
                 "pred": "pred", "pred_slot": "pred", "predict": "pred", "prediction": "pred"}


def _slot_of(rec) -> str | None:
    for f in ("slot", "stream", "tower"):
        v = rec.get(f)
        if isinstance(v, str) and v.lower() in _SLOT_ALIASES:
            return _SLOT_ALIASES[v.lower()]
    return None


def joint_lesion_curve(d, slot, kind="mlp"):
    """(blocks, deltas) for one SPS slot from an a12 joint JSON, or None."""
    les = d.get("lesions")
    if not isinstance(les, dict):
        return None
    pts = []
    for r in les.values():
        if not isinstance(r, dict) or _slot_of(r) != slot:
            continue
        if str(r.get("kind", r.get("component", ""))).lower() != kind:
            continue
        dv = r.get("delta", r.get("delta_nll"))
        if r.get("block") is None or dv is None:
            continue
        pts.append((int(r["block"]) + 1, float(dv)))
    if not pts:
        return None
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def joint_probe_curve(d, slot, tgt):
    """(relative depth, probe NLL) for one SPS slot from an a20 joint JSON, or None."""
    pr = d.get("probes")
    if not isinstance(pr, dict):
        return None
    pts = []
    for r in pr.values():
        if not isinstance(r, dict) or _slot_of(r) != slot or r.get("block") is None:
            continue
        if int(r["block"]) < 0 or not isinstance(r.get(tgt), dict):
            continue
        v = r[tgt].get("eval_nll")
        if v is None:
            continue
        pts.append((int(r["block"]), float(v)))
    if not pts:
        return None
    pts.sort()
    n = len(pts)
    return [(b + 1) / n for b, _ in pts], [v for _, v in pts]


def _n_layer(d):
    g = d.get("geometry") if isinstance(d.get("geometry"), dict) else {}
    for src in (g, d):
        for f in ("n_layer", "state_n_layer"):
            if isinstance(src.get(f), int) and src[f] > 0:
                return src[f]
    return None


def joint_slot_curves(analysis, key, fn, *args):
    """{slot: [per-seed (x, y)]} for a joint arm.  A seed is used only when BOTH slots
    parse and cover the full depth."""
    out = {"state": [], "pred": []}
    for run, d in result_seeds(analysis, key):
        cs = {slot: fn(d, slot, *args) for slot in out}
        bad = [slot for slot, c in cs.items() if c is None]
        if bad:
            warn(f"{analysis}_{run}.json: no {bad} slot records in a recognised joint layout "
                 f"(family={d.get('family')!r}, branch={d.get('branch')!r}); skipped")
            continue
        L = _n_layer(d)
        short = {slot: len(c[0]) for slot, c in cs.items() if L is not None and len(c[0]) != L}
        if short:
            warn(f"{analysis}_{run}.json: incomplete ({short} of {L} blocks); skipped")
            continue
        if out["state"] and any(cs[s][0] != out[s][0][0] for s in out):
            warn(f"{analysis}_{run}.json: block grid differs from the first seed; skipped")
            continue
        for slot in out:
            out[slot].append(cs[slot])
    return out


def mech_sps_for(analysis):
    """The SPS arm a mechanistic body panel uses: PS.BODY_SPS when its joint-slot JSON is
    complete, else the tied SPS arm (reported)."""
    for k in (PS.BODY_SPS, SPS):
        c = joint_slot_curves(analysis, k, *((joint_probe_curve, "P1")
                                              if analysis == "a20_role_probe"
                                              else (joint_lesion_curve,)))
        if c["state"]:
            if k != PS.BODY_SPS:
                warn(f"{analysis}: no complete JSON for {PS.BODY_SPS}; falling back to {k}")
            return k
    return PS.BODY_SPS


# ---------------------------------------------------------------------------------------
# a2 read reach / a13 read depth
# ---------------------------------------------------------------------------------------
def _a2_curves(analysis, key, sweep, ctrl_of):
    curves = []
    for run, d in result_seeds(analysis, key):
        if "sweeps" not in d:
            continue
        sw = d["sweeps"]
        caps = [c for c in A2_CAPS if f"{sweep}{c}_mask" in sw]
        if caps != list(A2_CAPS):   # file still being written / partial sweep
            warn(f"{analysis}_{run}.json: caps {caps} != {list(A2_CAPS)}; skipped")
            continue
        ctrl = ctrl_of(d)
        curves.append((run, [sw[f"{sweep}{c}_mask"]["val_nll"] - ctrl for c in caps]))
    return curves


def a2_curves(key):
    """[(run, [delta at each A2_CAPS])]: prediction may read only the N most recent state
    positions, farther keys masked (a2_knockout pred_near{N}_mask)."""
    return _a2_curves("a2_knockout", key, "pred_near",
                      lambda d: d["sweeps"]["control_uncapped"]["val_nll"])


def a2_ctx_all_curves(key):
    """The Transformer analogue: EVERY layer's attention restricted to the nearest N tokens
    (a2_ctx_knockout all_near{N}_mask)."""
    return _a2_curves("a2_ctx_knockout", key, "all_near",
                      lambda d: d.get("control_uncapped_nll",
                                      d["sweeps"]["control_uncapped"]["val_nll"]))


def a2_means(key, ab):
    """[[delta at each A2_CAPS] per seed] with far keys masked (ab='mask') or mean-replaced
    (ab='mean')."""
    cs = []
    for _, d in result_seeds("a2_knockout", key):
        sw = d.get("sweeps") or {}
        keys = [f"pred_near{c}_{ab}" for c in A2_CAPS]
        if "control_uncapped" in sw and all(x in sw for x in keys):
            cs.append([sw[x]["val_nll"] - sw["control_uncapped"]["val_nll"] for x in keys])
    return cs


def a13_valid(run, d) -> bool:
    """a13 read-depth caps/floors are only meaningful with read_source=pred_proj.  With
    read_source=state_kv the prediction tower reuses the state tower's own K/V, so a cap
    below the native level substitutes a K/V source that was never trained as a read."""
    rs = (d.get("geometry") or {}).get("read_source")
    if rs == "state_kv":
        warn(f"a13_read_depth_cap_{run}.json: read_source=state_kv -> read-depth sweep "
             "invalid (untrained K/V source); not drawn")
        return False
    return True


def a13_seeds(k):
    """result_seeds for a13, minus runs whose read-depth sweep is invalid (a13_valid)."""
    return [(r, d) for r, d in result_seeds("a13_read_depth_cap", k) if a13_valid(r, d)]


def a13_curve(d, which):
    """K -> delta for caps/floors, dropping K beyond the state depth (clamped duplicates)
    and the native (identity, delta == 0 by construction) setting."""
    maxK = d["geometry"].get("max_available_level", d["geometry"]["state_n_layer"])
    native = d["geometry"].get("native_read_levels") or d["geometry"].get("read_levels")
    ks, ys = [], []
    for k, rec in sorted(d[which].items(), key=lambda kv: int(kv[0])):
        K = int(k)
        if K > maxK or rec["read_levels"] == native:
            continue
        ks.append(K)
        ys.append(rec["delta"])
    return ks, ys


def a13_m_curves(k, which):
    """[(m list, delta list)] per seed against m = the number of state levels prediction may
    read: a cap at K admits the m=K shallowest levels, a floor at K the m=L_s-K+1 deepest.
    Unlike K, m is count-matched between caps and floors."""
    curves = []
    for _, d in a13_seeds(k):
        maxK = d["geometry"].get("max_available_level", d["geometry"]["state_n_layer"])
        ks, ys = a13_curve(d, which)
        curves.append((list(ks) if which == "caps" else [maxK - K + 1 for K in ks], ys))
    if not curves or not curves[0][0] or any(c[0] != curves[0][0] for c in curves):
        return []
    return curves


# ---------------------------------------------------------------------------------------
# a12 whole-layer lesions
# ---------------------------------------------------------------------------------------
LESION_KIND = "block"   # whole-layer lesions (attention + MLP), the numbers the text quotes


def lesion_curve(d, stream, kind=LESION_KIND):
    """(1-indexed layers, loss increase) of one tower of a two-tower / standard a12 JSON."""
    recs = [r for r in d["lesions"].values() if r["stream"] == stream and r["kind"] == kind]
    recs.sort(key=lambda r: r["block"])
    return [r["block"] + 1 for r in recs], [r["delta"] for r in recs]


# ---------------------------------------------------------------------------------------
# a20 role probes
# ---------------------------------------------------------------------------------------
def probe_curve(d, tower, tgt):
    """(1-indexed block, probe NLL) for one tower of a two-tower / standard a20 JSON."""
    recs = [r for r in d["probes"].values() if r["tower"] == tower and r["block"] >= 0]
    recs.sort(key=lambda r: r["block"])
    return [r["block"] + 1 for r in recs], [r[tgt]["eval_nll"] for r in recs]


def init_files(run):
    """the untrained-init role-probe files of one run"""
    return sorted(RESULTS.glob(f"a20_role_probe_{run}_init*.json"))


def init_runs(k):
    """seed-rule runs of arm k with an untrained-init role-probe file"""
    return pick_seeds(k, lambda r: bool(init_files(r)))


BIGRAM_FLOOR = {"P1": "bigram_next_given_cur", "P3": "bigram_prev_given_cur"}


def probe_floors(tg, keys):
    """(untrained-init floor, bigram floor or None) for probe target `tg`: the lowest probe
    loss any untrained-init site reaches over the `keys` arms, and the mean JM-bigram floor
    recorded in those files (next | current for P1, previous | current for P3)."""
    init, bigram = [], []
    for k in keys:
        for run in init_runs(k):
            for p in init_files(run):
                d = json.loads(p.read_text())
                init += [r[tg]["eval_nll"] for n, r in d["probes"].items()
                         if n != "final" and isinstance(r.get(tg), dict)]
                fl = d.get("floors") or {}
                b = BIGRAM_FLOOR.get(tg)
                if b and b in fl:
                    bigram.append(float(fl[b]))
    return (min(init) if init else None), (float(np.mean(bigram)) if bigram else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh-arch-stats", action="store_true",
                    help=f"rebuild {ARCH_STATS.name} from the Hydra configs (CPU, ~30 s)")
    a = ap.parse_args()
    if a.refresh_arch_stats:
        refresh_arch_stats()


if __name__ == "__main__":
    main()
