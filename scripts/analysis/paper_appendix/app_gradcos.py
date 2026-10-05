"""Appendix C.5 ("Fig 7", app_gradcos): state/prediction gradient cosine on the SHARED
weights, per tensor, shown as a DISTRIBUTION per layer (not a single summary dot).

a16 stores exactly one cos(g_state, g_pred) value per shared TENSOR (one linear weight
matrix, or -- for SPS/AF-SPS -- one gating vector) at the final (20B) checkpoint; it is not
a per-microbatch or per-head sample (n_micro_batches in the JSON is a gradient-accumulation
count for the full-batch estimator, not repeated draws).  So the richest honest unit for a
"distribution per layer" is: pool that layer's weight tensors (q, k, v, o, MLP gate/up/down,
and -- for AF-SPS -- its adaptive gating matrices).  A layer typically contributes 4 points
(shared-attention-only arms) or 7 (arms that also share the MLP); see the printed
"points/layer" line for the exact count per panel.  RMSNorm gain vectors (kind norm_attn /
norm_mlp) are EXCLUDED from this distribution and from the step-3 pooled summary -- they
sit on a wildly different scale (absmean routinely 0.5-1.2 vs ~0.02-0.2 for the weight
matrices; see the a12/mine_results docstrings) and are a separate story.  The embedding /
tied LM head (kind embed_head, one tensor total, block -1) is likewise not part of any
layer's distribution (n=1, nothing to distribute); it is drawn as a single point at "E".

Panel: per layer, one box (IQR + median, whiskers to min/max, no separate fliers since n is
already small) plus every underlying tensor as a jittered dot -- trained in the arm colour,
untrained-init in grey, drawn side by side at each layer.  A dotted line marks cos = 0.

Panels (a)-(e) match the body figure's five shared-weight arms exactly:
  (a) Two-tower 12+12 (shared attn. only) -- attention only, no MLP/embed tensors exist for
      this arm (its MLPs are never shared, so a16 never computes their cross-stream cosine).
  (b) AF-SPS, restricted to the 69 tensors present in BOTH the trained and untrained-init
      JSON (the init JSON additionally reports 24 pre-faithful-split norm-gain tensors that
      have no trained counterpart; see RESTRICT_INTERSECTION).
  (c) SPS (tied head; 2 seeds, pooled).
  (d) Shared Two-tower 12+12.
  (e) Shared Sequential 12+12 (2 seeds, pooled).
The untied-head SPS variant (SPS_UNTIED) and the a17 part-B training-trajectory ladder are
reported in the console output only (not drawn), as before.

Step-3 numbers (printed, for the caption/text): pooling every non-structural, non-norm-gain
attention/MLP matrix, trained vs its architecture's untrained-init control, PAIRED by tensor
name (matrices pooled across seeds where an arm has more than one) --
  * mean cosine with a percentile bootstrap 95% CI (10000 resamples), trained and init;
  * the fraction of matrices with cosine > 0, trained and init;
  * a paired Wilcoxon signed-rank test (trained vs init, one pair per matrix).
Joint-SPS's two structurally-fixed tensors (see `_structural`) are excluded throughout, as
in the original single-dot figure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paper_data as D  # noqa: E402  (puts src/ and scripts/ on sys.path)
from paper_appendix import app_gradcos_stats as GS  # noqa: E402
from plotting import paper_style as PS  # noqa: E402  (applies rcParams)
from plotting.paper_style import ARMS, MANIFEST  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ANALYSIS = "a16_grad_orthogonality"
STEM = "app_gradcos"

# RMSNorm gain vectors: a separate story (see module docstring), excluded from the
# distribution and from the step-3 pooled summary.
NORM_KINDS = {"norm_attn", "norm_mlp"}
EMBED_KIND = "embed_head"

# AF-SPS: the untrained-init a16 JSON reports cos_state_pred for 24 more tensors than the
# trained one (all state_h.*.{attention_norm,mlp_norm} -- norm gains under the pre-faithful
# param split, structurally absent from the trained run's grad accounting).  93 (init) vs
# 69 (trained) non-None tensors; trained's 69 are a strict subset of init's 93.  Restricted
# to that intersection so a block's trained vs init distribution is over the SAME tensors.
RESTRICT_INTERSECTION = {D.AFSPS}

# (arm key, init-control runs, panel title) in panel order; several init runs are averaged
# per layer.  Trained/init lists are index-paired (see module docstring "Step-3 numbers"):
# each init run corresponds to the trained seed at the same position in ARMS[key]["seeds"]
# (minus any seed with no a16 JSON, which simply drops out of both).
DIST_ARMS = MANIFEST["gradcos"]["panels"]
INIT_C = PS.MUTED
A17 = D.RESULTS / "a17"


def _complete(d, untrained):
    if d is None:
        return False
    g = d.get("gate")
    ok_gate = g.get("passed", False) if isinstance(g, dict) else bool(g)
    t = d.get("tensors") or []
    return (ok_gate and bool(d.get("untrained")) == untrained
            and any(r.get("cos_state_pred") is not None for r in t))


def _structural(d, r):
    """Joint-SPS tensors whose cosine is fixed by construction (see AF-SPS/joint note in
    the module docstring): the last block under the consumer split, and the untied-head
    embedding, both a16-`run_joint` exclusions."""
    if d.get("branch") != "joint":
        return False
    cfg = d.get("config") or {}
    n_layer = cfg.get("n_layer")
    if n_layer is not None and int(r["block"]) == int(n_layer) - 1:
        return True
    return r.get("kind") == "embed" and cfg.get("tie_lm_head") is False


def _tensor_names(d):
    return {r["name"] for r in d.get("tensors", []) if r.get("cos_state_pred") is not None}


def _weight_by_block(records, allowed=None):
    """{block: [cos, ...]} pooled across `records`, weight tensors only (excludes norm
    gains, the embedding/head, and structurally-fixed joint-SPS tensors)."""
    out = {}
    for d in records:
        for r in d["tensors"]:
            c = r.get("cos_state_pred")
            if c is None or _structural(d, r):
                continue
            if r["kind"] in NORM_KINDS or r["kind"] == EMBED_KIND:
                continue
            if allowed is not None and r["name"] not in allowed:
                continue
            out.setdefault(int(r["block"]), []).append(float(c))
    return out


def _embed_vals(records, allowed=None):
    out = []
    for d in records:
        for r in d["tensors"]:
            c = r.get("cos_state_pred")
            if c is None or r["kind"] != EMBED_KIND:
                continue
            if allowed is not None and r["name"] not in allowed:
                continue
            out.append(float(c))
    return out


def _weight_dict(d, allowed=None):
    """{tensor_name: cos} for one record's weight tensors (see `_weight_by_block`)."""
    out = {}
    for r in d["tensors"]:
        c = r.get("cos_state_pred")
        if c is None or _structural(d, r):
            continue
        if r["kind"] in NORM_KINDS or r["kind"] == EMBED_KIND:
            continue
        if allowed is not None and r["name"] not in allowed:
            continue
        out[r["name"]] = float(c)
    return out


def _paired_weight_arrays(trained_records, init_records, allowed=None):
    """Trained/init cosines paired by tensor name, index-matching trained_records[k] with
    init_records[k] (see DIST_ARMS docstring), then concatenated across k (seeds pooled).
    A count mismatch is warned and truncated to the shorter list."""
    n = min(len(trained_records), len(init_records))
    if len(trained_records) != len(init_records):
        D.warn(f"{STEM}: {len(trained_records)} trained vs {len(init_records)} init "
                 f"records -- pairing the first {n} of each in list order")
    pt, pi = [], []
    for k in range(n):
        wt, wi = _weight_dict(trained_records[k], allowed), _weight_dict(init_records[k], allowed)
        _, tv, iv = GS.pair_by_tensor_name(wt, wi)
        pt.append(tv)
        pi.append(iv)
    t = np.concatenate(pt) if pt else np.array([])
    i = np.concatenate(pi) if pi else np.array([])
    return t, i


def _summary_stats(label, trained_records, init_records, allowed=None, verbose=True):
    """Step-3 numbers: pooled mean +- bootstrap 95% CI, frac(cos>0), paired Wilcoxon,
    trained vs untrained-init, over attention+MLP matrices (paired by tensor name)."""
    t, i = _paired_weight_arrays(trained_records, init_records, allowed)
    mt, lot, hit = GS.bootstrap_mean_ci(t)
    mi, loi, hii = GS.bootstrap_mean_ci(i)
    ft, fi = GS.frac_positive(t), GS.frac_positive(i)
    n, w, pval = GS.paired_wilcoxon(t, i)
    if verbose:
        print(f"  {STEM} caption ({label}, n={n} paired matrices): mean cos trained "
              f"{mt:.4f} [{lot:.4f}, {hit:.4f}] vs init {mi:.4f} [{loi:.4f}, {hii:.4f}]; "
              f"frac(cos>0) trained {ft:.2f} vs init {fi:.2f}; "
              f"paired Wilcoxon W={w:.1f} p={pval:.2e}")
    return {"n": n, "mean_trained": mt, "ci_trained": (lot, hit), "mean_init": mi,
            "ci_init": (loi, hii), "frac_pos_trained": ft, "frac_pos_init": fi,
            "wilcoxon_p": pval}


def load_arm(key, init_runs, verbose=True):
    """(trained [(run, a16 JSON)], untrained-init [a16 JSON], tensor names both share or
    None): the complete, gate-passing a16 JSONs of an arm's seeds and init controls."""
    tr = []
    for run in ARMS[key]["seeds"]:
        d = D.result(ANALYSIS, run)
        if d is None:
            continue
        if _complete(d, untrained=False):
            tr.append((run, d))
        elif verbose:
            D.warn(f"{STEM}: {run} a16 JSON incomplete / gate failed -- left out")
    init, used = [], []
    for ir in init_runs:
        d = D.result(ANALYSIS, ir)
        if d is None:
            continue
        if not _complete(d, untrained=True):
            if verbose:
                D.warn(f"{STEM}: {ir} incomplete -- init control left out")
            continue
        init.append(d)
        used.append(ir)
    if verbose:
        print(f"  {STEM} {key}: seeds {[r for r, _ in tr]}, init {used or 'none'}"
              if tr else f"  {STEM} {key}: no a16 JSON -- left out")
    allowed = None
    if key in RESTRICT_INTERSECTION and tr:
        allowed = set.intersection(*[_tensor_names(d) for _, d in tr],
                                   *[_tensor_names(d) for d in init])
    return tr, init, allowed


def summaries() -> dict:
    """{arm: _summary_stats} for every panel and printed arm (the numbers of App. grad)"""
    out = {}
    for key, init_runs, *_ in MANIFEST["gradcos"]["panels"] + MANIFEST["gradcos"]["printed"]:
        tr, init, allowed = load_arm(key, init_runs, verbose=False)
        out[key] = _summary_stats(key, [d for _, d in tr], init, allowed, verbose=False)
    return out


def _jitter(n, w, seed):
    return np.random.default_rng(seed).uniform(-w, w, n) if n else np.array([])


def _box(ax, data, positions, color, width, seed):
    """One side (trained OR init) of the per-layer distribution: a narrow box (IQR +
    median, whiskers to min/max) plus every point jittered on top."""
    present = [(p, d) for p, d in zip(positions, data) if d]
    if not present:
        return
    pos, dat = zip(*present)
    ax.boxplot(dat, positions=list(pos), widths=width, showfliers=False, whis=(0, 100),
               patch_artist=True, zorder=2,
               boxprops=dict(facecolor=to_rgba(color, 0.30), edgecolor=color, linewidth=0.6),
               medianprops=dict(color=color, linewidth=1.0),
               whiskerprops=dict(color=color, linewidth=0.6),
               capprops=dict(color=color, linewidth=0.6))
    for k, (p, d) in enumerate(present):
        xj = p + _jitter(len(d), width * 0.32, seed=seed + k)
        ax.plot(xj, d, ls="none", marker="o", ms=1.6, mfc=color, mec="none", alpha=0.75,
                zorder=3)


def _dist_panel(ax, key, trained_records, init_records, allowed, title, has_embed):
    color = ARMS[key]["color"]
    bw = 0.30
    tw = _weight_by_block(trained_records, allowed)
    iw = _weight_by_block(init_records, allowed) if init_records else {}
    blocks = sorted(set(tw) | set(iw))
    layers = [b + 1 for b in blocks]  # 1-indexed
    _box(ax, [tw.get(b, []) for b in blocks], [x - 0.19 for x in layers], color, bw, seed=1)
    _box(ax, [iw.get(b, []) for b in blocks], [x + 0.19 for x in layers], INIT_C, bw, seed=101)
    xticks, xticklabels = list(layers), [str(l) if l % 3 == 0 or l == 1 else "" for l in layers]
    if has_embed:
        te, ie = _embed_vals(trained_records, allowed), _embed_vals(init_records or [], allowed)
        ax.plot(-0.19 + np.zeros(len(te)), te, ls="none", marker="D", ms=2.6, mfc=color,
                mec="white", mew=0.3, zorder=4)
        ax.plot(0.19 + np.zeros(len(ie)), ie, ls="none", marker="D", ms=2.6, mfc=INIT_C,
                mec="white", mew=0.3, zorder=4)
        xticks = [0] + xticks
        xticklabels = ["E"] + xticklabels
    ax.axhline(0.0, color=PS.INK, lw=0.6, ls=":", zorder=1)
    ax.set_title(title, fontsize=7, pad=2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(min(xticks) - 0.6, max(xticks) + 0.6)
    ax.tick_params(axis="both", labelsize=7)
    # console-only: points/layer + frac(cos<0), so the caption can state both honestly
    n_per_layer = sorted({len(tw[b]) for b in blocks}) if blocks else []
    allt = np.concatenate([tw[b] for b in blocks]) if blocks else np.array([])
    alli = np.concatenate([iw[b] for b in blocks if b in iw]) if iw else np.array([])
    print(f"  {STEM} {key}: {len(blocks)} layers, {n_per_layer} points/layer "
          f"(weight matrices only); frac cos<0 trained {np.mean(allt < 0):.2f}" +
          (f" vs init {np.mean(alli < 0):.2f}" if alli.size else ""))


def _ladder(tag):
    pts = []
    for p in sorted(A17.glob(f"a17_trajectory_partB_{tag}_*B.json")):
        b = p.stem.rsplit("_", 1)[-1][:-1]
        if not b.isdigit():
            continue
        d = json.loads(p.read_text())
        if d.get("arm") != tag:
            continue
        last = d.get("last_block_excluded")
        c = np.array([r["cos_state_pred"] for r in d["tensors"]
                      if r["cos_state_pred"] is not None and not r.get("stream_exclusive")
                      and r["block"] != last])
        if len(c):
            pts.append((float(d.get("tokens_B", int(b))), float(np.mean(c < 0)),
                        float(np.percentile(c, 10))))
    return sorted(pts)


RC7 = {"xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.labelsize": 7.5}


def make():
    with plt.rc_context(RC7):
        return _make()


def _make():
    arms = []
    for key, init_runs, title in DIST_ARMS:
        tr, init, allowed = load_arm(key, init_runs)
        if tr:
            arms.append((key, tr, init, allowed, title))
    ladders = [(tag, pts) for tag in MANIFEST["gradcos"]["ladder"] if (pts := _ladder(tag))]
    if not arms:
        print(f"SKIPPED {STEM}: no a16 JSONs")
        return None

    fig, axs = plt.subplots(2, 3, figsize=(PS.FULL_W, 4.0), sharey=True)
    axs = axs.ravel()
    axs[-1].axis("off")
    for pi, (key, tr, init, allowed, title) in enumerate(arms):
        if allowed is not None:
            print(f"  {STEM} {key}: restricted to the trained/init tensor-name "
                  f"intersection ({len(allowed)} tensors)")
        has_embed = any(any(r.get("kind") == EMBED_KIND for r in d["tensors"]) for _, d in tr)
        ax = axs[pi]
        _dist_panel(ax, key, [d for _, d in tr], init, allowed, title, has_embed)
        PS.panel_letter(ax, "abcdef"[pi])
        if pi % 3 == 0:
            ax.set_ylabel(r"$\cos(g_{\mathrm{state}}, g_{\mathrm{pred}})$", fontsize=7.5)
        ax.set_xlabel("layer", fontsize=7.5)
        _summary_stats(title.replace("\n", " "), [d for _, d in tr], init, allowed)

    h = [Line2D([], [], color=PS.INK, lw=0, marker="s", ms=6, mfc=to_rgba(PS.INK, 0.30),
                mec=PS.INK, label="trained")]
    h.append(Line2D([], [], color=PS.INK, lw=0, marker="s", ms=6, mfc=to_rgba(INIT_C, 0.30),
                    mec=INIT_C, label="initialization"))
    h.append(Line2D([], [], ls="none", marker="D", ms=3.2, mfc=PS.INK, mec="white", mew=0.3,
                    label="embedding / head (E)"))
    # the legend sits in the grid's empty sixth cell instead of a row below the panels
    axs[-1].legend(handles=h, loc="center left", frameon=False, fontsize=7)

    for tag, pts in ladders:  # the a17 ladder: caption numbers only
        x, fn, p10 = map(np.array, zip(*pts))
        print(f"  {STEM} (ladder, not drawn) {tag}: B={list(x)} "
              f"frac_neg={np.round(fn, 3).tolist()} p10={np.round(p10, 3).tolist()}")
    drawn = {k for k, *_ in arms}
    for key, init_runs in MANIFEST["gradcos"]["printed"]:
        tr, init, _ = load_arm(key, init_runs, verbose=False)
        if tr and key not in drawn:
            print(f"  {STEM} (not drawn, caption only) {key}: seeds {[r for r, _ in tr]}")
            _summary_stats(key, [d for _, d in tr], init)
    fig.canvas.draw()
    PS.check_inside(fig, [a for a in axs if a.axison], STEM)
    return PS.save(fig, STEM)


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    make()
