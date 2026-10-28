#!/usr/bin/env python3
"""The figures of the paper, in one style (src/plotting/paper_style.py).

  fig_teaser          Figure 1 schematic (fig_arch_paper.py; no data)
  fig_state_access    (sec. 3)   read window sweep | previous-token probe per state layer |
                                 next-token probe per prediction layer
  fig_read_depth      (sec. 4)   read depth (last / first m levels) | whole-layer lesions
                                 Two-tower | SPS | Sequential (one shared log y axis)
                      (both drawn at FULL_W from the panel_* functions below)
  fig_frontier        (a)-(c) val loss vs cumulative training compute, zoomed at the
                      matched pre-decay, Transformer-cost and SPS-cost points; (d) the full
                      training curves of every body model
  app_ctrl_dist       appendix: head-ablation cost vs 100 control draws (paper_appendix/)
  app_gradcos         appendix: state/prediction gradient cosines (paper_appendix/)
  app_depth_split     appendix: per-layer lesions of the 3+9 / 6+6 / 9+3 Sequential splits
                      (paper_appendix/)
  fig_gain_diag       not in the paper: next-token probe | lesions Sequential, the earlier
                      layout of panels now in fig_state_access (c) and fig_read_depth (d)
  fig_frontier_full   not in the paper: fig_frontier (d) on its own

  -> <outdir>/<stem>.pdf   (default paper/figures, plotting.paper_style.OUT_DIR)

Every number comes from paper_data.py (ledger, result JSONs, arch_stats.json); the arms
come from paper_manifest.yaml.

Usage:  python scripts/analysis/paper_figures.py [--only fig_teaser,...] [--outdir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper_data as D  # noqa: E402  (puts src/ and scripts/ on sys.path)
from paper_data import INTER, SEQ, SEQ6, SPS  # noqa: E402
from plotting import paper_style as PS  # noqa: E402  (applies rcParams)
from plotting.paper_style import ARMS  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import blended_transform_factory, offset_copy  # noqa: E402


LOSS_LABEL = "Validation loss (nats/token)"


def assert_inside(ax, xs=(), ys=(), what=""):
    """The axis limits are chosen for today's data: fail loudly if new data falls outside."""
    for vals, (lo, hi), axis in ((xs, ax.get_xlim(), "x"), (ys, ax.get_ylim(), "y")):
        v = np.asarray([x for x in np.ravel(vals) if np.isfinite(x)], float)
        assert not v.size or (lo <= v.min() and v.max() <= hi), \
            f"{what}: {axis} data [{v.min():.4g}, {v.max():.4g}] outside limits [{lo:.4g}, {hi:.4g}]"


# =======================================================================================
# shared small helpers
# =======================================================================================
def stream_handles(color=PS.INK, markers=None):
    """`markers` (optional {"state": m, "pred": m}) also draws the stream's marker shape
    on the legend line, for figures (e.g. fig_read_depth) that use shape, not just
    line style, to tell state and prediction apart."""
    markers = markers or {}
    return [Line2D([], [], color=color, ls=PS.STREAM_LS["state"], lw=PS.MEAN_LW,
                   marker=markers.get("state", "none"), markersize=4, mfc=color, mec=color,
                   label=PS.STREAM_LABEL["state"]),
            Line2D([], [], color=color, ls=PS.STREAM_LS["pred"], lw=PS.MEAN_LW,
                   marker=markers.get("pred", "none"), markersize=4, mfc=color, mec=color,
                   label=PS.STREAM_LABEL["pred"])]


def seed_handles():
    """ONE compact entry for the seed rule (shaded band = min-max over the seeds, line/marker
    = mean; a single-seed arm has no band). The bands themselves are drawn in each arm's own
    colour, not grey, so the swatch is hatched with a thin outline -- a generic placeholder
    for "shaded band", not a claim that the band is this colour."""
    return [Patch(facecolor=PS.MUTED, edgecolor=PS.INK, linewidth=0.4, hatch="////",
                  alpha=PS.SEED_BAND_ALPHA + 0.15,
                  label=f"band: seed range ({PS.N_MEAN_SEEDS} seeds, model colour)")]


def compact_legend(fig, handles, ncol=None, loc="outside lower center"):
    """One compact shared legend (<= ~6 entries) for a body figure."""
    return fig.legend(handles=handles, loc=loc, ncol=ncol or len(handles),
                      columnspacing=1.0, handletextpad=0.4, handlelength=1.6)


# =======================================================================================
# read-access / read-depth panels
# =======================================================================================
READ_DEEP_LS, READ_SHALLOW_LS = "-", "--"   # panel_read_depth: deepest / shallowest only
# panel_read_depth: the two curves also differ in hue and marker (not only line style), so
# "last m levels" (m deepest) vs "first m levels" (m shallowest) is unmistakable;
# vermillion is used by no other model in fig_read_depth.
READ_DEEP_STYLE = dict(color=PS.PALETTE["two_tower"], marker="o")
READ_SHALLOW_STYLE = dict(color=PS.PALETTE["afsps"], marker="D")


def panel_read_window(ax, title="(a) Recent positions only"):
    """How far back prediction reads (a2 window sweep, far keys masked): loss increase vs
    the N most recent readable state positions, for SPS, Two-tower 12+12 and Sequential
    12+12 (pred_near{N}_mask), plus the untied Transformer with ALL layers restricted to the
    nearest N tokens (a2_ctx_knockout), drawn in black.  Seeds as a shaded min-max band +
    mean, log y.  Returns ({key: n seeds}, every plotted y).  (The y label is left to the
    caller.)"""
    keys_a = [PS.BODY_SPS, INTER, SEQ]
    n_of, drawn = {}, []
    x = np.log2(D.A2_CAPS)
    for k in keys_a:
        curves = D.a2_curves(k)
        if curves:
            PS.plot_seeded_line(ax, x, [c for _, c in curves], k, ms=3.4)
            n_of[k] = len(curves)
            drawn += [c for _, c in curves]
    # untied Transformer, all layers restricted to the nearest N tokens (a2_ctx_knockout);
    # drawn in black to set it apart from the two-stream arms' hues
    t_key = PS.BODY_TRANSFORMER
    t_curves = D.a2_ctx_all_curves(t_key)
    if t_curves:
        PS.plot_seeded_line(ax, x, [c for _, c in t_curves], t_key, ms=3.4, color="black")
        n_of[t_key] = len(t_curves)
        drawn += [c for _, c in t_curves]
    ax.set_xticks(x, [str(c) for c in D.A2_CAPS])
    ax.set_xlabel("readable state positions $N$")
    ax.set_title(title, pad=3)
    ax.set_yscale("log")
    return n_of, drawn


def panel_read_depth(ax, title="(b) Deepest vs. shallowest levels"):
    """Count-matched read-depth test, Two-tower 12+12 (a13): cost of restricting the read
    to the m DEEPEST state levels only (floors, m = L_s-K+1, solid) vs the m SHALLOWEST
    levels only (caps, m = K, dashed), on a SHARED m axis.  Seeds as a shaded min-max band
    (hairline edges: the shallow-only range is <= ~0.10 nats, narrower than the line) +
    mean.  The two curves are labelled directly: "last m levels" (deepest only) and
    "first m levels" (shallowest only).  The uncapped model (m = L_s, cost exactly
    0) is not drawn.  Returns (number of seeds, every plotted y).  (The y label is left to
    the caller.)"""
    k = INTER
    n, drawn = 0, []
    ends = {}
    for which, ls, sty in (("floors", READ_DEEP_LS, READ_DEEP_STYLE),
                           ("caps", READ_SHALLOW_LS, READ_SHALLOW_STYLE)):
        curves = D.a13_m_curves(k, which)
        if curves:
            m = PS.plot_seeded_line(ax, curves[0][0], [c[1] for c in curves], k, ls=ls,
                                    ms=3.0, lw=1.1, band_edge_lw=0.35, **sty)
            st = np.vstack([c[1] for c in curves])
            drawn.append(st)
            print(f"[fig_read_depth] (a) {which} seed range per m "
                  f"{curves[0][0]}: {np.round(st.max(0) - st.min(0), 4).tolist()}")
            n = max(n, len(curves))
            ends[which] = (curves[0][0][0], float(m[0]))
    ax.set_title(title, pad=3)
    ax.set_xlabel("readable state levels $m$")
    ax.set_xticks(range(2, 12, 2))
    ax.set_xlim(0.4, 11.6)
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 1.6)
    if "caps" in ends:      # empty upper-right corner, above the falling shallow curve
        ax.text(0.97, 0.97, "first $m$ levels", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color=READ_SHALLOW_STYLE["color"],
                linespacing=1.0)
    if "floors" in ends:    # in the empty lower-left corner, under the deep curve
        ax.text(0.03, 0.03, "last $m$ levels", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7, color=READ_DEEP_STYLE["color"],
                linespacing=1.0)
    return n, drawn


# =======================================================================================
# role-probe panels
# =======================================================================================
def _floor_line(ax, y, ls, text=None, fontsize=7):
    ax.axhline(y, color=PS.MUTED, lw=0.7, ls=ls, zorder=1, alpha=0.8)
    if text:
        ax.annotate(text, xy=(1, y), xycoords=("axes fraction", "data"), xytext=(-1, 1.0),
                    textcoords="offset points", ha="right", va="bottom", fontsize=fontsize,
                    color=PS.MUTED)


def _role_curves(k, sps_k, tg):
    """{stream: [(x, y) per seed]} for arm k and probe target tg ('single' for the
    Transformer, 'state'/'pred' otherwise)."""
    if k == sps_k:
        sc = D.joint_slot_curves("a20_role_probe", sps_k, D.joint_probe_curve, tg)
        out = {}
        for slot in ("state", "pred"):
            if sc[slot]:
                xb = [round(v * len(sc[slot][0][0])) for v in sc[slot][0][0]]
                out[slot] = [(xb, c[1]) for c in sc[slot]]
        return out
    rs = D.result_seeds("a20_role_probe", k)
    towers = ("single",) if k == PS.BODY_TRANSFORMER else ("state", "pred")
    return {t: [D.probe_curve(d, t, tg) for _, d in rs] for t in towers if rs}


def _init_curve(k, sps_k, tg, stream):
    """Mean untrained-init probe curve (layers, NLL) of arm k's `stream` over every
    a20_role_probe_<seed run>_init*.json of the arm, or None."""
    curves = []
    for run in D.init_runs(k):
        for pth in D.init_files(run):
            d = json.loads(pth.read_text())
            if k == sps_k:
                c = D.joint_probe_curve(d, stream, tg)
                if c is None:
                    continue
                c = ([round(v * len(c[0])) for v in c[0]], c[1])
            else:
                c = D.probe_curve(d, stream, tg)
            if c[0]:
                curves.append((pth.name, c))
    if not curves or any(c[0] != curves[0][1][0] for _, c in curves):
        return None
    print(f"[roleprobe] {tg} untrained {k}/{stream}: {[n for n, _ in curves]}")
    return curves[0][1][0], np.mean([c[1] for _, c in curves], 0)


TRANSFORMER_LINE_COLOR = "#000000"   # distinct from ARMS[T]'s grey
UNTRAINED_LS = (0, (3, 1.6))
# probe target -> (stream drawn in colour, default title, y limits).  y NOT shared between
# the two targets: P1's curves span only ~3.4-6.1 nats (plus the ~7-7.4 untrained line), so
# its own tighter range magnifies the <= ~0.11-nat next-token seed bands ~1.6x, enough to
# see them; P3 keeps the full range its ~1.7-6 nat curves need
ROLE_PANELS = {"P1": ("pred", "(a) next token, prediction or single stream", (3.2, 7.7)),
               "P3": ("state", "(b) previous token, state or single stream", (0.9, 7.9))}


def _role_color(k):
    return TRANSFORMER_LINE_COLOR if k == PS.BODY_TRANSFORMER else None  # None -> arm colour


def role_models():
    """(SPS arm, models) of the role-probe panels: Transformer, SPS, Two-tower 12+12,
    Sequential 12+12 at 12 layers, untied head."""
    sps_k = D.mech_sps_for("a20_role_probe")
    return sps_k, (PS.BODY_TRANSFORMER, sps_k, INTER, SEQ)


def role_handles(models, nseeds):
    """One legend line per role-probe model drawn (Transformer in black)."""
    return [Line2D([], [], ls="-", label=PS.body_label(k), **PS.marker_kw(k, color=_role_color(k)))
            for k in models if k in nseeds]


def panel_roleprobe(ax, tg, sps_k, models, title=None):
    """One role-probe panel for probe target `tg` (a20): per-layer probe loss of the stream
    that carries the role (P1 next token -> prediction streams, P3 previous token -> state
    streams; the Transformer's single stream in both, in BLACK) for `models`, seeds as a
    shaded min-max band + mean (P1: thinner mean line and outlined band, its ranges are
    narrow); one shared grey dashed UNTRAINED-init line (mean over models and init files);
    dotted bigram floor.  Sets title / x axis / y limits (asserted to contain every plotted
    y), not the y label.
    Returns ({model: n seeds}, (n untrained models, cross-model spread) or None)."""
    hot, default_title, ylim = ROLE_PANELS[tg]
    nseeds, untrained, drawn = {}, None, []
    # only the stream that carries the role is drawn (per-model "other stream" lines were
    # illegible at print size and carried no per-model claim)
    ic_curves = []
    for k in models:
        cs = _role_curves(k, sps_k, tg)
        for stream, curves in cs.items():
            if not curves:
                continue
            nseeds[k] = len(curves)
            x = curves[0][0]
            ys = [c[1] for c in curves]
            if stream in (hot, "single"):
                # next-token seed ranges are <= ~0.11 nats, about the width of a full mean
                # line on this full 7-nat axis: a thinner mean line plus a light band with
                # hairline min/max edges keeps them visible.  P3's ranges (up to ~0.8 nats)
                # are already visible with the default seed band.
                thin = dict(lw=1.0, band_alpha=0.30, band_edge_lw=0.35) if tg == "P1" else {}
                PS.plot_seeded_line(ax, x, ys, k, ls="-", ms=3.0, zorder=4,
                                    color=_role_color(k), **thin)
                drawn.append(ys)
                st_ = np.vstack(ys)
                print(f"[roleprobe] {tg} {PS.body_label(k)}/{stream} max seed range "
                      f"{float(np.max(st_.max(0) - st_.min(0))):.3f} nats")
                ic = _init_curve(k, sps_k, tg, stream)
                if ic is not None:
                    ic_curves.append(ic)
                else:
                    D.warn(f"roleprobe: no untrained curve for {k}/{stream}")
    # untrained-init curves are near-identical across models (max cross-model spread
    # <=0.11 nats for P3/state; the P1/pred spread is larger, up to ~0.5-0.7 nats at layer 1,
    # driven by the Transformer's differently-scaled init, but all four sit in a tight
    # ~7-7.6 nat band far above every trained curve and the bigram floor, and the text
    # reports them as one number) -- one shared grey line replaces the 4 per-model ones.
    if ic_curves and all(c[0] == ic_curves[0][0] for c in ic_curves):
        xb = ic_curves[0][0]
        m = np.mean([c[1] for c in ic_curves], 0)
        ax.plot(xb, m, color=PS.MUTED, lw=0.9, ls=UNTRAINED_LS, alpha=0.75, zorder=2)
        drawn.append(m)
        # labelled directly (like the bigram line), just under its right end
        ax.annotate("untrained", xy=(xb[-1], m[-1]), xytext=(0, -2.5),
                    textcoords="offset points", ha="right", va="top", fontsize=7,
                    color=PS.MUTED)
        stack = np.vstack([c[1] for c in ic_curves])
        spread = float(np.max(stack.max(0) - stack.min(0)))   # worst per-layer cross-model gap
        untrained = (len(ic_curves), spread)
    _, bigram = D.probe_floors(tg, [PS.BODY_TRANSFORMER, sps_k, INTER, SEQ])
    if bigram is not None:
        _floor_line(ax, bigram, ":", "bigram")
        drawn.append(bigram)
    print(f"[roleprobe] {tg}: bigram floor {bigram}")
    ax.set_title(default_title if title is None else title, pad=3)
    ax.set_xlabel("layer")
    ax.set_xticks(range(1, 13), [str(b) if b % 2 else "" for b in range(1, 13)])
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(*ylim)
    assert_inside(ax, ys=np.concatenate([np.ravel(v) for v in drawn]), what=f"roleprobe {tg}")
    return nseeds, untrained


# =======================================================================================
# whole-layer lesion panels
# =======================================================================================
LESION_KIND = D.LESION_KIND

# Stream shape, on top of line style: every arm's marker (square/diamond/circle) already
# encodes ARCHITECTURE elsewhere, but within one lesion panel both streams share that same
# shape/color, so a state and a prediction point can be hard to tell apart where their
# curves cross or run close together (worst for SPS, whose "tied" fill is already the same
# hollow circle used for the exact-zero marker).  Overriding the shape by stream -- circle
# for state, triangle for prediction -- disambiguates within a panel without disturbing
# the arm-identity shape used by every other figure.
STREAM_MARKER = {"state": "o", "pred": "^"}
ZERO_MARKER = "x"   # deliberately NOT a stream shape, so it never reads as a data point


def _plot_zero_marker(ax, xs, color):
    """Draw an exact-zero cost as an explicit OFF-AXIS marker: an x sitting just below the
    axis frame (blended data-x / axes-fraction-y transform, nudged a few points under the
    spine), clip_on=False so it is not cut off.  On a log axis a literal 0 cannot be
    plotted in range, and drawing it at the visible floor reads as a (very small) real
    value; the legend entry for this marker states what it means."""
    trans = offset_copy(blended_transform_factory(ax.transData, ax.transAxes),
                        fig=ax.figure, x=0, y=-3.2, units="points")
    ax.plot(xs, [0.0] * len(xs), ls="none", marker=ZERO_MARKER, ms=4.2, mfc="none",
            mec=color, mew=1.1, zorder=5, clip_on=False, transform=trans)


def _draw_lesions(ax, k, per_stream):
    """per_stream: {stream: [(blocks, deltas) per seed]}.  Exact zeros (a block whose
    output nothing reads) cannot sit on a log axis: they are cut out of the curve and
    drawn as a hollow marker OFF the axis, just below the frame (see _plot_zero_marker),
    so it is never mistaken for a real small value.  Returns the zero blocks per stream."""
    zeros = {}
    for stream, curves in per_stream.items():
        if not curves:
            continue
        x = curves[0][0]
        ys = [np.where(np.asarray(c[1], float) > 0, c[1], np.nan) for c in curves]
        # seed spread = the shaded min-max band only (its right edge still spans the
        # layer-12 range, e.g. Sequential 12+12's state 0.559-2.255 nats)
        PS.plot_seeded_line(ax, x, ys, k, ls=PS.STREAM_LS[stream], ms=2.6,
                            marker=STREAM_MARKER[stream])
        st = np.vstack(ys)
        print(f"[lesion] {k} {stream} per-seed block deltas at layer {x[-1]}: "
              f"{np.round(st[:, -1], 4).tolist()}")
        z = [b for j, b in enumerate(x) if all(c[1][j] <= 0 for c in curves)]
        if z:
            zeros[stream] = z
            _plot_zero_marker(ax, z, ARMS[k]["color"])
    return zeros


def lesion_sps():
    """(SPS arm, its joint-slot a12 curves, whether both slots are complete)."""
    sps_k = D.mech_sps_for("a12_depth_lesion")
    sps = D.joint_slot_curves("a12_depth_lesion", sps_k, D.joint_lesion_curve, LESION_KIND)
    return sps_k, sps, bool(sps["state"]) and bool(sps["pred"])


def _lesion_per_stream(k, joint=None):
    """{stream: [(blocks, deltas) per seed]}: the SPS slot curves `joint`, else k's tower
    lesions from its a12 JSONs."""
    if joint is not None:
        return joint
    rs = D.result_seeds("a12_depth_lesion", k)
    return {s: [D.lesion_curve(d, s) for _, d in rs] for s in ("state", "pred")}


def panel_lesion(ax, k, joint=None, title=None, letter=None):
    """Per-layer WHOLE-LAYER (a12 kind "block") zero-ablation cost of arm `k`: state (solid,
    circles) and prediction (dashed, triangles) per layer, log y; seeds as a shaded min-max
    band + mean; an exact zero as an off-axis x.  `joint` = the SPS slot curves
    (lesion_sps) for the one shared stack, else tower lesions are read from a12 JSONs.
    Title defaults to the model name; `letter` adds a bold panel letter.  Sets title /
    x axis / log y, not the y label or limits.  Returns the exact-zero layers per stream
    ({} if none)."""
    per = _lesion_per_stream(k, joint)
    if joint is not None:
        for run, d in D.result_seeds("a12_depth_lesion", k):
            g = d.get("gates", {}).get("dead_end_last_state_block", {})
            if g and not g.get("passed"):
                D.warn(f"a12_depth_lesion_{run}.json: dead-end gate failed")
    for s_ in per:
        print(f"[lesion] {k} {s_} block deltas (mean):",
              np.round(np.mean([c[1] for c in per[s_]], 0), 3).tolist())
    z = _draw_lesions(ax, k, per)
    if z:
        print(f"[lesion] exact-zero blocks for {k}: {z}")
    name = "SPS" if k in (SPS, PS.BODY_SPS) else PS.body_label(k)
    ax.set_title(name if title is None else title, pad=3)
    ax.set_yscale("log")
    ax.set_xlabel("layer")
    ax.set_xticks(range(1, 13), [str(b) if b % 2 else "" for b in range(1, 13)])
    ax.set_xlim(0.4, 12.6)
    if letter:
        PS.panel_letter(ax, letter)
    return z


def assert_lesion_inside(ax, k, joint=None):
    """every positive lesion cost of arm k lies within ax's (final) y limits"""
    costs = np.concatenate([c[1] for cs in _lesion_per_stream(k, joint).values() for c in cs])
    assert_inside(ax, ys=costs[costs > 0], what=f"lesion {k}")


def zero_handle(label="exactly 0 (output unread)"):
    return Line2D([], [], ls="none", marker=ZERO_MARKER, ms=4.2, mfc="none", mec=PS.MUTED,
                  mew=1.1, label=label)


# =======================================================================================
# argument-ordered composites (re-layouts of the panels above; no new data)
#   fig_state_access  (a) panel_read_window  (b) panel_roleprobe P3  (c) panel_roleprobe P1
#   fig_read_depth    (a) panel_read_depth   (b) panel_lesion Two-tower  (c) SPS
#                     (d) Sequential         -- (b)-(d) share one log y axis
#   fig_gain_diag     (a) panel_roleprobe P1  (b) panel_lesion Sequential (not in the paper)
# Authored at FULL_W: include at \linewidth (1:1, fonts stay at 7pt).
# =======================================================================================
def _seed_band_label(ns):
    """Seed-band legend label stating the seed count actually drawn."""
    ns = set(ns)
    return f"band: range over {ns.pop()} seeds" if len(ns) == 1 else "band: seed range"


def _stream_labels(ax, k, where=(("state", 12, -5, 4, "right", "bottom"),
                                 ("pred", 8, 0, 5, "center", "bottom"))):
    """Label the state / prediction lesion curves of arm `k` directly (instead of legend
    entries): per stream, text at the seed-mean point of layer x, offset (dx, dy) points.
    Defaults suit Sequential 12+12: "state" just above-left of its layer-12 end, and
    "prediction" above its layer-8 point, where the state curve runs well below it."""
    rs = D.result_seeds("a12_depth_lesion", k)
    col = ARMS[k]["color"]
    for stream, x, dx, dy, ha, va in where:
        ys = [dict(zip(*D.lesion_curve(d, stream))).get(x) for _, d in rs]
        ys = [y for y in ys if y is not None and y > 0]
        if ys:
            ax.annotate(PS.STREAM_LABEL[stream], xy=(x, float(np.mean(ys))),
                        xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                        fontsize=7, color=col)


def fig_state_access():
    """What the prediction can reach in the state (sec. 3):
      (a) panel_read_window  loss increase vs N most recent readable state positions
      (b) panel_roleprobe P3 previous-token probe loss per state layer (Transformer: its
          single stream); untrained-init and bigram references labelled in the panel.
      (c) panel_roleprobe P1 next-token probe loss per prediction layer (Transformer: its
          single stream), same references, beside (b) so the two probes compare directly.
    Same four models, colours and markers in all panels (Transformer black); one legend."""
    fig, (aa, ab, ac) = plt.subplots(1, 3, figsize=(PS.FULL_W, 2.3))
    n_a, drawn_a = panel_read_window(aa, title="(a) Read window")
    aa.set_ylabel("Loss increase (nats)")
    sps_k, models = role_models()
    if sps_k != PS.BODY_SPS:
        D.warn(f"fig_state_access: (b) uses {sps_k}, (a) uses {PS.BODY_SPS}")
    n_b, ur = panel_roleprobe(ab, "P3", sps_k, models, title="(b) Previous-token probe")
    ab.set_ylabel("Probe loss (nats)")
    n_c, uc = panel_roleprobe(ac, "P1", sps_k, models, title="(c) Next-token probe")
    ac.set_ylabel("Probe loss (nats)")
    print(f"[fig_state_access] seeds (a): { {PS.body_label(k): n for k, n in n_a.items()} } "
          f"(b): { {PS.body_label(k): n for k, n in n_b.items()} } "
          f"(c): { {PS.body_label(k): n for k, n in n_c.items()} }; untrained (b): {ur} "
          f"(c): {uc}")
    handles = role_handles(models, {**n_a, **n_b, **n_c})
    handles += seed_handles()
    handles[-1].set_label(_seed_band_label(list(n_a.values()) + list(n_b.values())
                                           + list(n_c.values())))
    compact_legend(fig, handles)
    fig.canvas.draw()
    assert_inside(aa, ys=np.concatenate([np.ravel(v) for v in drawn_a]),
                  what="fig_state_access (a)")
    PS.check_inside(fig, [aa, ab, ac], "fig_state_access")
    return PS.save(fig, "fig_state_access")


READ_DEPTH_LESION_YMIN = 8e-3


def fig_read_depth():
    """Which state depth the prediction reads, and which layers matter (sec. 4):
      (a) panel_read_depth    Two-tower 12+12, deepest-only vs shallowest-only read levels
      (b) panel_lesion        Two-tower 12+12 per-layer whole-layer lesion cost
      (c) panel_lesion        SPS (slot lesions of the shared stack)
      (d) panel_lesion        Sequential 12+12 per-layer whole-layer lesion cost
    (b)-(d) share ONE log y axis (same scale), so SPS's unread final state (exact zero,
    off-axis x) sits beside Sequential's costly final state; (a) keeps its own axis and
    labels its curves directly, so the legend carries only the lesion encoding."""
    sps_k, sps, have_sps = lesion_sps()
    ncol = 4 if have_sps else 3
    fig = plt.figure(figsize=(PS.FULL_W, 2.4))
    gs = fig.add_gridspec(1, ncol)
    aa = fig.add_subplot(gs[0])
    ab = fig.add_subplot(gs[1])
    axes = [aa, ab]
    na, drawn_a = panel_read_depth(aa, title="(a) Two-tower 12+12\nread depth")
    aa.set_ylabel("Loss increase (nats)")
    zb = panel_lesion(ab, INTER, title="(b) Two-tower 12+12\nlesions")
    zc = {}
    if have_sps:
        ac = fig.add_subplot(gs[2], sharey=ab)
        axes.append(ac)
        zc = panel_lesion(ac, sps_k, sps, title="(c) SPS\nlesions")
        ac.tick_params(labelleft=False)
    else:
        D.warn("fig_read_depth: joint SPS a12 incomplete -> SPS panel dropped")
    ad = fig.add_subplot(gs[ncol - 1], sharey=ab)
    axes.append(ad)
    zd = panel_lesion(ad, SEQ, title="(d) Sequential 12+12\nlesions")
    ad.tick_params(labelleft=False)
    # bottom just under the lowest seed-range edge (~0.013 nats, SPS state layer 8); the
    # exact zero is drawn below the frame either way
    ab.set_ylim(READ_DEPTH_LESION_YMIN, None)
    ab.set_ylabel("Loss increase (nats)")
    nb = len(D.result_seeds("a12_depth_lesion", INTER))
    nc = len(sps["state"]) if have_sps else nb
    nd = len(D.result_seeds("a12_depth_lesion", SEQ))
    print(f"[fig_read_depth] seeds: (a) {na}, (b) {nb}, (c) {nc}, (d) {nd}; SPS arm {sps_k}")
    sh = stream_handles(markers=STREAM_MARKER)
    sh[0].set_label("state layer")
    sh[1].set_label("prediction layer")
    handles = sh
    if zb or zc or zd:
        handles.append(zero_handle("zero cost (unread layer)"))
    handles += seed_handles()
    handles[-1].set_label(_seed_band_label([na, nb, nc, nd]))
    compact_legend(fig, handles)
    fig.canvas.draw()
    assert_inside(aa, ys=np.concatenate([np.ravel(v) for v in drawn_a]),
                  what="fig_read_depth (a)")
    assert_lesion_inside(ab, INTER)
    if have_sps:
        assert_lesion_inside(ac, sps_k, sps)
    assert_lesion_inside(ad, SEQ)
    PS.check_inside(fig, axes, "fig_read_depth")
    return PS.save(fig, "fig_read_depth")


def fig_gain_diag():
    """Not in the paper (its panels are fig_state_access (c) and fig_read_depth (d)):
      (a) panel_roleprobe P1  next-token probe loss per prediction layer (Transformer: its
          single stream); untrained-init and bigram references labelled in the panel
      (b) panel_lesion        Sequential 12+12 per-layer whole-layer lesion cost, its two
          curves labelled directly (state / prediction)."""
    fig, (aa, ab) = plt.subplots(1, 2, figsize=(PS.FULL_W, 2.3))
    sps_k, models = role_models()
    n_a, ur = panel_roleprobe(aa, "P1", sps_k, models, title="(a) Next-token probe")
    aa.set_ylabel("Probe loss (nats)")
    zb = panel_lesion(ab, SEQ, title="(b) Sequential 12+12: lesions")
    _stream_labels(ab, SEQ)
    ab.set_ylabel("Loss increase (nats)")
    nb = len(D.result_seeds("a12_depth_lesion", SEQ))
    print(f"[fig_gain_diag] seeds (a): { {PS.body_label(k): n for k, n in n_a.items()} } "
          f"(b): {nb}; untrained (a): {ur}")
    handles = role_handles(models, n_a)
    handles += seed_handles()
    handles[-1].set_label(_seed_band_label(list(n_a.values()) + [nb]))
    if zb:
        handles.append(zero_handle())
    compact_legend(fig, handles)
    fig.canvas.draw()
    assert_lesion_inside(ab, SEQ)
    PS.check_inside(fig, [aa, ab], "fig_gain_diag")
    return PS.save(fig, "fig_gain_diag")


# =======================================================================================
# fig_frontier
# =======================================================================================
# line style per tower group: the reader must see 12+12 vs 6+6 at a glance; the references
# keep their own hue and a thin solid line
GROUP_LS = {"12+12": "-", "6+6": (0, (3.2, 1.6)), "ref": "-"}
GROUP_LW = {"12+12": 1.15, "6+6": 1.15, "ref": 0.85}


def _frontier_layout(fig, top, bottom, ml=0.03, mr=0.03, mt=0.03, mb=0.03, gap=0.10,
                     vgap=0.10, top_frac=0.31, weights=(1.25, 1.0, 1.0), iters=4):
    """Place the axes by hand, in inches, from their RENDERED extents: each top panel's
    tight bbox (tick labels, title, xlabel, and the out-of-axes value/name label column)
    is measured, and the data widths are what is left of the figure after every panel's
    overhang plus `gap` between neighbours -- so no text can cross into a neighbouring
    panel or past the figure edge.  The bottom panel spans the full width, left-aligned
    with panel (a).  Iterated because titles/ticks shift slightly with the axes size."""
    W, H = fig.get_size_inches()
    for _ in range(iters):
        r = fig.canvas.get_renderer()
        dpi = fig.dpi

        def oh(ax):
            b, t = ax.get_window_extent(r), ax.get_tightbbox(r)
            return ((b.x0 - t.x0) / dpi, (t.x1 - b.x1) / dpi,
                    (b.y0 - t.y0) / dpi, (t.y1 - b.y1) / dpi)
        o = [oh(ax) for ax in top]
        od = oh(bottom)
        avail = W - ml - mr - sum(a[0] + a[1] for a in o) - gap * (len(top) - 1)
        ws = [avail * w / sum(weights) for w in weights]
        # vertical: [mb | od.bottom | (d) | od.top | vgap | top.bottom | top row | top.top | mt]
        tb, tt = max(a[2] for a in o), max(a[3] for a in o)
        hv = H - mt - mb - od[2] - od[3] - vgap - tb - tt
        ht, hd = hv * top_frac, hv * (1 - top_frac)
        y_top = mb + od[2] + hd + od[3] + vgap + tb
        x = ml + max(o[0][0], od[0])
        x0 = x
        for ax, a, w in zip(top, o, ws):
            if ax is not top[0]:
                x += a[0]
            ax.set_position([x / W, y_top / H, w / W, ht / H])
            x += w + a[1] + gap
        # recompute avail so panel (a)'s left edge (shifted by the (d) ylabel) still fits
        shift = x0 - (ml + o[0][0])
        if shift > 1e-4:
            ws = [w - shift * wt / sum(weights) for w, wt in zip(ws, weights)]
            x = x0
            for ax, a, w in zip(top, o, ws):
                if ax is not top[0]:
                    x += a[0]
                ax.set_position([x / W, y_top / H, w / W, ht / H])
                x += w + a[1] + gap
        bottom.set_position([x0 / W, (mb + od[2]) / H, (W - mr - od[1] - x0) / W, hd / H])
        fig.canvas.draw()


def _frontier_legend(ax, cur, grp):
    """Grouped legend (one column per tower group, group name as a bold header) in
    fig_frontier (d) and fig_frontier_full; it is the key for fig_frontier (a)-(c) as well."""
    def hdr(t):
        return Line2D([], [], ls="none", marker="none", label=t)
    cols = []
    for g, title in (("12+12", "12+12 (solid)"), ("6+6", "6+6 (dashed)"),
                     ("ref", "References")):
        ks = [k for k in cur if grp[k] == g]
        if not ks:
            continue
        hs = [hdr(title)]
        for k in ks:
            hs.append(Line2D([], [], lw=GROUP_LW[g], ls=GROUP_LS[g], label=PS.body_label(k),
                             **PS.marker_kw(k, ms=2.6)))
        cols.append(hs)
    nrow = max(len(c) for c in cols)
    handles = []
    for c in cols:
        handles += c + [Line2D([], [], ls="none", label="")] * (nrow - len(c))
    leg = ax.legend(handles=handles, ncol=len(cols), loc="lower left", fontsize=7,
                    handlelength=2.6, borderaxespad=0.4, labelspacing=0.2,
                    columnspacing=1.2, handletextpad=0.5)
    for t in leg.get_texts():
        if t.get_text() in {c[0].get_label() for c in cols}:
            t.set_fontweight("bold")
            t.set_position((-18, 0))     # header text flush with the handles


def _full_panel(ax, d, what, title=None):
    """Validation loss vs cumulative TRAINING compute (tokens x 3 x forward FLOPs/token, log
    x) over each model's WHOLE run, final LR decay included, from the checkpoint ladder in
    the ledger (per-seed curves faint, seed mean bold with checkpoint markers); 12+12 solid,
    6+6 dashed, Transformer and SPS thin lines in their own hue; the grouped legend.  No
    grey shading, no comparison-point vlines, no in-plot key (Fig 3 feedback: "remove the
    grey stuff"); fig_frontier adds thin (a)/(b)/(c) brackets (_mark_zoom_windows)."""
    cur, grp = d["cur"], d["grp"]
    for k, (x, m, seeds) in cur.items():
        g, col = grp[k], ARMS[k]["color"]
        for xs, ys in seeds:
            ax.plot(xs, ys, color=col, lw=0.7, alpha=0.45, ls=GROUP_LS[g], zorder=2)
        ax.plot(x, m, lw=GROUP_LW[g], ls=GROUP_LS[g], zorder=4 if g != "ref" else 3,
                dash_capstyle="butt", **PS.marker_kw(k, ms=1.9))
        print(f"[{what}] {PS.body_label(k)} ({grp[k]}): {max(1, len(seeds))} "
              f"seed(s), {len(x)} pts, final {m[-1]:.4f} at {x[-1]:.0f} PFLOPs")
    ax.set_xscale("log")
    tv = [2e3, 5e3, 1e4, 2e4, 5e4]
    ax.set_xticks(tv, ["2k", "5k", "10k", "20k", "50k"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    x_end = max(x[-1] for x, _, _ in cur.values())
    # tight enough that the data fills the panel but still clear of the rightmost markers.
    ax.set_xlim(1.9e3, x_end * 1.06)
    ax.set_xlabel("Training compute (PFLOPs)")
    ax.set_ylabel(LOSS_LABEL)
    ax.set_ylim(2.66, 3.40)
    for x, m, seeds in cur.values():   # the first checkpoints sit left of the window
        for xs, ys in [(x, m)] + seeds:
            shown = np.asarray(xs) >= ax.get_xlim()[0]
            assert_inside(ax, np.asarray(xs)[shown], np.asarray(ys)[shown], what=what)
    ax.grid(True, axis="x", which="major", color=PS.GRID, lw=0.5)
    if title:
        ax.set_title(title, loc="left", fontsize=7, pad=3)
    _frontier_legend(ax, cur, grp)


def _mark_zoom_windows(ax, windows):
    """Thin bracket + letter along the top of (d) for each zoom panel's x window
    [(letter, (x0, x1), y_axes_frac)], clipped to (d)'s x range.  No shading."""
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    lo, hi = ax.get_xlim()
    for letter, (x0, x1), y in windows:
        x0, x1 = max(x0, lo), min(x1, hi)
        h = 0.025
        ax.plot([x0, x0, x1, x1], [y - h, y, y, y - h], transform=tr, color=PS.MUTED,
                lw=0.6, zorder=1, solid_capstyle="butt", clip_on=False)
        ax.text(np.sqrt(x0 * x1), y + 0.008, letter, transform=tr, ha="center", va="bottom",
                fontsize=7, color=PS.MUTED)


def fig_frontier_full():
    """fig_frontier's panel (d) on its own: the full validation loss vs cumulative training
    compute curves of every body model, one panel with the legend, sized for \\linewidth."""
    d = D.frontier()
    fig, aa = plt.subplots(figsize=(PS.FULL_W, 2.6))
    _full_panel(aa, d, "fig_frontier_full")
    fig.canvas.draw()
    PS.check_inside(fig, [aa], "fig_frontier_full")
    if d["missing"]:
        D.warn(f"fig_frontier_full: no ladder curve for {d['missing']}")
    return PS.save(fig, "fig_frontier_full")


def fig_frontier():
    """Four panels.  (d, bottom) validation loss vs cumulative TRAINING compute (tokens x 3 x
    forward FLOPs/token) for every body model, from the checkpoint ladder in the ledger
    (seed-rule runs: per-seed curves faint, seed mean bold); 12+12 solid, 6+6 dashed,
    Transformer and SPS thin reference lines in their own hue; (d) carries the curves,
    markers, the legend and thin (a)/(b)/(c) brackets marking each zoom window.
    Top row, zoomed windows of the same curves (seed lines thin, same line styles), one
    per comparison point, each model marked on its curve (filled = separate weights, open =
    shared) with 'value name' printed in a label column right of the panel's axes:
      (a) matched compute before any final LR decay: every model at the Transformer's
          18B-token `_pre_decay` checkpoint compute (90% of its total; the Transformer and
          6+6 models start their final decay there), marked by a dotted line.  Transformer /
          6+6: that checkpoint; 12+12 / SPS: linear interpolation between the two
          checkpoints around it;
      (b) low compute, after LR decay: final losses of the models that end at the
          Transformer's total compute (Transformer, 6+6);
      (c) high compute, after LR decay: final losses of the models that end at ~45k PFLOPs
          (12+12, SPS)."""
    d = D.frontier()
    cur, grp, tc, tT = d["cur"], d["grp"], d["tc"], d["tT"]
    # the two post-decay comparison points: models ending within 2% of the Transformer's
    # compute (short) and the rest (long_, ending at x_hi on average)
    short = [k for k, (x, _, _) in cur.items() if abs(x[-1] - tT) / tT < 0.02]
    long_ = [k for k in cur if k not in short]
    x_hi = float(np.mean([cur[k][0][-1] for k in long_]))

    # explicit layout (no constrained_layout): each top panel's value/name label column is
    # drawn OUTSIDE its data axes, in a gutter reserved to its right; _frontier_layout sizes
    # every gutter from the rendered text extents so no label can cross into the next panel
    # or past the figure edge.
    fig = plt.figure(figsize=(PS.FULL_W, 3.4))
    fig.set_layout_engine("none")
    axs = [fig.add_axes([0.1 + 0.3 * i, 0.7, 0.1, 0.2]) for i in range(3)]
    aa = fig.add_axes([0.1, 0.1, 0.8, 0.5])

    # ---- (d) main panel: full training curves + legend -----------------------------------
    _full_panel(aa, d, "fig_frontier (d)", title="(d)")

    # ---- (a)-(c) zoomed curves ---------------------------------------------------------
    # labels are the full body names, never abbreviations: if one overflows at 7pt, print
    # fewer decimals instead
    def _stack_labels(targets, step):
        """Evenly-spaced (uniform `step`) label y-positions, in the same top-to-bottom order
        as `targets`, chosen to minimise total squared distance to `targets` -- i.e. the one
        vertical shift of an evenly-spaced column that sits, in aggregate, as close as
        possible to where the points actually are.  Keeps every zoom panel's label column
        both internally uniform (Fig 3 feedback: labels were "off from each other" -- ad hoc
        greedy stacking left some pairs almost touching and others with a big unused gap) and
        anchored to the real data rather than an arbitrary top-down cascade."""
        n = len(targets)
        if n <= 1:
            return list(targets)
        c = float(np.mean(targets)) + step * (n - 1) / 2.0
        return [c - i * step for i in range(n)]

    def _upto(xs, ys, stop):
        """(xs, ys) truncated at x = stop, ending on the value linearly interpolated at stop
        (so a curve ends exactly at the comparison point, not at its last checkpoint)."""
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        if stop is None or xs[-1] <= stop:
            return xs, ys
        keep = xs < stop
        return (np.append(xs[keep], stop), np.append(ys[keep], np.interp(stop, xs, ys)))

    def zoom(ax, title, xlim, ylim, xticks, marks, faint, show_faint=True, gap_mult=1.0,
             stop=None):
        """The ladder curves in a window; `marks` {key: (x, y)} get the arm's marker and a
        stacked 'value name' label in the gutter RIGHT of the axes (outside the data area;
        _frontier_layout reserves that gutter from the measured text width); curves in `faint` are
        drawn faint (lower alpha, thinner) and their marker/leader/label are de-emphasised,
        not hidden, unless `show_faint` is False, in which case they are left out of the
        panel entirely; `stop` truncates EVERY curve (mean and seeds) at x = stop."""
        for k, (x, m, seeds) in cur.items():
            x, m = _upto(x, m, stop)
            seeds = [_upto(xs, ys, stop) for xs, ys in seeds]
            g, col = grp[k], ARMS[k]["color"]
            kw = dict(color=col, lw=GROUP_LW[g] + 0.1, ls=GROUP_LS[g], dash_capstyle="butt",
                      zorder=4 if g != "ref" else 3)
            if k in faint:
                # de-emphasised but still legibly solid next to the bold dashed 6+6 curves
                if show_faint:
                    ax.plot(x, m, **{**kw, "alpha": 0.42, "lw": max(0.75, kw["lw"] * 0.75),
                                     "zorder": 2})
                continue
            for xs, ys in seeds:
                ax.plot(xs, ys, color=col, lw=0.7, alpha=0.45, ls=GROUP_LS[g], zorder=2)
            ax.plot(x, m, **kw)
        y0, y1 = ylim
        gap = 0.125 * gap_mult * (y1 - y0)
        # two arms whose true values are within a marker's own height (e.g. Two-tower 6+6 vs
        # SPS in (a): 2.957 vs 2.960) draw their marker GLYPHS on top of each other -- dodge
        # the drawn marker position by the minimum readable separation, but keep the printed
        # value (`py`, below) and the leader line's start point exact.
        mgap = 0.32 * gap
        xl = xlim[1]        # leaders run to the axes' right edge; text sits just past it
        # marker glyphs: greedy top-down dodge, just enough to keep overlapping glyphs
        # readable (mgap is small relative to the label gap below).
        order = sorted(marks, key=lambda k: -marks[k][1])
        py_ms = []
        for k in order:
            _, py = marks[k]
            py_ms.append(py if not py_ms else min(py, py_ms[-1] - mgap))
        # labels: one evenly-spaced column (see _stack_labels), least-squares-anchored to
        # the (dodged) marker positions so the column sits as close as it can to the real
        # data while every gap between labels is identical.
        yts = _stack_labels(py_ms, gap)
        for k, py_m, yt in zip(order, py_ms, yts):
            px, py = marks[k]
            is_faint = k in faint
            ax.plot([px], [py_m], ls="none", zorder=8,
                    **PS.marker_kw(k, ms=3.6 * (0.72 if is_faint else 1.0),
                                   alpha=0.55 if is_faint else 1.0))
            if is_faint:
                # leader line: a mid-tint of the arm's OWN hue (not a neutral grey, which
                # reads as the Transformer's colour) so the de-emphasis is visible but the
                # arm identity survives; kept lighter than the label TEXT (below), which
                # must stay >=7pt-legible even for a de-emphasised curve -- a 0.30 lighten
                # on the text itself ("2.910 Two-tower 12+12" at 0.30 was near-invisible
                # against white) undershoots that bar.
                leader_col = PS.lighten(ARMS[k]["color"], 0.30)
                text_col = PS.lighten(ARMS[k]["color"], 0.08)
            else:
                leader_col = text_col = (
                    ARMS[k]["color"] if ARMS[k]["color"] != PS.C_TRANSFORMER else PS.MUTED)
            leader_alpha = 0.55 if is_faint else 1.0
            leader_lw = 0.32 if is_faint else 0.45
            # Elbow leader: bend vertically right at the marker's own x, then run
            # horizontally into the label at yt.  Unlike a single diagonal line to (xl, yt),
            # the horizontal run for every label sits only at its own yt, so leader lines
            # never cross another label's text -- only the marker cluster itself, where a
            # short vertical stack is expected and unambiguous.
            ax.plot([px, px, xl], [py_m, yt, yt], color=leader_col, lw=leader_lw,
                    alpha=leader_alpha, zorder=7, solid_capstyle="butt", dash_capstyle="butt",
                    clip_on=False)
            ax.annotate(f"{py:.3f} {PS.body_label(k)}", xy=(1.0, yt),
                        xycoords=("axes fraction", "data"), xytext=(2, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=7,
                        color=text_col, zorder=9, alpha=1.0, annotation_clip=False)
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(y0, y1)
        assert_inside(ax, [p[0] for p in marks.values()], [p[1] for p in marks.values()],
                      what=f"fig_frontier {title}")
        ax.set_xticks(xticks, [f"{v / 1e3:g}k" for v in xticks])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.tick_params(labelsize=7, length=2, pad=1.5)
        ax.grid(True, axis="y", color=PS.GRID, lw=0.4)
        ax.set_title(title, loc="left", fontsize=7, pad=3)
        ax.set_xlabel("PFLOPs", fontsize=7, labelpad=1.5)

    # panel (a) shows all 6 body models in full colour (every model must be clearly visible)
    FADE_A = set()
    report = {}
    # (a) matched compute before any final decay
    va = {k: D.value_at(cur, k, tc) for k in cur}
    # every curve stops AT the matched point: past it the Transformer and 6+6 models enter
    # their final LR decay and drop steeply, and that drop (not the comparison) is what the
    # eye followed when (a) showed it.  The post-decay trajectories are in (b)-(d).
    win_a, win_b, win_c = (1.4e4, tc * 1.04), (2.05e4, 3.1e4), (3.7e4, 5.5e4)
    zoom(axs[0], f"(a) Matched pre-decay, {tc / 1e3:.1f}k PFLOPs", win_a,
         (2.86, 3.03), [1.5e4, 2e4], {k: (tc, v[0]) for k, v in va.items()},
         faint=FADE_A, gap_mult=1.35, stop=tc)
    axs[0].axvline(tc, color=PS.MUTED, lw=0.7, ls=":", zorder=1)
    report["(a)"] = va
    # (b) low compute, after decay: ends of the Transformer and 6+6.  The 12+12/SPS models
    # (long_) are already shown, in full colour and labelled, in (a) and (d); showing them
    # faint and unlabelled here again only adds clutter, so this panel omits them entirely.
    vb = {k: D.value_at(cur, k, None) for k in short}
    zoom(axs[1], f"(b) Transformer-cost, {tT / 1e3:.1f}k PFLOPs", win_b,
         (2.78, 3.00), [2e4, 2.5e4, 3e4], {k: (cur[k][0][-1], vb[k][0]) for k in short},
         faint=long_, show_faint=False)
    report["(b)"] = vb
    # (c) high compute, after decay: ends of the 12+12 and SPS; same reasoning as (b) for
    # dropping the Transformer/6+6 curves.
    vc = {k: D.value_at(cur, k, None) for k in long_}
    zoom(axs[2], f"(c) SPS-cost, $\\approx${x_hi / 1e3:.0f}k PFLOPs", win_c,
         (2.68, 2.94), [4e4, 5e4], {k: (cur[k][0][-1], vc[k][0]) for k in long_},
         faint=short, show_faint=False)
    report["(c)"] = vc
    # (a) and (c) do not overlap in x; (b) overlaps (a), so its bracket sits lower
    _mark_zoom_windows(aa, [("(a)", win_a, 0.90), ("(b)", win_b, 0.79), ("(c)", win_c, 0.90)])
    axs[0].set_ylabel("Val. loss", fontsize=7, labelpad=2)
    for t, r in report.items():
        print(f"[fig_frontier] {t}: " + str({PS.body_label(k): (round(v[0], 4),
              [round(s, 4) for s in v[1]], "checkpoint" if v[2] else "interpolated")
              for k, v in sorted(r.items(), key=lambda kv: kv[1][0])}))

    _frontier_layout(fig, axs, aa)
    PS.check_inside(fig, axs + [aa], "fig_frontier")
    print(f"[fig_frontier] a = {tc:.1f} PFLOPs (Transformer 18B pre-decay), b = {tT:.1f}, "
          f"c = {x_hi:.1f} (mean end of {[PS.body_label(k) for k in long_]}: "
          f"{[round(float(cur[k][0][-1])) for k in long_]}); Transformer-cost / SPS-cost = "
          f"{tT / cur[PS.BODY_SPS][0][-1]:.3f}; "
          f"Sequential 6+6 - SPS = {vb[SEQ6][0] - vc[PS.BODY_SPS][0]:+.4f}")
    if d["missing"]:
        D.warn(f"fig_frontier: no ladder curve for {d['missing']}")
    return PS.save(fig, "fig_frontier")


def fig_teaser():
    """Figure 1, the architecture schematic (fig_arch_paper.py)."""
    import fig_arch_paper
    return PS.save(fig_arch_paper.draw(), "fig_teaser")


def appendix(name):
    """make() of an appendix figure module (paper_appendix/<name>.py)."""
    def make():
        import importlib
        return importlib.import_module(f"paper_appendix.{name}").make()
    make.__name__ = name
    return make


FIGS = {f.__name__: f for f in (fig_teaser, fig_state_access, fig_read_depth, fig_gain_diag,
                                  fig_frontier, fig_frontier_full,
                                  appendix("app_ctrl_dist"), appendix("app_gradcos"),
                                  appendix("app_depth_split"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma-separated subset of " + ",".join(FIGS))
    ap.add_argument("--outdir", default=str(PS.OUT_DIR))
    a = ap.parse_args()
    PS.OUT_DIR = Path(a.outdir)
    for n in [n for n in a.only.split(",") if n] or list(FIGS):
        FIGS[n]()


if __name__ == "__main__":
    main()
