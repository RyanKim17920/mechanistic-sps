"""Figure 1 of the paper (fig:teaser): the four architectures, in two rows.

  top row (centred): (a) Transformer, (b) SPS; bottom row: (c) Two-tower, (d) Sequential.
  Panels are measured (rendered bounding boxes) and placed with one equal gutter and
  equal outer margins per row, so spacing stays even if a label changes.

  -> paper/figures/fig_teaser.pdf (paper_figures.py fig_teaser)

A print-size, stripped-down redraw of the authors' own architecture diagram.  It keeps its
design -- one "Attention is all you need"-style tower per stream, per-slot input tokens
at the bottom, per-slot loss arrows at the top, an N x repeat frame, and the K,V read
drawn as a wire into the prediction tower's cross-attention -- and drops everything that
does not distinguish one architecture from another (Norm / Add bars, residual bypasses,
the Linear / Softmax split).

What each panel must show, checked against the model code (src/modeling/models/):

  (a) Transformer  one stream; every x_t emits p(x_{t+1}).
  (b) SPS          ONE stack over the interleaved sequence x_1 <p> x_2 <p> ...
                   (sps/core.py add_predict_tokens: x_t at slot 2t, <p> at 2t+1; <p> is a
                   row of the same input table).  Every slot attends all causal x keys plus
                   the <p> keys of the last 64 tokens (triton_sps_flash_attention.py, window
                   64 from s_sps_w64_20b_fw100) -- so state slots also read <p>.  Loss only
                   at the <p> slots (lm_head(x[:, 1::2])).  Read: the <p> query at layer l
                   attends the x keys layer l computes, i.e. state level l-1.
  (c) Two-tower    s_two_tower_w0_equal(6)_20b: separate towers and tables; state tower is a
                   plain causal LM that never reads prediction; prediction layer l reads, via
                   its OWN norm + K,V projection (read_source=pred_proj), the OUTPUT of state
                   layer l (read_map=post, f(i)=i+1 0-indexed); pred_window=0; the prediction
                   tower's input is x_t from its own table (predict_embedding=separate).
  (d) Sequential   s_two_tower_seq12/seq6_20b: as (c) but read_map=final -- every prediction
                   layer reads the state tower's FINAL output.

Depth is drawn as N (12 / 6+6 / 12+12 are in the text).  Nothing here is a measurement.

Usage: .venv/bin/python scripts/analysis/fig_arch_paper.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from plotting import paper_style as PS  # noqa: E402  (applies rcParams)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

INK, MUTED = PS.INK, PS.MUTED
C = {k: PS.ARMS[PS.MANIFEST["roles"][r]]["color"] for k, r in {
    "tf": "transformer", "sps": "sps", "tt": "two_tower12", "seq": "sequential12"}.items()}
DASH = (0, (2.2, 1.2))
FRAME = "#A0A0A0"
F_EMB = "#EFEFEF"      # embedding / head fill (neutral: hue is reserved for architecture)

FS = 7.0               # box text, token labels  (>= 7 pt at final size)
FS_S = 7.0             # wire labels, subtitles, N x
FS_T = 8.0             # panel titles

W, H = PS.FULL_W, 3.18
TW = 0.88              # a 3-slot tower
GAPT = 0.40            # gap between the two towers of a panel (holds only the K,V wire)
W_SPS = 2.10           # the 6-slot SPS stack (wide enough for the <predict> token labels)
PADL = 0.24            # panel left edge -> tower x (room for the N x label; titles start here)
# vertical layout of one row, relative to the row's base y0
Y_TOK, Y_EMB, H_EMB = 0.045, 0.15, 0.13
Y_ATT, H_ATT = 0.37, 0.24
Y_FFN, H_FFN = 0.67, 0.12
FR_LO, FR_HI = 0.33, 0.845
Y_HEAD, H_HEAD = 0.90, 0.12
Y_SUB = 1.285          # subtitle bottom (clear of the output labels, which end near 1.23)
Y_TITLE = 1.40
ROW = 1.60             # row pitch: content of a row spans ~0 .. 1.52

P = r"$\langle$predict$\rangle$"   # the one learned token every prediction slot receives
X3 = [r"$x_1$", r"$x_2$", r"$x_3$"]
OUT3 = [r"$p(x_2)$", r"$p(x_3)$", r"$p(x_4)$"]


# ----------------------------------------------------------------------------- primitives
def box(ax, x, y, w, h, text, fc="white", ls="-", fs=FS, ec=INK, lw=0.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.025",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=3))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=INK, zorder=4, linespacing=1.0)


def arrow(ax, p, q, color=INK, lw=0.6, head=True, z=2, ms=5.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>" if head else "-", mutation_scale=ms,
                                 color=color, lw=lw, zorder=z, shrinkA=0, shrinkB=0))


def wire(ax, pts, color, lw=0.9):
    """The K,V read path: polyline ending in an arrowhead, dot at its source."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        arrow(ax, a, b, color=color, lw=lw, head=False, z=5)
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, z=5, ms=6)
    ax.plot(*pts[0], "o", ms=2.4, color=color, zorder=6)


def slots(x, w, n):
    return [x + w * (i + 0.5) / n for i in range(n)]


def tokens(ax, x, w, y0, labels):
    for sx, lab in zip(slots(x, w, len(labels)), labels):
        ax.text(sx, y0 + Y_TOK, lab, ha="center", va="center", fontsize=FS, color=INK)
        arrow(ax, (sx, y0 + Y_TOK + 0.055), (sx, y0 + Y_EMB))


def outputs(ax, x, w, y0, labels):
    yt = y0 + Y_HEAD + H_HEAD
    for sx, lab in zip(slots(x, w, len(labels)), labels):
        if lab is None:
            continue
        arrow(ax, (sx, yt), (sx, yt + 0.085))
        ax.text(sx, yt + 0.095, lab, ha="center", va="bottom", fontsize=FS, color=INK)


def vseg(ax, x, y0, a, b, ls="-"):
    ax.plot([x, x], [y0 + a, y0 + b], color=INK, lw=0.6, ls=ls, zorder=2)


def frame(ax, x0, x1, y0, lo=FR_LO, hi=FR_HI, label=True):
    ax.add_patch(Rectangle((x0, y0 + lo), x1 - x0, hi - lo, fc="none", ec=FRAME, lw=0.6,
                           ls=(0, (2.5, 1.5)), zorder=1))
    if label:
        ax.text(x0 - 0.025, y0 + (lo + hi) / 2, r"$N\times$", ha="right", va="center",
                fontsize=FS_S, color=INK)


def title(ax, x, y0, tag, name, color, sub):
    ax.text(x, y0 + Y_TITLE, f"({tag})", ha="left", va="bottom", fontsize=FS_T,
            fontweight="bold", color=INK)
    ax.text(x + 0.20, y0 + Y_TITLE, name, ha="left", va="bottom", fontsize=FS_T,
            fontweight="bold", color=color)
    if sub:
        ax.text(x + 0.20, y0 + Y_SUB, sub, ha="left", va="bottom", fontsize=FS_S,
                color=MUTED)


def column(ax, x, y0, emb, attn, ls="-", ffn="MLP", head=True):
    """One tower: embedding -> attention -> MLP (-> LM head).  ls="--" = prediction stream."""
    lsb = "-" if ls == "-" else DASH
    xm = x + TW / 2
    box(ax, x, y0 + Y_EMB, TW, H_EMB, emb, fc=F_EMB, ls=lsb)
    vseg(ax, xm, y0, Y_EMB + H_EMB, Y_ATT)
    box(ax, x, y0 + Y_ATT, TW, H_ATT, attn, ls=lsb)
    if ffn:
        vseg(ax, xm, y0, Y_ATT + H_ATT, Y_FFN)
        box(ax, x, y0 + Y_FFN, TW, H_FFN, ffn, ls=lsb)
    if head:
        vseg(ax, xm, y0, Y_FFN + H_FFN, Y_HEAD)
        box(ax, x, y0 + Y_HEAD, TW, H_HEAD, "LM head", fc=F_EMB, ls=lsb)


# --------------------------------------------------------------------------------- panels
def panel_transformer(ax, x, y0):
    column(ax, x, y0, "Embedding", "Causal attention")
    tokens(ax, x, TW, y0, X3)
    frame(ax, x - 0.04, x + TW + 0.04, y0)
    outputs(ax, x, TW, y0, OUT3)
    title(ax, x - PADL, y0, "a", "Transformer", C["tf"], "one stack, one slot per token")


def panel_sps(ax, x, y0):
    w = W_SPS
    xm = x + w / 2
    box(ax, x, y0 + Y_EMB, w, H_EMB, "Embedding (one table)", fc=F_EMB)
    vseg(ax, xm, y0, Y_EMB + H_EMB, Y_ATT)
    box(ax, x, y0 + Y_ATT, w, H_ATT,
        "Causal attention:\nstate: all; " + P + ": last 64")
    vseg(ax, xm, y0, Y_ATT + H_ATT, Y_FFN)
    box(ax, x, y0 + Y_FFN, w, H_FFN, "MLP")
    vseg(ax, xm, y0, Y_FFN + H_FFN, Y_HEAD)
    box(ax, x, y0 + Y_HEAD, w, H_HEAD, "LM head", fc=F_EMB)
    tokens(ax, x, w, y0, [X3[0], P, X3[1], P, X3[2], P])
    frame(ax, x - 0.04, x + w + 0.04, y0)
    outputs(ax, x, w, y0, [None, OUT3[0], None, OUT3[1], None, OUT3[2]])
    # a <p> slot's query at layer l is answered by x keys/values layer l itself computes
    # from its input, i.e. state level l-1 (the layer below); shown as a small self-loop
    # off the attention box's right edge, from that input (bottom, level l-1) into the box
    # (level l), so it reads as "this layer's K,V come from one level down"
    col = C["sps"]
    xr = x + w + 0.05
    y_lo = y0 + Y_ATT + 0.02
    y_hi = y0 + Y_ATT + H_ATT - 0.02
    ax.add_patch(FancyArrowPatch((xr, y_lo), (xr, y_hi), connectionstyle="arc3,rad=1.15",
                                 arrowstyle="-|>", mutation_scale=5.0, color=col, lw=0.7,
                                 zorder=5, shrinkA=0, shrinkB=0))
    ax.plot(xr, y_lo, "o", ms=2.0, color=col, zorder=6)
    ax.text(xr + 0.24, (y_lo + y_hi) / 2, "K,V: state entering\nlayer $\\ell$",
            ha="left", va="center", fontsize=FS_S, color=col, linespacing=1.0)
    # consequence of that read: the state slots' output of the LAST layer feeds nothing
    # (no later layer reads it; the loss is taken only at the prediction slots)
    ax.text(xr + 0.05, y0 + Y_HEAD + H_HEAD / 2, "last layer's state\noutput: unread",
            ha="left", va="center", fontsize=FS_S, color=MUTED, linespacing=1.0)
    title(ax, x - PADL, y0, "b", "SPS", C["sps"], "one stack, interleaved slots")


def _towers(ax, xs, y0, pred_tokens):
    xp = xs + TW + GAPT
    column(ax, xs, y0, "State embedding", "Causal attention", head=False)
    column(ax, xp, y0, "Pred. embedding", "Cross-attention", ls="--")
    tokens(ax, xs, TW, y0, X3)
    tokens(ax, xp, TW, y0, pred_tokens)
    outputs(ax, xp, TW, y0, OUT3)
    return xp


def panel_two_tower(ax, xs, y0):
    col = C["tt"]
    xp = _towers(ax, xs, y0, X3)
    # prediction layer l reads state LEVEL l, i.e. state layer l's OUTPUT (after its MLP):
    # the wire leaves the top of the state MLP (as in (d)), still INSIDE the shared N x
    # frame, runs down the middle of the gap and enters the prediction cross-attention
    xm = xs + TW / 2
    yo = y0 + (Y_FFN + H_FFN + FR_HI) / 2
    vseg(ax, xm, y0, Y_FFN + H_FFN, (Y_FFN + H_FFN + FR_HI) / 2)
    xr = xs + TW + 0.5 * GAPT
    ya = y0 + Y_ATT + H_ATT / 2
    wire(ax, [(xm, yo), (xr, yo), (xr, ya), (xp, ya)], col)
    # label above the state tower (it has no LM head), clear of the frame
    ax.text(xs, y0 + FR_HI + 0.045, "K,V: same-depth state", ha="left", va="bottom",
            fontsize=FS_S, color=col)
    frame(ax, xs - 0.04, xp + TW + 0.04, y0)
    title(ax, xs - PADL, y0, "c", "Two-tower", col, "separate towers")


def panel_seq(ax, xs, y0):
    col = C["seq"]
    xp = _towers(ax, xs, y0, X3)
    xm = xs + TW / 2
    frame(ax, xs - 0.04, xs + TW + 0.04, y0)
    frame(ax, xp - 0.04, xp + TW + 0.04, y0, label=False)
    ax.text(xp + TW + 0.065, y0 + (FR_LO + FR_HI) / 2, r"$N\times$", ha="left", va="center",
            fontsize=FS_S, color=INK)
    yt = y0 + FR_HI + 0.07                                    # the state tower's final output
    vseg(ax, xm, y0, Y_FFN + H_FFN, FR_HI + 0.07)
    # every prediction layer reads the FINAL state: ONE wire leaves the top of the state
    # tower (outside its N x frame), runs down the middle of the gap and enters the
    # cross-attention of the prediction layer, which is drawn once inside its own N x
    # frame -- so each of its N repetitions reads that same single source (the label says
    # so).  Contrast (c): one shared frame, layer l <- level l.
    xr = xs + TW + 0.5 * GAPT
    ya = y0 + Y_ATT + H_ATT / 2
    wire(ax, [(xm, yt), (xr, yt), (xr, ya), (xp, ya)], col)
    ax.text(xs, yt + 0.045, "K,V: final state, every layer", ha="left", va="bottom",
            fontsize=FS_S, color=col)
    title(ax, xs - PADL, y0, "d", "Sequential", col, "separate towers")


def _canvas():
    fig = plt.figure(figsize=(W, H), layout="none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.grid(False)
    return fig, ax


def _extent(fn):
    """Horizontal extent (xmin, xmax) of a panel drawn at x=0, from rendered bboxes."""
    fig, ax = _canvas()
    fn(ax, 1.0, 0.2)
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    xs = []
    for a in [*ax.texts, *ax.patches, *ax.lines]:
        bb = a.get_window_extent(r)
        (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        xs += [x0, x1]
    plt.close(fig)
    return min(xs) - 1.0, max(xs) - 1.0


def _row(panels):
    ext = [_extent(fn) for fn in panels]
    return ext, [b - a for a, b in ext]


def draw():
    fig, ax = _canvas()
    M = 0.04                                                  # outer margin, all four sides
    y1, y2 = M + ROW, M
    top = [panel_transformer, panel_sps]
    bot = [panel_two_tower, panel_seq]
    ext_t, w_t = _row(top)
    ext_b, w_b = _row(bot)
    # the bottom row is the wide one: it sets the gutter G, used by both rows
    G = W - 2 * M - sum(w_b)
    for row, ext, w, y in ((bot, ext_b, w_b, y2), (top, ext_t, w_t, y1)):
        x = (W - (sum(w) + G * (len(w) - 1))) / 2              # centre the row
        for fn, (a, _), wi in zip(row, ext, w):
            fn(ax, x - a, y)
            x += wi + G
    return fig


def main():
    PS.save(draw(), "fig_teaser")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
