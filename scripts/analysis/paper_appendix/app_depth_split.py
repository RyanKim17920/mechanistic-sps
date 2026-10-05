"""app_depth_split: per-layer lesion profiles of the three Sequential depth splits.

Three Sequential arms with the same 12 layers split differently between the towers --
3+9, 6+6 (the depth-split reference) and 9+3 (manifest roles sequential3p9, sequential6,
sequential9p3).  Panels: state attention, state MLP and prediction MLP (prediction
attention is printed only); the x-axis is the layer index WITHIN that tower, so each arm
only extends as far as its own tower is deep.  Every seed-rule run that has a JSON is
drawn (paper_data.result_seeds).

JSON keys read
  a12_depth_lesion_<run>.json : lesions[*].{stream, kind, block, delta},
                                geometry.{state_n_layer, pred_n_layer}, control_gate_passed
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paper_data as D  # noqa: E402  (puts src/ and scripts/ on sys.path)
from plotting import paper_style as PS  # noqa: E402  (applies rcParams)
from plotting.paper_style import ARMS  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STEM = "app_depth_split"
SEQ39, SEQ6, SEQ93 = (D.ROLES[r] for r in ("sequential3p9", "sequential6", "sequential9p3"))
SPLITS = (SEQ39, SEQ6, SEQ93)
PANELS = (("state", "attn", "state attention"), ("state", "mlp", "state MLP"),
          ("pred", "mlp", "prediction MLP"))
SHORT = {SEQ39: "3+9", SEQ6: "6+6", SEQ93: "9+3"}


def _lesion(d, stream, kind):
    recs = sorted((r for r in d["lesions"].values()
                   if r["stream"] == stream and r["kind"] == kind), key=lambda r: r["block"])
    return [r["block"] + 1 for r in recs], [r["delta"] for r in recs]


RC7 = {"xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.labelsize": 7.5}


def make():
    with plt.rc_context(RC7):
        return _make()


def _make():
    data = {}
    for k in SPLITS:
        rs = [(r, d) for r, d in D.result_seeds("a12_depth_lesion", k)
              if d.get("control_gate_passed", True)]
        if rs:
            data[k] = rs
        else:
            D.warn(f"{STEM}: no a12 JSON for {k}")
    if not data:
        print(f"SKIPPED {STEM}: no a12 depth-lesion JSON for any depth split")
        return None
    for k, rs in data.items():
        print(f"[{STEM}] {ARMS[k]['label']}: {[r for r, _ in rs]}")

    # three panels: state attention (layer 1 decisive), state MLP and prediction MLP (the
    # shorter tower is load-bearing at every layer).  Prediction attention is printed only.
    fig, axes = plt.subplots(1, len(PANELS), figsize=(PS.FULL_W, 2.2), sharey=True)
    maxb = 1
    for j, (ax, (stream, kind, title)) in enumerate(zip(axes, PANELS)):
        for k, rs in data.items():
            curves = [_lesion(d, stream, kind) for _, d in rs]
            xs = curves[0][0]
            if any(c[0] != xs for c in curves):
                D.warn(f"{STEM}: block grids differ across seeds of {k}")
                continue
            maxb = max(maxb, len(xs))
            m = PS.plot_seeded_line(ax, xs, [c[1] for c in curves], k, ms=3.0)
            above = (j == 0 and k == SEQ6)   # avoid the 9+3 line at layer 7
            ax.annotate(SHORT[k], xy=(xs[-1], m[-1]), xytext=(0, 6) if above else (4, 0),
                        textcoords="offset points", ha="center" if above else "left",
                        va="bottom" if above else "center",
                        fontsize=7.5, color=ARMS[k]["color"])
            print(f"[{STEM}] {ARMS[k]['label']} {stream} {kind}: "
                  f"{np.round(m, 4).tolist()}")
        ax.set_yscale("log")
        ax.set_title(title, pad=2)
        ax.set_xlabel("layer within tower")
        if j == 0:
            ax.set_ylabel("loss increase (nats)")
    for k, rs in data.items():   # not drawn; caption numbers
        c = [_lesion(d, "pred", "attn")[1] for _, d in rs]
        print(f"[{STEM}] (not drawn) {ARMS[k]['label']} pred attn: "
              f"{np.round(np.mean(c, 0), 4).tolist()}")
    for ax in axes:
        ax.set_xticks(range(1, maxb + 1, 2))
        ax.set_xlim(0.6, maxb + 1.6)
    for ax, letter in zip(axes, "abc"):
        PS.panel_letter(ax, letter)

    return PS.save(fig, STEM)


if __name__ == "__main__":
    make()
