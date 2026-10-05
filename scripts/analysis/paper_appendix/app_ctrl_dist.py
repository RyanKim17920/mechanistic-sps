"""app_ctrl_dist: is the previous-token-head ablation cost larger than that of random
head sets?  100-draw control distributions for every body run, with all layers eligible
as controls and with layer 1 excluded (post hoc).

For each run the state tower's top-3 previous-token heads are jointly zeroed and the
cost is the HEADLINE metric (ctrl_summary.headline): top-1 accuracy lost on natural-text
induction tokens the intact model predicts correctly (induction_clean_top1) -- the same
number Tab. circuit and its p values use (k asserted equal via p_plus1).  Each control draw zeroes 3 other state-tower heads on the
same sequences:
  random_any       3 heads uniformly at random (never a previous-token head)
  pattern_matched  per treated head, one of the k=5 nearest non-previous-token heads in
                   (entropy, log distance, sink mass, output norm)
  -exL1            the same, with every layer-1 head removed from the control pool.
                   POST HOC: chosen after seeing that the controls that beat the
                   previous-token set almost always contained a layer-1 head.
Two panels (random / pattern-matched pool); one row per body model (one run each), with
an upper strip for the all-layer pool (filled dots) and a lower one with layer 1
excluded (open dots), both in the model's colour (grey is the Transformer's colour, so
the pool is encoded by fill, not by a grey strip).  Each faint dot is one control draw, placed at its cost RELATIVE to the
run's own ablation cost (so the ablation sits at 1 in every row).  Draws beyond the
x-range are collected in one edge marker per row.  The right-hand number is k.

p+1 (add-one permutation p-value):   p = (k + 1) / (n + 1),
k = number of control draws whose cost is >= the ablation cost, n = number of draws.
It is recomputed from the draws and asserted equal to the stored one; the ablation cost
is asserted identical between the all-layer and -exL1 files of a run.

Only a9 (natural-text induction tokens) is drawn, one run per body model; the k of the
other five runs is printed.  The a8 exL1 files
(head patching, two-tower family only, a different quantity) are not.

Reads scripts/analysis/results/
  a9_patching_<run>_ctrl-random_any-n100[-exL1].json
  a9_patching_<run>_ctrl-pattern_matched-n100-k5[-exL1].json
keys: -ctrl_draws[i].sets.induction_clean_top1.d_acc_control (per-draw cost),
      ctrl_summary.headline.{metric, prev_cost, p_plus1, n_draws},
      -natural.induction_clean_top1.d_acc (ablation cost, cross-checked against prev_cost),
      n_ctrl_draws, ctrl_mode, ctrl_exclude_layers.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paper_data as D  # noqa: E402  (puts src/ and scripts/ on sys.path)
from plotting import paper_style as PS  # noqa: E402  (applies rcParams)
from plotting.paper_style import ARMS, MANIFEST  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

STEM = "app_ctrl_dist"
SET = "induction_clean_top1"   # headline metric: accuracy lost on these tokens
XMAX = 1.6          # control cost / ablation cost; draws beyond collect at the edge
XMIN = -0.15

# one run per body model, top to bottom; the k of the other runs with controls is printed
RUNS = MANIFEST["ctrl_dist"]["runs"]
ALL_RUNS = RUNS + MANIFEST["ctrl_dist"]["printed"]
# two panels (random / pattern-matched pool); each model row has two strips: control heads
# drawn from all layers (filled dots, upper) and with layer 1 excluded (open dots, lower),
# both in the model colour.
POOLS = [("random_any", "Random control heads",
          ("ctrl-random_any-n100", "ctrl-random_any-n100-exL1")),
         ("pattern_matched", "Pattern-matched control heads",
          ("ctrl-pattern_matched-n100-k5", "ctrl-pattern_matched-n100-k5-exL1"))]
PANELS = [(m, suf, "") for m, _, sufs in POOLS for suf in sufs]
FS = 7.0   # >= 7pt at printed size (this figure is included at \linewidth = 5.5in, 1:1)


def row_label(run):
    key = PS.arm_of(run)
    lab = PS.body_label(key) if key in PS.body_arms() else ARMS[key]["label"].split(",")[0]
    if run not in RUNS:
        seeds = ARMS[key]["seeds"]
        lab += f", seed {seeds.index(run) + 1}"
    return lab


def extract(d, mode, exl1):
    """Headline metric (same as make_tables.ctrl_p / Tab. circuit): top-1 accuracy lost on
    induction tokens the intact model predicts correctly (induction_clean_top1)."""
    assert d["ctrl_mode"] == mode, (d["run"], d["ctrl_mode"], mode)
    assert (d.get("ctrl_exclude_layers") or []) == ([1] if exl1 else []), d["run"]
    h = d["ctrl_summary"]["headline"]
    assert h["metric"] == "accuracy lost on " + SET, (d["run"], h["metric"])
    costs = np.array([-x["sets"][SET]["d_acc_control"] for x in d["ctrl_draws"]
                      if x["sets"].get(SET, {}).get("n")], float)
    prev = float(h["prev_cost"])
    assert abs(prev + d["natural"][SET]["d_acc"]) < 1e-9, d["run"]
    assert len(costs) == h["n_draws"] == d["n_ctrl_draws"], d["run"]
    k = int((costs >= prev - 1e-9).sum())
    p = (k + 1) / (len(costs) + 1)
    assert abs(p - h["p_plus1"]) < 1e-9, (d["run"], p, h["p_plus1"])
    return costs, prev, k, p


def make():
    data = {}
    for run in ALL_RUNS:
        for mode, suffix, _ in PANELS:
            d = D.result("a9_patching", f"{run}_{suffix}")
            if d is None:
                D.warn(f"{STEM}: no {suffix} file for {run} (row left empty)")
                continue
            data[(run, suffix)] = extract(d, mode, suffix.endswith("exL1"))
    if not data:
        print(f"SKIPPED {STEM}: no a9 n100 control files")
        return None
    for r in ALL_RUNS:   # the ablation itself does not depend on the control pool
        prevs = {round(data[(r, s)][1], 12) for _, s, _ in PANELS if (r, s) in data}
        assert len(prevs) <= 1, (r, prevs)
    runs = [r for r in RUNS if any((r, s) in data for _, s, _ in PANELS)]

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(PS.FULL_W, 3.1), sharey=True,
                             layout="constrained")
    ys = np.arange(len(runs))[::-1] * 1.0
    DY = 0.2   # half-offset of the two strips inside a row
    for ax, (_mode, ttl, sufs) in zip(axes, POOLS):
        for y, run in zip(ys, runs):
            c_arm = ARMS[PS.arm_of(run)]["color"]
            for suffix, dy, filled in ((sufs[0], DY, True), (sufs[1], -DY, False)):
                if (run, suffix) not in data:
                    continue
                costs, prev, k, p = data[(run, suffix)]
                r = costs / prev
                inside = r[r <= XMAX]
                jit = rng.uniform(-0.11, 0.11, len(inside))
                if filled:
                    ax.plot(np.clip(inside, XMIN, None), y + dy + jit, ls="none", marker="o",
                            ms=1.8, mfc=c_arm, mec="none", alpha=0.5, zorder=2)
                else:
                    ax.plot(np.clip(inside, XMIN, None), y + dy + jit, ls="none", marker="o",
                            ms=2.1, mfc="none", mec=c_arm, mew=0.45, alpha=0.6, zorder=2)
                if int((r > XMAX).sum()):
                    ax.plot([XMAX], [y + dy], ls="none", marker=">", ms=4.0,
                            mfc=c_arm if filled else "white", mec=c_arm, mew=0.8,
                            zorder=3, clip_on=False)
                ax.text(1.06, y + dy, str(k), transform=ax.get_yaxis_transform(),
                        ha="left", va="center", fontsize=FS,
                        color=PS.INK if k else "#C8C8C8",
                        fontweight="bold" if (k and filled) else "normal")
        ax.axvline(1.0, color=PS.INK, lw=1.0, zorder=3)
        ax.set_xlim(XMIN, XMAX)
        ax.set_xticks([0, 0.5, 1, 1.5], ["0", "0.5", "1", "1.5"])
        ax.tick_params(labelsize=FS)
        ax.tick_params(axis="y", length=0)
        ax.set_title(ttl, pad=3, fontsize=FS + 0.5)
        ax.grid(False)
        for y in ys[:-1]:
            ax.axhline(y - 0.5, color=PS.GRID, lw=0.5, zorder=0)
        ax.text(1.06, len(runs) - 0.45, "k", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=FS, color=PS.MUTED, style="italic")
        ax.set_xlabel("control cost / ablation cost", fontsize=FS)
    axes[0].set_yticks(ys, [row_label(r) for r in runs])
    for t, r in zip(axes[0].get_yticklabels(), runs):
        c = ARMS[PS.arm_of(r)]["color"]
        t.set_color(PS.MUTED if c == PS.C_TRANSFORMER else c)
        t.set_fontsize(FS)
    axes[0].set_ylim(-0.55, len(runs) - 0.45)
    # strip key below the panels (off the data): fill encodes the control pool
    handles = [Line2D([], [], ls="none", marker="o", ms=3.2, mfc=PS.INK, mec=PS.INK,
                      label="control heads from all layers (upper strip)"),
               Line2D([], [], ls="none", marker="o", ms=3.2, mfc="none", mec=PS.INK, mew=0.7,
                      label="layer 1 excluded, post hoc (lower strip)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False,
               fontsize=FS, handletextpad=0.3, columnspacing=1.6)
    for run in ALL_RUNS:
        if (run, PANELS[0][1]) not in data:
            continue
        print(f"[{STEM}] {row_label(run):32s} prev={data[(run, PANELS[0][1])][1]:.4f} k:",
              {s.replace('ctrl-', ''): data[(run, s)][2] for _, s, _ in PANELS
               if (run, s) in data}, "(drawn)" if run in runs else "(table only)")
    fig.canvas.draw()
    PS.check_inside(fig, list(axes), STEM)
    return PS.save(fig, STEM)


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    make()
