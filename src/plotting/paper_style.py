"""One visual language for every BODY figure of the paper.

Every paper generator (scripts/analysis/paper_figures.py, make_tables.py, paper_appendix/)
imports this module and nothing else for colour, marker, label or layout decisions, so
the same arm looks the same in every figure.

ENCODING (fixed; do not re-derive per figure)
  * hue          = architecture / read-alignment family
  * marker fill  = cross-stream weight SHARING: FILLED = separate towers, OPEN = shared weights
                   ("tied" is reserved for the LM head <-> token table; see ARMS[k]['head'])
                   (SPS runs both streams through ONE stack, so every SPS arm is open;
                   the "untied head" variants only untie the LM head, not the streams)
  * line style   = stream: solid = state, dashed = prediction
  * attention vs MLP never gets its own hue (use hatching / line weight instead)

Raw run names never reach a figure: every artist label comes from `ARMS[...]["label"]`
(or `STREAM_LABEL`); the arms themselves are listed in scripts/analysis/paper_manifest.yaml.

Importing this module registers the vendored fonts (src/plotting/fonts/) and applies the
rcParams (7 pt base, pdf fonttype 42,
left+bottom spines, light y grid).  Figures are authored at final size
(`FULL_W` inches) with constrained_layout and saved WITHOUT bbox="tight".
"""
from __future__ import annotations

import warnings
from pathlib import Path

import plotting.mpl_cache  # noqa: F401  (pin MPLCONFIGDIR before matplotlib)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib import font_manager  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "paper" / "figures"

# ---------------------------------------------------------------------------------------
# geometry / typography
# ---------------------------------------------------------------------------------------
FULL_W = 5.5
GRID = "#E6E6E6"
INK = "#222222"
MUTED = "#6E6E6E"

RC = {
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "axes.titleweight": "regular",
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.borderaxespad": 0.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.6,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "lines.linewidth": 1.3,
    "lines.markersize": 4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 150,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.02,
    "figure.constrained_layout.w_pad": 0.02,
}
# The paper's fonts ship in src/plotting/fonts/: Nimbus Roman (text) and the STIX fonts
# that mathtext uses. matplotlib embeds a figure's fonts in the order of their file paths,
# so loading all of them from this one directory makes the PDF bytes independent of where
# the repo, the venv or any system font is installed.
FONT_DIR = Path(__file__).resolve().parent / "fonts"
for _font in sorted(FONT_DIR.rglob("*.[ot]tf")):
    font_manager.fontManager.addfont(str(_font))
# findfont keeps the first of equally good matches: put the vendored copies first
font_manager.fontManager.ttflist.sort(key=lambda f: not f.fname.startswith(str(FONT_DIR)))
font_manager.fontManager._findfont_cached.cache_clear()
plt.rcParams.update(RC)

# line widths / alphas for the seed rule
MEAN_LW = 1.5
MEAN_MS = 4.2
SEED_BAND_ALPHA = 0.20   # the seed rule: n>=2 shades the min-max seed range, mean on top

# ---------------------------------------------------------------------------------------
# arms: scripts/analysis/paper_manifest.yaml is the one list of paper models, seeds and roles
# ---------------------------------------------------------------------------------------
PALETTE = {"transformer": "#7F7F7F", "sps": "#E69F00", "afsps": "#D55E00",
           "two_tower": "#0072B2", "asym": "#56B4E9", "sequential": "#009E73",
           "sequential_light": "#7FCDB4", "sequential_mid": "#00805E",
           "sequential_dark": "#004D38"}
C_TRANSFORMER = PALETTE["transformer"]

MANIFEST = yaml.safe_load((REPO / "scripts" / "analysis" / "paper_manifest.yaml").read_text())
ARMS: dict[str, dict] = {k: {**a, "color": PALETTE[a["palette"]]}
                         for k, a in MANIFEST["arms"].items()}
BODY_SPS = MANIFEST["body"]["sps"]
BODY_TRANSFORMER = MANIFEST["body"]["transformer"]
BODY_HEAD = MANIFEST["body"]["head"]
# SEED RULE for every reported mean: exactly N_MEAN_SEEDS seeds, taken in the arm's
# `mean_seeds` order (default: `seeds` order) among the runs that have the data; extra seeds
# appear only in the full roster (paper_data.mean_runs).
N_MEAN_SEEDS = MANIFEST["n_mean_seeds"]


def body_arms() -> list[str]:
    """Transformer, SPS, Two-tower 12+12, Sequential 12+12, Two-tower 6+6, Sequential 6+6."""
    return list(MANIFEST["body"]["arms"])


def body_label(key: str) -> str:
    """Body-figure name: drops the qualifier the caption states once (head group, variant)."""
    return ARMS[key].get("body_label", ARMS[key]["label"])


STREAM_LS = {"state": "-", "pred": "--", "single": "-"}
STREAM_LABEL = {"state": "state", "pred": "prediction", "single": "single stream"}

# seed run name -> primary key
SEED_TO_ARM = {s: k for k, a in ARMS.items() for s in a["seeds"]}


def arm_of(run: str) -> str:
    return SEED_TO_ARM[run]


def marker_kw(key: str, ms: float = MEAN_MS, alpha: float = 1.0, marker: str | None = None,
              color: str | None = None) -> dict:
    """Marker kwargs obeying the fill rule (filled = separate towers, open = shared weights).
    `marker` / `color` override the arm's canonical shape / hue for one figure without
    touching the fill rule."""
    a = ARMS[key]
    c = color or a["color"]
    return {"marker": marker or a["marker"], "markersize": ms, "markeredgecolor": c,
            "markerfacecolor": ("white" if a["shared"] else c), "markeredgewidth": 0.9,
            "color": c, "alpha": alpha}


def mean_seed_order(key: str) -> list[str]:
    a = ARMS[key]
    return list(a.get("mean_seeds", a["seeds"]))


# ---------------------------------------------------------------------------------------
# seed rule helpers
# ---------------------------------------------------------------------------------------
def plot_seeded_line(ax, x, ys, key, ls="-", label=None, zorder=3, mean_marker=True,
                     ms=None, marker=None, color=None, lw=None, band_alpha=None,
                     band_edge_lw=0.0, range_bars=False):
    """ys: list of per-seed y arrays on a shared x.  SEED RULE: n>=2 shades the min-max
    seed range as a light band (no per-seed line/dots) plus the mean as the line; n==1
    draws the single line with no band.  A per-seed dot at each x is easy to miss at this
    print size, so the range is a filled band instead -- still exactly two seeds's worth of
    information (min and max), just visible.  `marker` overrides the arm's canonical shape
    (e.g. to distinguish two streams of the same arm) for the mean marker.  `color`
    overrides the arm's canonical hue (e.g. to pull one arm's colour out of a clash with
    another element in one figure) for both the band and the mean line/marker.
    Visibility options for seed ranges narrower than the mean line itself: `lw` thins the
    mean line, `band_alpha` overrides SEED_BAND_ALPHA, `band_edge_lw` > 0 outlines the band's
    min and max edges in the arm hue, and `range_bars` adds a vertical min-max bar at every
    x (a band collapses to zero width at the last x, so an endpoint range is otherwise
    invisible)."""
    ys = [np.asarray(y, float) for y in ys]
    x = np.asarray(x, float)
    c = color or ARMS[key]["color"]
    n = len(ys)
    ms = MEAN_MS * 0.8 if ms is None else ms
    st = np.vstack(ys)
    if n >= 2:
        with warnings.catch_warnings():   # an all-NaN column (an exact zero on a log axis)
            warnings.simplefilter("ignore", RuntimeWarning)
            lo = np.nanmin(st, axis=0)
            hi = np.nanmax(st, axis=0)
        ok = ~(np.isnan(lo) | np.isnan(hi))
        if ok.any():
            ax.fill_between(x[ok], lo[ok], hi[ok], color=c,
                            alpha=SEED_BAND_ALPHA if band_alpha is None else band_alpha, lw=0,
                            zorder=zorder - 2)
            if band_edge_lw > 0:
                for edge in (lo, hi):
                    ax.plot(x[ok], edge[ok], color=c, lw=band_edge_lw, alpha=0.7, ls="-",
                            zorder=zorder - 1, solid_capstyle="butt")
            if range_bars:
                ax.vlines(x[ok], lo[ok], hi[ok], color=c, lw=1.0, alpha=0.6,
                          zorder=zorder - 1)
    with np.errstate(all="ignore"):   # an all-NaN column (e.g. an exact zero cut from a log axis) stays NaN
        m = np.where(np.isnan(st).all(0), np.nan, np.nanmean(np.where(np.isnan(st), 0, st), 0)
                     * st.shape[0] / np.maximum((~np.isnan(st)).sum(0), 1))
    kw = marker_kw(key, ms=ms, marker=marker, color=color) if mean_marker else {"color": c}
    ax.plot(x, m, ls=ls, lw=MEAN_LW if lw is None else lw, label=label, zorder=zorder, **kw)
    return m


def lighten(hex_color: str, frac: float = 0.55) -> str:
    """Blend `hex_color` toward white by `frac` (0 = unchanged, 1 = white).  Used to fade an
    arm's own hue (e.g. de-emphasised curves/labels) without falling back to a neutral grey
    that could be mistaken for a different arm's colour (e.g. the Transformer's grey)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * frac) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def panel_letter(ax, letter, x=-0.02, y=1.0):
    """Bold '(a)' at the axes' top-left, outside the data area."""
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction",
                xytext=(-2, 4), textcoords="offset points", ha="right", va="bottom",
                fontsize=8, fontweight="bold", color=INK, annotation_clip=False)


def check_inside(fig, groups, name):
    """Assert-style report: every text artist's rendered bbox lies inside the figure canvas,
    and the tight bboxes of the panels in `groups` do not intersect.  Prints the result."""
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    bad = []
    # tick labels outside the view interval are never drawn, so they are checked through the
    # axes' tight bbox (drawn ticks only); every other text artist individually
    ticklabs = {id(t) for ax in fig.axes for t in ax.get_xticklabels(which="both")
                + ax.get_yticklabels(which="both")}
    items = [(t.get_text(), t.get_window_extent(r)) for t in fig.findobj(
        lambda a: hasattr(a, "get_text") and a.get_visible() and a.get_text()
        and id(a) not in ticklabs)]
    items += [(f"axes {i} tight bbox", ax.get_tightbbox(r)) for i, ax in enumerate(fig.axes)]
    for txt, e in items:
        if e.x0 < fb.x0 - 0.5 or e.x1 > fb.x1 + 0.5 or e.y0 < fb.y0 - 0.5 or e.y1 > fb.y1 + 0.5:
            bad.append(("outside figure", txt, tuple(round(float(v), 1) for v in e.extents)))
    tbs = [ax.get_tightbbox(r) for ax in groups]
    for i in range(len(tbs)):
        for j in range(i + 1, len(tbs)):
            if tbs[i].overlaps(tbs[j]):
                bad.append(("panels overlap", i, j))
    print(f"[{name}] layout check: " + ("OK, all text inside canvas, no panel overlap"
                                        if not bad else f"PROBLEMS {bad}"))
    return not bad


def save(fig, stem: str, outdir: Path | str | None = None):
    """-> <outdir>/<stem>.pdf, without a CreationDate so that it is reproducible."""
    out = Path(outdir) if outdir else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / f"{stem}.pdf"
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(f"WROTE {pdf}", flush=True)
    return pdf
