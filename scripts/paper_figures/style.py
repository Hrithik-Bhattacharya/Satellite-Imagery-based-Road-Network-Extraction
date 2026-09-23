"""
Shared figure style for every paper figure (local and Kaggle-generated).

Palette: the dataviz reference categorical palette, validated on a white surface
(scripts/validate_palette.js: all hard gates pass; aqua/yellow sit below 3:1
contrast, so bars in those colors always carry visible value labels).
Colors follow the model, never its rank, in every figure.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CONTEXT_GRAY = "#b9b7ae"   # non-highlighted reference bars

# Entity -> color, fixed across all figures.
MODEL_COLORS = {
    "final": BLUE,        # v2, run 2, epoch 53 (best_model_v2.pth)
    "baseline": ORANGE,   # June baseline, no attention gates (best_model_new.pth)
    "run1": AQUA,         # v2, run 1, epoch 18
    "collapsed": YELLOW,  # v2, collapsed run, epoch 46
}
MODEL_LABELS = {
    "final": "Proposed (v2, final)",
    "baseline": "Baseline (no gates)",
    "run1": "v2, run 1 (ep. 18)",
    "collapsed": "v2, collapsed (ep. 46)",
}

# Error maps: correct pixels recede to neutral ink so errors carry the color.
TP_COLOR, FP_COLOR, FN_COLOR = "#52514e", ORANGE, BLUE

# Paper widths (IEEE): single column 3.5 in, double column 7.16 in.
COL_W, PAGE_W = 3.5, 7.16


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.facecolor": SURFACE,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "lines.linewidth": 1.5,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,   # embed TrueType -- required by most IEEE/Overleaf checks
        "ps.fonttype": 42,
    })


def save(fig, out_dir, name):
    """Writes PNG (300 DPI, for slides/Word) and PDF (vector, for LaTeX)."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)


def panel_label(ax, letter):
    ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, fontweight="bold", color=INK)


def image_axes(ax):
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
