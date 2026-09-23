"""
Figures for the internship report (docs/report/Internship_Report.docx).

Two are diagrams of the implemented system (they describe code, not results); the other two
are built only from real model outputs already on disk:
  fig_r_collapse_panels   the output panels of figures/real/fig_collapse.png, without its histogram
  fig_r_outputs           sample tiles and the deployed pipeline's road mask (deck_assets.py)

Usage (repo root):  python scripts/report/report_figures.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG = os.path.join(REPO, "figures", "real")
OUT = os.path.join(FIG, "report")
TILES = ["100034", "117991", "115714", "102408"]

INK, GREY, FILL, KEY = "#1a1a1a", "#6b6b6b", "#f2f2f2", "#dcdcdc"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})


def node(ax, x, y, w, h, label, key=False, size=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=KEY if key else FILL, ec=INK, lw=0.7))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size, color=INK,
            linespacing=1.25, fontweight="bold" if key else "normal")


def link(ax, p, q, dashed=False, color=INK):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.7, color=color,
                                 linestyle=(0, (3, 2)) if dashed else "-", shrinkA=0, shrinkB=0))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def fig_pipeline():
    W_, H_ = 6.6, 3.05
    fig, ax = canvas(W_, H_)
    bh, gap = 0.62, 0.17
    lanes = [
        ("Training", 2.12, ["DeepGlobe\ntraining tiles", "Augmentation\n256² crops, flips,\ncolour jitter",
                            "MobileViT v2\nnetwork", "Combined loss\nBCE + Dice\n+ clDice",
                            "Checkpoint\nselection\nval IoU + gate"]),
        ("Inference", 0.72, ["Satellite\ntile", "4 flip\nTTA", "Trained\nnetwork\n(ONNX)", "Road\nprobability\nmap",
                             "Hysteresis\n+ closing", "Canopy gap\nbridging", "Road\nmask"]),
    ]
    boxes = {}
    for name, y, steps in lanes:
        n = len(steps)
        bw = (W_ - (n - 1) * gap) / n
        ax.text(0, y + bh + 0.1, name.upper(), fontsize=8, color=GREY, fontweight="bold")
        for i, s in enumerate(steps):
            x = i * (bw + gap)
            node(ax, x, y, bw, bh, s, key=("MobileViT" in s or "bridging" in s), size=8)
            boxes[(name, i)] = (x, y, bw)
            if i:
                link(ax, (x - gap, y + bh / 2), (x, y + bh / 2))
    x, y, bw = boxes[("Training", 4)]
    xi, yi, bwi = boxes[("Inference", 2)]
    # selected checkpoint -> exported network used at inference
    ax.plot([x + bw / 2, x + bw / 2], [y, 1.72], color=INK, lw=0.7, ls=(0, (3, 2)))
    ax.plot([x + bw / 2, xi + bwi / 2], [1.72, 1.72], color=INK, lw=0.7, ls=(0, (3, 2)))
    link(ax, (xi + bwi / 2, 1.72), (xi + bwi / 2, yi + bh), dashed=True)
    ax.text((x + bw / 2 + xi + bwi / 2) / 2, 1.77, "ONNX export (sigmoid in graph, dynamic H × W)",
            ha="center", va="bottom", fontsize=7.5, color=GREY)
    ax.text(W_ / 2, 0.4, "Shaded boxes: the two core parts, the network and canopy gap bridging.",
            ha="center", fontsize=7.5, color=GREY)
    return fig


def fig_network():
    W_, H_ = 6.6, 2.95
    fig, ax = canvas(W_, H_)
    enc = [("Input", "3 × H"), ("Strip conv\nstem", "32 × H/2"), ("MV2 ↓", "64 × H/4"),
           ("MobileViT v2", "64 × H/4"), ("MV2 ↓ +\nMobileViT v2", "96 × H/8"),
           ("MV2 ↓ +\nMobileViT v2\nbottleneck", "128 × H/16")]
    n, gap = len(enc), 0.16
    bw, bh = (W_ - (n - 1) * gap) / n, 0.78
    ye, yd = 1.78, 0.42
    xs = [i * (bw + gap) for i in range(n)]
    ax.text(0, ye + bh + 0.1, "ENCODER", fontsize=8, color=GREY, fontweight="bold")
    ax.text(xs[1], yd + bh + 0.1, "DECODER", fontsize=8, color=GREY, fontweight="bold")
    for i, (t, d) in enumerate(enc):
        node(ax, xs[i], ye, bw, bh, f"{t}\n{d}", key="MobileViT" in t, size=7.8)
        if i:
            link(ax, (xs[i] - gap, ye + bh / 2), (xs[i], ye + bh / 2))
    dec = [("Up + 1×1 conv", "1 × H"), ("Up + strip conv", "32 × H/2"), ("Up + strip conv", "64 × H/4"),
           ("Up + strip conv", "96 × H/8")]
    dxs = xs[1:5]
    for (t, d), x in zip(dec, dxs):
        node(ax, x, yd, bw, bh, f"{t}\n{d}", size=7.8)
    link(ax, (xs[5] + bw / 2, ye), (dxs[3] + bw, yd + bh / 2))
    for i in (3, 2, 1):
        link(ax, (dxs[i], yd + bh / 2), (dxs[i - 1] + bw, yd + bh / 2))
    for i, src in ((1, 1), (2, 3), (3, 4)):          # stem, stage-2 and stage-3 outputs feed the decoder
        link(ax, (xs[src] + bw / 2, ye), (dxs[i] + bw / 2, yd + bh), dashed=True, color=GREY)
        mx, my = (xs[src] + dxs[i]) / 2 + bw / 2, (ye + yd + bh) / 2
        ax.text(mx, my, "AG", ha="center", va="center", fontsize=7, color=INK,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=GREY, lw=0.6))
    ax.text(0, 0.05, "Dashed: skip connections, each filtered by an attention gate (AG). "
                     "Labels give channels × spatial size for an H × H input.", fontsize=7.5, color=GREY)
    return fig


def fig_collapse_panels():
    """The three output columns of fig_collapse.png, cut before its histogram panel."""
    im = np.asarray(Image.open(os.path.join(FIG, "fig_collapse.png")).convert("RGB"))
    white = (im > 245).all(axis=2).all(axis=0)
    w = im.shape[1]
    # first fully white column gap after 60 % of the width separates the panels from the histogram
    cut = next(x for x in range(int(w * 0.6), w) if white[x])
    rows = ~(im[:, :cut] > 245).all(axis=2).all(axis=1)
    bottom = np.nonzero(rows)[0].max() + 6
    return Image.fromarray(im[:bottom, :cut])


def fig_outputs():
    fig, axes = plt.subplots(2, 4, figsize=(6.6, 3.55), gridspec_kw=dict(wspace=0.04, hspace=0.06))
    for j, t in enumerate(TILES):
        tile = Image.open(os.path.join(FIG, "deck", f"tile_{t}.jpg"))
        mask = np.asarray(Image.open(os.path.join(FIG, "deck", f"mask_{t}.png")))
        axes[0, j].imshow(tile)
        axes[1, j].imshow(mask, cmap="gray", vmin=0, vmax=255, interpolation="antialiased")
        axes[1, j].set_xlabel(f"Tile {t}  ({100 * (mask > 0).mean():.1f}% road)", fontsize=8.5)
        for ax in axes[:, j]:
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#bbbbbb"); s.set_linewidth(0.5)
    axes[0, 0].set_ylabel("Input image", fontsize=9)
    axes[1, 0].set_ylabel("Model output", fontsize=9)
    return fig


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, make in (("fig_r_pipeline", fig_pipeline), ("fig_r_network", fig_network),
                       ("fig_r_outputs", fig_outputs)):
        fig = make()
        fig.savefig(os.path.join(OUT, f"{name}.png"))
        plt.close(fig)
    fig_collapse_panels().save(os.path.join(OUT, "fig_r_collapse_panels.png"))
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
