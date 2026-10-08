"""Draws the system overview and network architecture figures for the paper.

Every value shown is taken from the code: backend/src/models/mobilevit_v2.py,
backend/src/utils/loss.py, backend/src/utils/graph_postprocess.py, and
figures/real/measurements/efficiency.json. Text is shrunk automatically until
it fits inside its box.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "paper", "figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})

NAVY, BLUE, TEAL, PURPLE = "#1b2a49", "#1f4e9c", "#1d6b52", "#3d2b7a"
GOLD, RED, MAROON, GREY = "#b8860b", "#c0392b", "#8e1b1b", "#5f6368"
LIGHT = {NAVY: "#eef1f7", BLUE: "#edf3fc", TEAL: "#ecf6f1", PURPLE: "#f1eef9",
         GOLD: "#fbf6e9", RED: "#fcefee", MAROON: "#f8ecec", GREY: "#f3f4f5"}


class Canvas:
    def __init__(self, w, h):
        self.fig = plt.figure(figsize=(w, h))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w); self.ax.set_ylim(0, h); self.ax.axis("off")
        self.r = self.fig.canvas.get_renderer()

    def fit(self, t, x0, x1, y0=None, y1=None):
        """Shrink text t until its bounding box lies within the given data limits."""
        inv = self.ax.transData.inverted()
        for _ in range(40):
            bb = inv.transform(t.get_window_extent(self.r))
            ok = bb[0][0] >= x0 and bb[1][0] <= x1
            if y0 is not None:
                ok = ok and bb[0][1] >= y0 and bb[1][1] <= y1
            if ok:
                return t
            t.set_fontsize(t.get_fontsize() * 0.95)
        return t

    def card(self, x, y, w, h, num, title, bullets, color, size=6.2):
        ax = self.ax
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07",
                                    fc="white", ec=color, lw=1.3, zorder=2))
        top = y + h
        if num is not None:
            ax.add_patch(Circle((x + 0.17, top - 0.17), 0.11, fc=color, ec="none", zorder=3))
            ax.text(x + 0.17, top - 0.172, str(num), color="white", fontsize=6.5, weight="bold",
                    ha="center", va="center", zorder=4)
            tx = x + 0.33
        else:
            tx = x + 0.1
        t = ax.text(tx, top - 0.17, title, color=color, fontsize=7.2, weight="bold", ha="left", va="center", zorder=4)
        self.fit(t, tx, x + w - 0.06)
        ax.plot([x + 0.08, x + w - 0.08], [top - 0.33, top - 0.33], color=color, lw=0.6, alpha=0.6, zorder=3)
        body = "\n".join("•  " + b for b in bullets)
        t = ax.text(x + 0.1, top - 0.4, body, fontsize=size, ha="left", va="top", linespacing=1.45, zorder=4, color="#202124")
        self.fit(t, x + 0.08, x + w - 0.06, y + 0.04, top - 0.36)

    def arrow(self, x0, y0, x1, y1, color=BLUE, label=None, lpos=None, lw=1.4, ls="-", rad=0.0, fs=5.6):
        self.ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                         arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22", color=color, lw=lw,
                                         linestyle=ls, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))
        if label:
            lx, ly = lpos if lpos else ((x0 + x1) / 2, (y0 + y1) / 2 + 0.06)
            self.ax.text(lx, ly, label, color=color, fontsize=fs, ha="center", va="bottom", zorder=5,
                         bbox=dict(fc="white", ec="none", pad=0.5))

    def save(self, name):
        self.fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
        self.fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)


def system_overview():
    c = Canvas(7.16, 3.0)
    w, h, yt = 1.55, 1.25, 1.72
    xs = [0.02, 1.88, 3.74, 5.6]
    c.card(xs[0], yt, w, h, 1, "Satellite tile", [
        "0.5 m per pixel, RGB", "1024 x 1024 tile", "Training: random", "256 x 256 crops, flips"], NAVY)
    c.card(xs[1], yt, w, h, 2, "MobileViT v2 network", [
        "1.60 M parameters", r"Separable attention $O(N\cdot d)$", r"Strip conv $1{\times}3 \rightarrow 3{\times}1$",
        "Channel shift, no weights", "Attention gated skips"], BLUE)
    c.card(xs[2], yt, w, h, 3, "Postprocessing", [
        "Four flip TTA", r"Hysteresis $T_h{=}0.35,\ T_l{=}0.12$", r"Closing $5{\times}5$",
        r"Gap bridging $d{\leq}220$ px", r"End point angle $\theta{\leq}65^\circ$"], TEAL)
    c.card(xs[3], yt, w, h, 4, "Graph, deployment", [
        r"Skeleton to $G=(V,E)$", r"Betweenness $C_B(v)$", r"Resilience index $R$",
        "ONNX Runtime, 6.9 MB", "0.91 s per tile on CPU"], PURPLE)
    ym = yt + h / 2
    c.arrow(xs[0] + w, ym, xs[1], ym, NAVY, "RGB", (xs[0] + w + 0.155, ym + 0.05))
    c.arrow(xs[1] + w, ym, xs[2], ym, BLUE, r"$P$", (xs[1] + w + 0.155, ym + 0.05))
    c.arrow(xs[2] + w, ym, xs[3], ym, TEAL, "mask", (xs[2] + w + 0.155, ym + 0.05))

    yb, hb = 0.02, 0.98
    c.card(0.02, yb, w, hb, 5, "Ground truth", [r"DeepGlobe road mask $Y$", "Binary, 0.5 m per pixel"], GOLD)
    c.card(1.88, yb, w, hb, 6, "Soft skeleton", [
        "Min and max pooling", r"$K=10$ iterations", r"Gives $S(P)$ and $S(Y)$"], RED)
    c.card(3.74, yb, 3.41, hb, 7, "Composite loss and model selection", [
        r"$\mathcal{L}=\alpha(e)\,\mathcal{L}_{BCE}+\beta\,\mathcal{L}_{Dice}+(1-\alpha-\beta)\,\mathcal{L}_{clDice}$",
        r"$\alpha$ decays from 0.50 to 0.15 over 40 epochs, $\beta=0.35$",
        "Reject epochs that mark more than 20% of pixels as road"], MAROON)
    yb_mid = yb + hb / 2
    c.arrow(0.02 + w, yb_mid, 1.88, yb_mid, GOLD, r"$Y$", (0.02 + w + 0.155, yb_mid + 0.05))
    c.arrow(1.88 + w, yb_mid, 3.74, yb_mid, RED, r"$S(\cdot)$", (1.88 + w + 0.155, yb_mid + 0.05))
    c.arrow(xs[1] + w * 0.4, yt, xs[1] + w * 0.4, yb + hb, BLUE, r"$P$", (xs[1] + w * 0.4 - 0.11, (yt + yb + hb) / 2 - 0.05))
    c.arrow(4.3, yb + hb, xs[1] + w * 0.85, yt, MAROON, "gradient", (4.1, (yt + yb + hb) / 2 - 0.04), ls="--")
    c.save("fig_system")


def network():
    c = Canvas(7.16, 4.45)
    ax = c.ax
    # ---- panel (a): strip convolution ----
    px, py, pw, ph = 0.02, 2.62, 3.5, 1.8
    ax.add_patch(FancyBboxPatch((px, py), pw, ph, boxstyle="round,pad=0,rounding_size=0.06", fc="white", ec=BLUE, lw=0.9, ls="--"))
    ax.text(px + pw / 2, py + ph - 0.13, "(a) Strip convolution, adopted from HPLNet", color=BLUE, fontsize=7.2,
            weight="bold", ha="center", va="center")
    s = 0.22
    def grid(x0, y0, nr, nc, labels, fc):
        for i in range(nr):
            for j in range(nc):
                ax.add_patch(Rectangle((x0 + j * s, y0 - (i + 1) * s), s, s, fc=fc, ec=NAVY, lw=0.6))
                ax.text(x0 + j * s + s / 2, y0 - (i + 0.5) * s, labels[i][j], fontsize=5.2, ha="center", va="center")
    gy = py + ph - 0.42
    grid(px + 0.25, gy, 3, 3, [[f"$w_{{{i}{j}}}$" for j in range(1, 4)] for i in range(1, 4)], "#dbe7f7")
    ax.text(px + 0.25 + 1.5 * s, gy - 3 * s - 0.12, r"$3{\times}3$: $9\,C_{in}C_{out}$", fontsize=5.6, ha="center", va="top")
    ax.text(px + 1.18, gy - 1.5 * s, r"$\rightarrow$", fontsize=10, ha="center", va="center", color=BLUE)
    grid(px + 1.45, gy - s, 1, 3, [[f"$a_{j}$" for j in range(1, 4)]], "#cfe0f5")
    ax.text(px + 1.45 + 1.5 * s, gy - 3 * s - 0.12, r"$1{\times}3$: $3\,C_{in}C_{out}$", fontsize=5.6, ha="center", va="top")
    ax.text(px + 2.28, gy - 1.5 * s, r"then", fontsize=5.8, ha="center", va="center", color=BLUE)
    grid(px + 2.6, gy, 3, 1, [[f"$b_{i}$"] for i in range(1, 4)], "#cfe0f5")
    ax.text(px + 2.6 + s / 2, gy - 3 * s - 0.12, r"$3{\times}1$: $3\,C_{in}C_{out}$", fontsize=5.6, ha="center", va="top")
    t = ax.text(px + pw / 2, py + 0.2, "Factorised kernel: 6 instead of 9 weights per channel pair,\nstrong response to horizontal and vertical line segments",
                fontsize=5.6, ha="center", va="center", color="#202124")
    c.fit(t, px + 0.05, px + pw - 0.05)

    # ---- panel (b): channel shift ----
    qx = 3.64
    ax.add_patch(FancyBboxPatch((qx, py), pw, ph, boxstyle="round,pad=0,rounding_size=0.06", fc="white", ec=TEAL, lw=0.9, ls="--"))
    ax.text(qx + pw / 2, py + ph - 0.13, "(b) Channel shift, adopted from HPLNet", color=TEAL, fontsize=7.2,
            weight="bold", ha="center", va="center")
    cols = ["#9ec5e8", "#f2c38b", "#b9dcb0", "#d7b8e6", "#e6e6e6"]
    names = ["up", "down", "left", "right", "unchanged"]
    bx, by = qx + 0.22, py + 0.5
    for k in range(5):
        wk = 0.18 if k < 4 else 0.5
        x0 = bx + (0.2 * k if k < 4 else 0.8)
        ax.add_patch(Rectangle((x0, by), wk, 0.8, fc=cols[k], ec=NAVY, lw=0.6))
        ax.text(x0 + wk / 2, by + 0.4, names[k], fontsize=5.0, ha="center", va="center", rotation=0 if k == 4 else 90)
    ax.text(bx + 0.66, by + 0.9, r"Features $Z\in\mathbb{R}^{C\times h\times w}$", fontsize=5.6, ha="center")
    ax.annotate("", xy=(bx + 1.72, by + 0.4), xytext=(bx + 1.38, by + 0.4),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.2))
    cx, cy = qx + 2.55, by + 0.4
    ax.add_patch(Rectangle((cx - 0.25, cy - 0.25), 0.5, 0.5, fc="#f7f7f7", ec=NAVY, lw=0.6))
    for (dx, dy), col in zip([(0, 1), (0, -1), (-1, 0), (1, 0)], cols):
        ax.annotate("", xy=(cx + dx * 0.55, cy + dy * 0.48), xytext=(cx + dx * 0.25, cy + dy * 0.25),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0))
    ax.text(cx, cy, "2 px", fontsize=5.4, ha="center", va="center")
    t = ax.text(qx + pw / 2, py + 0.2, "25% of channels in four groups, each moved by 2 pixels;\nno weights and no multiply operations",
                fontsize=5.6, ha="center", va="center", color="#202124")
    c.fit(t, qx + 0.05, qx + pw - 0.05)

    # ---- panel (c): encoder and decoder ----
    ax.text(0.02, 2.47, "(c) Encoder and decoder for a 256 x 256 input", color=NAVY, fontsize=7.2, weight="bold", ha="left", va="center")
    bw, bh = 1.22, 0.62
    xs = [0.55 + i * 1.33 for i in range(5)]
    ye = 1.62
    enc = [("Stem", "Strip conv, s 2", "32 x 128 x 128"),
           ("Encoder 1", "Inverted residual, s 2", "64 x 64 x 64"),
           ("Encoder 2", "MobileViT v2 (2), IR s 2", "96 x 32 x 32"),
           ("Encoder 3", "MobileViT v2 (2), IR s 2", "128 x 16 x 16"),
           ("Bottleneck", "MobileViT v2 (3)", "128 x 16 x 16")]
    def block(x, y, title, op, shape, color):
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0,rounding_size=0.05", fc=LIGHT[color], ec=color, lw=1.1))
        t1 = ax.text(x + bw / 2, y + bh - 0.12, title, fontsize=6.2, weight="bold", color=color, ha="center", va="center")
        t2 = ax.text(x + bw / 2, y + bh / 2 - 0.02, op, fontsize=5.3, ha="center", va="center")
        ax.plot([x + 0.08, x + bw - 0.08], [y + 0.17, y + 0.17], color=color, lw=0.5, alpha=0.5)
        t3 = ax.text(x + bw / 2, y + 0.085, shape, fontsize=5.3, ha="center", va="center", color="#202124")
        for t in (t1, t2, t3):
            c.fit(t, x + 0.04, x + bw - 0.04)
    ax.text(0.02, ye + bh / 2, "Input\n3 x 256\nx 256", fontsize=5.2, ha="left", va="center")
    c.arrow(0.4, ye + bh / 2, xs[0], ye + bh / 2, NAVY, lw=1.0)
    for i, e in enumerate(enc):
        block(xs[i], ye, *e, BLUE)
        if i:
            c.arrow(xs[i - 1] + bw, ye + bh / 2, xs[i], ye + bh / 2, BLUE, lw=1.0)
    yg, gw, gh = 0.9, 0.95, 0.36
    yd = 0.02
    dec_x = [xs[3], xs[2], xs[1]]
    src = [xs[3], xs[2], xs[0]]
    skip = ["96 x 32 x 32", "64 x 64 x 64", "32 x 128 x 128"]
    dec = [("Decoder 3", "Up x2, concat, strip conv", "96 x 32 x 32"),
           ("Decoder 2", "Up x2, concat, strip conv", "64 x 64 x 64"),
           ("Decoder 1", "Up x2, concat, strip conv", "32 x 128 x 128")]
    for i, x in enumerate(dec_x):
        gx = x + (bw - gw) / 2
        ax.add_patch(FancyBboxPatch((gx, yg), gw, gh, boxstyle="round,pad=0,rounding_size=0.04", fc=LIGHT[GOLD], ec=GOLD, lw=1.1))
        ax.text(gx + gw / 2, yg + gh / 2, f"Gate {3 - i}", fontsize=5.8, weight="bold", color=GOLD, ha="center", va="center")
        if src[i] == x:
            c.arrow(x + 0.3, ye, x + 0.3, yg + gh, GOLD, lw=0.9, ls="--")
            ax.text(x + 0.36, (ye + yg + gh) / 2, "skip\n" + skip[i], fontsize=4.8, color=GOLD, ha="left", va="center")
        else:
            c.arrow(src[i] + bw / 2, ye, gx + 0.15, yg + gh, GOLD, lw=0.9, ls="--")
            ax.text((src[i] + bw / 2 + gx + 0.15) / 2 - 0.08, (ye + yg + gh) / 2 - 0.1, "skip " + skip[i], fontsize=4.8, color=GOLD, ha="right", va="center")
        block(x, yd, *dec[i], TEAL)
        c.arrow(gx + gw / 2, yg, gx + gw / 2, yd + bh, GOLD, lw=0.9)
    c.arrow(xs[4] + bw / 2, ye, xs[3] + bw, yd + bh * 0.75, TEAL, lw=1.0)
    ax.text(xs[4] + 0.75, 1.05, "up x2,\ngating\nsignal", fontsize=4.8, color=TEAL, ha="left", va="center")
    c.arrow(xs[3], yd + bh / 2, xs[2] + bw, yd + bh / 2, TEAL, lw=1.0)
    c.arrow(xs[2], yd + bh / 2, xs[1] + bw, yd + bh / 2, TEAL, lw=1.0)
    block(xs[0] - 0.53, yd, "Head", "Up x2, 1x1 conv, sigmoid", "1 x 256 x 256", PURPLE)
    c.arrow(xs[1], yd + bh / 2, xs[0] - 0.53 + bw, yd + bh / 2, TEAL, lw=1.0)
    c.save("fig_network")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    system_overview()
    network()
    print("saved to", os.path.abspath(OUT))
