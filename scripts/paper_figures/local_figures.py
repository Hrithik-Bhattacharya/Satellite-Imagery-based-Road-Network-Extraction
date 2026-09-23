"""
Paper figures that can be produced from data available locally -- every panel is
a real measurement or a real model output; nothing is simulated.

  fig_efficiency          measured params / FLOPs / CPU latency (needs benchmark_efficiency.py first)
  fig_baseline_training   the June baseline run's real training log (wandb_local.json)
  fig_pipeline_stages     real tile through probability -> hysteresis -> canopy-gap bridging
  fig_collapse            collapsed vs. fixed checkpoint on real tiles
  fig_resolution          native-resolution vs. 256-resize validation protocol on real tiles

Accuracy against ground truth needs the DeepGlobe masks and is produced by
notebooks/evaluate_for_paper.ipynb on Kaggle.

Usage (repo root):  python scripts/paper_figures/local_figures.py
"""

import importlib.util
import json
import os
import sys

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(__file__)
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import style  # noqa: E402
from style import INK, INK_2, MUTED, BLUE, ORANGE, AQUA, YELLOW, CONTEXT_GRAY, MODEL_COLORS, MODEL_LABELS  # noqa: E402
from backend.src.models.mobilevit_v2 import load_mobilevit_checkpoint  # noqa: E402
from backend.src.utils.graph_postprocess import connect_canopy_gaps, hysteresis_threshold  # noqa: E402

OUT = os.path.join(REPO, "figures", "real")
SAMPLES = ["100034", "102408", "115714", "117991"]
CKPTS = {
    "final": "models/best_model_v2.pth",
    "baseline": "models/best_model_new.pth",
    "run1": "models/archive/best_model_v2_epoch18_iou0159.pth",
    "collapsed": "models/archive/best_model_v2_collapsed_epoch46.pth",
}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# Sequential single-hue ramp (palette blue 100 -> 700) with white at zero.
PROB_CMAP = LinearSegmentedColormap.from_list(
    "prob_blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
MASK_CMAP = ListedColormap(["#ffffff", INK])

_spec = importlib.util.spec_from_file_location("psi", os.path.join(REPO, "scripts", "predict_single_image.py"))
_psi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_psi)
predict_tta = _psi.predict_tta   # the exact TTA used in deployment


def load_rgb(tile_id):
    rgb = cv2.cvtColor(cv2.imread(os.path.join(REPO, "data", "samples", f"{tile_id}_sat.jpg")), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return rgb[: (h // 32) * 32, : (w // 32) * 32]


def to_tensor(rgb):
    return torch.from_numpy(((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))[None]


def predict(model, rgb, tta=False):
    x = to_tensor(rgb)
    if tta:
        return predict_tta(model, x, torch.device("cpu"))
    with torch.no_grad():
        return torch.sigmoid(model(x))[0, 0].numpy()


def postprocess(probs):
    """Deployed pipeline; returns (closed mask, final mask) so bridged pixels can be isolated."""
    hyst = hysteresis_threshold(probs, high_thresh=0.35, low_thresh=0.12)
    closed = cv2.morphologyEx(hyst, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    final = connect_canopy_gaps(closed, max_gap_dist=220.0, max_angle_deg=65.0, road_width=6)
    return hyst, closed, final


def n_components(mask):
    return cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)[0] - 1


# ---------------------------------------------------------------------------
def fig_efficiency():
    path = os.path.join(OUT, "measurements", "efficiency.json")
    if not os.path.exists(path):
        print("skip fig_efficiency: run benchmark_efficiency.py first")
        return
    d = json.load(open(path))
    names = ["Proposed", "LR-ASPP-MBv3", "DeepLabV3-MBv3", "D-LinkNet34", "U-Net"]
    names = [n for n in names if n in d["models"]]
    metrics = [
        ("params", 1e-6, "Parameters (millions)", "{:.1f} M"),
        ("gflops", 1.0, f"GFLOPs per {d['tile']}² tile", "{:,.0f}"),
        ("latency_ms", 1e-3, f"CPU latency per {d['tile']}² tile (s)", "{:.2f} s"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(style.PAGE_W, 1.9), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, (key, scale, xlabel, fmt), letter in zip(axes, metrics, "abc"):
        vals = [d["models"][n][key] * scale for n in names]
        colors = [BLUE if n == "Proposed" else CONTEXT_GRAY for n in names]
        ax.barh(y, vals, height=0.62, color=colors, edgecolor="white", linewidth=1.0)
        vmax = max(vals)
        for yi, v, n in zip(y, vals, names):
            label = fmt.format(v)
            if key == "latency_ms" and n == "Proposed" and "onnx" in d:
                label += f"  (ONNX Runtime: {d['onnx']['latency_ms'] / 1000:.2f} s)"
            ax.text(v + vmax * 0.02, yi, label, va="center", fontsize=7,
                    color=INK if n == "Proposed" else INK_2, fontweight="bold" if n == "Proposed" else "normal")
        ax.set_xlim(0, vmax * 1.32)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
        style.panel_label(ax, letter)
    noted = [n for n in names if "flops_note" in d["models"][n] or "latency_note" in d["models"][n]]
    axes[0].set_yticks(y)
    shown = {"Proposed": "Proposed (ours)", "LR-ASPP-MBv3": "LR-ASPP MBv3", "DeepLabV3-MBv3": "DeepLabV3 MBv3"}
    axes[0].set_yticklabels([shown.get(n, n) + ("†" if n in noted else "")
                             for n in names])
    foot =(f"Measured on {d['cpu']} ({d['torch_threads']} threads), PyTorch {d['torch'].split('+')[0]}, batch 1, "
            f"median of 3 to 7 runs. Reference models untrained: compute cost only.")
    if noted:
        foot += ("\n† Full 1024² pass exceeded this machine's free RAM: FLOPs counted at 256² and "
                 "scaled ×16 (exact for fully convolutional nets); latency timed as four 512² crops.")
    fig.text(0.0, -0.1 if noted else -0.06, foot, fontsize=6.5, color=MUTED, va="top" if noted else "baseline")
    style.save(fig, OUT, "fig_efficiency")


# ---------------------------------------------------------------------------
def fig_baseline_training():
    log = json.load(open(os.path.join(REPO, "wandb_local.json")))
    ep = sorted([r for r in log if "val/epoch_loss" in r], key=lambda r: r["epoch"])
    e = np.array([r["epoch"] for r in ep])   # 0-indexed, as logged and as stored in checkpoints
    best = min(ep, key=lambda r: r["val/epoch_cldice"])
    fig, axes = plt.subplots(1, 2, figsize=(style.PAGE_W, 2.1))
    panels = [("epoch_bce", "BCE loss component"), ("epoch_cldice", "Soft clDice loss component")]
    for ax, (key, title), letter in zip(axes, panels, "ab"):
        tr = np.array([r[f"train/{key}"] for r in ep])
        va = np.array([r[f"val/{key}"] for r in ep])
        ax.plot(e, tr, color=BLUE, lw=1.5, label="Train (crops at native resolution)")
        ax.plot(e, va, color=ORANGE, lw=1.5, label="Validation (tiles resized 4× smaller)")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Epoch")
        ax.set_xlim(0, e[-1])
        style.panel_label(ax, letter)
        if key == "epoch_cldice":
            be = best["epoch"]
            ax.plot([be], [best["val/epoch_cldice"]], "o", ms=5, color=ORANGE, mec="white", mew=1.0)
            ax.annotate(f"best val. epoch {be}\n({best['val/epoch_cldice']:.3f})", (be, best["val/epoch_cldice"]),
                        xytext=(be + 4, best["val/epoch_cldice"] - 0.17), fontsize=6.8, color=INK_2,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
            ax.text(e[-1], tr[-1] - 0.04, "train", color=BLUE, fontsize=7, ha="right", va="top")
            ax.text(e[-1], va[-1] + 0.03, "validation", color=ORANGE, fontsize=7, ha="right", va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    style.save(fig, OUT, "fig_baseline_training")


# ---------------------------------------------------------------------------
def fig_pipeline_stages(final):
    """Picks, among the four sample tiles, the one where gap bridging added the most pixels."""
    runs = {}
    for t in SAMPLES:
        rgb = load_rgb(t)
        probs = predict(final, rgb, tta=True)
        hyst, closed, fin = postprocess(probs)
        runs[t] = (rgb, probs, hyst, closed, fin, int(((fin > 0) & ~(closed > 0)).sum()))
    tile = max(runs, key=lambda k: runs[k][5])
    rgb, probs, hyst, closed, fin, bridged_px = runs[tile]
    bridged = (fin > 0) & ~(closed > 0)

    fig, axes = plt.subplots(1, 4, figsize=(style.PAGE_W, 2.35))
    axes[0].imshow(rgb)
    im = axes[1].imshow(probs, cmap=PROB_CMAP, vmin=0, vmax=1)
    axes[2].imshow(closed > 0, cmap=MASK_CMAP, interpolation="nearest")
    overlay = np.ones((*fin.shape, 3))
    overlay[closed > 0] = np.array([11, 11, 11]) / 255
    overlay[bridged] = np.array(tuple(int(ORANGE[i:i + 2], 16) for i in (1, 3, 5))) / 255
    axes[3].imshow(overlay, interpolation="nearest")
    titles = [f"Input tile\n{tile}", "Road probability\n(4 flip TTA)",
              f"Hysteresis + closing\n{n_components(closed)} components",
              f"+ canopy gap bridging\n{n_components(fin)} component{'s' if n_components(fin) != 1 else ''}"]
    for ax, letter, t in zip(axes, "abcd", titles):
        style.image_axes(ax)
        ax.set_title(t, loc="left", fontsize=7.5)
        style.panel_label(ax, letter)
    fig.subplots_adjust(wspace=0.08, bottom=0.2)
    pos = axes[1].get_position()
    cax = fig.add_axes([pos.x0, pos.y0 - 0.09, pos.width, 0.035])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=2)
    axes[3].legend(handles=[Patch(color=INK, label="model output"), Patch(color=ORANGE, label="bridged")],
                   loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2, fontsize=6.8, handlelength=1.0)
    style.save(fig, OUT, "fig_pipeline_stages")
    return {"tile": tile, "bridged_pixels": bridged_px,
            "components_before": n_components(closed), "components_after": n_components(fin)}


# ---------------------------------------------------------------------------
def fig_collapse(models):
    tiles = ["100034", "115714"]
    fig = plt.figure(figsize=(style.PAGE_W, 3.9))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.08, hspace=0.18)
    stats = {}
    heads = ["Input tile", "Collapsed run (ep. 46)", "Proposed (final)"]
    for row, t in enumerate(tiles):
        rgb = load_rgb(t)
        pc = predict(models["collapsed"], rgb)
        pf = predict(models["final"], rgb)
        for col, img in enumerate([rgb, pc, pf]):
            ax = fig.add_subplot(gs[row, col])
            if col == 0:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap=PROB_CMAP, vmin=0, vmax=1)
                ax.text(0.03, 0.04, f"{100 * (img > 0.5).mean():.1f}% of pixels > 0.5", transform=ax.transAxes,
                        fontsize=6.5, color=INK, va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
            if row == 0:
                ax.set_title(heads[col], loc="left", fontsize=7.5)
            style.image_axes(ax)
            for s in ax.spines.values():
                s.set_visible(True); s.set_color("#e1e0d9"); s.set_linewidth(0.5)
        stats[t] = {"collapsed_pos_frac": float((pc > 0.5).mean()), "final_pos_frac": float((pf > 0.5).mean())}

    ax = fig.add_subplot(gs[:, 3])
    bins = np.linspace(0, 1, 41)
    for key in ["collapsed", "final"]:
        vals = np.concatenate([predict(models[key], load_rgb(t)).ravel() for t in SAMPLES])
        h, _ = np.histogram(vals, bins=bins)
        ax.step(bins[:-1], 100 * h / h.sum(), where="post", color=MODEL_COLORS[key], lw=1.5,
                label=MODEL_LABELS[key])
        stats[f"{key}_all_tiles_pos_frac"] = float((vals > 0.5).mean())
    ax.set_yscale("log")
    ax.set_xlabel("Predicted road probability")
    ax.set_ylabel("% of pixels (4 tiles, log)")
    ax.set_title("Probability distribution", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=6.8, ncol=1)
    ax.yaxis.set_label_position("right"); ax.yaxis.tick_right()
    style.save(fig, OUT, "fig_collapse")
    return stats


# ---------------------------------------------------------------------------
def fig_resolution(final):
    tiles = ["117991", "115714", "100034"]
    fig, axes = plt.subplots(len(tiles), 3, figsize=(style.COL_W * 1.45, 1.62 * len(tiles)))
    stats = {}
    for r, t in enumerate(tiles):
        rgb = load_rgb(t)
        native = predict(final, rgb) > 0.5
        small = predict(final, cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)) > 0.5
        small_up = cv2.resize(small.astype(np.uint8), rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
        inter, union = (native & small_up).sum(), (native | small_up).sum()
        stats[t] = {"native_pos_frac": float(native.mean()), "resized_pos_frac": float(small.mean()),
                    "agreement_iou": float(inter / union) if union else 1.0}
        axes[r, 0].imshow(rgb)
        axes[r, 1].imshow(native, cmap=MASK_CMAP, interpolation="nearest")
        axes[r, 2].imshow(small_up, cmap=MASK_CMAP, interpolation="nearest")
        for ax, frac in ((axes[r, 1], native.mean()), (axes[r, 2], small.mean())):
            ax.text(0.03, 0.03, f"{100 * frac:.1f}% road", transform=ax.transAxes, fontsize=6.5, color=INK,
                    va="bottom", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
        for ax in axes[r]:
            style.image_axes(ax)
            for s in ax.spines.values():
                s.set_visible(True); s.set_color("#e1e0d9"); s.set_linewidth(0.5)
    axes[0, 0].set_title("Input (1024²)", loc="left", fontsize=7.5)
    axes[0, 2].set_title("Resized to 256²\n(validation during training)", loc="left", fontsize=7.5)
    axes[0, 1].set_title("Native resolution\n(training & deployment)", loc="left", fontsize=7.5)
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    style.save(fig, OUT, "fig_resolution")
    return stats


# ---------------------------------------------------------------------------
def fig_tta(final):
    t = "115714"
    rgb = load_rgb(t)
    out = {}
    for tta in (False, True):
        _, _, fin = postprocess(predict(final, rgb, tta=tta))
        out[tta] = fin > 0
    removed = out[False] & ~out[True]
    added = out[True] & ~out[False]
    fig, axes = plt.subplots(1, 3, figsize=(style.PAGE_W, 2.45))
    axes[0].imshow(rgb); axes[0].set_title("Input tile", loc="left")
    axes[1].imshow(out[False], cmap=MASK_CMAP, interpolation="nearest")
    axes[1].set_title(f"Single pass ({n_components(out[False].astype(np.uint8))} components)", loc="left")
    ov = np.ones((*rgb.shape[:2], 3))
    ov[out[True]] = np.array([11, 11, 11]) / 255
    ov[removed] = np.array([235, 104, 52]) / 255
    axes[2].imshow(ov, interpolation="nearest")
    axes[2].set_title(f"4 flip TTA ({n_components(out[True].astype(np.uint8))} components)", loc="left")
    axes[2].legend(handles=[Patch(color=INK, label="kept"), Patch(color=ORANGE, label="removed by TTA")],
                   loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=6.8, handlelength=1.0)
    for ax, letter in zip(axes, "abc"):
        style.image_axes(ax); style.panel_label(ax, letter)
    style.save(fig, OUT, "fig_tta")
    return {"tile": t, "components_single": n_components(out[False].astype(np.uint8)),
            "components_tta": n_components(out[True].astype(np.uint8)),
            "pixels_removed": int(removed.sum()), "pixels_added": int(added.sum())}


def main():
    style.apply()
    torch.set_grad_enabled(False)
    models = {k: load_mobilevit_checkpoint(os.path.join(REPO, v))[0] for k, v in CKPTS.items()}
    stats = {}
    fig_efficiency()
    fig_baseline_training()
    print("pipeline stages..."); stats["pipeline_stages"] = fig_pipeline_stages(models["final"])
    print("collapse...");        stats["collapse"] = fig_collapse(models)
    print("resolution...");      stats["resolution"] = fig_resolution(models["final"])
    os.makedirs(os.path.join(OUT, "measurements"), exist_ok=True)
    with open(os.path.join(OUT, "measurements", "local_figure_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
