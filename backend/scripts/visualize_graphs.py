"""
visualize_graphs.py – Member 4 (Validation & MLOps)
Generate side-by-side graph overlay visualisations comparing predicted road
graphs against ground-truth road graphs.

For each matched (pred, gt) pair:
  - Draws the skeleton
  - Overlays GT graph (green) and Pred graph (red)
  - Highlights matched / mismatched nodes
  - Saves PNG to outputs/graph_comparisons/

Usage:
    cd backend
    python scripts/visualize_graphs.py --gt_dir path/gt --pred_dir path/pred
                                       [--max_images 10]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))
sys.path.insert(0, str(_SCRIPT_DIR.parent / "src"))


def _load_mask(path: str) -> np.ndarray:
    try:
        import cv2
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    except ImportError:
        from PIL import Image
        return np.array(Image.open(path).convert("L"))


def _draw_graph_on_image(ax, G, color: str, alpha: float = 0.8, node_size: int = 12):
    """Draw a NetworkX graph onto a matplotlib axis."""
    for u, v, d in G.edges(data=True):
        pix = d.get("pixels")
        if pix is not None and len(pix) > 1:
            pix = np.array(pix)
            ax.plot(pix[:, 1], pix[:, 0], color=color, linewidth=1.2, alpha=alpha)
        else:
            ax.plot([u[1], v[1]], [u[0], v[0]], color=color, linewidth=1.2, alpha=alpha)
    if G.number_of_nodes() > 0:
        ys = [n[0] for n in G.nodes()]
        xs = [n[1] for n in G.nodes()]
        ax.scatter(xs, ys, color=color, s=node_size, zorder=5, alpha=alpha)


def visualize_pair(pred_mask: np.ndarray, gt_mask: np.ndarray, title: str, out_path: Path):
    """Create a three-panel comparison: GT overlay | Pred overlay | Side-by-side."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not installed.")
        return

    from src.utils.graph_adapter import mask_to_graph

    try:
        G_gt   = mask_to_graph(gt_mask)
        G_pred = mask_to_graph(pred_mask)
    except Exception as e:
        print(f"  [SKIP] Graph construction failed: {e}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=11)

    # Panel 1: GT mask + GT graph
    axes[0].imshow(gt_mask, cmap="gray", vmin=0, vmax=255)
    _draw_graph_on_image(axes[0], G_gt, color="lime")
    axes[0].set_title(f"Ground Truth\n(nodes={G_gt.number_of_nodes()}, edges={G_gt.number_of_edges()})")
    axes[0].axis("off")

    # Panel 2: Pred mask + Pred graph
    axes[1].imshow(pred_mask, cmap="gray", vmin=0, vmax=255)
    _draw_graph_on_image(axes[1], G_pred, color="red")
    axes[1].set_title(f"Prediction\n(nodes={G_pred.number_of_nodes()}, edges={G_pred.number_of_edges()})")
    axes[1].axis("off")

    # Panel 3: Combined overlay on GT mask
    axes[2].imshow(gt_mask, cmap="gray", vmin=0, vmax=255, alpha=0.4)
    axes[2].imshow(pred_mask, cmap="hot", vmin=0, vmax=255, alpha=0.3)
    _draw_graph_on_image(axes[2], G_gt,   color="lime",  alpha=0.8)
    _draw_graph_on_image(axes[2], G_pred, color="red",   alpha=0.7)
    axes[2].set_title("Overlay\ngreen=GT  red=Pred")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualise GT vs Pred graph overlays.")
    parser.add_argument("--gt_dir",     required=True, type=str)
    parser.add_argument("--pred_dir",   required=True, type=str)
    parser.add_argument("--output_dir", default="outputs",  type=str)
    parser.add_argument("--max_images", default=10,         type=int)
    args = parser.parse_args()

    gt_dir   = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    comp_dir = Path(args.output_dir) / "graph_comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    gt_files   = {f.stem: f for f in gt_dir.iterdir()   if f.suffix.lower() in exts}
    pred_files = {f.stem: f for f in pred_dir.iterdir() if f.suffix.lower() in exts}
    common = sorted(set(gt_files) & set(pred_files))[: args.max_images]

    print(f"Generating graph comparison visualisations for {len(common)} images ...")

    for stem in common:
        pred_mask = _load_mask(pred_files[stem])
        gt_mask   = _load_mask(gt_files[stem])
        if pred_mask is None or gt_mask is None:
            continue

        out_png = comp_dir / f"{stem}_comparison.png"
        print(f"  {stem} ...", end=" ", flush=True)
        visualize_pair(pred_mask, gt_mask, title=stem, out_path=out_png)
        print(f"saved → {out_png.name}")

    print(f"\nAll comparisons saved to: {comp_dir}")


if __name__ == "__main__":
    main()
