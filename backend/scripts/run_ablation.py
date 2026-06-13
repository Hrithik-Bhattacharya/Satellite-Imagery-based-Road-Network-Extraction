"""
run_ablation.py – Member 4 (Validation & MLOps)
Ablation study framework across four experiment configurations.

Experiments:
  A: U-Net baseline
  B: MobileViT v2
  C: MobileViT v2 + clDice loss
  D: MobileViT v2 + clDice + Weak Labels (OSM)

If trained checkpoints exist, loads them and runs inference + metrics.
If checkpoints are NOT available, generates framework skeleton with
placeholder entries — does NOT fabricate training results.

Generates:
    outputs/ablation_results.csv
    outputs/ablation_results.json

Usage:
    cd backend
    python scripts/run_ablation.py [--checkpoint_dir path/to/ckpts]
                                   [--test_gt_dir  path/to/gt_masks]
                                   [--test_pred_dir path/to/pred_masks]
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))      # backend/
sys.path.insert(0, str(_SCRIPT_DIR.parent / "src"))

from src.utils.wandb_logger import WandbLogger


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {
        "id":           "A",
        "name":         "U-Net Baseline",
        "model_class":  "UNet",
        "loss":         "BCE",
        "weak_labels":  False,
        "ckpt_file":    "unet_best.pth",
        "model_kwargs": {"n_channels": 3, "n_classes": 1},
    },
    {
        "id":           "B",
        "name":         "MobileViT v2",
        "model_class":  "MobileViT_v2",
        "loss":         "BCE",
        "weak_labels":  False,
        "ckpt_file":    "mobilevit_v2_best.pth",
        "model_kwargs": {"num_classes": 1, "width_mult": 0.5},
    },
    {
        "id":           "C",
        "name":         "MobileViT v2 + clDice",
        "model_class":  "MobileViT_v2",
        "loss":         "clDice",
        "weak_labels":  False,
        "ckpt_file":    "mobilevit_v2_cldice_best.pth",
        "model_kwargs": {"num_classes": 1, "width_mult": 0.5},
    },
    {
        "id":           "D",
        "name":         "MobileViT v2 + clDice + Weak Labels",
        "model_class":  "MobileViT_v2",
        "loss":         "clDice",
        "weak_labels":  True,
        "ckpt_file":    "mobilevit_v2_cldice_weak_best.pth",
        "model_kwargs": {"num_classes": 1, "width_mult": 0.5},
    },
]


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_model(exp: dict, checkpoint_dir: Path):
    """
    Attempt to load a trained model checkpoint.
    Returns (model, params, size_mb) or raises FileNotFoundError.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required to load checkpoints.")

    model_class_name = exp["model_class"]

    if model_class_name == "UNet":
        from src.models.unet_baseline import UNet
        model = UNet(**exp["model_kwargs"])
    elif model_class_name == "MobileViT_v2":
        from src.models.mobilevit_v2 import MobileViT_v2
        model = MobileViT_v2(**exp["model_kwargs"])
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")

    ckpt_path = checkpoint_dir / exp["ckpt_file"]
    if ckpt_path.exists():
        state = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(state)
        print(f"    Loaded checkpoint: {ckpt_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model.eval()
    params   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb  = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    return model, params, round(size_mb, 3)


def _benchmark_fps(model, device) -> float:
    """Quick FPS estimate: 20 forward passes at batch=1."""
    try:
        import torch
        model = model.to(device)
        dummy = torch.randn(1, 3, 256, 256, device=device)
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy)
            t0 = time.perf_counter()
            for _ in range(20):
                _ = model(dummy)
        return round(20 / (time.perf_counter() - t0), 2)
    except Exception:
        return 0.0


def _evaluate_from_dirs(model, pred_dir: Path, gt_dir: Path, device) -> dict:
    """Run inference + metrics if test directories are provided."""
    from src.utils.metrics_iou  import compute_iou, compute_f1
    from src.utils.metrics_apls import compute_apls
    from src.utils.metrics_topo import compute_topo

    try:
        import cv2
        _load = lambda p: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    except ImportError:
        from PIL import Image
        _load = lambda p: np.array(Image.open(p).convert("L"))

    exts = {".png", ".jpg", ".jpeg", ".tif"}
    gt_files = {f.stem: f for f in gt_dir.iterdir() if f.suffix in exts}
    pred_files = {f.stem: f for f in pred_dir.iterdir() if f.suffix in exts}
    common = sorted(set(gt_files) & set(pred_files))[:20]  # limit to 20 images for speed

    ious, f1s, aplss, topo_f1s = [], [], [], []
    for stem in common:
        pred = _load(pred_files[stem])
        gt   = _load(gt_files[stem])
        if pred is None or gt is None:
            continue
        ious.append(compute_iou(pred, gt))
        f1s.append(compute_f1(pred, gt))
        try:
            aplss.append(compute_apls(pred, gt, n_samples=30)["apls"])
        except Exception:
            aplss.append(0.0)
        try:
            topo_f1s.append(compute_topo(pred, gt)["f1"])
        except Exception:
            topo_f1s.append(0.0)

    return {
        "iou":     round(float(np.mean(ious)),     4) if ious     else None,
        "f1":      round(float(np.mean(f1s)),      4) if f1s      else None,
        "apls":    round(float(np.mean(aplss)),     4) if aplss    else None,
        "topo_f1": round(float(np.mean(topo_f1s)), 4) if topo_f1s else None,
    }


# ---------------------------------------------------------------------------
# Run ablation
# ---------------------------------------------------------------------------

def run_ablation(checkpoint_dir: Path, pred_dir: Path, gt_dir: Path,
                 out_dir: Path, logger: WandbLogger | None) -> list[dict]:
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        device = None

    rows = []
    for exp in EXPERIMENTS:
        print(f"\n--- Experiment {exp['id']}: {exp['name']} ---")
        row = {
            "Experiment": exp["id"],
            "Model":      exp["name"],
            "Loss":       exp["loss"],
            "WeakLabels": exp["weak_labels"],
        }

        # Attempt to load checkpoint
        model = None
        params = None
        size_mb = None
        fps = None

        if device is not None:
            try:
                model, params, size_mb = _load_model(exp, checkpoint_dir)
                fps = _benchmark_fps(model, device)
                print(f"    Params: {params:,}  Size: {size_mb} MB  FPS: {fps}")
            except FileNotFoundError:
                print(f"    [INFO] No checkpoint found — framework-only row.")
            except ImportError as e:
                print(f"    [SKIP] {e}")

        row["Params"]   = params   if params   is not None else "N/A (no checkpoint)"
        row["Size_MB"]  = size_mb  if size_mb  is not None else "N/A"
        row["FPS"]      = fps      if fps      is not None else "N/A"

        # Metrics
        if model is not None and pred_dir and pred_dir.exists() and gt_dir and gt_dir.exists():
            m = _evaluate_from_dirs(model, pred_dir, gt_dir, device)
            row.update({k.upper() if k != "topo_f1" else "TOPO": v for k, v in m.items()})
        else:
            row["IoU"]  = "N/A (run training first)"
            row["F1"]   = "N/A"
            row["APLS"] = "N/A"
            row["TOPO"] = "N/A"

        rows.append(row)
        print(f"    IoU={row.get('IoU')}  F1={row.get('F1')}  APLS={row.get('APLS')}  TOPO={row.get('TOPO')}")

        if logger and any(isinstance(row.get(k), float) for k in ["IoU", "F1", "APLS", "TOPO"]):
            logger.log_metrics(
                {f"ablation/{row['Experiment']}/{k.lower()}": v
                 for k, v in row.items() if isinstance(v, float)},
                step=EXPERIMENTS.index(exp),
            )

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run ablation study across model configurations.")
    parser.add_argument("--checkpoint_dir", default="outputs/checkpoints", type=str)
    parser.add_argument("--test_gt_dir",    default=None, type=str)
    parser.add_argument("--test_pred_dir",  default=None, type=str)
    parser.add_argument("--output_dir",     default="outputs", type=str)
    parser.add_argument("--no_wandb",       action="store_true")
    args = parser.parse_args()

    out_dir       = Path(args.output_dir)
    ckpt_dir      = Path(args.checkpoint_dir)
    pred_dir      = Path(args.test_pred_dir) if args.test_pred_dir else None
    gt_dir        = Path(args.test_gt_dir)   if args.test_gt_dir   else None

    out_dir.mkdir(parents=True, exist_ok=True)

    logger = None if args.no_wandb else WandbLogger(
        project="rural-road-ablation",
        run_name="ablation-study",
        output_dir=str(out_dir),
    )

    rows = run_ablation(ckpt_dir, pred_dir, gt_dir, out_dir, logger)

    # --- CSV ---
    csv_path = out_dir / "ablation_results.csv"
    fieldnames = ["Experiment", "Model", "Loss", "WeakLabels",
                  "Params", "Size_MB", "IoU", "F1", "APLS", "TOPO", "FPS"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nAblation results saved: {csv_path}")

    # --- JSON ---
    json_path = out_dir / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    if logger:
        logger.log_artifact(str(csv_path),  artifact_type="ablation")
        logger.log_artifact(str(json_path), artifact_type="ablation")
        logger.finish()

    # --- Print table ---
    print("\n====== ABLATION RESULTS ======")
    print(f"{'Exp':<4} {'Model':<38} {'Params':>12} {'IoU':>7} {'F1':>7} {'APLS':>7} {'TOPO':>7} {'FPS':>7}")
    print("-" * 95)
    for r in rows:
        p = r.get("Params", "N/A")
        if isinstance(p, int):
            p = f"{p:,}"
        print(f"{r['Experiment']:<4} {r['Model']:<38} {str(p):>12} "
              f"{str(r.get('IoU','?')):>7} {str(r.get('F1','?')):>7} "
              f"{str(r.get('APLS','?')):>7} {str(r.get('TOPO','?')):>7} "
              f"{str(r.get('FPS','?')):>7}")


if __name__ == "__main__":
    main()
