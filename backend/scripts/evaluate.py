"""
evaluate.py – Member 4 (Validation & MLOps)
CLI evaluation script: computes IoU, Dice, Precision, Recall, F1, APLS, TOPO
for all (prediction, ground-truth) mask pairs in given directories.

Usage:
    python scripts/evaluate.py --gt_dir path/to/gt --pred_dir path/to/preds

Outputs:
    outputs/evaluation_results.csv    – per-image metric table
    outputs/evaluation_summary.json   – aggregate statistics
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Make sure backend/src is on the path when running from backend/
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))        # backend/
sys.path.insert(0, str(_SCRIPT_DIR.parent / "src"))  # backend/src

try:
    import cv2
    def _load_mask(path: str) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        # Ensure mask is binary/probability in [0,1]
        if img is None:
            return None
        return (img.astype(np.float32) / 255.0)
except ImportError:
    from PIL import Image
    def _load_mask(path: str) -> np.ndarray:
        img = np.array(Image.open(path).convert("L"))
        return (img.astype(np.float32) / 255.0)

from src.utils.metrics_iou  import compute_iou, compute_precision, compute_recall, compute_f1
from src.utils.metrics_dice import compute_dice
from src.utils.metrics_apls import compute_apls
from src.utils.metrics_topo import compute_topo
from src.utils.wandb_logger import WandbLogger

_MASK_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

def find_pairs(gt_dir: Path, pred_dir: Path) -> list[tuple[Path, Path]]:
    gt_files = {f.stem: f for f in gt_dir.iterdir() if f.suffix.lower() in _MASK_EXTS}
    pred_files = {f.stem: f for f in pred_dir.iterdir() if f.suffix.lower() in _MASK_EXTS}
    common = sorted(set(gt_files) & set(pred_files))
    if not common:
        print(f"[WARNING] No matching stems found between {gt_dir} and {pred_dir}")
    return [(pred_files[s], gt_files[s]) for s in common]

def evaluate_pair(pred_path: Path, gt_path: Path) -> dict:
    pred = _load_mask(str(pred_path))
    gt   = _load_mask(str(gt_path))
    if pred is None or gt is None:
        return {"error": "Could not load image"}

    metrics = {
        "filename": pred_path.name,
        "iou":       compute_iou(pred, gt),
        "dice":      compute_dice(pred, gt),
        "precision": compute_precision(pred, gt),
        "recall":    compute_recall(pred, gt),
        "f1":        compute_f1(pred, gt),
    }

    try:
        apls_result = compute_apls(pred, gt, n_samples=50)
        metrics["apls"]   = apls_result["apls"]
        metrics["routes"] = apls_result["routes"]
    except Exception as e:
        metrics["apls"]   = 0.0
        metrics["routes"] = 0
        metrics["apls_error"] = str(e)

    try:
        topo_result = compute_topo(pred, gt)
        metrics["topo_precision"] = topo_result["precision"]
        metrics["topo_recall"]    = topo_result["recall"]
        metrics["topo_f1"]        = topo_result["f1"]
    except Exception as e:
        metrics["topo_precision"] = 0.0
        metrics["topo_recall"]    = 0.0
        metrics["topo_f1"]        = 0.0
        metrics["topo_error"]     = str(e)

    return metrics

def aggregate(results: list[dict]) -> dict:
    numeric_keys = [k for k in results[0] if k not in ("filename", "error") and
                    not k.endswith("_error") and isinstance(results[0][k], (int, float))]
    summary = {}
    for k in numeric_keys:
        vals = [r[k] for r in results if k in r and isinstance(r[k], (int, float))]
        summary[k] = {
            "mean": float(np.mean(vals)) if vals else 0.0,
            "std":  float(np.std(vals))  if vals else 0.0,
            "min":  float(np.min(vals))  if vals else 0.0,
            "max":  float(np.max(vals))  if vals else 0.0,
        }
    return summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate road segmentation predictions.")
    parser.add_argument("--gt_dir",   required=True, type=str, help="Directory of ground-truth masks")
    parser.add_argument("--pred_dir", required=True, type=str, help="Directory of predicted masks")
    parser.add_argument("--output_dir", default="outputs", type=str, help="Output directory")
    parser.add_argument("--n_samples", default=50, type=int, help="Route samples for APLS")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    gt_dir   = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not gt_dir.exists():
        print(f"[ERROR] GT directory not found: {gt_dir}")
        sys.exit(1)
    if not pred_dir.exists():
        print(f"[ERROR] Pred directory not found: {pred_dir}")
        sys.exit(1)

    pairs = find_pairs(gt_dir, pred_dir)
    print(f"Found {len(pairs)} image pairs to evaluate.")
    if not pairs:
        print("Nothing to evaluate.")
        sys.exit(0)

    logger = None if args.no_wandb else WandbLogger(
        project="rural-road-evaluation",
        run_name="evaluate",
        output_dir=str(out_dir),
    )

    results = []
    t0 = time.time()
    for i, (pred_path, gt_path) in enumerate(pairs, 1):
        print(f"  [{i}/{len(pairs)}] {pred_path.name} ...", end=" ", flush=True)
        r = evaluate_pair(pred_path, gt_path)
        results.append(r)

        if logger:
            logger.log_metrics({k: v for k, v in r.items() if isinstance(v, (int, float))}, step=i)

        print(f"IoU={r.get('iou', 0):.4f}  F1={r.get('f1', 0):.4f}  "
              f"APLS={r.get('apls', 0):.4f}  TOPO_F1={r.get('topo_f1', 0):.4f}")

    elapsed = time.time() - t0
    print(f"\nEvaluation complete in {elapsed:.1f}s")

    csv_path = out_dir / "evaluation_results.csv"
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved per-image results: {csv_path}")

    summary = aggregate(results)
    summary["n_images"] = len(results)
    summary["elapsed_seconds"] = elapsed
    summary_path = out_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")

    print("\n===== EVALUATION SUMMARY =====")
    for metric in ["iou", "dice", "precision", "recall", "f1", "apls", "topo_f1"]:
        if metric in summary:
            m = summary[metric]
            print(f"  {metric:<18s}  mean={m['mean']:.4f}  std={m['std']:.4f}")

    if logger:
        logger.log_metrics({f"summary/{k}": v["mean"] for k, v in summary.items()
                            if isinstance(v, dict) and "mean" in v})
        logger.log_artifact(str(csv_path),     artifact_type="evaluation")
        logger.log_artifact(str(summary_path), artifact_type="evaluation")
        logger.finish()

if __name__ == "__main__":
    main()
