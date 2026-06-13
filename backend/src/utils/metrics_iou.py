"""
metrics_iou.py – Member 4 (Validation & MLOps)
Pixel-level segmentation metrics: IoU, Precision, Recall, F1.

Supports:
  - NumPy arrays  (H, W) or (B, H, W) uint8 / bool / float
  - PyTorch tensors of the same shapes
Returns scalar floats in all cases.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy_binary(arr, threshold: float = 0.5) -> np.ndarray:
    """Convert tensor or numpy array to a flat binary numpy array."""
    if _TORCH_AVAILABLE and isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32)
    return (arr > threshold).astype(np.float32)


def _tp_fp_fn(pred: np.ndarray, gt: np.ndarray):
    """Return (TP, FP, FN) over entire flattened arrays."""
    p = pred.ravel()
    g = gt.ravel()
    tp = float(np.sum(p * g))
    fp = float(np.sum(p * (1.0 - g)))
    fn = float(np.sum((1.0 - p) * g))
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_iou(pred, gt, threshold: float = 0.5, eps: float = 1e-7) -> float:
    """
    Intersection over Union (Jaccard Index).

    Args:
        pred: binary prediction mask – numpy (H,W)|(B,H,W) or torch tensor.
        gt:   binary ground-truth mask – same shape as pred.
        threshold: binarisation threshold when inputs are soft probabilities.
        eps: small value to avoid division by zero.

    Returns:
        Scalar float IoU.
    """
    pred = _to_numpy_binary(pred, threshold)
    gt   = _to_numpy_binary(gt,   threshold)
    tp, fp, fn = _tp_fp_fn(pred, gt)
    return float(tp / (tp + fp + fn + eps))


def compute_precision(pred, gt, threshold: float = 0.5, eps: float = 1e-7) -> float:
    """Pixel-level Precision: TP / (TP + FP)."""
    pred = _to_numpy_binary(pred, threshold)
    gt   = _to_numpy_binary(gt,   threshold)
    tp, fp, _ = _tp_fp_fn(pred, gt)
    return float(tp / (tp + fp + eps))


def compute_recall(pred, gt, threshold: float = 0.5, eps: float = 1e-7) -> float:
    """Pixel-level Recall: TP / (TP + FN)."""
    pred = _to_numpy_binary(pred, threshold)
    gt   = _to_numpy_binary(gt,   threshold)
    tp, _, fn = _tp_fp_fn(pred, gt)
    return float(tp / (tp + fn + eps))


def compute_f1(pred, gt, threshold: float = 0.5, eps: float = 1e-7) -> float:
    """F1 score: harmonic mean of Precision and Recall."""
    prec = compute_precision(pred, gt, threshold, eps)
    rec  = compute_recall(pred,  gt, threshold, eps)
    return float(2.0 * prec * rec / (prec + rec + eps))


def compute_all_pixel_metrics(pred, gt, threshold: float = 0.5) -> dict:
    """Convenience wrapper returning all pixel metrics as a dict."""
    return {
        "iou":       compute_iou(pred,       gt, threshold),
        "precision": compute_precision(pred, gt, threshold),
        "recall":    compute_recall(pred,    gt, threshold),
        "f1":        compute_f1(pred,        gt, threshold),
    }
