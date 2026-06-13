"""
metrics_dice.py – Member 4 (Validation & MLOps)
Dice / Sørensen coefficient for binary segmentation masks.

Supports:
  - NumPy arrays  (H, W) or (B, H, W)
  - PyTorch tensors of the same shapes
Returns a scalar float.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _to_numpy_binary(arr, threshold: float = 0.5) -> np.ndarray:
    if _TORCH_AVAILABLE and isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32)
    return (arr > threshold).astype(np.float32)


def compute_dice(pred, gt, threshold: float = 0.5, eps: float = 1e-7) -> float:
    """
    Dice Coefficient: 2*TP / (2*TP + FP + FN).

    Args:
        pred: predicted binary mask – numpy (H,W)|(B,H,W) or torch tensor.
        gt:   ground-truth binary mask – same shape.
        threshold: binarisation threshold for soft inputs.
        eps: smoothing constant to prevent division by zero.

    Returns:
        Scalar float Dice score in [0, 1].
    """
    pred = _to_numpy_binary(pred, threshold)
    gt   = _to_numpy_binary(gt,   threshold)

    p = pred.ravel()
    g = gt.ravel()

    intersection = float(np.sum(p * g))
    union        = float(np.sum(p) + np.sum(g))

    return float((2.0 * intersection + eps) / (union + eps))


def compute_soft_dice(pred_prob, gt, eps: float = 1e-7) -> float:
    """
    Soft Dice loss-compatible metric operating on probability maps.
    Useful for ranking model confidence without hard thresholding.
    """
    if _TORCH_AVAILABLE and isinstance(pred_prob, torch.Tensor):
        pred_prob = pred_prob.detach().cpu().numpy()
    if _TORCH_AVAILABLE and isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()

    p = np.asarray(pred_prob, dtype=np.float32).ravel()
    g = np.asarray(gt,        dtype=np.float32).ravel()

    intersection = float(np.sum(p * g))
    union        = float(np.sum(p) + np.sum(g))

    return float((2.0 * intersection + eps) / (union + eps))
