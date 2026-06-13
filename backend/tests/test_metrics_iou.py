"""
test_metrics_iou.py – Member 4 unit tests
Tests for IoU, Precision, Recall, F1 on synthetic masks.
No DeepGlobe dataset required.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.metrics_iou import (
    compute_iou,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_all_pixel_metrics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_mask(H: int = 64, W: int = 64, fill: str = "partial") -> np.ndarray:
    """Create simple synthetic binary masks (uint8, 0/255)."""
    mask = np.zeros((H, W), dtype=np.uint8)
    if fill == "full":
        mask[:] = 255
    elif fill == "partial":
        mask[H // 4: 3 * H // 4, W // 4: 3 * W // 4] = 255
    elif fill == "half":
        mask[:, : W // 2] = 255
    elif fill == "empty":
        pass
    return mask


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

class TestIoU:
    def test_perfect_match(self):
        m = make_mask(fill="partial")
        assert compute_iou(m, m) == pytest.approx(1.0, abs=1e-4)

    def test_no_overlap(self):
        pred = make_mask(fill="empty")
        gt   = make_mask(fill="full")
        assert compute_iou(pred, gt) == pytest.approx(0.0, abs=1e-4)

    def test_partial_overlap(self):
        pred = make_mask(fill="half")
        gt   = make_mask(fill="full")
        iou = compute_iou(pred, gt)
        # half the mask is predicted → IoU = 0.5/1.0 = 0.5
        assert 0.3 < iou < 0.7

    def test_empty_pred_and_gt(self):
        m = make_mask(fill="empty")
        # Both empty → IoU = 0 / (0 + eps) ≈ 0
        assert compute_iou(m, m) == pytest.approx(0.0, abs=1e-3)

    def test_numpy_float_input(self):
        m = make_mask(fill="partial").astype(np.float32) / 255.0
        assert compute_iou(m, m) == pytest.approx(1.0, abs=1e-4)

    def test_batch_shape(self):
        m = make_mask(fill="partial")
        m_batch = m[np.newaxis, :]  # (1, H, W)
        assert compute_iou(m_batch, m_batch) == pytest.approx(1.0, abs=1e-4)

    def test_torch_tensor(self):
        try:
            import torch
            m = make_mask(fill="partial").astype(np.float32) / 255.0
            t = torch.from_numpy(m)
            assert compute_iou(t, t) == pytest.approx(1.0, abs=1e-4)
        except ImportError:
            pytest.skip("PyTorch not installed")


# ---------------------------------------------------------------------------
# Precision / Recall
# ---------------------------------------------------------------------------

class TestPrecisionRecall:
    def test_all_true_positives(self):
        m = make_mask(fill="partial")
        assert compute_precision(m, m) == pytest.approx(1.0, abs=1e-4)
        assert compute_recall(m, m)    == pytest.approx(1.0, abs=1e-4)

    def test_all_false_positives(self):
        pred = make_mask(fill="full")
        gt   = make_mask(fill="empty")
        assert compute_precision(pred, gt) == pytest.approx(0.0, abs=1e-4)

    def test_all_false_negatives(self):
        pred = make_mask(fill="empty")
        gt   = make_mask(fill="full")
        assert compute_recall(pred, gt) == pytest.approx(0.0, abs=1e-4)

    def test_symmetry_iou_and_f1(self):
        pred = make_mask(fill="partial")
        gt   = make_mask(fill="full")
        # Swapping pred/gt changes precision vs recall but F1 relationship should hold
        iou  = compute_iou(pred, gt)
        f1   = compute_f1(pred,  gt)
        assert 0.0 <= iou <= 1.0
        assert 0.0 <= f1  <= 1.0


# ---------------------------------------------------------------------------
# F1
# ---------------------------------------------------------------------------

class TestF1:
    def test_perfect(self):
        m = make_mask(fill="partial")
        assert compute_f1(m, m) == pytest.approx(1.0, abs=1e-4)

    def test_zero(self):
        pred = make_mask(fill="empty")
        gt   = make_mask(fill="full")
        assert compute_f1(pred, gt) == pytest.approx(0.0, abs=1e-4)

    def test_harmonic_mean_property(self):
        pred = make_mask(fill="half")
        gt   = make_mask(fill="full")
        p = compute_precision(pred, gt)
        r = compute_recall(pred,    gt)
        expected_f1 = 2.0 * p * r / (p + r + 1e-7)
        assert compute_f1(pred, gt) == pytest.approx(expected_f1, abs=1e-5)


# ---------------------------------------------------------------------------
# compute_all_pixel_metrics
# ---------------------------------------------------------------------------

class TestAllMetrics:
    def test_returns_all_keys(self):
        m = make_mask(fill="partial")
        result = compute_all_pixel_metrics(m, m)
        assert set(result.keys()) == {"iou", "precision", "recall", "f1"}

    def test_perfect_scores(self):
        m = make_mask(fill="partial")
        result = compute_all_pixel_metrics(m, m)
        for v in result.values():
            assert v == pytest.approx(1.0, abs=1e-4)

    def test_values_in_range(self):
        pred = make_mask(fill="half")
        gt   = make_mask(fill="partial")
        result = compute_all_pixel_metrics(pred, gt)
        for v in result.values():
            assert 0.0 <= v <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
