"""
test_metrics_dice.py – Member 4 unit tests
Tests for Dice coefficient on synthetic masks.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.metrics_dice import compute_dice, compute_soft_dice


def _mask(fill: str = "partial", H: int = 64, W: int = 64) -> np.ndarray:
    m = np.zeros((H, W), dtype=np.uint8)
    if fill == "full":
        m[:] = 255
    elif fill == "partial":
        m[H // 4: 3 * H // 4, W // 4: 3 * W // 4] = 255
    elif fill == "half":
        m[:, : W // 2] = 255
    return m


class TestDice:
    def test_perfect(self):
        m = _mask("partial")
        assert compute_dice(m, m) == pytest.approx(1.0, abs=1e-4)

    def test_no_overlap(self):
        pred = np.zeros((64, 64), dtype=np.uint8)
        gt   = np.full((64, 64), 255, dtype=np.uint8)
        assert compute_dice(pred, gt) < 0.01

    def test_range(self):
        pred = _mask("half")
        gt   = _mask("full")
        d = compute_dice(pred, gt)
        assert 0.0 <= d <= 1.0

    def test_symmetric(self):
        """Dice should be symmetric: dice(A,B) == dice(B,A)"""
        a = _mask("partial")
        b = _mask("half")
        assert compute_dice(a, b) == pytest.approx(compute_dice(b, a), abs=1e-6)

    def test_batch_input(self):
        m = _mask("partial")
        mb = m[np.newaxis, :]  # (1, H, W)
        assert compute_dice(mb, mb) == pytest.approx(1.0, abs=1e-4)

    def test_float_normalised_input(self):
        m = _mask("partial").astype(np.float32) / 255.0
        assert compute_dice(m, m) == pytest.approx(1.0, abs=1e-4)

    def test_torch_input(self):
        try:
            import torch
            m = _mask("partial").astype(np.float32) / 255.0
            t = torch.from_numpy(m)
            assert compute_dice(t, t) == pytest.approx(1.0, abs=1e-4)
        except ImportError:
            pytest.skip("PyTorch not installed")


class TestSoftDice:
    def test_perfect_prob(self):
        m = _mask("partial").astype(np.float32) / 255.0
        assert compute_soft_dice(m, m) == pytest.approx(1.0, abs=1e-4)

    def test_range(self):
        p = _mask("half").astype(np.float32) / 255.0
        g = _mask("partial").astype(np.float32) / 255.0
        assert 0.0 <= compute_soft_dice(p, g) <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
