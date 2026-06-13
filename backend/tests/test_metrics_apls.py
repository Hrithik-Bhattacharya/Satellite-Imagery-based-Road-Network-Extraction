"""
test_metrics_apls.py – Member 4 unit tests
Tests for APLS metric on synthetic road masks.
No DeepGlobe dataset required.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _has_graph_builder() -> bool:
    """Check that Member 2's graph_builder can be found."""
    try:
        from src.utils.graph_adapter import _import_graph_builder
        _import_graph_builder()
        return True
    except (ImportError, Exception):
        return False


# ---------------------------------------------------------------------------
# Synthetic mask helpers
# ---------------------------------------------------------------------------

def _make_line_mask(H: int = 128, W: int = 128) -> np.ndarray:
    """Create a horizontal road line mask."""
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[H // 2 - 2: H // 2 + 2, :] = 255
    return mask


def _make_cross_mask(H: int = 128, W: int = 128) -> np.ndarray:
    """Create a cross (T-junction) road mask."""
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[H // 2 - 2: H // 2 + 2, :] = 255          # horizontal
    mask[:, W // 2 - 2: W // 2 + 2] = 255           # vertical
    return mask


def _make_empty_mask(H: int = 64, W: int = 64) -> np.ndarray:
    return np.zeros((H, W), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_graph_builder(), reason="graph_builder (Member 2) not on PYTHONPATH")
class TestAPLS:
    def test_perfect_match_returns_one(self):
        """Same mask for pred and GT should yield APLS close to 1."""
        from src.utils.metrics_apls import compute_apls
        mask = _make_line_mask()
        result = compute_apls(mask, mask, n_samples=20)
        assert "apls" in result
        assert result["apls"] >= 0.8, f"Expected APLS ≈ 1.0, got {result['apls']}"

    def test_empty_pred_returns_zero(self):
        """Empty prediction against non-empty GT should yield APLS = 0."""
        from src.utils.metrics_apls import compute_apls
        pred = _make_empty_mask()
        gt   = _make_line_mask(64, 64)
        result = compute_apls(pred, gt, n_samples=10)
        assert result["apls"] == pytest.approx(0.0, abs=1e-6)
        assert result["routes"] == 0

    def test_both_empty_returns_zero(self):
        from src.utils.metrics_apls import compute_apls
        empty = _make_empty_mask()
        result = compute_apls(empty, empty, n_samples=10)
        assert result["apls"] == 0.0

    def test_result_in_range(self):
        from src.utils.metrics_apls import compute_apls
        pred = _make_line_mask()
        gt   = _make_cross_mask()
        result = compute_apls(pred, gt, n_samples=20)
        assert 0.0 <= result["apls"] <= 1.0

    def test_routes_key_present(self):
        from src.utils.metrics_apls import compute_apls
        mask = _make_line_mask()
        result = compute_apls(mask, mask, n_samples=10)
        assert "routes" in result
        assert isinstance(result["routes"], int)

    def test_cross_vs_line_lower_than_perfect(self):
        """Cross mask vs line mask should score lower than line vs line."""
        from src.utils.metrics_apls import compute_apls
        line  = _make_line_mask()
        cross = _make_cross_mask()
        perfect_score = compute_apls(line, line, n_samples=20)["apls"]
        mismatch_score = compute_apls(line, cross, n_samples=20)["apls"]
        assert mismatch_score <= perfect_score


# ---------------------------------------------------------------------------
# Fallback test (no graph_builder) — structural only
# ---------------------------------------------------------------------------

class TestAPLSReturnStructure:
    """Tests that don't require graph_builder — verify return structure only."""

    def test_compute_apls_returns_dict(self):
        """compute_apls should return a dict even if graph construction fails."""
        try:
            from src.utils.metrics_apls import compute_apls
            mask = np.zeros((32, 32), dtype=np.uint8)
            result = compute_apls(mask, mask, n_samples=5)
            assert isinstance(result, dict)
            assert "apls" in result
            assert "routes" in result
        except ImportError:
            pytest.skip("metrics_apls not importable")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
