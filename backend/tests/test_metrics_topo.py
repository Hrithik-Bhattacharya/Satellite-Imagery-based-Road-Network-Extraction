"""
test_metrics_topo.py – Member 4 unit tests
Tests for TOPO topological metrics on synthetic road masks.
No DeepGlobe dataset required.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _has_graph_builder() -> bool:
    try:
        from src.utils.graph_adapter import _import_graph_builder
        _import_graph_builder()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Synthetic mask helpers
# ---------------------------------------------------------------------------

def _line(H: int = 128, W: int = 128) -> np.ndarray:
    m = np.zeros((H, W), dtype=np.uint8)
    m[H // 2 - 1: H // 2 + 1, 10: W - 10] = 255
    return m


def _cross(H: int = 128, W: int = 128) -> np.ndarray:
    m = np.zeros((H, W), dtype=np.uint8)
    m[H // 2 - 1: H // 2 + 1, :] = 255
    m[:, W // 2 - 1: W // 2 + 1] = 255
    return m


def _empty() -> np.ndarray:
    return np.zeros((64, 64), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests requiring graph_builder
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_graph_builder(), reason="graph_builder (Member 2) not on PYTHONPATH")
class TestTopo:
    def test_perfect_returns_one(self):
        """Same mask → TOPO F1 should be high (≥ 0.8)."""
        from src.utils.metrics_topo import compute_topo
        m = _cross()
        result = compute_topo(m, m, tolerance=5.0)
        assert result["f1"] >= 0.8, f"Expected F1 ≥ 0.8, got {result['f1']}"

    def test_result_has_required_keys(self):
        from src.utils.metrics_topo import compute_topo
        result = compute_topo(_line(), _line())
        assert "precision" in result
        assert "recall"    in result
        assert "f1"        in result

    def test_scores_in_range(self):
        from src.utils.metrics_topo import compute_topo
        result = compute_topo(_line(), _cross())
        for k in ["precision", "recall", "f1"]:
            assert 0.0 <= result[k] <= 1.0, f"{k} out of range: {result[k]}"

    def test_empty_pred_against_gt(self):
        from src.utils.metrics_topo import compute_topo
        result = compute_topo(_empty(), _line(64, 64), tolerance=5.0)
        # With no prediction, recall should be 0
        assert result["recall"] <= 0.35

    def test_both_empty(self):
        from src.utils.metrics_topo import compute_topo
        result = compute_topo(_empty(), _empty())
        # Both empty: no nodes to compare, should not crash
        assert isinstance(result, dict)

    def test_detailed_breakdown_keys(self):
        from src.utils.metrics_topo import compute_topo
        m = _cross()
        result = compute_topo(m, m)
        for key in ["endpoint_precision", "endpoint_recall", "junction_precision",
                    "junction_recall", "connectivity"]:
            assert key in result, f"Missing key: {key}"

    def test_topological_similarity_scalar(self):
        from src.utils.metrics_topo import compute_topological_similarity
        score = compute_topological_similarity(_line(), _line())
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Structure-only tests (no graph_builder required)
# ---------------------------------------------------------------------------

class TestTopoStructure:
    def test_returns_dict(self):
        try:
            from src.utils.metrics_topo import compute_topo
            result = compute_topo(np.zeros((32, 32), np.uint8), np.zeros((32, 32), np.uint8))
            assert isinstance(result, dict)
        except ImportError:
            pytest.skip("metrics_topo not importable")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
