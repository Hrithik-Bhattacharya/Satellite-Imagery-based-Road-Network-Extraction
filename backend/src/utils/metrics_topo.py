"""
metrics_topo.py – Member 4 (Validation & MLOps)
Topological preservation metrics for road network graphs.

Evaluates:
  - Endpoint preservation   (degree-1 nodes)
  - Junction preservation   (degree>2 nodes)
  - Connectivity preservation (connected-component structure)
  - Topological similarity score (combined)

All functions operate on NetworkX graphs produced by graph_adapter.mask_to_graph().

Returns:
    {
        "precision": float,
        "recall":    float,
        "f1":        float,
    }
"""

from __future__ import annotations

import math
import numpy as np

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False

from .graph_adapter import mask_to_graph


# ---------------------------------------------------------------------------
# Node-type extractors
# ---------------------------------------------------------------------------

def _get_endpoints(G) -> list:
    """Return all degree-1 nodes (road endpoints / dead-ends)."""
    return [n for n in G.nodes() if G.degree(n) == 1]


def _get_junctions(G) -> list:
    """Return all degree>2 nodes (road intersections)."""
    return [n for n in G.nodes() if G.degree(n) > 2]


# ---------------------------------------------------------------------------
# Spatial matching helpers
# ---------------------------------------------------------------------------

def _nearest_distance(node, candidates) -> float:
    """Euclidean distance from node to the closest candidate node."""
    if not candidates:
        return float("inf")
    y0, x0 = node
    dists = [math.sqrt((y0 - y1) ** 2 + (x0 - x1) ** 2) for y1, x1 in candidates]
    return min(dists)


def _match_nodes(pred_nodes, gt_nodes, tolerance: float) -> tuple:
    """
    Compute precision / recall of node matching within a spatial tolerance.

    A predicted node is a true positive if there is a GT node within `tolerance`
    pixels.  A GT node is recalled if there is a predicted node within tolerance.

    Returns (precision, recall, f1).
    """
    eps = 1e-7

    if not gt_nodes and not pred_nodes:
        return 1.0, 1.0, 1.0

    if not gt_nodes:
        return 0.0, 1.0, 0.0   # nothing to recall, all pred are FP

    if not pred_nodes:
        return 1.0, 0.0, 0.0   # nothing predicted

    # Precision: fraction of pred nodes that are close to any GT node
    tp_pred = sum(
        1 for p in pred_nodes if _nearest_distance(p, gt_nodes) <= tolerance
    )
    precision = tp_pred / (len(pred_nodes) + eps)

    # Recall: fraction of GT nodes covered by any pred node
    tp_gt = sum(
        1 for g in gt_nodes if _nearest_distance(g, pred_nodes) <= tolerance
    )
    recall = tp_gt / (len(gt_nodes) + eps)

    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return float(precision), float(recall), float(f1)


# ---------------------------------------------------------------------------
# Connectivity metric
# ---------------------------------------------------------------------------

def _connectivity_preservation(G_pred, G_gt) -> float:
    """
    Measure how well the predicted graph preserves the connectivity structure
    of the GT graph.

    Score = 1 - |n_cc_gt - n_cc_pred| / (n_cc_gt + eps)
    Clamped to [0, 1].
    """
    if not _NX_AVAILABLE:
        return 0.0

    n_gt   = nx.number_connected_components(G_gt)   if G_gt.number_of_nodes()   > 0 else 0
    n_pred = nx.number_connected_components(G_pred) if G_pred.number_of_nodes() > 0 else 0

    if n_gt == 0:
        return 1.0 if n_pred == 0 else 0.0

    return float(max(0.0, 1.0 - abs(n_gt - n_pred) / (n_gt + 1e-7)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_topo(
    pred_mask,
    gt_mask,
    tolerance: float = 5.0,
    min_edge_length: int = 2,
) -> dict:
    """
    Compute topological graph metrics between predicted and GT masks.

    Evaluates endpoint preservation, junction preservation, and connectivity.

    Args:
        pred_mask:       predicted binary road mask (numpy or torch, H×W).
        gt_mask:         ground-truth binary road mask (numpy or torch, H×W).
        tolerance:       max pixel distance for two nodes to be considered matched.
        min_edge_length: passed to graph_adapter.mask_to_graph().

    Returns:
        dict with keys: "precision", "recall", "f1"
        (averaged across endpoint, junction, and connectivity sub-metrics)
    """
    if not _NX_AVAILABLE:
        raise ImportError("networkx is required for TOPO metrics.")

    try:
        G_pred = mask_to_graph(pred_mask, min_edge_length=min_edge_length)
        G_gt   = mask_to_graph(gt_mask,   min_edge_length=min_edge_length)
    except Exception as e:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "error": str(e)}

    # --- endpoint score ---
    ep_pred = _get_endpoints(G_pred)
    ep_gt   = _get_endpoints(G_gt)
    ep_prec, ep_rec, ep_f1 = _match_nodes(ep_pred, ep_gt, tolerance)

    # --- junction score ---
    jn_pred = _get_junctions(G_pred)
    jn_gt   = _get_junctions(G_gt)
    jn_prec, jn_rec, jn_f1 = _match_nodes(jn_pred, jn_gt, tolerance)

    # --- connectivity score (used as equal third component) ---
    conn = _connectivity_preservation(G_pred, G_gt)

    # Macro-average across the three sub-metrics
    precision = (ep_prec + jn_prec + conn) / 3.0
    recall    = (ep_rec  + jn_rec  + conn) / 3.0
    eps = 1e-7
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    return {
        "precision":          float(precision),
        "recall":             float(recall),
        "f1":                 float(f1),
        # detailed breakdown
        "endpoint_precision": float(ep_prec),
        "endpoint_recall":    float(ep_rec),
        "endpoint_f1":        float(ep_f1),
        "junction_precision": float(jn_prec),
        "junction_recall":    float(jn_rec),
        "junction_f1":        float(jn_f1),
        "connectivity":       float(conn),
    }


def compute_topological_similarity(pred_mask, gt_mask, tolerance: float = 5.0) -> float:
    """
    Single scalar topological similarity score derived from TOPO F1.
    Range [0, 1] — 1 means topologically identical graphs.
    """
    result = compute_topo(pred_mask, gt_mask, tolerance=tolerance)
    return result.get("f1", 0.0)
