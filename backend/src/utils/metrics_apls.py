"""
metrics_apls.py – Member 4 (Validation & MLOps)
Average Path Length Similarity (APLS) metric for road network evaluation.

Reference:
    van Etten et al., "City-Scale Road Extraction from Satellite Imagery" (2018).
    SpaceNet challenge evaluation methodology.

Internally uses graph_adapter.mask_to_graph() to convert masks to graphs,
then compares shortest path lengths between matched node pairs.

Returns:
    {
        "apls":   float in [0, 1],   1 = perfect match
        "routes": int                number of sampled route pairs evaluated
    }

Handles gracefully:
  - disconnected graphs
  - empty masks / graphs
  - missing paths (treated as infinite cost)
"""

from __future__ import annotations

import math
import random
import numpy as np

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False

from .graph_adapter import mask_to_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _euclidean(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _all_pairs_shortest_paths(G, weight: str = "length"):
    """
    Return a dict of dicts: path_lengths[src][dst] = shortest path length.
    Uses networkx's single-source Dijkstra for each node in the graph.
    Limits to graphs ≤ 500 nodes to keep runtime tractable.
    """
    if not _NX_AVAILABLE:
        return {}

    nodes = list(G.nodes())
    N = len(nodes)
    if N == 0:
        return {}

    # Subsample if graph is large to keep evaluation fast
    MAX_NODES = 200
    if N > MAX_NODES:
        nodes = random.sample(nodes, MAX_NODES)

    path_lengths = {}
    for src in nodes:
        lengths = nx.single_source_dijkstra_path_length(G, src, weight=weight)
        path_lengths[src] = dict(lengths)

    return path_lengths


def _sample_node_pairs(G, n_samples: int = 100):
    """Sample up to n_samples unique node pairs from the graph."""
    nodes = list(G.nodes())
    N = len(nodes)
    if N < 2:
        return []
    pairs = set()
    attempts = 0
    max_attempts = n_samples * 10
    while len(pairs) < n_samples and attempts < max_attempts:
        a, b = random.sample(nodes, 2)
        pairs.add((min(a, b), max(a, b)))
        attempts += 1
    return list(pairs)


# ---------------------------------------------------------------------------
# Core APLS computation
# ---------------------------------------------------------------------------

def _compute_apls_from_graphs(G_pred, G_gt, n_samples: int = 100) -> dict:
    """
    Compute APLS between two NetworkX graphs.

    Strategy:
      1. For each GT node, find the spatially nearest node in Pred.
      2. Sample pairs of GT nodes that are connected.
      3. Compare GT path length vs Pred path length (via mapped nodes).
      4. APLS = mean(1 - |L_gt - L_pred| / L_gt) clamped to [0,1].
    """
    if G_gt.number_of_nodes() == 0 or G_pred.number_of_nodes() == 0:
        return {"apls": 0.0, "routes": 0}

    gt_nodes  = list(G_gt.nodes())
    pred_nodes = list(G_pred.nodes())

    # Build spatial mapping: each GT node → nearest Pred node
    # (simple nearest-neighbour in pixel space)
    pred_arr = np.array(pred_nodes, dtype=np.float32)  # (M, 2)

    def nearest_pred(gt_node):
        diff = pred_arr - np.array(gt_node, dtype=np.float32)
        dists = np.sum(diff ** 2, axis=1)
        return pred_nodes[int(np.argmin(dists))]

    node_map = {n: nearest_pred(n) for n in gt_nodes}

    # Sample connected pairs in the GT graph
    pairs = _sample_node_pairs(G_gt, n_samples)
    if not pairs:
        return {"apls": 0.0, "routes": 0}

    scores = []
    for (a, b) in pairs:
        # GT path length
        try:
            l_gt = nx.shortest_path_length(G_gt, a, b, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        if l_gt <= 0:
            continue

        # Mapped Pred path length
        pa = node_map.get(a)
        pb = node_map.get(b)
        if pa is None or pb is None or pa == pb:
            scores.append(0.0)
            continue

        try:
            l_pred = nx.shortest_path_length(G_pred, pa, pb, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Disconnected in prediction — full penalty
            scores.append(0.0)
            continue

        score = max(0.0, 1.0 - abs(l_gt - l_pred) / (l_gt + 1e-7))
        scores.append(score)

    if not scores:
        return {"apls": 0.0, "routes": 0}

    return {"apls": float(np.mean(scores)), "routes": len(scores)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_apls(
    pred_mask,
    gt_mask,
    n_samples: int = 100,
    min_edge_length: int = 2,
) -> dict:
    """
    Compute APLS metric from raw binary road masks.

    Args:
        pred_mask: predicted binary road mask (numpy or torch, H×W).
        gt_mask:   ground-truth binary road mask (numpy or torch, H×W).
        n_samples: number of node-pair routes to sample for evaluation.
        min_edge_length: minimum edge length passed to graph simplification.

    Returns:
        dict with keys:
            "apls"   – float in [0, 1]
            "routes" – int, number of evaluated routes
    """
    if not _NX_AVAILABLE:
        raise ImportError("networkx is required for APLS. Install it with: pip install networkx")

    try:
        G_pred = mask_to_graph(pred_mask, min_edge_length=min_edge_length)
        G_gt   = mask_to_graph(gt_mask,   min_edge_length=min_edge_length)
    except Exception as e:
        # graph construction failed — return zero score, don't crash
        return {"apls": 0.0, "routes": 0, "error": str(e)}

    return _compute_apls_from_graphs(G_pred, G_gt, n_samples=n_samples)


def compute_graph_connectivity_similarity(pred_mask, gt_mask) -> float:
    """
    Graph connectivity similarity: ratio of connected components match.
    Complementary to APLS — measures structural completeness.

    Returns a float in [0, 1].
    """
    try:
        G_pred = mask_to_graph(pred_mask)
        G_gt   = mask_to_graph(gt_mask)
    except Exception:
        return 0.0

    if G_gt.number_of_nodes() == 0:
        return 1.0 if G_pred.number_of_nodes() == 0 else 0.0

    n_gt   = nx.number_connected_components(G_gt)   if G_gt.number_of_nodes()   > 0 else 0
    n_pred = nx.number_connected_components(G_pred) if G_pred.number_of_nodes() > 0 else 0

    if n_gt == 0:
        return 1.0

    # Similarity: 1 - normalised difference
    return float(max(0.0, 1.0 - abs(n_gt - n_pred) / (n_gt + 1e-7)))
