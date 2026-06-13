"""
graph_adapter.py – Member 4 (Validation & MLOps)

Bridge between raw binary road masks and the NetworkX graph representation
produced by Member 2's graph_builder module.

Only imports from Member 2:
    get_skeleton_from_mask
    build_graph_from_skeleton
    simplify_graph

Pipeline:
    mask → skeleton → raw graph → simplified graph

COMPATIBILITY NOTE:
    Member 2's graph_builder.build_graph_from_skeleton contains a Python scoping
    bug: the loop variable `nx` in `for ny, nx in _neighbors(...)` shadows the
    module-level `import networkx as nx` throughout the entire function.
    We monkey-patch the function with a corrected re-implementation that uses
    `nc` (col) instead of `nx` as the column loop variable.
    We do NOT modify graph_builder.py.

Usage:
    from backend.src.utils.graph_adapter import mask_to_graph
    G = mask_to_graph(mask_array)   # returns networkx.Graph
"""

from __future__ import annotations

import sys
import os
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Dynamic import of Member 2's graph_builder
# ---------------------------------------------------------------------------

_SEARCH_PATHS = [
    # Merged repo layout: backend/ sits next to graph_builder.py
    os.path.join(os.path.dirname(__file__), "..", "..", ".."),
    # Side-by-side member repos (standard project layout)
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "..",
        "Member 2", "Satellite-Imagery-based-Road-Network-Extraction-hrithik"
    ),
    # CI / Docker flat layout
    "/workspace/graph",
]


def _import_graph_builder():
    """Locate and import graph_builder from Member 2."""
    try:
        import graph_builder as gb
        return gb
    except ImportError:
        pass

    for candidate in _SEARCH_PATHS:
        candidate = os.path.abspath(candidate)
        if os.path.isfile(os.path.join(candidate, "graph_builder.py")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            try:
                import graph_builder as gb
                return gb
            except ImportError:
                continue
                
    # Fallback: deep search in the workspace root (e.g. after merge)
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    if os.path.exists(workspace_root):
        for root, dirs, files in os.walk(workspace_root):
            if "graph_builder.py" in files:
                if root not in sys.path:
                    sys.path.insert(0, root)
                try:
                    import graph_builder as gb
                    return gb
                except ImportError:
                    continue

    raise ImportError(
        "Cannot locate graph_builder.py (Member 2). "
        "Add the folder containing graph_builder.py to PYTHONPATH."
    )


def _patch_graph_builder(gb_module):
    """
    Monkey-patch build_graph_from_skeleton to fix the nx-variable shadowing bug.

    The original function uses `for ny, nx in _neighbors(...)` which makes `nx`
    a local variable in the entire function scope, shadowing `import networkx as nx`.
    Our fixed version uses `nr, nc` (row, col) as loop variables.
    """
    import networkx as _networkx
    import numpy as _np

    def _build_graph_fixed(skel):
        """Corrected build_graph_from_skeleton — variable nx renamed to nc."""
        if skel.dtype != _np.uint8:
            skel = skel.astype(_np.uint8)

        H, W = skel.shape
        pts = set(zip(*_np.where(skel > 0)))

        degree = {}
        for p in pts:
            row, col = p
            cnt = 0
            for nr, nc in gb_module._neighbors(row, col, (H, W)):
                if (nr, nc) in pts:
                    cnt += 1
            degree[p] = cnt

        nodes = {p for p, d in degree.items() if d == 1 or d > 2}

        if len(nodes) == 0 and len(pts) > 0:
            nodes.add(next(iter(pts)))

        G = _networkx.Graph()

        for n in nodes:
            G.add_node(n)

        visited = set()

        def walk_path(start, neighbor):
            path = [start]
            cur = neighbor
            prev = start
            while True:
                path.append(cur)
                visited.add(cur)
                if cur in nodes and cur != start:
                    return cur, path
                nexts = [
                    p2 for p2 in gb_module._neighbors(cur[0], cur[1], (H, W))
                    if p2 in pts and p2 != prev
                ]
                if not nexts:
                    return cur, path
                prev, cur = cur, nexts[0]

        for n in nodes:
            row, col = n
            for nr, nc in gb_module._neighbors(row, col, (H, W)):
                nb = (nr, nc)
                if nb in pts and nb not in visited:
                    target, path = walk_path(n, nb)
                    if target is None:
                        continue
                    a, b = tuple(n), tuple(target)
                    if a == b:
                        continue
                    if G.has_edge(a, b):
                        continue
                    G.add_edge(a, b, pixels=path, length=len(path))

        return G

    gb_module.build_graph_from_skeleton = _build_graph_fixed
    return gb_module


# Module-level lazy cache
_gb = None


def _get_gb():
    global _gb
    if _gb is None:
        raw = _import_graph_builder()
        _gb = _patch_graph_builder(raw)
    return _gb


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mask_to_graph(mask, min_edge_length: int = 2):
    """
    Convert a binary road mask to a simplified NetworkX graph.

    Pipeline: mask → skeleton → graph → simplify_graph

    Args:
        mask: binary mask as:
              - np.ndarray (H, W), dtype uint8 (0/255) or bool or float [0,1]
              - torch.Tensor (H, W) or (1, H, W) or (1, 1, H, W)
        min_edge_length: minimum edge length for simplify_graph.

    Returns:
        networkx.Graph with nodes keyed by (row, col) tuples and edges carrying
        'pixels' and 'length' attributes.
        Returns an empty Graph if the mask is blank or skeleton is empty.
    """
    import networkx as nx
    gb = _get_gb()

    # --- normalise input to 2-D numpy uint8 ---
    if _TORCH_AVAILABLE and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    mask = np.asarray(mask, dtype=np.float32)

    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask.squeeze(0)

    if mask.ndim != 2:
        raise ValueError(f"mask_to_graph expects a 2-D array, got shape {mask.shape}")

    if mask.max() <= 1.0:
        mask = (mask * 255.0).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    if mask.max() == 0:
        return nx.Graph()

    # --- pipeline ---
    skeleton = gb.get_skeleton_from_mask(mask)
    if skeleton.max() == 0:
        return nx.Graph()

    G_raw = gb.build_graph_from_skeleton(skeleton)

    if G_raw.number_of_nodes() == 0:
        return nx.Graph()

    G = gb.simplify_graph(G_raw, min_length=min_edge_length)
    return G


def get_graph_stats(G) -> dict:
    """Return basic graph statistics as a flat dict."""
    import networkx as nx
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    total_length = sum(d.get("length", 1) for _, _, d in G.edges(data=True))
    components = nx.number_connected_components(G) if num_nodes > 0 else 0
    return {
        "nodes":               num_nodes,
        "edges":               num_edges,
        "total_edge_length":   total_length,
        "connected_components": components,
    }
