import numpy as np
from skimage.morphology import skeletonize
import networkx as nx
from collections import deque


def get_skeleton_from_mask(mask):
    """Convert binary mask (0/255 or boolean) to skeleton (bool ndarray)."""
    if mask.dtype != bool:
        binary = mask > 127
    else:
        binary = mask
    return skeletonize(binary).astype(np.uint8)


def _neighbors(y, x, shape):
    H, W = shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                yield ny, nx


def build_graph_from_skeleton(skel):
    """Build a NetworkX graph from a 1-pixel-wide skeleton image.

    Nodes are placed at endpoints and junctions (degree != 2). Edges
    follow skeleton paths between nodes and carry a `pixels` list and `length`.
    Returns an undirected NetworkX Graph with nodes keyed by (y,x) tuples.
    """
    if skel.dtype != np.uint8:
        skel = skel.astype(np.uint8)

    H, W = skel.shape
    pts = set(zip(*np.where(skel > 0)))

    # degree for skeleton pixels
    degree = {}
    for p in pts:
        y, x = p
        cnt = 0
        for ny, nx in _neighbors(y, x, (H, W)):
            if (ny, nx) in pts:
                cnt += 1
        degree[p] = cnt

    # nodes are endpoints (deg==1) or junctions (deg>2)
    nodes = {p for p, d in degree.items() if d == 1 or d > 2}

    # If closed loops exist (all deg==2), pick an arbitrary pixel as a node
    if len(nodes) == 0 and len(pts) > 0:
        nodes.add(next(iter(pts)))

    G = nx.Graph()

    # add nodes to graph
    for n in nodes:
        G.add_node(n)

    visited = set()

    def walk_path(start, neighbor):
        """Walk from start pixel into neighbor until another node is reached."""
        path = [start]
        cur = neighbor
        prev = start
        while True:
            path.append(cur)
            visited.add(cur)
            if cur in nodes and cur != start:
                return cur, path
            # find next neighbors excluding previous
            nexts = [p for p in _neighbors(cur[0], cur[1], (H, W)) if p in pts and p != prev]
            if not nexts:
                # dead end
                return cur, path
            # pick the next that wasn't visited if possible
            prev, cur = cur, nexts[0]

    # For each node, start walking down each outgoing branch
    for n in nodes:
        y, x = n
        for nb in _neighbors(y, x, (H, W)):
            if nb in pts and nb not in visited:
                target, path = walk_path(n, nb)
                if target is None:
                    continue
                # determine edge endpoints (n and target)
                a, b = tuple(n), tuple(target)
                if a == b:
                    continue
                if G.has_edge(a, b):
                    # if edge exists, maybe keep the shorter representation
                    continue
                G.add_edge(a, b, pixels=path, length=len(path))

    return G


def simplify_graph(G, min_length=2):
    """Remove tiny spurious edges (optionally) by pruning edges shorter than min_length."""
    H = G.copy()
    for u, v, d in list(G.edges(data=True)):
        if d.get("length", 0) < min_length:
            H.remove_edge(u, v)
    # remove isolated nodes
    for n in list(H.nodes()):
        if H.degree(n) == 0:
            H.remove_node(n)
    return H


if __name__ == "__main__":
    # quick smoke test using a synthetic T-junction
    import cv2
    import matplotlib.pyplot as plt

    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.line(img, (20, 128), (236, 128), color=255, thickness=3)
    cv2.line(img, (128, 20), (128, 128), color=255, thickness=3)

    sk = get_skeleton_from_mask(img)
    G = build_graph_from_skeleton(sk)
    print("Nodes:", len(G.nodes()))
    print("Edges:", len(G.edges()))

    # visualize
    plt.imshow(sk, cmap="gray")
    ys = [n[0] for n in G.nodes()]
    xs = [n[1] for n in G.nodes()]
    plt.scatter(xs, ys, c="red")
    for u, v, d in G.edges(data=True):
        pix = np.array(d["pixels"])
        plt.plot(pix[:, 1], pix[:, 0], c="yellow")
    plt.gca().invert_yaxis()
    plt.show()
