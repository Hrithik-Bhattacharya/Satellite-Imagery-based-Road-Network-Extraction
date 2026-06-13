**Graph Builder — Overview**

This repository contains utilities to convert 1-pixel-wide skeletonized road masks into a graph representation (nodes and edges) suitable for downstream analysis or training.

**Files**
- `graph_builder.py`: Skeleton -> NetworkX graph conversion. Key functions:
  - `get_skeleton_from_mask(mask)` — skeletonize a binary mask (returns uint8 skeleton).
  - `build_graph_from_skeleton(skel)` — builds a graph where nodes are endpoints/junctions and edges carry pixel paths and length.
  - `simplify_graph(G, min_length=2)` — remove short spurious edges and isolated nodes.
- `test_graph.py`: Small test harness that generates a synthetic mask, runs skeletonization, builds the graph, prints summary, and visualizes results.
- `requirements.txt`: Python dependencies.

**Quick Start**

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

2. Run the test/visualization script:

```bash
python test_graph.py
```

This opens a matplotlib window showing the original synthetic mask and the skeleton overlaid with detected graph nodes and edges.

**Using with your data**

1. Generate or load a mask where roads are white (255) and background is 0. You can use the `simulate_weak_labels_deepglobe` output from `weak_labels.py` or OSM-rasterized masks.
2. Produce a skeleton (optional — `build_graph_from_skeleton` accepts an already-skeletonized image):

```python
from graph_builder import get_skeleton_from_mask, build_graph_from_skeleton
sk = get_skeleton_from_mask(mask)
G = build_graph_from_skeleton(sk)
```

3. Map pixel coordinates to geographic coordinates if your source image is georeferenced: use the image affine transform from `rasterio.open(...).transform` to convert pixel (row, col) to world coordinates.

**Output & Export**

- The graph `G` is a `networkx.Graph` with node keys as `(row, col)` tuples.
- Each edge contains attributes:
  - `pixels`: ordered list of pixel coordinates along the edge (from node A to node B)
  - `length`: number of pixels in the path

You can export the graph using NetworkX exporters, or convert nodes/edges to GeoJSON by mapping pixel coords to geographic coords.

**Next steps / Suggestions**
- Add a function to export graphs to GeoJSON/GraphML.
- Add a CLI to batch-process a directory of masks and save graphs.
- Integrate with `weak_labels.py` to convert rasterized OSM centerlines directly into graphs and merge with model outputs.

**Contact / Notes**
This doc was generated alongside the code in `graph_builder.py` and `test_graph.py` to help member 2 start using the graph extraction pipeline.
