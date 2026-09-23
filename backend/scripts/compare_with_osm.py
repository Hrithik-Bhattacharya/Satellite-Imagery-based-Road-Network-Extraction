import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
import networkx as nx

# Add repo root to sys path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from backend.src.models.mobilevit_v2 import MobileViT_v2
from backend.src.utils.graph_builder import get_skeleton_from_mask, build_graph_from_skeleton, simplify_graph

try:
    import osmnx as ox
except ImportError:
    print("Please install osmnx: pip install osmnx")
    sys.exit(1)

def load_model(device):
    model_path = os.path.join(repo_root, "backend", "models", "best_model.pth")
    model = MobileViT_v2(num_classes=1, width_mult=1.0)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def main():
    print("--- OSM Comparison Tool ---")
    
    # 1. Define bounding box (e.g., a rural area in India)
    # format: (north, south, east, west)
    bbox = (28.7041, 28.6941, 77.1125, 77.1025) 
    
    print(f"Fetching OpenStreetMap road graph for bbox {bbox}...")
    try:
        # Get drivable roads from OSM (OSMnx v2+ takes bbox tuple)
        G_osm = ox.graph_from_bbox(bbox=bbox, network_type='drive')
        osm_edges = len(G_osm.edges())
        osm_nodes = len(G_osm.nodes())
        print(f"[OSM] Found {osm_nodes} nodes and {osm_edges} edges.")
    except Exception as e:
        print(f"Failed to fetch OSM data: {e}")
        osm_edges = 0
        
    print("\nSimulating model inference over the same bounding box...")
    
    # 2. Run model
    # (In a real scenario, we would use an XYZ tile service to download the satellite image
    # for this exact bounding box. Here we just run on a dummy image for demonstration)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    
    # Dummy inference on a blank image to simulate the pipeline
    img_tensor = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
    pred_mask = preds.squeeze().cpu().numpy()
    pred_mask = (pred_mask * 255).astype(np.uint8)
    
    skel = get_skeleton_from_mask(pred_mask)
    G_pred = build_graph_from_skeleton(skel)
    G_pred = simplify_graph(G_pred, min_length=2)
    
    pred_edges = len(G_pred.edges())
    pred_nodes = len(G_pred.nodes())
    
    print(f"[AI Model] Extracted {pred_nodes} nodes and {pred_edges} edges.")
    
    # 3. Compare and compute missing roads
    # This is a conceptual comparison. A real script would project coordinates 
    # and compute spatial intersection.
    
    print("\n--- Results ---")
    if pred_edges > osm_edges:
        print(f"🔥 Missing Roads Discovered: {pred_edges - osm_edges} more edges extracted by AI than mapped on OSM!")
    else:
        print(f"OSM has more mapped edges in this bounding box.")

if __name__ == "__main__":
    main()
