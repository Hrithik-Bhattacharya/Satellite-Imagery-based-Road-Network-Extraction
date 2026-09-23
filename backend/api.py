from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import cv2
import numpy as np
from PIL import Image
import io
import os
import sys

# Ensure backend src is in path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from backend.src.models.mobilevit_v2 import MobileViT_v2
from backend.src.utils.graph_builder import get_skeleton_from_mask, build_graph_from_skeleton, simplify_graph

app = FastAPI(title="Topological Rural Road Extraction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
model = None

@app.on_event("startup")
def load_model():
    global model
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
    print("Model loaded successfully.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(image)
    
    # Preprocess
    orig_h, orig_w = img_np.shape[:2]
    img_resized = cv2.resize(img_np, (256, 256))
    img_tensor = torch.tensor(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
    pred_mask = preds.squeeze().cpu().numpy()
    pred_mask = (pred_mask * 255).astype(np.uint8)
    
    # Post-process
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_closed = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)
    
    final_mask = cv2.resize(mask_closed, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # Extract Graph
    skel = get_skeleton_from_mask(final_mask)
    G = build_graph_from_skeleton(skel)
    G = simplify_graph(G, min_length=5)
    
    nodes = [{"id": f"{int(n[0])}_{int(n[1])}", "x": int(n[1]), "y": int(n[0])} for n in G.nodes()]
    edges = []
    for u, v, d in G.edges(data=True):
        pix = d.get("pixels", [])
        if len(pix) > 0:
            path = [{"x": int(p[1]), "y": int(p[0])} for p in pix]
            edges.append({"source": f"{int(u[0])}_{int(u[1])}", "target": f"{int(v[0])}_{int(v[1])}", "path": path})
            
    return {"nodes": nodes, "edges": edges, "image_width": int(orig_w), "image_height": int(orig_h)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
