"""
FastAPI REST endpoint for rural road extraction.
Uses ONNX Runtime — no PyTorch required.

Run:
    uvicorn backend.api:app --reload
    # or: make api

POST /predict   — upload a satellite tile, receive road mask PNG + graph JSON
GET  /health    — liveness check
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from PIL import Image
import io, os, sys, base64

import onnxruntime as ort
import scipy.ndimage as ndi

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from backend.src.utils.graph_builder import get_skeleton_from_mask, build_graph_from_skeleton, simplify_graph

# ── constants matching predict_onnx.py ────────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
HIGH_THRESH = 0.35
LOW_THRESH  = 0.12
CLOSE_K     = 5

app = FastAPI(title="Rural Road Extraction API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── model loading ──────────────────────────────────────────────────────────────
ONNX_PATH = os.path.join(repo_root, "models", "mobilevit_v2.onnx")
_session = None

def get_session():
    global _session
    if _session is None:
        if not os.path.exists(ONNX_PATH):
            raise RuntimeError(f"ONNX model not found at {ONNX_PATH}")
        so = ort.SessionOptions()
        so.enable_cpu_mem_arena  = False
        so.enable_mem_pattern    = False
        so.inter_op_num_threads  = 2
        so.intra_op_num_threads  = 4
        _session = ort.InferenceSession(ONNX_PATH, sess_options=so,
                                        providers=["CPUExecutionProvider"])
    return _session


# ── preprocessing / postprocessing ────────────────────────────────────────────
def preprocess(img_np: np.ndarray) -> np.ndarray:
    x = img_np.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None]            # (1, 3, H, W)


def postprocess(prob: np.ndarray) -> np.ndarray:
    """Hysteresis threshold → morphological close → binary uint8 mask."""
    strong = prob >= HIGH_THRESH
    weak   = prob >= LOW_THRESH
    filled = ndi.binary_fill_holes(strong)
    connected = weak & ndi.binary_dilation(filled, iterations=3)
    mask = (connected * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def run_tta(session, x: np.ndarray) -> np.ndarray:
    """4-flip test-time augmentation, returns averaged probability map."""
    inp = session.get_inputs()[0].name
    probs = []
    for flip in [None, 0, 1, (0, 1)]:
        xi = np.flip(x, flip).copy() if flip is not None else x
        p  = session.run(None, {inp: xi})[0][0, 0]
        if flip is not None:
            p = np.flip(p, flip)
        probs.append(p)
    return np.mean(probs, axis=0)


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(ONNX_PATH),
            "model_exists": os.path.exists(ONNX_PATH)}


@app.post("/predict")
async def predict(file: UploadFile = File(...), tta: bool = True):
    """
    Upload a satellite tile (JPG/PNG). Returns:
      - mask_b64: base64-encoded PNG of the binary road mask
      - road_frac: fraction of pixels predicted as road
      - graph: {nodes, edges} road network graph
    """
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    try:
        session = get_session()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    x = preprocess(img_np)
    prob = run_tta(session, x) if tta else session.run(
        None, {session.get_inputs()[0].name: x})[0][0, 0]

    mask = postprocess(prob)
    road_frac = float((mask > 127).mean())

    # encode mask as base64 PNG
    _, buf = cv2.imencode(".png", mask)
    mask_b64 = base64.b64encode(buf).decode()

    # build road graph
    try:
        skel = get_skeleton_from_mask(mask)
        G    = build_graph_from_skeleton(skel)
        G    = simplify_graph(G, min_length=5)
        nodes = [{"id": f"{int(n[0])}_{int(n[1])}", "x": int(n[1]), "y": int(n[0])}
                 for n in G.nodes()]
        edges = []
        for u, v, d in G.edges(data=True):
            pix  = d.get("pixels", [])
            path = [{"x": int(p[1]), "y": int(p[0])} for p in pix]
            edges.append({"source": f"{int(u[0])}_{int(u[1])}",
                          "target": f"{int(v[0])}_{int(v[1])}", "path": path})
        graph = {"nodes": nodes, "edges": edges}
    except Exception:
        graph = {"nodes": [], "edges": []}

    return JSONResponse({
        "mask_b64":    mask_b64,
        "road_frac":   round(road_frac, 4),
        "image_width": img_np.shape[1],
        "image_height": img_np.shape[0],
        "tta":         tta,
        "graph":       graph,
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
