import io
import os
import sys

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from backend.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_exists"] is True


def test_predict_endpoint():
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post("/predict?tta=false", files={"file": ("dummy.png", buf, "image/png")})

    assert response.status_code == 200
    data = response.json()
    assert data["image_width"] == 256 and data["image_height"] == 256
    assert {"mask_b64", "road_frac", "graph"} <= set(data)
    assert {"nodes", "edges"} <= set(data["graph"])
