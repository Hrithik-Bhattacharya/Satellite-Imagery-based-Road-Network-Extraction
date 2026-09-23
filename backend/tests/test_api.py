import os
import sys
import pytest
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import io

# Add repo root to sys path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from backend.api import app, load_model

client = TestClient(app)

def test_predict_endpoint():
    # Force load model for the test client
    load_model()
    
    # Create a dummy image
    img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    image = Image.fromarray(img_array)
    
    # Save to BytesIO
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # Send request
    response = client.post(
        "/predict",
        files={"file": ("dummy.png", img_byte_arr, "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "nodes" in data
    assert "edges" in data
    assert "image_width" in data
    assert "image_height" in data
    assert data["image_width"] == 256
    assert data["image_height"] == 256
