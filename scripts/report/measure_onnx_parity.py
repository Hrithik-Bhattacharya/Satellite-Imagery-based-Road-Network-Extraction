"""
Measures PyTorch vs ONNX Runtime agreement for the current model and writes
figures/real/measurements/onnx_parity.json (read by build_report.py).

Uses random inputs at a square and a non-square size, so it also checks that height and
width were exported as dynamic axes. Sizes stay small because a full 1024² PyTorch pass can
exceed free RAM on the laptop used for the report.

Usage (repo root):  python scripts/report/measure_onnx_parity.py
"""

import json
import os
import sys

import numpy as np
import onnxruntime as ort
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from backend.src.models.mobilevit_v2 import load_mobilevit_checkpoint  # noqa: E402

SIZES = [(256, 256), (256, 384), (512, 512)]


def main():
    torch.set_grad_enabled(False)
    model, _ = load_mobilevit_checkpoint(os.path.join(REPO, "models", "best_model_v2.pth"))
    model.eval()
    session = ort.InferenceSession(os.path.join(REPO, "models", "mobilevit_v2.onnx"),
                                   providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    gen = torch.Generator().manual_seed(0)
    runs = []
    for h, w in SIZES:
        x = torch.randn(1, 3, h, w, generator=gen)
        ref = torch.sigmoid(model(x)).numpy()          # the ONNX graph includes the sigmoid
        out = session.run(None, {name: x.numpy()})[0]
        runs.append({"height": h, "width": w, "max_abs_diff": float(np.abs(ref - out).max())})
    result = {"checkpoint": "models/best_model_v2.pth", "onnx": "models/mobilevit_v2.onnx", "runs": runs,
              "max_abs_diff": max(r["max_abs_diff"] for r in runs)}
    path = os.path.join(REPO, "figures", "real", "measurements", "onnx_parity.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
