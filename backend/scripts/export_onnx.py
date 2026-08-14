"""
export_onnx.py -- Export a trained MobileViT_v2 checkpoint to ONNX for edge deployment.

Usage (from repo root):
    python backend/scripts/export_onnx.py \
        --checkpoint models/best_model_new.pth \
        --output models/mobilevit_v2.onnx

The exported graph includes the sigmoid activation, so ONNX Runtime output is a
[0, 1] probability map directly -- no need to reimplement sigmoid on the edge device.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import MobileViT_v2


class SigmoidWrapper(nn.Module):
    """Bakes the sigmoid into the exported graph so ONNX output is a probability map."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


def find_default_checkpoint(repo_root: str) -> str:
    candidates = [
        os.path.join(repo_root, "models", "best_model_v2.pth"),
        os.path.join(repo_root, "models", "best_model_new.pth"),
        os.path.join(repo_root, "models", "best_model.pth"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[1]  # best_model_new.pth, used for the error message if nothing found


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Pass --checkpoint pointing at a trained .pth file (see models/)."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint) else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"    Missing (randomly initialized): {missing}")
    if unexpected:
        print(f"    Unexpected (ignored): {unexpected}")


def verify_parity(torch_model: nn.Module, onnx_path: str, dummy_input: torch.Tensor) -> float:
    """Runs the exported ONNX graph and compares it against the PyTorch model, at both the
    exact export shape AND a different (non-square) shape -- the latter is the real test that
    height/width were actually exported as dynamic axes rather than baked in from dummy_input.
    Returns the largest max-abs-diff seen across both checks."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    worst = 0.0
    test_inputs = [dummy_input, torch.randn(1, 3, 320, 384)]
    for test_input in test_inputs:
        with torch.no_grad():
            torch_out = torch_model(test_input).numpy()
        onnx_out = session.run(None, {"input_image": test_input.numpy()})[0]
        worst = max(worst, float(np.abs(torch_out - onnx_out).max()))
    return worst


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    width_mult: float,
    img_size: int,
    opset: int,
) -> None:
    print("Initializing ONNX Export for Edge Deployment...")

    # 1. Initialize model and load TRAINED weights (width_mult must match training --
    #    every checkpoint in this repo was trained at width_mult=1.0).
    base_model = MobileViT_v2(num_classes=1, width_mult=width_mult)
    print(f"Loading trained weights from {checkpoint_path}...")
    load_checkpoint(base_model, checkpoint_path)
    base_model.eval()

    model = SigmoidWrapper(base_model)
    model.eval()

    # 2. Dummy input -- batch size 1 for edge inference, dynamic axis still allows batching.
    dummy_input = torch.randn(1, 3, img_size, img_size)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # 3. Export
    try:
        print(f"Exporting model to {output_path}...")
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input_image"],
            output_names=["road_probability"],
            # The architecture is fully-convolutional / patch-based (no dims baked in beyond
            # divisibility by 32), and predict_single_image.py relies on running inference on
            # full-resolution tiles rather than fixed 256x256 crops -- so height/width must be
            # dynamic here too, not just batch, or the exported graph only accepts the exact
            # img_size used for the dummy export input.
            dynamic_axes={
                "input_image": {0: "batch_size", 2: "height", 3: "width"},
                "road_probability": {0: "batch_size", 2: "height", 3: "width"},
            },
        )
        print("SUCCESS: Model exported to ONNX.")
        print(f"File size: {os.path.getsize(output_path) / (1024 * 1024):.3f} MB")
    except Exception as e:
        print("FAILED: Error during ONNX export.")
        print(e)
        raise

    # 4. Verify PyTorch <-> ONNX Runtime parity so a silent export bug can't ship.
    try:
        max_diff = verify_parity(model, output_path, dummy_input)
        status = "OK" if max_diff < 1e-3 else "WARNING: large discrepancy"
        print(f"Parity check (PyTorch vs ONNX Runtime, random input): max abs diff = {max_diff:.6f} [{status}]")
    except ImportError:
        print("onnxruntime not installed -- skipping parity check (pip install onnxruntime).")


if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser = argparse.ArgumentParser(description="Export a trained MobileViT_v2 checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=str, default=find_default_checkpoint(repo_root),
                         help="Path to a trained .pth checkpoint (default: auto-detect in models/)")
    parser.add_argument("--output", type=str, default=os.path.join(repo_root, "models", "mobilevit_v2.onnx"),
                         help="Output .onnx path")
    parser.add_argument("--width_mult", type=float, default=1.0,
                         help="Must match the width_mult the checkpoint was trained with (default 1.0)")
    parser.add_argument("--img_size", type=int, default=256, help="Export input tile size (must be divisible by 32)")
    parser.add_argument("--opset", type=int, default=18,
                         help="ONNX opset version. 18 matches what torch's exporter natively "
                              "produces for this graph -- lower values trigger a version-downgrade "
                              "step that fails on this model's Pad op and silently falls back to 18 "
                              "anyway, so there's no benefit to requesting less.")
    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.output, args.width_mult, args.img_size, args.opset)
