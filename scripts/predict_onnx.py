"""
predict_onnx.py -- Edge/on-device inference reference implementation.

Runs the exported ONNX model (see backend/scripts/export_onnx.py) through ONNX Runtime
instead of PyTorch, then applies the same hysteresis-threshold + canopy-gap-bridging
postprocessing used by scripts/predict_single_image.py.

Deliberately has NO PyTorch / CUDA / albumentations dependency -- only onnxruntime,
opencv-python, numpy and scipy, so it runs on a companion computer / Jetson-class
device or any machine without a full training environment installed.

Usage:
    python scripts/predict_onnx.py data/samples/117991_sat.jpg \
        --model models/mobilevit_v2.onnx \
        --output results/prediction_117991_onnx.png
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.src.utils.graph_postprocess import connect_canopy_gaps, hysteresis_threshold

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def find_default_model(repo_root: str) -> str:
    candidates = [
        os.path.join(repo_root, "models", "mobilevit_v2.onnx"),
        os.path.join(repo_root, "mobilevit_v2.onnx"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def pick_providers(requested: str | None) -> list:
    available = ort.get_available_providers()
    if requested:
        if requested not in available:
            raise ValueError(f"Provider '{requested}' not available. Available: {available}")
        return [requested]
    # Prefer a hardware accelerator if this ONNX Runtime build has one, else CPU.
    preferred_order = [
        "CUDAExecutionProvider",      # Jetson / discrete GPU
        "TensorrtExecutionProvider",
        "CoreMLExecutionProvider",    # iOS / macOS
        "NnapiExecutionProvider",     # Android
        "CPUExecutionProvider",
    ]
    return [p for p in preferred_order if p in available] or ["CPUExecutionProvider"]


def predict_tta(session: ort.InferenceSession, input_name: str, output_name: str, input_tensor: np.ndarray) -> np.ndarray:
    """Same 4-way flip TTA as scripts/predict_single_image.py's predict_tta(), reimplemented
    in plain NumPy since this script has no PyTorch dependency by design."""
    variants = [
        (lambda x: x, lambda p: p),                            # identity
        (lambda x: x[:, :, :, ::-1], lambda p: p[:, :, :, ::-1]),   # horizontal
        (lambda x: x[:, :, ::-1, :], lambda p: p[:, :, ::-1, :]),   # vertical
        (lambda x: x[:, :, ::-1, ::-1], lambda p: p[:, :, ::-1, ::-1]),  # both
    ]
    probs_sum = None
    for forward_fn, inverse_fn in variants:
        variant_input = np.ascontiguousarray(forward_fn(input_tensor))
        probs = inverse_fn(session.run([output_name], {input_name: variant_input})[0])
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return probs_sum / len(variants)


def preprocess(image_bgr: np.ndarray) -> tuple[np.ndarray, int, int]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    new_h, new_w = (h // 32) * 32, (w // 32) * 32
    resized = cv2.resize(image_rgb, (new_w, new_h))
    normed = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    chw = normed.transpose(2, 0, 1)[None, ...]  # (1, 3, H, W)
    return np.ascontiguousarray(chw), new_h, new_w


def postprocess(
    probs: np.ndarray,
    high_thresh: float,
    low_thresh: float,
    max_gap_dist: float,
    max_angle_deg: float,
    road_width: int,
) -> np.ndarray:
    mask_hyst = hysteresis_threshold(probs, high_thresh=high_thresh, low_thresh=low_thresh)
    kernel_close = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(mask_hyst, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    mask_connected = connect_canopy_gaps(
        mask_closed, max_gap_dist=max_gap_dist, max_angle_deg=max_angle_deg, road_width=road_width
    )
    return mask_connected


def predict(
    image_path: str,
    model_path: str,
    output_path: str,
    provider: str | None,
    high_thresh: float,
    low_thresh: float,
    max_gap_dist: float,
    max_angle_deg: float,
    road_width: int,
    use_tta: bool,
) -> None:
    if not os.path.exists(model_path):
        print(f"Error: ONNX model not found at {model_path}. Export one first with "
              f"backend/scripts/export_onnx.py.")
        sys.exit(1)

    providers = pick_providers(provider)
    print(f"ONNX Runtime providers: {providers}")
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: could not load image {image_path}")
        sys.exit(1)

    input_tensor, h, w = preprocess(image_bgr)

    t0 = time.perf_counter()
    if use_tta:
        probs = predict_tta(session, input_name, output_name, input_tensor)
    else:
        probs = session.run([output_name], {input_name: input_tensor})[0]
    latency_ms = (time.perf_counter() - t0) * 1000
    probs = probs.squeeze()

    pos_frac = float((probs > 0.5).mean())
    print(f"Inference: {latency_ms:.1f} ms | tile {w}x{h} | positive-pixel fraction @0.5: {pos_frac*100:.1f}%")
    if pos_frac > 0.20:
        print("WARNING: predicted positive fraction is unusually high for a road mask -- "
              "check the model checkpoint this ONNX file was exported from.")

    mask = postprocess(probs, high_thresh, low_thresh, max_gap_dist, max_angle_deg, road_width)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, mask)
    print(f"Saved connected road mask to {output_path}")


if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description="Edge/on-device inference via ONNX Runtime (no PyTorch required)")
    parser.add_argument("image_path", type=str, help="Path to the input satellite image")
    parser.add_argument("--model", type=str, default=find_default_model(repo_root), help="Path to exported .onnx model")
    parser.add_argument("--output", type=str, default="prediction_onnx.png", help="Output mask path (PNG)")
    parser.add_argument("--provider", type=str, default=None,
                         help="Force a specific ONNX Runtime execution provider (default: auto-pick best available)")
    parser.add_argument("--high_thresh", type=float, default=0.35, help="Hysteresis high threshold")
    parser.add_argument("--low_thresh", type=float, default=0.12, help="Hysteresis low threshold")
    parser.add_argument("--max_gap_dist", type=float, default=220.0, help="Max canopy-gap bridging distance (px)")
    parser.add_argument("--max_angle_deg", type=float, default=65.0, help="Max angle deviation for gap bridging")
    parser.add_argument("--road_width", type=int, default=6, help="Stroke width for bridged gap segments (px)")
    parser.add_argument("--no-tta", dest="use_tta", action="store_false",
                         help="Disable 4-way flip test-time augmentation (on by default; costs 4x forward passes)")
    args = parser.parse_args()

    predict(
        args.image_path, args.model, args.output, args.provider,
        args.high_thresh, args.low_thresh, args.max_gap_dist, args.max_angle_deg, args.road_width,
        args.use_tta,
    )
