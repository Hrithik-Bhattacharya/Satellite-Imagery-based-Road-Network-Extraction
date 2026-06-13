"""
benchmark_model.py – Member 4 (Validation & MLOps)
Automatically benchmarks U-Net baseline, MobileViT v2, and the ONNX model.

Measures:
  - Parameter count  (verified against README-claimed numbers)
  - Model file size  (MB)
  - FPS / throughput (frames per second)
  - Latency          (ms per image)

Generates:
    outputs/benchmark_report.json

Usage:
    cd backend
    python scripts/benchmark_model.py
"""

import sys
import os
import json
import time
import math
from pathlib import Path

import numpy as np

# Ensure backend root is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))  # backend/

# ---- README-claimed parameter counts for verification ----
_CLAIMED_PARAMS = {
    "MobileViT_v2": 478_849,
    "UNet":         31_037_633,
}

_WARMUP_ITERS = 5
_BENCH_ITERS  = 20
_INPUT_SHAPE  = (1, 3, 256, 256)   # (B, C, H, W)


# ---------------------------------------------------------------------------
# PyTorch benchmarking helpers
# ---------------------------------------------------------------------------

def _count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _model_size_mb(model) -> float:
    """Estimate in-memory parameter size in MB (float32)."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / (1024 ** 2)


def _bench_torch(model, device, n_warmup: int = _WARMUP_ITERS, n_iters: int = _BENCH_ITERS) -> dict:
    """Run forward-pass timing loop. Returns latency_ms and fps."""
    import torch
    model.eval()
    dummy = torch.randn(*_INPUT_SHAPE, device=device)

    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            _ = model(dummy)

        # Timed runs
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    latency_ms = (elapsed / n_iters) * 1000.0
    fps = n_iters / elapsed
    return {"latency_ms": round(latency_ms, 3), "fps": round(fps, 2)}


def _verify_params(model_name: str, actual: int) -> dict:
    """Compare actual param count to README-claimed value."""
    claimed = _CLAIMED_PARAMS.get(model_name)
    if claimed is None:
        return {"claimed": "unknown", "actual": actual, "match": "n/a"}
    match = claimed == actual
    return {
        "claimed": claimed,
        "actual":  actual,
        "match":   match,
        "delta":   actual - claimed,
    }


def _benchmark_torch_model(model_name: str, ModelClass, model_kwargs: dict, device) -> dict:
    """Full benchmark for a single PyTorch model."""
    import torch
    print(f"\n  Benchmarking {model_name} ...")
    try:
        model = ModelClass(**model_kwargs).to(device)
        model.eval()
        params = _count_parameters(model)
        size_mb = _model_size_mb(model)
        timing = _bench_torch(model, device)
        param_check = _verify_params(model_name, params)

        print(f"    Params     : {params:,}  (claimed: {param_check['claimed']}, match: {param_check['match']})")
        print(f"    Size       : {size_mb:.2f} MB")
        print(f"    Latency    : {timing['latency_ms']} ms")
        print(f"    FPS        : {timing['fps']}")

        return {
            "model":      model_name,
            "parameters": params,
            "size_mb":    round(size_mb, 3),
            **timing,
            "param_verification": param_check,
            "status": "ok",
        }
    except Exception as e:
        print(f"    [ERROR] {e}")
        return {"model": model_name, "status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# ONNX benchmarking
# ---------------------------------------------------------------------------

def _benchmark_onnx(onnx_path: str, device_str: str = "cpu") -> dict:
    """Benchmark the ONNX model using onnxruntime."""
    print(f"\n  Benchmarking ONNX model: {onnx_path} ...")
    try:
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device_str == "cuda"
            else ["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(onnx_path, providers=providers)
        input_name = session.get_inputs()[0].name
        dummy_np = np.random.randn(*_INPUT_SHAPE).astype(np.float32)

        # Warmup
        for _ in range(_WARMUP_ITERS):
            session.run(None, {input_name: dummy_np})

        t0 = time.perf_counter()
        for _ in range(_BENCH_ITERS):
            session.run(None, {input_name: dummy_np})
        elapsed = time.perf_counter() - t0

        latency_ms = (elapsed / _BENCH_ITERS) * 1000.0
        fps        = _BENCH_ITERS / elapsed
        size_mb    = os.path.getsize(onnx_path) / (1024 ** 2) if os.path.isfile(onnx_path) else 0.0

        print(f"    Size       : {size_mb:.3f} MB")
        print(f"    Latency    : {latency_ms:.3f} ms")
        print(f"    FPS        : {fps:.2f}")

        return {
            "model":      "MobileViT_v2_ONNX",
            "size_mb":    round(size_mb, 3),
            "latency_ms": round(latency_ms, 3),
            "fps":        round(fps, 2),
            "parameters": "n/a (ONNX)",
            "status": "ok",
        }
    except ImportError:
        print("    [SKIP] onnxruntime not installed. Install with: pip install onnxruntime")
        return {"model": "MobileViT_v2_ONNX", "status": "skipped", "reason": "onnxruntime not installed"}
    except Exception as e:
        print(f"    [ERROR] {e}")
        return {"model": "MobileViT_v2_ONNX", "status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark all road-extraction models.")
    parser.add_argument("--onnx_path",  default=None, type=str, help="Path to ONNX model file")
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--device",     default="auto", choices=["auto", "cpu", "cuda"],
                        help="Inference device")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Device selection ---
    try:
        import torch
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        print(f"[Benchmark] Device: {device}")
        _TORCH_OK = True
    except ImportError:
        print("[Benchmark] PyTorch not installed – skipping PyTorch model benchmarks.")
        _TORCH_OK = False
        device = None

    results = []

    if _TORCH_OK:
        # Resolve Member 1's path dynamically
        workspace_root = os.path.abspath(os.path.join(_SCRIPT_DIR.parent, "..", "..", ".."))
        if os.path.exists(workspace_root):
            for root, dirs, files in os.walk(workspace_root):
                if "unet_baseline.py" in files and os.path.basename(os.path.dirname(root)) == "src":
                    # root is something like .../backend/src/models
                    backend_dir = os.path.dirname(os.path.dirname(root))
                    if backend_dir not in sys.path:
                        sys.path.insert(0, backend_dir)
                    break

        # Try importing models from Member 1
        try:
            from src.models.unet_baseline import UNet
            r = _benchmark_torch_model(
                "UNet",
                UNet,
                {"n_channels": 3, "n_classes": 1},
                device,
            )
            results.append(r)
        except ImportError as e:
            print(f"\n  [SKIP] UNet: {e}")
            results.append({"model": "UNet", "status": "skipped", "reason": str(e)})

        try:
            from src.models.mobilevit_v2 import MobileViT_v2
            r = _benchmark_torch_model(
                "MobileViT_v2",
                MobileViT_v2,
                {"num_classes": 1, "width_mult": 0.5},
                device,
            )
            results.append(r)
        except ImportError as e:
            print(f"\n  [SKIP] MobileViT_v2: {e}")
            results.append({"model": "MobileViT_v2", "status": "skipped", "reason": str(e)})

    # --- ONNX benchmark ---
    # Resolve ONNX path: prefer arg, then search standard locations
    onnx_candidates = [
        args.onnx_path,
        str(_SCRIPT_DIR.parent / "outputs" / "mobilevit_v2.onnx"),
        "mobilevit_v2.onnx",
        "../outputs/mobilevit_v2.onnx",
    ]
    onnx_path = None
    for c in onnx_candidates:
        if c and os.path.isfile(c):
            onnx_path = c
            break

    if onnx_path:
        r = _benchmark_onnx(onnx_path, device_str=str(device) if device else "cpu")
        results.append(r)
    else:
        print("\n  [SKIP] ONNX model not found. Run export_onnx.py first.")
        results.append({
            "model":  "MobileViT_v2_ONNX",
            "status": "skipped",
            "reason": "ONNX file not found",
        })

    # --- Build report ---
    report = {
        "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
        "device":          str(device) if device else "cpu",
        "input_shape":     list(_INPUT_SHAPE),
        "warmup_iters":    _WARMUP_ITERS,
        "benchmark_iters": _BENCH_ITERS,
        "models":          results,
    }

    report_path = out_dir / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Benchmark] Report saved: {report_path}")

    # Print comparison table
    print("\n====== BENCHMARK COMPARISON ======")
    print(f"{'Model':<22} {'Params':>12} {'Size(MB)':>10} {'Latency(ms)':>13} {'FPS':>8} {'Status':>10}")
    print("-" * 80)
    for r in results:
        name    = r.get("model",      "")
        params  = r.get("parameters", "n/a")
        size    = r.get("size_mb",    "n/a")
        lat     = r.get("latency_ms", "n/a")
        fps     = r.get("fps",        "n/a")
        status  = r.get("status",     "")
        if isinstance(params, int):
            params = f"{params:,}"
        print(f"{name:<22} {str(params):>12} {str(size):>10} {str(lat):>13} {str(fps):>8} {status:>10}")

    return report


if __name__ == "__main__":
    main()
