"""
Measures compute cost of the proposed model and reference architectures on this
machine, and writes figures/real/measurements/efficiency.json. Every number is measured:
parameter counts from the instantiated modules, FLOPs via torch's FlopCounterMode
(conv/matmul FLOPs = 2 x MACs), latency as the median wall-clock time of repeated
batch-1 forward passes on a full 1024x1024 DeepGlobe-sized tile.

Each model is measured in its own subprocess so memory from one never affects the
next (a 1024^2 U-Net pass alone needs ~1.5 GB).

Usage (repo root):  python scripts/paper_figures/benchmark_efficiency.py
"""

import json
import os
import platform
import statistics
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "figures", "real", "measurements", "efficiency.json")
TILE = 1024
WARMUP = 1
RUNS_FAST, RUNS_SLOW, SLOW_S = 7, 3, 4.0   # fewer repeats once a single pass exceeds SLOW_S


def cpu_name():
    if platform.system() == "Windows":
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
            if out:
                return out.splitlines()[0].strip()
        except Exception:
            pass
    return platform.processor() or "unknown CPU"


def _time(fn):
    for _ in range(WARMUP):
        t0 = time.perf_counter(); fn(); first = time.perf_counter() - t0
    runs = RUNS_SLOW if first > SLOW_S else RUNS_FAST
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn(); samples.append((time.perf_counter() - t0) * 1000)
    return {"latency_ms": statistics.median(samples), "latency_min_ms": min(samples),
            "latency_max_ms": max(samples), "runs": runs}


def worker(name, flops_tile=TILE, lat_tile=TILE):
    """Runs inside a subprocess; prints one JSON line. flops_tile < TILE counts FLOPs at a
    smaller input and scales by pixel count -- exact for fully convolutional networks, and
    only used when the full-size counting pass exhausts memory."""
    import numpy as np
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.dirname(__file__))
    torch.manual_seed(0)
    x = torch.randn(1, 3, TILE, TILE)

    if name == "Proposed (ONNX)":
        import onnxruntime as ort
        path = os.path.join(REPO, "models", "mobilevit_v2.onnx")
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        r = _time(lambda: sess.run(None, {inp: x.numpy()}))
        size = os.path.getsize(path) + (os.path.getsize(path + ".data") if os.path.exists(path + ".data") else 0)
        r.update(file_mb_total=size / 2**20, graph_file_mb=os.path.getsize(path) / 2**20, ort_version=ort.__version__)
        print(json.dumps(r)); return

    if name == "Proposed":
        from backend.src.models.mobilevit_v2 import load_mobilevit_checkpoint
        model, _ = load_mobilevit_checkpoint(os.path.join(REPO, "models", "best_model_v2.pth"))
        cite = "this work"
    else:
        from reference_models import reference_models
        ctor, cite = reference_models()[name]
        model = ctor()
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    with torch.inference_mode():
        counter = FlopCounterMode(display=False)
        with counter:
            model(x if flops_tile == TILE else torch.randn(1, 3, flops_tile, flops_tile))
        flops = counter.get_total_flops() * (TILE / flops_tile) ** 2
        if lat_tile == TILE:
            r = _time(lambda: model(x))
        else:   # the full tile as (TILE/lat_tile)^2 sequential crops, as a memory-limited device would run it
            crops = [x[..., i:i + lat_tile, j:j + lat_tile] for i in range(0, TILE, lat_tile) for j in range(0, TILE, lat_tile)]
            r = _time(lambda: [model(c) for c in crops])
    r.update(params=params, gflops=flops / 1e9, weights_mb_fp32=params * 4 / 2**20, citation=cite)
    if flops_tile != TILE:
        r["flops_note"] = f"counted at {flops_tile}^2 and scaled x{(TILE / flops_tile) ** 2:.0f} (fully convolutional)"
    if lat_tile != TILE:
        r["latency_note"] = f"1024^2 tile timed as {(TILE // lat_tile) ** 2} sequential {lat_tile}^2 crops (full-tile pass exceeds free RAM)"
    print(json.dumps(r))


def main():
    import torch
    names = ["Proposed", "Proposed (ONNX)", "LR-ASPP-MBv3", "DeepLabV3-MBv3", "D-LinkNet34", "U-Net"]
    results = {"tile": TILE, "cpu": cpu_name(), "torch_threads": torch.get_num_threads(),
               "torch": torch.__version__, "models": {}}
    only = set(sys.argv[1:])            # optionally re-measure just some models, keeping the rest
    if only and os.path.exists(OUT):
        prev = json.load(open(OUT))
        results["models"].update({k: v for k, v in prev.get("models", {}).items() if k not in only})
        if "onnx" in prev and "Proposed (ONNX)" not in only:
            results["onnx"] = prev["onnx"]
    for name in names:
        if only and name not in only:
            continue
        print(f"{name}...", flush=True)
        line, p = [], None
        for flops_tile, lat_tile in ((TILE, TILE), (TILE // 2, TILE), (TILE // 4, TILE // 2)):
            p = subprocess.run([sys.executable, __file__, "--worker", name, str(flops_tile), str(lat_tile)],
                               capture_output=True, text=True, env={**os.environ, "PYTHONWARNINGS": "ignore"})
            line = [l for l in p.stdout.splitlines() if l.startswith("{")]
            if p.returncode == 0 and line:
                break
            print(f"  attempt (FLOPs at {flops_tile}^2, latency at {lat_tile}^2) failed ({p.returncode}); retrying smaller")
        if not line:
            print(f"  FAILED ({p.returncode}): {p.stderr.strip()[-400:]}")
            continue
        r = json.loads(line[-1])
        if name == "Proposed (ONNX)":
            results["onnx"] = r
        else:
            results["models"][name] = r
        print("  ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2], *(int(a) for a in sys.argv[3:5]))
    else:
        main()
