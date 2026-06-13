"""
generate_comparative_benchmark.py – Member 4 (Validation & MLOps)
Merges benchmark + evaluation results into a single comparative table.

Reads:
    outputs/benchmark_report.json
    outputs/evaluation_summary.json  (optional)
    outputs/ablation_results.csv     (optional)

Generates:
    outputs/comparative_benchmark.csv
    (columns: Model, Params, Size_MB, FPS, Latency_ms, IoU, F1, APLS, TOPO)

Usage:
    cd backend
    python scripts/generate_comparative_benchmark.py
"""

import csv
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs", type=str)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load sources
    bench_path  = out_dir / "benchmark_report.json"
    eval_path   = out_dir / "evaluation_summary.json"
    ablation_csv = out_dir / "ablation_results.csv"

    bench_data  = json.loads(bench_path.read_text())  if bench_path.exists()  else {}
    eval_data   = json.loads(eval_path.read_text())   if eval_path.exists()   else {}
    ablation_rows = []
    if ablation_csv.exists():
        with open(ablation_csv) as f:
            ablation_rows = list(csv.DictReader(f))

    # Build ablation lookup by experiment name
    ablation_lookup = {r.get("Model","").strip(): r for r in ablation_rows}

    # Global eval metrics (aggregate, not per-model)
    def _mean(key):
        if key in eval_data and isinstance(eval_data[key], dict):
            return eval_data[key].get("mean", "N/A")
        return "N/A"

    headers = ["Model", "Params", "Size_MB", "FPS", "Latency_ms", "IoU", "F1", "APLS", "TOPO"]
    rows = []

    for m in bench_data.get("models", []):
        model_name = m.get("model", "?")
        p = m.get("parameters", "N/A")
        if isinstance(p, int):
            p = f"{p:,}"

        # Try to get per-experiment metrics from ablation
        abl = ablation_lookup.get(model_name.replace("_", " ").strip(), {})

        row = {
            "Model":       model_name,
            "Params":      str(p),
            "Size_MB":     str(m.get("size_mb",    "N/A")),
            "FPS":         str(m.get("fps",        "N/A")),
            "Latency_ms":  str(m.get("latency_ms", "N/A")),
            "IoU":         str(abl.get("IoU",  _mean("iou"))),
            "F1":          str(abl.get("F1",   _mean("f1"))),
            "APLS":        str(abl.get("APLS", _mean("apls"))),
            "TOPO":        str(abl.get("TOPO", _mean("topo_f1"))),
        }
        rows.append(row)

    out_path = out_dir / "comparative_benchmark.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Comparative Benchmark] Saved: {out_path}")
    print(f"\n{'Model':<25} {'Params':>12} {'MB':>6} {'FPS':>7} {'Lat(ms)':>9} {'IoU':>7} {'F1':>7} {'APLS':>7} {'TOPO':>7}")
    print("-" * 95)
    for r in rows:
        print(f"{r['Model']:<25} {r['Params']:>12} {r['Size_MB']:>6} {r['FPS']:>7} "
              f"{r['Latency_ms']:>9} {r['IoU']:>7} {r['F1']:>7} {r['APLS']:>7} {r['TOPO']:>7}")


if __name__ == "__main__":
    main()
