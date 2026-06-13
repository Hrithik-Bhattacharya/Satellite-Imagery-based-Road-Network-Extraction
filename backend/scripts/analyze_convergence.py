"""
analyze_convergence.py – Member 4 (Validation & MLOps)
Reads a training log CSV and generates convergence visualisations and a
structured JSON report.

Expected input columns (flexible – any subset is fine):
    epoch, train_loss, val_loss, train_iou, val_iou, train_f1, val_f1,
    train_apls, val_apls, lr

Generates:
    outputs/loss_curve.png
    outputs/metric_curve.png
    outputs/convergence_report.json

Usage:
    cd backend
    python scripts/analyze_convergence.py --log_csv outputs/training_logs.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))  # backend/


def _smooth(values: list, window: int = 3) -> list:
    """Simple moving average for curve smoothing."""
    if len(values) < window:
        return values
    out = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        out.append(float(np.mean(values[lo:hi])))
    return out


def _find_convergence_epoch(values: list, patience: int = 5, delta: float = 1e-4) -> int:
    """Return epoch index where improvement < delta for `patience` consecutive epochs."""
    if len(values) < patience:
        return -1
    best = values[0]
    no_improve = 0
    for i, v in enumerate(values[1:], 1):
        if best - v > delta:
            best = v
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                return i - patience + 1
    return -1


def load_log(csv_path: Path) -> dict:
    """Load training log CSV into a column dict."""
    import csv
    columns: dict[str, list] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                k = k.strip()
                try:
                    val = float(v)
                except (ValueError, TypeError):
                    val = v
                columns.setdefault(k, []).append(val)
    return columns


def plot_curves(columns: dict, out_dir: Path):
    """Generate loss and metric PNG curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend for servers
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not installed. Skipping plot generation.")
        return

    epochs = columns.get("epoch", list(range(1, len(next(iter(columns.values()))) + 1)))

    # --- Loss curve ---
    fig, ax = plt.subplots(figsize=(9, 5))
    if "train_loss" in columns:
        ax.plot(epochs, columns["train_loss"], label="Train Loss", color="steelblue")
    if "val_loss" in columns:
        ax.plot(epochs, columns["val_loss"], label="Val Loss", color="tomato")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Convergence – Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_png = out_dir / "loss_curve.png"
    plt.savefig(str(loss_png), dpi=150)
    plt.close(fig)
    print(f"  Saved: {loss_png}")

    # --- Metric curve ---
    metric_pairs = [
        ("train_iou",  "val_iou",  "IoU",  "forestgreen",   "limegreen"),
        ("train_f1",   "val_f1",   "F1",   "darkorange",    "gold"),
        ("train_apls", "val_apls", "APLS", "mediumpurple",  "plum"),
    ]
    plotted = False
    fig, ax = plt.subplots(figsize=(9, 5))
    for train_col, val_col, label, c_tr, c_val in metric_pairs:
        if train_col in columns:
            ax.plot(epochs, columns[train_col], label=f"Train {label}", color=c_tr)
            plotted = True
        if val_col in columns:
            ax.plot(epochs, columns[val_col], label=f"Val {label}", color=c_val, linestyle="--")
            plotted = True
    if plotted:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("Training Convergence – Metric Curves")
        ax.legend(ncol=2, fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        metric_png = out_dir / "metric_curve.png"
        plt.savefig(str(metric_png), dpi=150)
        plt.close(fig)
        print(f"  Saved: {metric_png}")


def build_report(columns: dict) -> dict:
    """Build convergence_report.json content."""
    n_epochs = max(len(v) for v in columns.values() if isinstance(v, list))
    report = {"n_epochs": n_epochs}

    # Best values
    for col, higher_better in [
        ("val_iou",   True),  ("val_f1",    True),
        ("val_apls",  True),  ("val_loss",  False),
        ("train_loss",False),
    ]:
        if col in columns:
            vals = [v for v in columns[col] if isinstance(v, float)]
            if vals:
                if higher_better:
                    best_val = max(vals)
                    best_ep  = vals.index(best_val)
                else:
                    best_val = min(vals)
                    best_ep  = vals.index(best_val)
                report[f"best_{col}"] = {"value": round(best_val, 5), "epoch": best_ep + 1}

    # Convergence epoch
    if "val_loss" in columns:
        loss_vals = [v for v in columns["val_loss"] if isinstance(v, float)]
        conv_ep = _find_convergence_epoch(loss_vals)
        report["convergence_epoch"] = conv_ep + 1 if conv_ep >= 0 else "not detected"

    # Final values
    report["final"] = {}
    for col in ["val_iou", "val_f1", "val_apls", "val_loss"]:
        if col in columns:
            vals = [v for v in columns[col] if isinstance(v, float)]
            if vals:
                report["final"][col] = round(vals[-1], 5)

    return report


def main():
    parser = argparse.ArgumentParser(description="Analyse training convergence logs.")
    parser.add_argument("--log_csv",    required=True, type=str,
                        help="Path to training_logs.csv")
    parser.add_argument("--output_dir", default="outputs", type=str)
    args = parser.parse_args()

    log_path = Path(args.log_csv)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        print(f"[ERROR] Log file not found: {log_path}")
        print("        Provide a CSV with columns like: epoch,train_loss,val_loss,val_iou,val_f1")
        sys.exit(1)

    print(f"[Convergence] Loading log: {log_path}")
    columns = load_log(log_path)
    print(f"  Detected columns: {list(columns.keys())}")
    print(f"  Epochs found: {max(len(v) for v in columns.values())}")

    print("[Convergence] Generating plots ...")
    plot_curves(columns, out_dir)

    print("[Convergence] Building report ...")
    report = build_report(columns)

    report_path = out_dir / "convergence_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_path}")

    print("\n===== CONVERGENCE SUMMARY =====")
    for k, v in report.items():
        if k != "final":
            print(f"  {k:<30s} {v}")
    if "final" in report:
        print("  Final epoch metrics:")
        for k, v in report["final"].items():
            print(f"    {k:<26s} {v}")


if __name__ == "__main__":
    main()
