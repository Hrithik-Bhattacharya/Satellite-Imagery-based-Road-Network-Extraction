"""Validation curve of the v4 run from models/v4_training/history.csv (one column wide)."""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
rows = list(csv.DictReader(open(os.path.join(REPO, "models", "v4_training", "history.csv"))))
col = lambda k: [float(r[k]) for r in rows]
ep = col("epoch")
plt.rcParams.update({"font.family": "serif", "font.size": 8})
fig, ax = plt.subplots(figsize=(3.45, 2.2))
ax.plot(ep, [100 * v for v in col("val_iou")], color="#9aa5b1", lw=0.9, label="Model weights")
ax.plot(ep, [100 * v for v in col("ema_iou")], color="#c0392b", lw=1.3, label="EMA weights")
ax.set_xlabel("Epoch"); ax.set_ylabel("Selection set IoU (%)")
ax.set_ylim(10, 65); ax.set_xlim(0, 160); ax.grid(alpha=0.3); ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
out = os.path.join(REPO, "docs", "paper", "figures", "fig_training_curves.pdf")
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), dpi=250, bbox_inches="tight")
print("saved", out)
