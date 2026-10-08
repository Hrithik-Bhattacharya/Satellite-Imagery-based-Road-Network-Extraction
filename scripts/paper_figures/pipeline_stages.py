"""Postprocessing stages on one tile with the deployed ONNX model (no PyTorch needed).

Writes docs/paper/figures/fig_pipeline_stages.pdf and
figures/real/measurements/pipeline_stages.json.
"""
import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from backend.src.utils.graph_postprocess import hysteresis_threshold, connect_canopy_gaps

TILE = "117991"
OUT_FIG = os.path.join(REPO, "docs", "paper", "figures", "fig_pipeline_stages.pdf")
OUT_JSON = os.path.join(REPO, "figures", "real", "measurements", "pipeline_stages.json")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

so = ort.SessionOptions()
so.enable_cpu_mem_arena = False
so.enable_mem_pattern = False
sess = ort.InferenceSession(os.path.join(REPO, "models", "mobilevit_v2.onnx"), sess_options=so,
                            providers=["CPUExecutionProvider"])
name = sess.get_inputs()[0].name

img = cv2.cvtColor(cv2.imread(os.path.join(REPO, "data", "samples", f"{TILE}_sat.jpg")), cv2.COLOR_BGR2RGB)
x = np.ascontiguousarray(((img.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None])
acc = None
for ax in [None, (2,), (3,), (2, 3)]:
    xi = np.ascontiguousarray(np.flip(x, ax)) if ax else x
    p = sess.run(None, {name: xi})[0]
    p = np.flip(p, ax) if ax else p
    acc = p if acc is None else acc + p
prob = (acc / 4).squeeze()

hyst = hysteresis_threshold(prob, 0.35, 0.12)
closed = cv2.morphologyEx(hyst, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
bridged = connect_canopy_gaps(closed)


def components(m):
    return int(cv2.connectedComponents((m > 127).astype(np.uint8), connectivity=8)[0] - 1)


stats = {"tile": TILE, "components_after_closing": components(closed), "components_after_bridging": components(bridged),
         "bridged_pixels": int(((bridged > 127) & (closed <= 127)).sum()),
         "road_fraction_final": float((bridged > 127).mean())}
json.dump(stats, open(OUT_JSON, "w"), indent=2)
print(stats)

plt.rcParams.update({"font.family": "serif", "font.size": 7})
fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.05), gridspec_kw={"wspace": 0.04})
over = np.zeros((*closed.shape, 3), np.uint8) + 255
over[closed > 127] = (20, 20, 20)
new = (bridged > 127) & (closed <= 127)
over[new] = (230, 90, 20)
panels = [(img, None, "(a) Input"),
          (prob, "inferno", "(b) Probability"),
          (255 - closed, "gray", f"(c) Threshold: {stats['components_after_closing']} parts"),
          (over, None, f"(d) Bridged: {stats['components_after_bridging']} parts")]
for a, (im, cmap, title) in zip(axs, panels):
    a.imshow(im, cmap=cmap, vmin=0, vmax=1 if cmap == "inferno" else 255)
    a.set_title(title, fontsize=6.6)
    a.set_xticks([]); a.set_yticks([])
fig.savefig(OUT_FIG, bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT_FIG.replace(".pdf", ".png"), dpi=250, bbox_inches="tight", pad_inches=0.02)
print("saved", OUT_FIG)
