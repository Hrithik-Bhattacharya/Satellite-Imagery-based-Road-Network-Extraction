import os, glob, sys
import numpy as np
import cv2
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

REPO = r"C:\Users\Dillu\Desktop\EL\road-extraction\Satellite-Imagery-based-Road-Network-Extraction"
sys.path.insert(0, REPO)
from backend.src.utils.graph_postprocess import hysteresis_threshold, connect_canopy_gaps

ONNX = os.path.join(REPO, "models", "mobilevit_v2.onnx")
TILES = sorted(glob.glob(os.path.join(REPO, "data", "samples", "*.jpg")))
OUT = os.path.join(REPO, "docs", "paper", "figures", "fig_predictions.png")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

so = ort.SessionOptions()
so.enable_cpu_mem_arena = False
so.enable_mem_pattern = False
sess = ort.InferenceSession(ONNX, sess_options=so, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name

def predict(img):
    x = ((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
    x = np.ascontiguousarray(x)
    s = None
    for ax in [None, (2,), (3,), (2, 3)]:
        xi = np.ascontiguousarray(np.flip(x, ax)) if ax else x
        p = sess.run(None, {inp: xi})[0]
        p = np.flip(p, ax) if ax else p
        s = p if s is None else s + p
    prob = (s / 4).squeeze()
    mask = hysteresis_threshold(prob, 0.35, 0.12)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return connect_canopy_gaps(mask)

def overlay(img, mask):
    ov = img.astype(np.float32)
    r = mask > 127
    ov[r] = ov[r] * 0.25 + np.array([255, 60, 0], np.float32) * 0.75
    return ov.clip(0, 255).astype(np.uint8)

res = []
for t in TILES:
    img = cv2.cvtColor(cv2.imread(t), cv2.COLOR_BGR2RGB)
    m = predict(img)
    res.append((os.path.basename(t).split("_")[0], img, m, overlay(img, m)))
    print(res[-1][0], f"{(m > 127).mean()*100:.2f}%")

fig = plt.figure(figsize=(7.16, 4.9))
g = gs.GridSpec(4, 6, figure=fig, wspace=0.03, hspace=0.04, left=0.01, right=0.99, top=0.95, bottom=0.01)
heads = ["Image", "Mask", "Overlay"] * 2
for i, (name, rgb, mask, ov) in enumerate(res):
    r, c0 = i // 2, (i % 2) * 3
    for k, im in enumerate([rgb, mask, ov]):
        ax = fig.add_subplot(g[r, c0 + k])
        ax.imshow(im, cmap="gray" if k == 1 else None, vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(heads[c0 + k], fontsize=8, family="serif")
        if k == 0:
            ax.text(0.03, 0.95, name, transform=ax.transAxes, fontsize=6, color="white", va="top",
                    family="serif", bbox=dict(fc="black", ec="none", alpha=0.6, pad=1))
fig.savefig(OUT, dpi=300, facecolor="white")
print("saved", OUT)
