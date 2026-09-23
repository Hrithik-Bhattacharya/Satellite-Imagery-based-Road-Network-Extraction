"""
Slide imagery from real model outputs: each sample tile overlaid with the deployed
pipeline's road mask (4-flip TTA -> hysteresis -> closing -> canopy-gap bridging).
Model-predicted road is drawn in blue, segments added by gap bridging in orange.

Usage (repo root):  python scripts/presentation/deck_assets.py
"""

import json
import os
import sys

import cv2
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "paper_figures"))
from backend.src.models.mobilevit_v2 import load_mobilevit_checkpoint  # noqa: E402
from local_figures import SAMPLES, load_rgb, predict, postprocess  # noqa: E402

OUT = os.path.join(REPO, "figures", "real", "deck")
BLUE = np.array([42, 120, 214], np.float32)      # #2a78d6
ORANGE = np.array([235, 104, 52], np.float32)    # #eb6834


def overlay(rgb, road, bridged):
    img = rgb.astype(np.float32) * 0.78            # slight dim so the mask reads clearly
    road_px = road & ~bridged
    img[road_px] = img[road_px] * 0.15 + BLUE * 0.85
    img[bridged] = img[bridged] * 0.1 + ORANGE * 0.9
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_grad_enabled(False)
    model, meta = load_mobilevit_checkpoint(os.path.join(REPO, "models", "best_model_v2.pth"))
    stats = {"checkpoint_epoch": meta.get("epoch")}
    for t in SAMPLES:
        rgb = load_rgb(t)
        _, closed, final = postprocess(predict(model, rgb, tta=True))
        road, bridged = final > 0, (final > 0) & ~(closed > 0)
        # Thicken the 6 px bridge strokes slightly so they stay visible at slide scale.
        bridged = cv2.dilate(bridged.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        cv2.imwrite(os.path.join(OUT, f"overlay_{t}.jpg"),
                    cv2.cvtColor(overlay(rgb, road, bridged), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(os.path.join(OUT, f"tile_{t}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        # Plain model output for the input-vs-output slide: final road mask, white on black.
        cv2.imwrite(os.path.join(OUT, f"mask_{t}.png"), (final > 0).astype(np.uint8) * 255)
        stats[t] = {"road_frac": float(road.mean()), "bridged_px": int(bridged.sum())}
        n, labels, cc, _ = cv2.connectedComponentsWithStats(bridged.astype(np.uint8), connectivity=8)
        if n > 1:
            # The bridged segment with the most vegetation along it (mean Excess-Green index,
            # 2g - r - b on chromatic coordinates) -- i.e. the gap most likely under canopy.
            f = rgb.astype(np.float32)
            exg = (2 * f[..., 1] - f[..., 0] - f[..., 2]) / (f.sum(axis=2) + 1e-6)
            cands = [k for k in range(1, n) if cc[k, cv2.CC_STAT_AREA] >= 150]
            if cands:
                k = max(cands, key=lambda k: exg[labels == k].mean())
                x, y, w, h = (int(v) for v in cc[k, :4])
                H_, W_ = bridged.shape
                stats[t]["canopy_bridge_bbox"] = [x / W_, y / H_, (x + w) / W_, (y + h) / H_]
                stats[t]["canopy_bridge_mean_exg"] = float(exg[labels == k].mean())
                stats[t]["tile_mean_exg"] = float(exg.mean())
        print(t, stats[t])
    json.dump(stats, open(os.path.join(OUT, "overlay_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
