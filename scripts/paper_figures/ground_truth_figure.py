"""Paper version of the Kaggle ground truth figure (fig_eval_qualitative_wide.png).

The Kaggle evaluation labels the prediction row with an old model name and adds a
footnote under the legend. This script relabels the row as "Prediction", drops the
footnote (the caption in main.tex explains the tile choice), and saves the result as
docs/paper/figures/fig_ground_truth.png. The image data are not changed.
"""
import os

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO, "figures", "real", "v4", "paper_results", "figures", "fig_eval_qualitative_wide.png")
OUT = os.path.join(REPO, "docs", "paper", "figures", "fig_ground_truth.png")
FONT = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")

im = Image.open(SRC).convert("RGB")
gray = np.asarray(im.convert("L"))

# Measure the "Ground truth" row label so the new label matches its size and position.
label = gray[440:790, :230] < 128
ys, xs = np.where(label.any(1))[0], np.where(label.any(0))[0]
size = next(s for s in range(20, 60) if ImageFont.truetype(FONT, s).getbbox("Ground truth")[2] >= ys.max() - ys.min())
font = ImageFont.truetype(FONT, size)

draw = ImageDraw.Draw(im)
draw.rectangle([0, 840, 231, 1185], fill="white")          # old prediction row label
bbox = font.getbbox("Prediction")
text = Image.new("RGB", (bbox[2] + 4, bbox[3] + 4), "white")
ImageDraw.Draw(text).text((2, 0), "Prediction", font=font, fill=(12, 12, 12))
text = text.rotate(90, expand=True)
im.paste(text, (xs.min() - 2, (848 + 1170) // 2 - text.size[1] // 2))

im = im.crop((0, 0, im.size[0], 1262))                       # drop the footnote line
left = max(0, np.where((np.asarray(im.convert("L")) < 245).any(0))[0].min() - 8)
im = im.crop((left, 0, im.size[0], im.size[1]))
im.save(OUT, dpi=(300, 300))
print("saved", OUT, im.size)
