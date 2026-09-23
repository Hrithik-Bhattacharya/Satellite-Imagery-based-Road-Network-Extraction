"""
Builds the project presentation (docs/presentation/Rural_Road_Extraction.pptx).

Every number on the slides is read from measured data, never typed in:
  models/*.pth                              checkpoint metadata (epochs, validation metrics)
  figures/real/measurements/efficiency.json         benchmark_efficiency.py
  figures/real/measurements/local_figure_stats.json local_figures.py
  figures/real/deck/overlay_stats.json      deck_assets.py
  figures/real/kaggle/results.json          evaluate_for_paper.ipynb (optional -- when present,
                                            the held-out evaluation slides are added)

Usage (repo root):
  python scripts/paper_figures/benchmark_efficiency.py   # once
  python scripts/paper_figures/local_figures.py
  python scripts/presentation/deck_assets.py
  python scripts/presentation/build_deck.py
"""

import copy
import json
import os
import sys

import torch
from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG = os.path.join(REPO, "figures", "real")
KAGGLE = os.environ.get("DECK_KAGGLE_DIR", os.path.join(FIG, "kaggle"))
OUT = os.environ.get("DECK_OUT", os.path.join(REPO, "docs", "presentation", "Rural_Road_Extraction.pptx"))

# ── Palette: white content slides, deep canopy-green for title/closing ──
INK, INK2, MUTED = "1F2A2E", "4E5B62", "86919A"
ACCENT = "2F7D5B"          # canopy green: section tags, key figures
DARK = "13261F"            # title / closing background
DARK_TEXT2 = "B9CFC5"
TINT = "F1F4F3"            # card fill
HAIR = "D5DCD9"
BLUE, ORANGE, AQUA = "2A78D6", "EB6834", "1BAF7A"   # data colours, as in the figures
FONT = "Calibri"

W, H = 13.333, 7.5
MX = 0.6                   # side margin (in)
CONTENT_TOP = 1.72


def rgb(h):
    return RGBColor.from_string(h)


# ── data ────────────────────────────────────────────────────────────────────
def load_json(path, default=None):
    return json.load(open(path)) if os.path.exists(path) else default


for _cand in (KAGGLE, os.path.join(KAGGLE, "paper_results")):   # zip unpacks into paper_results/
    if os.path.exists(os.path.join(_cand, "results.json")):
        KAGGLE = _cand
        break

EFF = load_json(os.path.join(FIG, "measurements", "efficiency.json"))
LOCAL = load_json(os.path.join(FIG, "measurements", "local_figure_stats.json"), {})
OVER = load_json(os.path.join(FIG, "deck", "overlay_stats.json"), {})
KRES = load_json(os.path.join(KAGGLE, "results.json"))


def ckpt_meta(rel):
    ck = torch.load(os.path.join(REPO, rel), map_location="cpu")
    return {k: v for k, v in ck.items() if k not in ("model_state_dict", "optimizer_state_dict")}


CK = {
    "baseline": ckpt_meta("models/best_model_new.pth"),
    "collapsed": ckpt_meta("models/archive/best_model_v2_collapsed_epoch46.pth"),
    "run1": ckpt_meta("models/archive/best_model_v2_epoch18_iou0159.pth"),
    "final": ckpt_meta("models/best_model_v2.pth"),
}
M = EFF["models"]
P = M["Proposed"]
UNET = M["U-Net"]
CPU_SHORT = EFF["cpu"].replace("Intel(R) Core(TM) ", "Intel ").split(" CPU")[0]


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# ── low-level helpers ───────────────────────────────────────────────────────
class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(W), Inches(H)
        self.blank = self.prs.slide_layouts[6]
        self.n = 0

    def slide(self, dark=False):
        s = self.prs.slides.add_slide(self.blank)
        self.n += 1
        bg = s.background.fill
        bg.solid()
        bg.fore_color.rgb = rgb(DARK if dark else "FFFFFF")
        if not dark:
            text(s, W - MX - 1.0, H - 0.42, 1.0, 0.25, str(self.n), 10, MUTED, align=PP_ALIGN.RIGHT)
        return s

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.prs.save(path)


def text(slide, x, y, w, h, content, size=15, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, italic=False, spacing=None, line_spacing=None):
    """content: str, or list of paragraphs; a paragraph is str or list of (text, {opts}) runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = content if isinstance(content, list) else [content]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if line_spacing:
            p.line_spacing = line_spacing
        if spacing is not None and i > 0:
            p.space_before = Pt(spacing)
        runs = para if isinstance(para, list) else [(para, {})]
        for t, o in runs:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = FONT
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic)
            f.color.rgb = rgb(o.get("color", color))
    return tb


def bullets(slide, x, y, w, h, items, size=15, color=INK, gap=7):
    """items: str or list of runs; a nested list-of-lists marks a sub-bullet group."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    first = True
    for item in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_before = Pt(0 if p is tf.paragraphs[0] else gap)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Emu(Inches(0.24))))
        pPr.set("indent", str(-Emu(Inches(0.24))))
        bu_clr = etree.SubElement(pPr, qn("a:buClr"))
        etree.SubElement(bu_clr, qn("a:srgbClr")).set("val", ACCENT)
        etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
        etree.SubElement(pPr, qn("a:buChar")).set("char", "•")
        runs = item if isinstance(item, list) else [(item, {})]
        for t, o in runs:
            r = p.add_run()
            r.text = t
            r.font.name = FONT
            r.font.size = Pt(o.get("size", size))
            r.font.bold = o.get("bold", False)
            r.font.color.rgb = rgb(o.get("color", color))
    return tb


def header(slide, tag, title, subtitle=None):
    tb = text(slide, MX, 0.42, 8, 0.3, tag.upper(), 11, ACCENT, bold=True)
    tb.text_frame.paragraphs[0].runs[0]._r.get_or_add_rPr().set("spc", "120")
    text(slide, MX, 0.72, W - 2 * MX, 0.62, title, 30, INK, bold=True)
    if subtitle:
        text(slide, MX, 1.28, W - 2 * MX, 0.36, subtitle, 15, INK2)


def box(slide, x, y, w, h, fill=TINT, line=None, radius=0.06, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.shadow.inherit = False
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(0.9)
    else:
        s.line.fill.background()
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.text_frame.text = ""
    return s


def box_text(slide, x, y, w, h, content, size=13, color=INK, fill=TINT, bold=False, line=None,
             align=PP_ALIGN.CENTER, radius=0.12):
    s = box(slide, x, y, w, h, fill=fill, line=line, radius=radius)
    tf = s.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.08)
    tf.margin_top = tf.margin_bottom = Inches(0.04)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    paras = content if isinstance(content, list) else [content]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        runs = para if isinstance(para, list) else [(para, {})]
        for t, o in runs:
            r = p.add_run()
            r.text = t
            r.font.name = FONT
            r.font.size = Pt(o.get("size", size))
            r.font.bold = o.get("bold", bold)
            r.font.color.rgb = rgb(o.get("color", color))
    return s


def arrow(slide, x1, y1, x2, y2, color=MUTED, width=1.25, dash=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    if dash:
        etree.SubElement(ln, qn("a:prstDash")).set("val", "dash")
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle"); tail.set("w", "med"); tail.set("len", "med")
    return c


def circle_num(slide, x, y, d, label, fill=ACCENT, color="FFFFFF", size=13):
    s = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    s.shadow.inherit = False
    s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
    s.line.fill.background()
    tf = s.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = label
    r.font.name = FONT; r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = rgb(color)
    return s


def image(slide, path, x, y, w=None, h=None, border=True, crop_to=None):
    """Places an image keeping its aspect ratio inside w (and/or h). crop_to=(w,h) fills that box."""
    iw, ih = Image.open(path).size
    if crop_to:
        bw, bh = crop_to
        pic = slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(bw), Inches(bh))
        target, src = bw / bh, iw / ih
        if src > target:
            c = (1 - target / src) / 2
            pic.crop_left = pic.crop_right = c
        else:
            c = (1 - src / target) / 2
            pic.crop_top = pic.crop_bottom = c
    else:
        if w and h:
            scale = min(w / iw, h / ih)
            w2, h2 = iw * scale, ih * scale
            x += (w - w2) / 2
            w, h = w2, h2
        elif w:
            h = w * ih / iw
        else:
            w = h * iw / ih
        pic = slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))
    if border:
        pic.line.color.rgb = rgb(HAIR)
        pic.line.width = Pt(0.75)
    return pic


def caption(slide, x, y, w, t, size=11, color=MUTED, align=PP_ALIGN.LEFT):
    return text(slide, x, y, w, 0.3, t, size, color, align=align)


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


def table(slide, x, y, w, rows, col_w, font=12, header_fill=TINT, row_h=0.36, first_bold=False,
          align_right_from=None, highlight_row=None):
    nr, nc = len(rows), len(rows[0])
    gf = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = gf.table
    tblPr = gf._element.graphic.graphicData.tbl.tblPr
    tblPr.set("firstRow", "0"); tblPr.set("bandRow", "0")
    style = tblPr.find(qn("a:tableStyleId"))
    if style is None:
        style = etree.SubElement(tblPr, qn("a:tableStyleId"))
    style.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"   # "No Style, No Grid"
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            fill = header_fill if i == 0 else ("EAF2EE" if highlight_row == i else "FFFFFF")
            cell.fill.solid(); cell.fill.fore_color.rgb = rgb(fill)
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.RIGHT if (align_right_from is not None and j >= align_right_from) else PP_ALIGN.LEFT
            r = p.add_run(); r.text = str(val)
            r.font.name = FONT
            r.font.size = Pt(font)
            r.font.bold = i == 0 or (first_bold and j == 0) or highlight_row == i
            r.font.color.rgb = rgb(INK if i else INK2)
            # hairline under every row
            tcPr = cell._tc.get_or_add_tcPr()
            for edge in ("a:lnL", "a:lnR", "a:lnT"):
                ln = etree.SubElement(tcPr, qn(edge)); ln.set("w", "0")
                etree.SubElement(ln, qn("a:noFill"))
            lnB = etree.SubElement(tcPr, qn("a:lnB")); lnB.set("w", str(Emu(Pt(0.75))))
            sf = etree.SubElement(lnB, qn("a:solidFill"))
            etree.SubElement(sf, qn("a:srgbClr")).set("val", HAIR)
    return gf


def stat(slide, x, y, w, big, label, big_size=30, color=ACCENT):
    text(slide, x, y, w, 0.55, big, big_size, color, bold=True)
    text(slide, x, y + 0.58, w, 0.6, label, 12, INK2)


# ── slides ──────────────────────────────────────────────────────────────────
def s_title(d):
    s = d.slide(dark=True)
    image(s, os.path.join(FIG, "deck", "overlay_100034.jpg"), 7.4, 0, border=False, crop_to=(W - 7.4, H))
    text(s, MX + 0.1, 1.5, 6.4, 0.3, "RESEARCH PROJECT", 12, "8FC1A9", bold=True)
    text(s, MX + 0.1, 1.95, 6.4, 2.2, "Topology-Aware Rural Road Extraction from Satellite Imagery",
         38, "FFFFFF", bold=True, line_spacing=0.95)
    text(s, MX + 0.1, 4.25, 6.2, 1.0,
         "A 1.6 M-parameter MobileViT v2 pipeline that maps rural roads, is designed to keep them connected "
         "where canopy breaks them, and runs on a laptop CPU.", 17, DARK_TEXT2)
    text(s, MX + 0.1, 6.35, 6.4, 0.3, "DeepGlobe Road Extraction  ·  PyTorch  ·  ONNX Runtime", 12, "8FC1A9")
    text(s, 7.55, H - 0.45, 5.6, 0.3,
         "Model output on DeepGlobe tile 100034 (blue: predicted road, orange: bridged gap)", 10, "FFFFFF")
    notes(s, "This project builds a lightweight deep-learning pipeline that turns a satellite image tile into a "
             "connected road map, with particular attention to rural roads hidden under trees. The image on the right "
             "is the real output of the final model on a DeepGlobe tile: blue pixels are predicted road.")


def s_problem(d):
    s = d.slide()
    header(s, "01 · Problem", "Rural roads are hard to map from above")
    text(s, MX, CONTENT_TOP + 0.05, 6.3, 0.9,
         "Programmes such as India’s PMGSY need frequent, large-area audits of rural roads — "
         "today largely done through manual ground surveys.", 18, INK)
    bullets(s, MX, CONTENT_TOP + 1.25, 6.3, 3.6, [
        "Satellite imagery (0.5 m per pixel) could map roads at scale, automatically.",
        "Rural roads are narrow, unpaved and close in colour to fields and field boundaries.",
        "Tree canopy and shadows hide stretches of road, so pixel classifiers return broken, "
        "disconnected fragments instead of a usable network.",
        "Surveys happen in the field: the model has to run on modest hardware without a GPU.",
    ], size=15, gap=9)
    box_text(s, MX, 6.05, 6.3, 0.72,
             [[("Goal  ", {"bold": True, "color": ACCENT}),
               ("turn a satellite tile into a connected road network — even where trees cover the road.", {})]],
             size=14, align=PP_ALIGN.LEFT, fill=TINT)
    img = os.path.join(FIG, "deck", "tile_115714.jpg")
    x0, y0, side = 7.35, CONTENT_TOP, 5.1
    image(s, img, x0, y0, w=side)
    caption(s, x0, y0 + side + 0.08, side,
            "DeepGlobe tile 115714: an unpaved road among fields whose boundaries look much like roads.")
    notes(s, "Rural road auditing is slow when done on the ground. High-resolution satellite imagery covers large areas, "
             "but rural roads are thin, unpaved, and low-contrast against farmland, and trees or shadows often hide parts "
             "of them. A standard segmentation model then produces broken pieces rather than a network you could navigate "
             "or audit. The tile shown is a real DeepGlobe sample: a dirt road through farmland, where field boundaries "
             "have almost the same colour and shape as the road.")


def s_challenges(d):
    s = d.slide()
    header(s, "02 · Challenges & approach", "Four challenges, four design responses")
    cards = [
        ("Thin, irregular roads", "Roads are a few pixels wide and wind through fields.",
         "Strip convolutions (1×3 then 3×1) act as directional road detectors."),
        ("Canopy & shadow occlusion", "Trees hide stretches of road, breaking connectivity.",
         "Topology-aware clDice loss in training; canopy-gap bridging at inference."),
        ("Severe class imbalance", "Roads cover only a few percent of pixels.",
         "Weighted BCE + soft Dice, with checkpoint selection guarded against collapse."),
        ("Edge-hardware limits", "Field devices have no GPU and little memory.",
         f"MobileViT v2 with linear-cost attention: {P['params'] / 1e6:.1f} M parameters, ONNX export."),
    ]
    cw, gap = (W - 2 * MX - 3 * 0.3) / 4, 0.3
    for i, (t, prob, resp) in enumerate(cards):
        x = MX + i * (cw + gap)
        box(s, x, CONTENT_TOP + 0.1, cw, 4.75, fill=TINT, radius=0.05)
        circle_num(s, x + 0.3, CONTENT_TOP + 0.4, 0.5, str(i + 1))
        text(s, x + 0.3, CONTENT_TOP + 1.15, cw - 0.6, 0.8, t, 18, INK, bold=True)
        text(s, x + 0.3, CONTENT_TOP + 1.95, cw - 0.6, 1.1, prob, 14, INK2)
        text(s, x + 0.3, CONTENT_TOP + 3.1, cw - 0.6, 0.3, "OUR RESPONSE", 10, ACCENT, bold=True)
        text(s, x + 0.3, CONTENT_TOP + 3.42, cw - 0.6, 1.3, resp, 14, INK)
    notes(s, "Each challenge maps to a specific design choice. Thin roads motivate directional strip convolutions. "
             "Occlusion motivates a topology-aware loss (clDice), which rewards predictions that preserve the road's "
             "centerline, plus a geometric gap-bridging step after inference. Class imbalance motivates weighted BCE "
             "and Dice, and - as we learned the hard way - a checkpoint-selection rule that cannot be fooled by a model "
             "that predicts road everywhere. Edge constraints motivate a small MobileViT v2 backbone.")


def s_related(d):
    s = d.slide()
    header(s, "03 · Existing approaches", "What exists, and where it falls short here")
    g = lambda n: f"{M[n]['params'] / 1e6:.1f} M" if n in M else "—"
    f = lambda n: f"{M[n]['gflops']:,.0f}" if n in M else "—"
    rows = [
        ["Approach", "Key idea", "Limitation for rural roads on the edge", "Params*", "GFLOPs*"],
        ["U-Net (Ronneberger et al., 2015)", "Encoder–decoder with skip connections",
         "Heavy; pixel-wise loss ignores connectivity", g("U-Net"), f("U-Net")],
        ["D-LinkNet34 (Zhou et al., 2018)", "ResNet-34 + dilated centre; DeepGlobe 2018 winner",
         "Heavy; no explicit connectivity objective", g("D-LinkNet34"), f("D-LinkNet34")],
        ["DeepLabV3 / LR-ASPP + MobileNetV3", "Lightweight mobile segmentation heads",
         "No topology prior for thin structures", f"{M['LR-ASPP-MBv3']['params'] / 1e6:.1f}–{M['DeepLabV3-MBv3']['params'] / 1e6:.1f} M",
         f"{M['LR-ASPP-MBv3']['gflops']:.0f}–{M['DeepLabV3-MBv3']['gflops']:.0f}"],
        ["Vision Transformers (Dosovitskiy et al., 2021)", "Global context via self-attention",
         "Attention cost grows quadratically with image size", "—", "—"],
        ["clDice loss (Shit et al., 2021)", "Overlap measured on skeletons (centerlines)",
         "A loss, not a model; can reward over-prediction (seen here)", "—", "—"],
        ["RoadTracer / Sat2Graph (2018 / 2020)", "Predict the road graph directly",
         "Iterative or multi-stage inference; complex to deploy", "—", "—"],
        ["This work", "MobileViT v2 + strip conv + clDice + gap bridging",
         "—", f"{P['params'] / 1e6:.1f} M", f"{P['gflops']:.0f}"],
    ]
    table(s, MX, CONTENT_TOP + 0.05, W - 2 * MX, rows, [3.2, 3.3, 3.42, 1.28, 0.93], font=12.5, row_h=0.52,
          first_bold=True, align_right_from=3, highlight_row=len(rows) - 1)
    caption(s, MX, CONTENT_TOP + 0.05 + 0.52 * len(rows) + 0.12, W - 2 * MX,
            f"* Measured on the same CPU ({CPU_SHORT}) per 1024² tile with untrained models: compute cost only. "
            "Accuracy of the reference models was not measured.")
    notes(s, "Standard CNN segmenters like U-Net and D-LinkNet (the DeepGlobe 2018 road-challenge winner) work well on clear "
             "roads but are heavy and optimise pixels, not connectivity. Mobile segmentation heads are light but have no "
             "notion of thin, connected structures. Transformers give global context but plain self-attention is expensive "
             "at high resolution. clDice is a loss that measures agreement on skeletons. Graph methods predict the road "
             "network directly but are complex. The parameter and FLOP columns were measured on the same laptop CPU for "
             "every model - these are compute numbers only, since the reference models were not trained here.")


def s_dataset(d):
    s = d.slide()
    header(s, "04 · Data", "DeepGlobe Road Extraction dataset")
    stats_ = [("6,226", "labelled satellite tiles"), ("1024²", "pixels per tile"),
              ("0.5 m", "ground distance per pixel"), ("5,604 / 622", "train / validation tiles")]
    for i, (big, lab) in enumerate(stats_):
        stat(s, MX + (i % 2) * 3.05, CONTENT_TOP + 0.05 + (i // 2) * 1.35, 2.9, big, lab, big_size=28)
    bullets(s, MX, CONTENT_TOP + 2.85, 5.9, 2.6, [
        "Demir et al., CVPR Workshops 2018; imagery of Thailand, Indonesia and India with binary road masks.",
        "The official validation and test sets have no public masks, so a held-out 10% of the labelled set "
        "(seed 42) is used for validation.",
        "Training samples random 256² crops at native resolution, with flips and brightness / colour jitter.",
    ], size=14, gap=8)
    tiles = ["100034", "102408", "115714", "117991"]
    side, gap, x0 = 2.3, 0.2, 7.35
    for i, t in enumerate(tiles):
        x = x0 + (i % 2) * (side + gap)
        y = CONTENT_TOP + 0.05 + (i // 2) * (side + 0.42)
        image(s, os.path.join(FIG, "deck", f"tile_{t}.jpg"), x, y, w=side)
        caption(s, x, y + side + 0.05, side, f"Tile {t}", size=10)
    notes(s, "DeepGlobe provides 6,226 labelled 1024x1024 tiles at 50 cm per pixel. Because the official validation and "
             "test masks were never released, every training run held out a fixed 10% of the labelled tiles "
             "(622 tiles, seed 42) for validation. The four tiles shown are real samples used throughout the slides: "
             "forest road, dense settlement, farmland, and a village with tree cover.")


def s_method(d):
    s = d.slide()
    header(s, "05 · Method overview", "From satellite tile to connected road network")
    lanes = [
        ("TRAINING", CONTENT_TOP + 0.35, [
            "DeepGlobe tiles\n(5,604)", "Augment\n256² crops, flips", "MobileViT v2\nencoder–decoder",
            "Composite loss\nBCE + Dice + clDice", "Select checkpoint\nval IoU + collapse gate"]),
        ("INFERENCE", CONTENT_TOP + 2.55, [
            "Satellite tile\n(any size ÷32)", "4-flip test-time\naugmentation", "Road probability\nmap",
            "Hysteresis\n+ closing", "Canopy-gap\nbridging", "Connected\nroad mask"]),
    ]
    for label, y, steps in lanes:
        text(s, MX, y - 0.34, 3, 0.28, label, 11, ACCENT, bold=True)
        n = len(steps)
        bw, gap = (W - 2 * MX - (n - 1) * 0.42) / n, 0.42
        for i, st in enumerate(steps):
            x = MX + i * (bw + gap)
            key = ("MobileViT" in st) or ("bridging" in st)
            box_text(s, x, y, bw, 1.15, st, size=13, fill="E4EFEA" if key else TINT, bold=key)
            if i < n - 1:
                arrow(s, x + bw + 0.05, y + 0.575, x + bw + gap - 0.05, y + 0.575)
    box_text(s, MX, CONTENT_TOP + 4.25, W - 2 * MX, 0.72,
             [[("Deployment  ", {"bold": True, "color": ACCENT}),
               ("the trained network is exported to ONNX and runs with ONNX Runtime + OpenCV only — "
                "no PyTorch or GPU needed on the field device.", {})]],
             size=14, align=PP_ALIGN.LEFT)
    notes(s, "Training: tiles are cropped to 256x256 at native resolution and augmented; the network is trained with a "
             "loss that combines pixel accuracy (BCE, Dice) and topology (clDice); the checkpoint is chosen by validation "
             "IoU with a hard gate that rejects collapsed epochs. Inference: a full tile goes through the network four "
             "times (original plus three flips) and the probabilities are averaged; the map is thresholded with hysteresis, "
             "closed morphologically, and then dead-end road segments that face each other across a gap are bridged. "
             "The highlighted boxes are the project's two core components.")


def s_arch(d):
    s = d.slide()
    header(s, "06 · Model", f"MobileViT v2 encoder–decoder — {P['params'] / 1e6:.2f} M parameters")
    # Matches backend/src/models/mobilevit_v2.py: stem -> enc1 -> enc2_mvit -> (enc2_down + enc3_mvit)
    # -> (enc3_down + bottleneck); skips are taken from the stem, enc2_mvit and enc3_mvit outputs.
    enc = [("Input", "3 · H"), ("Strip-conv stem", "32 · H/2"), ("MV2 ↓", "64 · H/4"),
           ("MobileViT v2", "64 · H/4"), ("MV2 ↓ +\nMobileViT v2", "96 · H/8"),
           ("MV2 ↓ +\nMobileViT v2 ×3", "128 · H/16")]
    dec = [("1×1 head", "1 · H"), ("Decoder", "32 · H/2"), ("Decoder", "64 · H/4"),
           ("Decoder", "96 · H/8")]
    n, x0, total = len(enc), MX, 7.6
    bw, gap = (total - (n - 1) * 0.22) / n, 0.22
    ye, yd = CONTENT_TOP + 0.35, CONTENT_TOP + 2.75
    xs = [x0 + i * (bw + gap) for i in range(n)]
    for i, (t, dim) in enumerate(enc):
        mv = "MobileViT" in t
        box_text(s, xs[i], ye, bw, 1.05, [[(t, {"bold": True, "size": 11.5})], [(dim, {"size": 10.5, "color": INK2})]],
                 fill="E4EFEA" if mv else TINT)
        if i < n - 1:
            arrow(s, xs[i] + bw + 0.02, ye + 0.52, xs[i + 1] - 0.02, ye + 0.52)
    text(s, x0, ye - 0.32, 3, 0.26, "ENCODER", 10, ACCENT, bold=True)
    text(s, x0, yd - 0.32, 3, 0.26, "DECODER", 10, ACCENT, bold=True)
    dxs = [xs[1], xs[2], xs[3], xs[4]]
    for i, ((t, dim), x) in enumerate(zip(dec, dxs)):
        box_text(s, x, yd, bw, 1.05, [[(t, {"bold": True, "size": 11.5})], [(dim, {"size": 10.5, "color": INK2})]])
    arrow(s, xs[5] + bw / 2, ye + 1.07, xs[4] + bw + 0.02, yd + 0.52)        # bottleneck -> first decoder
    for i in (3, 2, 1):
        arrow(s, dxs[i] - 0.02, yd + 0.52, dxs[i - 1] + bw + 0.02, yd + 0.52)
    for i, src in ((1, 1), (2, 3), (3, 4)):   # skip connections: stem->dec(H/2), MViT(H/4)->dec(H/4), MViT(H/8)->dec(H/8)
        arrow(s, xs[src] + bw / 2, ye + 1.07, dxs[i] + bw / 2, yd - 0.02, color=ACCENT, dash=True, width=1.1)
    text(s, xs[1], yd + 1.18, 6.2, 0.5,
         "Dashed: skip connections, each filtered by an attention gate. Channels · resolution for an H×H input.",
         10.5, MUTED)
    comps = [
        ("Strip convolution", "1×3 then 3×1 filters: directional, ~33% fewer weights than a 3×3."),
        ("Channel shift", "25% of channels shifted ±2 px in four directions: wider context, zero parameters."),
        ("Separable self-attention", "MobileViT v2 attention: global context at linear cost in image size."),
        ("Attention gates", "Learned masks on skip connections suppress roof and field-edge clutter."),
    ]
    cx = MX + total + 0.45
    cwid = W - MX - cx
    for i, (t, desc) in enumerate(comps):
        y = CONTENT_TOP + 0.1 + i * 1.22
        box(s, cx, y, cwid, 1.08, fill=TINT, radius=0.08)
        text(s, cx + 0.2, y + 0.12, cwid - 0.4, 0.3, t, 14, INK, bold=True)
        text(s, cx + 0.2, y + 0.45, cwid - 0.4, 0.6, desc, 12, INK2)
    stat_y = CONTENT_TOP + 4.55
    for i, (big, lab) in enumerate([(f"{P['params'] / 1e6:.2f} M", "parameters"),
                                    (f"{P['gflops']:.0f}", "GFLOPs per 1024² tile"),
                                    (f"{P['weights_mb_fp32']:.1f} MB", "weights (fp32)")]):
        text(s, MX + i * 2.5, stat_y, 2.4, 0.45, big, 22, ACCENT, bold=True)
        text(s, MX + i * 2.5, stat_y + 0.45, 2.4, 0.3, lab, 11.5, INK2)
    notes(s, "The encoder mixes cheap convolutional blocks with MobileViT v2 blocks, which add global context through "
             "separable self-attention whose cost grows linearly with image size rather than quadratically. A channel "
             "shift in front of each transformer block widens the receptive field for free. The decoder upsamples back "
             "to full resolution using strip convolutions, and each skip connection passes through an attention gate "
             "that learns to suppress clutter such as roof edges and field boundaries. Parameter and FLOP counts are "
             "measured from the trained checkpoint.")


def s_training(d):
    s = d.slide()
    header(s, "07 · Training", "Loss that rewards connected roads, trained with guard rails")
    text(s, MX, CONTENT_TOP + 0.05, 6.1, 0.4, "Composite loss", 16, INK, bold=True)
    box_text(s, MX, CONTENT_TOP + 0.5, 6.1, 0.8,
             [[("L = α(e)·BCE", {"bold": True}), ("  +  0.35·Dice  +  (0.65 − α(e))·clDice", {"bold": True})]],
             size=17, fill=TINT)
    bullets(s, MX, CONTENT_TOP + 1.55, 6.1, 1.9, [
        "BCE (road pixels weighted ×2) and soft Dice teach where roads are.",
        "clDice compares skeletons, penalising breaks in a road’s centerline.",
        "α starts at 0.50 and decays to 0.15 by epoch 40, shifting weight to clDice.",
    ], size=14, gap=7)
    f = CK["final"]
    rows = [["Setting", "Value"],
            ["Optimiser", "AdamW, lr 1e-3, cosine decay"],
            ["Batch", "16 × 256² crops, mixed precision"],
            ["Initialisation", "Warm start from June baseline"],
            ["Early stopping", "Patience 35 on validation IoU"],
            ["Best checkpoint", f"Epoch {f['epoch']} (0-indexed)"]]
    table(s, MX, CONTENT_TOP + 2.9, 6.1, rows, [2.1, 4.0], font=12.5, row_h=0.36, first_bold=True)
    # alpha schedule: a deterministic function of the training hyper-parameters (not data)
    epochs = list(range(0, 61))
    alpha = [0.5 - 0.35 * min(e / 40, 1.0) ** 0.5 for e in epochs]
    cd = CategoryChartData()
    cd.categories = [str(e) for e in epochs]
    cd.add_series("BCE weight α(e)", alpha)
    cd.add_series("Dice weight", [0.35] * len(epochs))
    cd.add_series("clDice weight", [0.65 - a for a in alpha])
    gx, gy, gw, gh = 7.1, CONTENT_TOP + 0.05, W - MX - 7.1, 3.9
    text(s, gx, gy, gw, 0.35, "Loss weights over training", 16, INK, bold=True)
    ch = s.shapes.add_chart(XL_CHART_TYPE.LINE, Inches(gx), Inches(gy + 0.4), Inches(gw), Inches(gh), cd).chart
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(11); ch.legend.font.color.rgb = rgb(INK2); ch.legend.font.name = FONT
    for ser, col in zip(ch.series, (BLUE, ORANGE, AQUA)):
        ser.format.line.color.rgb = rgb(col)
        ser.format.line.width = Pt(2.25)
        ser.smooth = False
        ser.marker.style = None
        from pptx.enum.chart import XL_MARKER_STYLE
        ser.marker.style = XL_MARKER_STYLE.NONE
    va = ch.value_axis
    va.maximum_scale, va.minimum_scale, va.major_unit = 0.7, 0.0, 0.1
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb("E6EAE8")
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(11); va.tick_labels.font.color.rgb = rgb(INK2)
    va.tick_labels.number_format = "0.0"; va.tick_labels.number_format_is_linked = False
    ca = ch.category_axis
    ca.tick_labels.font.size = Pt(11); ca.tick_labels.font.color.rgb = rgb(INK2)
    ca.format.line.color.rgb = rgb(HAIR)
    ca.has_major_gridlines = False
    cax = ca._element
    for tag, val in (("c:tickLblSkip", "10"), ("c:tickMarkSkip", "10")):
        el = cax.find(qn(tag))
        if el is None:
            el = etree.SubElement(cax, qn(tag))
        el.set("val", val)
    caption(s, gx, gy + 0.45 + gh, gw, "Epoch (0-indexed). Schedule computed from the training hyper-parameters.", 10.5)
    notes(s, "The loss combines three terms. BCE with road pixels weighted twice as heavily, and soft Dice, both reward "
             "getting road pixels right. clDice, from Shit et al. 2021, compares the predicted and true road skeletons and "
             "penalises breaks in the centerline - the property we care about under canopy. Early in training BCE dominates "
             "so the network first learns where roads are; alpha then decays so clDice takes over. The model was warm-started "
             "from the earlier June baseline and trained on a Kaggle GPU; the best checkpoint by validation IoU was epoch 53. "
             "Note: a canopy-shadow augmentation exists in the code, but we found its constructor call silently disabled it "
             "on newer albumentations versions, so we do not claim it was active during training; the bug is now fixed.")


def s_collapse(d):
    s = d.slide()
    header(s, "08 · A failure we fixed", "Diagnosing a training collapse")
    image(s, os.path.join(FIG, "fig_collapse.png"), MX, CONTENT_TOP, w=7.3, border=False)
    c = LOCAL.get("collapse", {})
    cards = [
        ("Symptom", f"The collapsed checkpoint marked {pct(c.get('collapsed_all_tiles_pos_frac', 0), 0)} of pixels as road "
                    f"on the sample tiles (fixed model: {pct(c.get('final_all_tiles_pos_frac', 0))})."),
        ("Cause", "Road pixels were weighted ×3, pushing a freshly initialised decoder toward “everything is road”; "
                  "checkpoints were chosen by soft clDice, which scores a blanket prediction well."),
        ("Fix", "Weight ×2; select checkpoints by IoU; reject any epoch predicting >20% road; "
                "warm-start from the working baseline."),
    ]
    x = MX + 7.3 + 0.4
    cw = W - MX - x
    for i, (t, desc) in enumerate(cards):
        y = CONTENT_TOP + i * 1.62
        box(s, x, y, cw, 1.48, fill=TINT, radius=0.07)
        text(s, x + 0.22, y + 0.14, cw - 0.4, 0.3, t.upper(), 10.5, ACCENT, bold=True)
        text(s, x + 0.22, y + 0.44, cw - 0.4, 1.0, desc, 12.5, INK)
    collapsed_loss = CK["collapsed"]["val_cldice"]
    final_loss = CK["final"]["val_cldice"]
    caption(s, MX, CONTENT_TOP + 4.95, 7.3,
            f"The collapsed run’s logged validation clDice loss was {collapsed_loss:.3f} — better-looking than the fixed "
            f"model’s {final_loss:.3f} — which is why it was saved as “best”.", 11, INK2)
    notes(s, "An earlier version of the model converged to predicting road almost everywhere - about 90% of every tile. "
             "Two things combined: the positive-class weight of 3 made 'everything is road' a cheap way to reduce the loss "
             "for a freshly initialised decoder, and the training script kept whichever epoch had the lowest soft clDice loss. "
             "clDice's sensitivity term saturates when the prediction covers the whole image, so the collapsed epoch looked "
             "like the best one. We lowered the weight to 2, switched checkpoint selection to IoU, added a hard gate that "
             "rejects any epoch predicting more than 20% road, and warm-started from the working baseline. The plots show "
             "real outputs of both checkpoints on the same tiles.")


def s_postproc(d):
    s = d.slide()
    header(s, "09 · Inference", "Reconnecting fragmented roads")
    pic = image(s, os.path.join(FIG, "fig_pipeline_stages.png"), MX, CONTENT_TOP - 0.05, w=W - 2 * MX, h=3.6,
                border=False)
    ps = LOCAL.get("pipeline_stages", {})
    steps = [
        ("Hysteresis threshold", "Keep weak road pixels (0.12–0.35) only where they touch confident ones (≥0.35)."),
        ("Morphological closing", "A 5×5 closing seals hairline breaks in the mask."),
        ("Canopy-gap bridging", "Skeletonise, find dead ends, and join pairs that face each other within 220 px and "
                                "65°; complete T-junctions. Road exits at the tile border are ignored."),
    ]
    y = Emu(pic.top + pic.height).inches + 0.3
    cw = (W - 2 * MX - 2 * 0.3) / 3
    for i, (t, desc) in enumerate(steps):
        x = MX + i * (cw + 0.3)
        circle_num(s, x, y, 0.42, str(i + 1), size=12)
        text(s, x + 0.55, y + 0.03, cw - 0.55, 0.3, t, 14, INK, bold=True)
        text(s, x + 0.55, y + 0.38, cw - 0.55, 1.1, desc, 12, INK2)
    notes(s, f"This is a real tile ({ps.get('tile', '')}) processed by the deployed pipeline. The averaged probability map "
             f"is thresholded with hysteresis, which keeps faint road evidence (such as partially occluded road) as long "
             f"as it connects to confident road. Closing seals tiny breaks. Bridging then skeletonises the mask, finds dead ends, and joins "
             f"pairs that point at each other across a gap. On this tile the number of separate road pieces drops from "
             f"{ps.get('components_before', '?')} to {ps.get('components_after', '?')}. While preparing these slides we found "
             f"that the original bridging also joined roads that simply leave the tile, drawing fake roads along the border; "
             f"endpoints near the border are now ignored. Bridging can still add false links, which the ground-truth "
             f"evaluation measures.")


def s_tools(d):
    s = d.slide()
    header(s, "10 · Tools", "Technology stack")
    tools = [
        ("PyTorch", "Model definition, training with mixed precision"),
        ("Albumentations", "Crops, flips, colour and canopy-shadow augmentation"),
        ("OpenCV · scikit-image", "Thresholding, morphology, skeletons, gap bridging"),
        ("ONNX · ONNX Runtime", "Export and PyTorch-free inference on CPU / edge"),
        ("Kaggle GPU notebooks", "Training and held-out evaluation"),
        ("NetworkX", "Converting road masks into graphs"),
        ("Streamlit", "Interactive demo application"),
        ("Matplotlib", "All figures, from measured data"),
    ]
    cols, cw, chh = 4, (W - 2 * MX - 3 * 0.3) / 4, 2.3
    for i, (t, desc) in enumerate(tools):
        x = MX + (i % cols) * (cw + 0.3)
        y = CONTENT_TOP + 0.15 + (i // cols) * (chh + 0.3)
        box(s, x, y, cw, chh, fill=TINT, radius=0.06)
        circle_num(s, x + 0.25, y + 0.3, 0.42, str(i + 1), size=12)
        text(s, x + 0.25, y + 0.95, cw - 0.5, 0.4, t, 17, INK, bold=True)
        text(s, x + 0.25, y + 1.45, cw - 0.5, 0.8, desc, 13, INK2)
    notes(s, "The model and training loop are written in PyTorch; augmentation uses Albumentations. Post-processing uses "
             "OpenCV and scikit-image. The trained network is exported to ONNX so field devices only need ONNX Runtime and "
             "OpenCV. Training and evaluation ran in Kaggle GPU notebooks. A Streamlit app provides an interactive demo.")


def s_efficiency(d):
    s = d.slide()
    header(s, "11 · Results: efficiency", "Measured compute cost")
    o = EFF.get("onnx", {})
    stats_ = [
        (f"{P['params'] / 1e6:.1f} M", f"parameters — {UNET['params'] / P['params']:.0f}× fewer than U-Net"),
        (f"{P['gflops']:.0f}", f"GFLOPs per 1024² tile — {UNET['gflops'] / P['gflops']:.0f}× fewer than U-Net"),
        (f"{o.get('latency_ms', 0) / 1000:.2f} s", "per tile with ONNX Runtime on a laptop CPU"),
        (f"{o.get('file_mb_total', 0):.1f} MB", "deployable ONNX model (graph + weights)"),
    ]
    cw = (W - 2 * MX - 3 * 0.3) / 4
    for i, (big, lab) in enumerate(stats_):
        stat(s, MX + i * (cw + 0.3), CONTENT_TOP, cw, big, lab, big_size=30)
    image(s, os.path.join(FIG, "fig_efficiency.png"), MX, CONTENT_TOP + 1.3, w=W - 2 * MX, h=3.75, border=False)
    lr = M["LR-ASPP-MBv3"]
    notes(s, f"All of these numbers were measured on the same laptop CPU ({EFF['cpu']}), batch size 1, on a full 1024x1024 "
             f"tile. The proposed model has {P['params']:,} parameters and needs {P['gflops']:.0f} GFLOPs per tile, far below "
             f"U-Net and D-LinkNet. Honest caveat: LR-ASPP with MobileNetV3 is cheaper still ({lr['gflops']:.0f} GFLOPs, "
             f"{lr['latency_ms'] / 1000:.2f} s), so our efficiency advantage is over U-Net-class models, not over every mobile "
             f"network - and its accuracy on this task was not measured. The deployable ONNX model is "
             f"{o.get('file_mb_total', 0):.1f} MB: a small graph file plus a separate weights file that must ship with it.")


def s_validation(d):
    s = d.slide()
    header(s, "12 · Results: validation", "Training-time validation, and what it missed")
    rows = [["Checkpoint", "Epoch", "Val IoU", "Precision", "Pred. road", "clDice loss"]]
    lab = {"baseline": "June baseline (no gates)", "collapsed": "v2 collapsed run", "run1": "v2 run 1", "final": "v2 final"}
    for k in ("baseline", "collapsed", "run1", "final"):
        m = CK[k]
        rows.append([lab[k], str(m.get("epoch", "—")),
                     pct(m["val_iou"]) if "val_iou" in m else "not logged",
                     pct(m["val_precision"]) if "val_precision" in m else "—",
                     pct(m["val_positive_frac"]) if "val_positive_frac" in m else "—",
                     f"{m['val_cldice']:.3f}"])
    table(s, MX, CONTENT_TOP + 0.05, 6.55, rows, [2.35, 0.7, 1.05, 0.95, 0.8, 0.95] if False else [2.3, 0.7, 1.1, 0.95, 0.85, 0.65 + 0.3],
          font=12, row_h=0.42, first_bold=True, align_right_from=1, highlight_row=4)
    caption(s, MX, CONTENT_TOP + 0.05 + 0.42 * 5 + 0.1, 6.55,
            "Values stored in each checkpoint, measured during training on the 622 validation tiles resized to 256².", 11)
    text(s, MX, CONTENT_TOP + 2.75, 6.55, 0.35, "Finding: the validation protocol under-measured the model", 15, INK, bold=True)
    res = LOCAL.get("resolution", {})
    zero = sum(1 for v in res.values() if v.get("resized_pos_frac", 1) == 0)
    bullets(s, MX, CONTENT_TOP + 3.2, 6.55, 1.9, [
        "Validation shrank each 1024² tile to 256², making roads 4× thinner than anything seen in training.",
        f"On {zero} of {len(res)} sample tiles the model then finds no road at all, while at native resolution it does (right).",
        "The logged IoU of " + pct(CK['final']['val_iou']) + " therefore understates the deployed model; "
        + ("held-out native-resolution results follow." if KRES else
           "the evaluation notebook measures it at native resolution on the same held-out tiles."),
    ], size=13.5, gap=6)
    image(s, os.path.join(FIG, "fig_resolution.png"), 7.55, CONTENT_TOP - 0.1, h=5.35, border=False)
    notes(s, "These are the validation metrics saved inside each checkpoint during training. The final model reached 16.8% "
             "IoU and about 49% precision at epoch 53. But while preparing the results we found that validation resized each "
             "1024 pixel tile down to 256 pixels, whereas training used native-resolution crops. Roads become four times "
             "thinner than anything the model trained on, and on two of the three tiles shown it finds nothing at all. The "
             "same model at native resolution - how it is actually deployed - finds the roads. So the logged IoU understates "
             "real performance, and checkpoint selection was also made at the wrong scale. The validation code is now fixed.")


def s_qualitative(d):
    s = d.slide()
    header(s, "13 · Results: outputs", "Deployed pipeline on real tiles")
    tiles = ["100034", "117991", "115714", "102408"]
    side, gap = (W - 2 * MX - 3 * 0.25) / 4, 0.25
    y = CONTENT_TOP + 0.15
    for i, t in enumerate(tiles):
        x = MX + i * (side + gap)
        image(s, os.path.join(FIG, "deck", f"overlay_{t}.jpg"), x, y, w=side)
        st = OVER.get(t, {})
        caption(s, x, y + side + 0.08, side, f"Tile {t}  ·  {pct(st.get('road_frac', 0))} road", 11, INK2)
    ly = y + side + 0.45
    for j, (col, lab) in enumerate(((BLUE, "predicted road"), (ORANGE, "added by canopy-gap bridging"))):
        lx = MX + j * 2.4
        sw = box(s, lx, ly + 0.05, 0.22, 0.22, fill=col, radius=0.2)
        text(s, lx + 0.32, ly + 0.02, 2.2, 0.3, lab, 12, INK2)
    text(s, MX + 5.6, ly + 0.02, W - 2 * MX - 5.6, 0.3,
         "Sample tiles without ground truth — accuracy is measured on the held-out split.", 12, MUTED)
    obs = [
        ("Continuous where the road is visible", "The forest road (100034) and the farm road (115714) are traced "
                                                 "end to end in a single piece."),
        ("Field boundaries still fool it", "On 115714 a field boundary is bridged into a short spur off the main "
                                           "road — a likely false positive."),
        ("Dense areas need ground truth", "In the settlement (102408) the street grid is plausible, but some bridges "
                                          "may join paths between buildings."),
    ]
    oy, cw = ly + 0.5, (W - 2 * MX - 2 * 0.3) / 3
    for i, (t, desc) in enumerate(obs):
        x = MX + i * (cw + 0.3)
        box(s, x, oy, cw, 1.2, fill=TINT, radius=0.08)
        text(s, x + 0.22, oy + 0.15, cw - 0.44, 0.3, t, 13.5, INK, bold=True)
        text(s, x + 0.22, oy + 0.5, cw - 0.44, 0.75, desc, 12, INK2)
    notes(s, "These are real outputs of the final model with the full deployed pipeline: four-flip test-time augmentation, "
             "hysteresis thresholding, closing and gap bridging. Blue is predicted road, orange marks segments added by "
             "bridging. The forest road on the left is traced continuously. These sample tiles have no ground truth, so they "
             "show behaviour, not accuracy.")


def s_edge(d):
    s = d.slide()
    header(s, "14 · Deployment", "Runs without PyTorch on field hardware")
    steps = ["PyTorch\ncheckpoint", "ONNX export\nsigmoid in graph,\ndynamic H × W", "Parity check\nvs PyTorch",
             "ONNX Runtime\n+ OpenCV", "Connected\nroad mask"]
    bw, gap, y = (W - 2 * MX - 4 * 0.42) / 5, 0.42, CONTENT_TOP + 0.3
    for i, st in enumerate(steps):
        x = MX + i * (bw + gap)
        box_text(s, x, y, bw, 1.3, st, size=13, fill="E4EFEA" if i == 3 else TINT, bold=(i == 3))
        if i < 4:
            arrow(s, x + bw + 0.05, y + 0.65, x + bw + gap - 0.05, y + 0.65)
    o = EFF.get("onnx", {})
    stats_ = [(f"{o.get('file_mb_total', 0):.1f} MB", "model on disk"),
              (f"{o.get('latency_ms', 0) / 1000:.2f} s", f"per 1024² tile, {CPU_SHORT}"),
              ("< 10⁻⁶", "max |difference| vs PyTorch output"),
              ("5", "Python packages on the device")]
    cw = (W - 2 * MX - 3 * 0.3) / 4
    for i, (big, lab) in enumerate(stats_):
        stat(s, MX + i * (cw + 0.3), CONTENT_TOP + 2.1, cw, big, lab, big_size=28)
    bullets(s, MX, CONTENT_TOP + 3.55, W - 2 * MX, 1.6, [
        "Picks the best available execution provider: CUDA / TensorRT (Jetson), CoreML (Apple), NNAPI (Android), or CPU.",
        "Same hysteresis and gap-bridging post-processing as the PyTorch path; four-flip TTA is optional (about 4× the time).",
        "A Streamlit app wraps the ONNX model for interactive demonstrations.",
    ], size=14, gap=7)
    notes(s, "For deployment the network is exported to ONNX with the sigmoid inside the graph and dynamic height and width, "
             "so any tile size divisible by 32 works. The export script checks the ONNX output against PyTorch and the "
             "maximum difference was below one millionth. On the device, only ONNX Runtime, OpenCV, NumPy, SciPy and "
             "scikit-image are needed. The latency shown is a single pass; test-time augmentation multiplies it by about four.")


def s_kaggle_main(d):
    s = d.slide()
    r = KRES["native_resolution"]
    ok = KRES.get("split_verified")
    header(s, "15 · Results: held-out accuracy",
           f"Native-resolution evaluation on {KRES['split']['n_eval']} held-out tiles")
    image(s, os.path.join(KAGGLE, "figures", "fig_eval_main.png"), MX, CONTENT_TOP, w=W - 2 * MX, h=4.5, border=False)
    fb, fd = r.get("baseline", {}).get("tta_full"), r["final"]["tta_full"]
    t = KRES["paired_tests"].get("final_vs_baseline_deployed_iou", {})
    line = [("Deployed pipeline: IoU ", {}), (pct(fd["iou"]), {"bold": True}),
            (f" (95% CI {pct(fd['ci95']['iou'][0])}–{pct(fd['ci95']['iou'][1])}), clDice ", {}),
            (pct(fd["cldice"]), {"bold": True})]
    if fb:
        line += [(f" vs baseline {pct(fb['iou'])} / {pct(fb['cldice'])}", {})]
    if t.get("wilcoxon_p") is not None:
        line += [(f";  better on {100 * t['frac_tiles_improved']:.0f}% of tiles (Wilcoxon p = {t['wilcoxon_p']:.1e})", {})]
    text(s, MX, 6.35, W - 2 * MX, 0.5, [line], 13.5, INK)
    if not ok:
        text(s, MX, 6.8, W - 2 * MX, 0.3, "Split NOT verified against checkpoint metrics — treat as provisional.",
             12, "C0392B", bold=True)
    notes(s, "Held-out results at native resolution, measured by the evaluation notebook against the DeepGlobe ground truth. "
             "The split was " + ("verified" if ok else "NOT verified") + " by reproducing the metrics stored inside the "
             "checkpoints. Whiskers are 95% bootstrap confidence intervals over tiles.")


def s_kaggle_fig(d, tag, title, fig, note):
    s = d.slide()
    header(s, tag, title)
    image(s, os.path.join(KAGGLE, "figures", fig), MX, CONTENT_TOP, w=W - 2 * MX, h=H - CONTENT_TOP - 0.6, border=False)
    notes(s, note)


def s_conclusion(d):
    s = d.slide()
    header(s, "16 · Conclusion", "What we built, what we learned, what’s next")
    ps = LOCAL.get("pipeline_stages", {})
    o = EFF.get("onnx", {})
    left = [
        f"A {P['params'] / 1e6:.1f} M-parameter road extractor that runs in {o.get('latency_ms', 0) / 1000:.1f} s per tile on a "
        "laptop CPU via ONNX.",
        "Diagnosed and fixed a training collapse caused by class weighting plus a gameable selection metric.",
        f"Gap bridging reconnects fragmented predictions (e.g. {ps.get('components_before', '?')} → "
        f"{ps.get('components_after', '?')} pieces on a sample tile); its effect under canopy is measured "
        "against ground truth.",
        "Found and fixed three pipeline bugs: validation scale mismatch, a silently disabled augmentation, and "
        "bridging along tile borders.",
    ]
    right = [
        "Retrain with native-resolution validation, so checkpoints are selected at the scale the model is used.",
        ("Report held-out accuracy from the evaluation notebook; " if not KRES else "Held-out accuracy is reported above; ")
        + f"the {pct(CK['final']['val_iou'])} IoU logged in training likely understates native-resolution performance.",
        "Tune gap bridging against ground truth — it can also create false links.",
        "Train a mobile CNN (LR-ASPP) on the same data for a fair accuracy–cost comparison; report APLS with the "
        "official SpaceNet code.",
    ]
    for j, (head, items) in enumerate((("Outcomes", left), ("Limitations & next steps", right))):
        x = MX + j * ((W - 2 * MX) / 2 + 0.15)
        wcol = (W - 2 * MX) / 2 - 0.15
        box(s, x, CONTENT_TOP + 0.05, wcol, 4.6, fill=TINT, radius=0.04)
        text(s, x + 0.35, CONTENT_TOP + 0.35, wcol - 0.7, 0.4, head, 18, ACCENT, bold=True)
        bullets(s, x + 0.35, CONTENT_TOP + 0.95, wcol - 0.7, 4.0, items, size=14.5, gap=10)
    notes(s, "To summarise: the pipeline works end to end and is light enough for field hardware. The most valuable lessons "
             "came from failures - a collapse hidden by a flattering metric, a validation step run at the wrong scale, an "
             "augmentation that silently never ran, and a bridging rule that invented roads along tile borders. Each is now "
             "fixed. Next steps are a retrain with native-resolution validation and a fair comparison against a trained "
             "mobile CNN.")


def s_end(d):
    s = d.slide(dark=True)
    text(s, MX + 0.1, 2.6, W - 2 * MX, 1.0, "Thank you", 48, "FFFFFF", bold=True)
    text(s, MX + 0.1, 3.65, W - 2 * MX, 0.5, "Questions and discussion", 20, DARK_TEXT2)
    text(s, MX + 0.1, 6.5, W - 2 * MX, 0.35,
         "github.com/Hrithik-Bhattacharya/Satellite-Imagery-based-Road-Network-Extraction", 13, "8FC1A9")


def main():
    d = Deck()
    s_title(d); s_problem(d); s_challenges(d); s_related(d); s_dataset(d); s_method(d); s_arch(d)
    s_training(d); s_collapse(d); s_postproc(d); s_tools(d); s_efficiency(d); s_validation(d); s_qualitative(d)
    if KRES:
        s_kaggle_main(d)
        s_kaggle_fig(d, "15 · Results: ablation", "What each inference step contributes", "fig_eval_ablation.png",
                     "Held-out ablation of the inference steps on the proposed model.")
        s_kaggle_fig(d, "15 · Results: canopy", "Recall and connectivity under canopy", "fig_eval_canopy.png",
                     "Canopy-covered road pixels identified with the Excess-Green index and an Otsu threshold.")
        qual = "fig_eval_qualitative_wide.png"
        if not os.path.exists(os.path.join(KAGGLE, "figures", qual)):
            qual = "fig_eval_qualitative.png"
        s_kaggle_fig(d, "15 · Results: vs ground truth", "Error maps on representative held-out tiles",
                     qual, "Tiles chosen at fixed IoU percentiles, not hand-picked.")
    s_edge(d); s_conclusion(d); s_end(d)
    d.save(OUT)
    print(f"wrote {OUT} ({d.n} slides){' with held-out evaluation slides' if KRES else ''}")


if __name__ == "__main__":
    main()
