"""
Builds docs/report/Literature_Validation.docx, the print version of the Literature Validation
doc (claude.ai artifact f26235c1-6004-4d60-a0d9-f57da34d7c09).

Measured numbers are read from the same measurement files the internship report uses, so the two
documents cannot drift apart. Numbers attributed to papers are cited, not measured, and are marked
as such in the text.

Usage (repo root):
  python scripts/report/build_validation_doc.py
  powershell -ExecutionPolicy Bypass -File scripts/report/export_pdf.ps1 \
      -Docx docs/report/Literature_Validation.docx -Pdf docs/report/Literature_Validation.pdf
"""

import json
import os

import torch
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG = os.path.join(REPO, "figures", "real")
OUT = os.environ.get("VALIDATION_OUT", os.path.join(REPO, "docs", "report", "Literature_Validation.docx"))
FONT = "Times New Roman"
TEXT_W = 6.27

EFF = json.load(open(os.path.join(FIG, "measurements", "efficiency.json")))
LOCAL = json.load(open(os.path.join(FIG, "measurements", "local_figure_stats.json")))
OVER = json.load(open(os.path.join(FIG, "deck", "overlay_stats.json")))
PARITY = json.load(open(os.path.join(FIG, "measurements", "onnx_parity.json")))
M = EFF["models"]
P, UNET, ONNX = M["Proposed"], M["U-Net"], EFF["onnx"]
LR = M["LR-ASPP-MBv3"]
CK = torch.load(os.path.join(REPO, "models", "best_model_v2.pth"), map_location="cpu")
F = {k: v for k, v in CK.items() if k not in ("model_state_dict", "optimizer_state_dict")}
PS, COL = LOCAL["pipeline_stages"], LOCAL["collapse"]
CPU = EFF["cpu"].replace("Intel(R) Core(TM) ", "Intel Core ").split(" CPU")[0]


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


class Doc:
    def __init__(self):
        self.d = Document()
        s = self.d.sections[0]
        s.page_width, s.page_height = Inches(8.27), Inches(11.69)
        for m in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
            setattr(s, m, Inches(1))
        st = self.d.styles["Normal"]
        st.font.name, st.font.size = FONT, Pt(11)
        st.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:cs"), FONT)
        self._page_numbers(s)

    def _page_numbers(self, section):
        p = section.footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for kind, text in (("begin", None), (None, " PAGE "), ("end", None)):
            r = p.add_run()
            r.font.name, r.font.size = FONT, Pt(10)
            el = OxmlElement("w:fldChar") if kind else OxmlElement("w:instrText")
            if kind:
                el.set(qn("w:fldCharType"), kind)
            else:
                el.set(qn("xml:space"), "preserve")
                el.text = text
            r._r.append(el)

    def _runs(self, p, text, size, bold=False, italic=False):
        """**bold** and *italic* markers inside text become runs."""
        import re
        for piece in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
            if not piece:
                continue
            b, i = bold, italic
            if piece.startswith("**") and piece.endswith("**"):
                piece, b = piece[2:-2], True
            elif piece.startswith("*") and piece.endswith("*"):
                piece, i = piece[1:-1], True
            r = p.add_run(piece)
            r.font.name, r.font.size, r.font.bold, r.font.italic = FONT, Pt(size), b, i
            r._r.get_or_add_rPr().get_or_add_rFonts().set(qn("w:cs"), FONT)
        return p

    def title(self, text, sub):
        p = self.d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(2)
        self._runs(p, text, 16, bold=True)
        q = self.d.add_paragraph()
        q.alignment = WD_ALIGN_PARAGRAPH.CENTER
        q.paragraph_format.space_after = Pt(14)
        self._runs(q, sub, 10.5, italic=True)

    def h(self, text, size=13):
        p = self.d.add_paragraph()
        pf = p.paragraph_format
        pf.space_before, pf.space_after, pf.keep_with_next = Pt(12), Pt(4), True
        self._runs(p, text, size, bold=True)

    def para(self, text, size=11, after=6, keep=False):
        p = self.d.add_paragraph()
        pf = p.paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.space_after, pf.line_spacing, pf.keep_with_next = Pt(after), 1.12, keep
        return self._runs(p, text, size)

    def bullets(self, items, size=11, numbered=False):
        for n, item in enumerate(items, 1):
            p = self.d.add_paragraph()
            pf = p.paragraph_format
            pf.left_indent, pf.first_line_indent = Inches(0.3), Inches(-0.18)
            pf.space_after, pf.line_spacing = Pt(3), 1.12
            pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            self._runs(p, (f"{n}.  " if numbered else "•  ") + item, size)

    def table(self, rows, widths, size=9):
        t = self.d.add_table(rows=len(rows), cols=len(rows[0]))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        for i, row in enumerate(rows):
            trPr = t.rows[i]._tr.get_or_add_trPr()
            trPr.append(OxmlElement("w:cantSplit"))
            if i == 0:
                trPr.append(OxmlElement("w:tblHeader"))
            for j, val in enumerate(row):
                cell = t.cell(i, j)
                cell.width = Inches(widths[j])
                p = cell.paragraphs[0]
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.line_spacing = 1.0
                self._runs(p, str(val), size, bold=(i == 0))
        for j, gc in enumerate(t._tbl.tblGrid.findall(qn("w:gridCol"))):
            gc.set(qn("w:w"), str(int(widths[j] * 1440)))
        layout = OxmlElement("w:tblLayout"); layout.set(qn("w:type"), "fixed")
        t._tbl.tblPr.append(layout)
        cm = OxmlElement("w:tblCellMar")
        for side, v in (("top", 25), ("bottom", 25), ("left", 70), ("right", 70)):
            e = OxmlElement(f"w:{side}"); e.set(qn("w:w"), str(v)); e.set(qn("w:type"), "dxa"); cm.append(e)
        t._tbl.tblPr.append(cm)
        self.para("", size=5, after=4)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.d.save(path)


def build():
    d = Doc()
    d.title("Literature Validation: Rural Road Extraction",
            "Project P29  ·  What the reviewed papers did, what we built, and how the two line up  "
            "·  20 September 2026")

    d.h("1. Purpose")
    d.para("This document checks our design against the papers we reviewed. For each paper it records what "
           "approach the authors used, what they reported, and what we took from it. It then sets out our own "
           "system, our measured results, what we optimised, and where our approach is stronger or weaker than "
           "the work we built on. It is a validation record: every design choice in the project should trace "
           "either to a paper we reviewed or to a measurement we made.")
    d.para("**How to read the labels.** Facts here come from three different levels of evidence. **Measured** "
           "means produced by a script in our repository, on the current model, and stored in a file. **Cited** "
           "means stated by the authors of a paper; we did not reproduce it, and it is not comparable with our "
           "own numbers because the dataset, the split and the protocol all differ. **Not recorded** means we "
           "did not capture it during the review, so it is marked rather than filled in from memory.")
    d.para("The distinction matters most for accuracy. We measured our compute cost thoroughly, but our only "
           "accuracy figure comes from validation during training, at a resolution we have since shown to be "
           "wrong. This document therefore makes no claim that our model beats any published model on accuracy.")

    d.h("2. The papers we reviewed")
    d.para("All figures in this table are cited: they are what the authors report, on their own data and "
           "protocol.", after=4)
    d.table([
        ["Paper", "Approach", "Dataset", "Reported results and model size", "Limitation for our case"],
        ["clDice (Shit et al., CVPR 2021)",
         "Similarity measured on morphological skeletons rather than masks; soft skeletons make it "
         "differentiable so it can be used as a loss",
         "Five public sets of tubular structures: Massachusetts Roads, DRIVE retina, CREMI, Vessap, and a 3D set",
         "Better connectivity, higher graph similarity and better volume scores than pixel losses; topology "
         "preserved up to homotopy equivalence",
         "A loss only. Needs a backbone and gives no architecture"],
        ["TopoRF-Net (Fu et al., Sensors 2025)",
         "Multi receptive field module, connectivity aware decoding, topology aware loss",
         "DeepGlobe Road (6,226 train / 1,240 val / 1,217 test); Massachusetts (1,108 / 14 / 49, 1 m)",
         "DeepGlobe road class IoU 69.76%, F1 82.18%, precision 85.50%, recall 79.12%; Massachusetts IoU "
         "59.68%. 56 M parameters; no FLOPs or speed reported",
         "Far too heavy for edge use, and sensitive to noisy labels"],
        ["Road MobileSeg (Qu et al., Sensors 2024)",
         "Mobile transformer backbone with coordinate attention and a micro token pyramid",
         "DeepGlobe, 6,226 images split 4,981 train / 1,245 test",
         "mIoU 71.52% / 73.36% / 74.76% for Tiny / Small / Base; 1.41 M, 2.83 M, 4.74 M parameters; 112, 157 "
         "and 295 ms on a Xiaomi 11 phone",
         "No connectivity aware learning, so roads still come out broken"],
        ["SAC loss (Shojaei et al., WACV Workshops 2025)",
         "Connectivity loss using proximity based weighting with asymmetric penalties, no skeleton needed",
         "Not recorded (paper text not retrievable for this review)",
         "Not recorded; the authors report reduced fragmentation of thin structures",
         "Depends on image resolution and needs manual calibration"],
        ["GLTDNet (Wang et al., IEEE 2025)",
         "Hybrid CNN and transformer, global and local feature enhancement with topological decoupling",
         "Cross domain road sets; exact names not recorded",
         "Precision 92.16%, F1 80.77%, IoU 67.75% in their cross domain setting; size not recorded",
         "Needs heavy tuning on heavily occluded imagery"],
        ["MViT-PCD (IEEE GRSL 2023)",
         "MobileViT backbone with multiscale feature differencing, for change detection",
         "Public Martian surface datasets",
         "Accuracy 97.2% and 82.9% on two sets, at 43.4 frames per second; size not recorded",
         "Weaker than CNNs on fine pixel level edges"],
        ["MobileViT v2 (Mehta and Rastegari, TMLR 2023)",
         "Separable self attention whose cost is linear in the number of tokens, inside a convolutional network",
         "ImageNet and downstream tasks",
         "75.6% ImageNet top 1 at about 3 M parameters, 3.2 times faster than MobileViT on a phone",
         "A classification backbone. Nothing about roads or connectivity"],
    ], [1.05, 1.5, 1.1, 1.6, 1.02])
    d.para("**Two things this table settles.** First, on the dataset we actually use, published road class IoU "
           "sits between roughly 58% and 70%: TopoRF-Net reports U-Net at 61.66% and D-LinkNet at 58.09% on "
           "DeepGlobe, against its own 69.76%. Second, a model of our size on that dataset is not unusual. Road "
           "MobileSeg Tiny reaches 71.52% mIoU with 1.41 M parameters, slightly smaller than ours. Being small "
           "is therefore not by itself a result; accuracy at that size would be, and that is what we have not "
           "yet measured.")
    d.para("**Two cautions about comparing numbers.** Road MobileSeg reports mIoU, the mean over road and "
           "background, which is always far higher than road only IoU because background is easy and dominant. "
           "TopoRF-Net reports road class IoU. The two cannot be read side by side. Parameter counts also vary "
           "by variant and by how they are counted: TopoRF-Net cites D-LinkNet at 218 M, while our own "
           "measurement of D-LinkNet34 gives 31.10 M. This is why we trust only our own measurements for cost.")
    d.para("**One correction to our review.** Our literature table lists MViT-PCD as a network for surface "
           "topographic detection by Liu et al. The paper is MViT-PCD: A Lightweight ViT-Based Network for "
           "Martian Surface Topographic Change Detection, in IEEE Geoscience and Remote Sensing Letters. It is "
           "a change detection paper on Mars imagery, not road extraction. It supports the idea that a mobile "
           "ViT backbone works on remote sensing imagery, but it is not road extraction evidence.")

    d.h("3. What each paper contributed to our design")
    d.para("The review pointed at one gap: the literature solves either small size or connectivity, rarely both "
           "at once. Our system attempts both, and each part traces to a specific paper.", after=4)
    d.table([
        ["From", "We took", "Where it lives in our code"],
        ["MobileViT v2, separable self attention",
         "A backbone giving global context at a cost linear in image size, so a whole 1024² tile can be "
         "processed at once on a CPU. Road pixels under a tree are only recoverable from context far along the "
         "road, which is why we did not use a pure CNN",
         "backend/src/models/mobilevit_v2.py, the MobileViT v2 blocks and the linear attention"],
        ["clDice",
         "The loss term that scores skeletons instead of masks, so a break in a road is punished even though it "
         "costs few pixels",
         "SoftClDiceLoss in the training notebook and backend/src/utils/loss.py"],
        ["TopoRF-Net, SAC loss",
         "The principle that connectivity must be trained for, not only cleaned up afterwards, and that the "
         "connectivity term should gain weight as training proceeds",
         "The alpha schedule, which moves weight from cross entropy to clDice over the first 40 epochs"],
        ["Road MobileSeg",
         "The target size: a road model of roughly 1 to 5 M parameters is a sensible goal on DeepGlobe",
         "Width and depth of our encoder, 1,604,657 parameters"],
        ["MViT-PCD",
         "Supporting evidence that a mobile ViT backbone behaves well on remote sensing imagery",
         "Choice of backbone family"],
        ["GLTDNet",
         "The warning that models trained in one region transfer badly to another, so any DeepGlobe result is an "
         "upper bound for Indian rural imagery",
         "Stated as a limit; not yet addressed in code"],
        ["D-LinkNet and the DeepGlobe challenge",
         "The dataset itself, and the reference point for what a strong trained model achieves on it",
         "data/samples, and the cost comparison in figures/real/measurements/efficiency.json"],
    ], [1.5, 2.85, 1.92])
    d.para("Two parts of our system do not come from the reviewed papers, and should be described as our own "
           "engineering rather than as literature backed:")
    d.bullets([
        "**Strip convolutions and channel shift.** Factorising a 3×3 convolution into 1×3 and 3×1 is a "
        "directional prior for long thin shapes, and shifting a quarter of the channels by two pixels widens "
        "the receptive field for free. Both are reasonable and cheap, but without an ablation we cannot say "
        "what either contributes.",
        "**Canopy gap bridging.** The literature closes gaps inside the loss. We also close them geometrically "
        "after inference, by joining road ends that face each other. This is the part of the pipeline with the "
        "least support in the papers, and the part most likely to invent roads that are not there.",
    ])

    d.h("4. Our system and what we measured")
    d.para(f"**Model.** A MobileViT v2 encoder and decoder, {P['params']:,} parameters. The encoder is a strip "
           "convolution stem, then three stages alternating MobileNetV2 inverted residual blocks with MobileViT "
           "v2 blocks, reducing the tile by 16 times, then a bottleneck of three transformer layers. The decoder "
           "upsamples back to full size and merges each skip connection through an attention gate. The head is a "
           "1×1 convolution, with the sigmoid baked into the exported model.")
    d.para("**Data.** DeepGlobe Road Extraction: 6,226 labelled RGB tiles, 1024×1024 at 0.5 m, over Thailand, "
           "Indonesia and India. The official test masks were never released, so we hold out 10% with seed 42: "
           "5,604 training and 622 validation tiles. Training samples are random 256² crops at native resolution.")
    d.para("**Training.** AdamW at 1×10⁻³, weight decay 1×10⁻⁴, cosine schedule to 1×10⁻⁵, batch of 16, gradient "
           "clipping at norm 1.0, mixed precision on a Kaggle GPU. The loss is α(e)·BCE + 0.35·Dice + "
           "(0.65 − α(e))·clDice, with α falling from 0.50 to 0.15 by epoch 40 and road pixels weighted twice in "
           "BCE. The saved model is the one with the best validation IoU, with any epoch predicting more than "
           "20% road rejected.")
    d.para(f"**Cost, all measured** on one laptop CPU ({CPU}, batch 1, one 1024² tile, median of 3 to 7 runs):",
           after=4)
    rows = [["Model", "Parameters", "GFLOPs", "Weights", "CPU latency"]]
    for key, label in (("Proposed", "Ours (PyTorch)"), (None, "Ours (ONNX Runtime)"),
                       ("LR-ASPP-MBv3", "LR-ASPP MobileNetV3"), ("DeepLabV3-MBv3", "DeepLabV3 MobileNetV3"),
                       ("D-LinkNet34", "D-LinkNet34"), ("U-Net", "U-Net")):
        if key is None:
            rows.append([label, f"{P['params'] / 1e6:.2f} M", f"{P['gflops']:,.1f}",
                         f"{ONNX['file_mb_total']:.1f} MB", f"{ONNX['latency_ms'] / 1000:.2f} s"])
            continue
        m = M[key]
        note = "†" if "latency_note" in m else ""
        rows.append([label, f"{m['params'] / 1e6:.2f} M", f"{m['gflops']:,.1f}",
                     f"{m['weights_mb_fp32']:.1f} MB", f"{m['latency_ms'] / 1000:.2f} s{note}"])
    d.table(rows, [1.85, 1.15, 0.95, 1.07, 1.25], size=9.5)
    d.para(f"† U-Net timed as four 512² crops, because a full tile exceeds this machine's free memory. Against "
           f"U-Net that is {UNET['params'] / P['params']:.1f} times fewer parameters, "
           f"{UNET['gflops'] / P['gflops']:.0f} times fewer operations and "
           f"{UNET['latency_ms'] / P['latency_ms']:.1f} times faster. Against LR-ASPP we are larger and slower, "
           "so “smallest” is not a claim we can make. The reference models were never trained here; these are "
           "cost numbers only.", size=10)
    d.para(f"**Accuracy, measured but weak.** The only accuracy figure we have is from validation during "
           f"training, on 622 tiles resized to 256²: IoU {pct(F['val_iou'], 2)}, precision "
           f"{pct(F['val_precision'], 2)}, road pixels {pct(F['val_positive_frac'], 2)}, soft clDice loss "
           f"{F['val_cldice']:.3f}, at epoch {F['epoch']}. Section 7 explains why that protocol understates the "
           "model, but even allowing for it, the figure sits far below the 58% to 70% road IoU that published "
           "models report on this dataset. Until the evaluation notebook is run at full resolution, the honest "
           "statement is that our accuracy is unproven and probably well behind the literature.")
    d.para(f"**Deployment, measured.** Exported to ONNX with dynamic height and width. PyTorch and ONNX Runtime "
           f"outputs differ by at most {PARITY['max_abs_diff'] * 1e7:.1f}×10⁻⁷ across 256², 256×384 and 512² "
           f"inputs, so the exported model is the same model. The deployable payload is "
           f"{ONNX['file_mb_total']:.1f} MB, and inference needs only ONNX Runtime, OpenCV, NumPy, SciPy and "
           "scikit-image.")

    d.h("5. What our predictions look like")
    d.para("We have no ground truth for the four sample tiles, so these are observations of behaviour, not "
           "accuracy. All four run through the full deployed pipeline: four flip test time augmentation, "
           "hysteresis thresholding at 0.35 and 0.12, a 5×5 closing, then canopy gap bridging.", after=4)
    d.table([
        ["Tile", "Scene", "Road predicted", "Bridged pixels", "What it looks like"],
        ["100034", "Forest track through dense canopy", pct(OVER["100034"]["road_frac"], 2),
         f"{OVER['100034']['bridged_px']:,}", "One continuous track down the tile, plus a fainter branch"],
        ["117991", "Village with fields and trees", pct(OVER["117991"]["road_frac"], 2),
         f"{OVER['117991']['bridged_px']:,}",
         f"A connected grid of lanes; bridging joins {PS['components_before']} pieces into "
         f"{PS['components_after']}"],
        ["115714", "Farmland, single road with a bend", pct(OVER["115714"]["road_frac"], 2),
         f"{OVER['115714']['bridged_px']:,}",
         "One clean continuous line; a short spur follows a field boundary and is probably false"],
        ["102408", "Dense settlement", pct(OVER["102408"]["road_frac"], 2),
         f"{OVER['102408']['bridged_px']:,}", "A full street network, blocks closed rather than broken"],
    ], [0.72, 1.55, 0.95, 0.85, 2.2], size=9.5)
    d.para("The masks are thin, roughly one road wide, and mostly continuous. That is the behaviour the clDice "
           "term is meant to produce, and it differs visibly from the thick blobs a pixel only loss tends to give.")
    d.para("**Failure modes visible in these same tiles.**", after=3)
    d.bullets([
        "**Invented links.** The spur on 115714 follows a field boundary. Bridging draws a straight 6 pixel line "
        "between road ends that face each other, and a field edge that lines up with a road end will be joined.",
        f"**Bridging does the heavy lifting on cluttered tiles.** On 117991 nearly "
        f"{OVER['117991']['bridged_px']:,} pixels are drawn by the geometric step rather than predicted by the "
        "network. A fair evaluation must report results with bridging on and off.",
        f"**Settlement tiles are dense.** At {pct(OVER['102408']['road_frac'], 2)} road, tile 102408 is near the "
        "kind of over prediction the 20% training gate exists to catch. It looks correct here, but the model is "
        "not conservative on built up areas.",
    ])
    d.para(f"For comparison, the collapsed checkpoint from the failed run marked "
           f"{pct(COL['collapsed_all_tiles_pos_frac'])} of pixels as road across the same four tiles, against "
           f"{pct(COL['final_all_tiles_pos_frac'])} for the current model. That is what a broken model looks "
           "like, and it is why the gate exists.")

    d.h("6. What we optimised, and why")
    d.para("Each row states the choice, the reason, and what evidence we have that it works. The last column is "
           "the honest part: for several choices the answer is none yet.", after=4)
    d.table([
        ["Choice", "Why", "Evidence we have"],
        ["MobileViT v2 backbone rather than a CNN or a full ViT",
         "Roads under canopy need context from far along the road. Full self attention costs the square of the "
         "number of patches, impossible on a 1024² tile on a CPU; separable attention is linear",
         f"Cost measured: {P['gflops']:.1f} GFLOPs and {ONNX['latency_ms'] / 1000:.2f} s per tile. No accuracy "
         "comparison against a CNN of the same size"],
        ["clDice in the loss, weight rising over training",
         "A pixel loss treats a break as a few wrong pixels. clDice scores skeletons, so a break costs much "
         "more. Starting with pixel losses avoids optimising topology before the model knows where roads are",
         "Predictions are thin and mostly continuous. No ablation against BCE and Dice alone"],
        ["Road class weight 2, not 3",
         "At weight 3 a freshly initialised network reduced its loss fastest by marking everything as road",
         f"The collapsed run at weight 3 marked {pct(COL['collapsed_all_tiles_pos_frac'])} of pixels road; at "
         f"weight 2 with a warm start the model sits at {pct(F['val_positive_frac'])}"],
        ["Warm start from trained weights",
         "A random start plus a positive class weight is what produced the collapse",
         "Verified from the weights: our model correlates 0.8 to 0.96 with the baseline in encoder and decoder, "
         "the collapsed run 0.00"],
        ["Select checkpoints by IoU, with a 20% road gate",
         "The soft clDice validation loss rewards a mask covering everything, because its sensitivity term "
         "saturates. It cannot be used to choose a model",
         f"The collapsed epoch scored 0.215 on that loss, better looking than our final {F['val_cldice']:.3f}, "
         "and was saved as best"],
        ["Hysteresis thresholding at 0.35 and 0.12",
         "A single threshold either loses faint road under canopy or floods the image. Hysteresis keeps weak "
         "pixels only where they touch confident road",
         "Qualitative only: faint roads survive in the sample tiles"],
        ["Canopy gap bridging",
         "Trees break a road into pieces; the audit use case needs a connected network",
         f"On tile {PS['tile']}, {PS['components_before']} pieces become {PS['components_after']}, with "
         f"{PS['bridged_pixels']:,} pixels added. No ground truth check that the additions are real"],
        ["Four flip test time augmentation",
         "A real road looks like a road under any flip; a one off artefact usually does not. Four forward "
         "passes, no retraining",
         "Cost measured. Effect on accuracy not measured"],
        ["Export to ONNX",
         "The field target has no GPU and often no PyTorch. ONNX Runtime is also faster on this CPU",
         f"{ONNX['latency_ms'] / 1000:.2f} s against {P['latency_ms'] / 1000:.2f} s in PyTorch, and outputs "
         f"agree to {PARITY['max_abs_diff'] * 1e7:.1f}×10⁻⁷"],
        ["Validate at full resolution (fixed after this model)",
         "Roads four times thinner than in training are barely detected, so the metric measured the wrong thing",
         "On 2 of 3 tiles the model found no road at all after resizing, against 1.6% and 1.1% at full resolution"],
    ], [1.5, 2.5, 2.27])
    d.para("**What we did not optimise.** We never tuned the bridging parameters (220 pixels, 65 degrees, 6 "
           "pixel width) against ground truth; they were set by eye on sample tiles. We never ran the ablation "
           "that would separate the contributions of strip convolutions, channel shift, attention gates and "
           "clDice. Both are the obvious next pieces of work.")

    d.h("7. What we learned")
    d.para("The most useful lessons came from things that went wrong, and all four were found by looking at "
           "predictions rather than at logged numbers.")
    d.para("**A metric can reward a broken model.** Training collapsed to marking nearly every pixel as road, "
           "and the soft clDice validation loss scored that better than the working model, because its "
           "sensitivity term saturates when the prediction covers the image. The collapsed epoch was duly saved "
           "as the best one. A loss used for training is not automatically safe for choosing a model, and any "
           "selection metric needs a sanity gate. Ours rejects epochs predicting more than 20% road.")
    d.para("**Validation ran at the wrong scale.** Tiles were resized from 1024² to 256² for validation, making "
           "roads four times thinner than in the training crops. On 2 of 3 sample tiles the model then found no "
           "road at all, and scored accordingly. Every training time number we have comes from that protocol. "
           "Validate at the resolution you deploy at, or the number measures the protocol rather than the model.")
    d.para("**An augmentation that never ran.** The canopy shadow augmentation, the one aimed squarely at our "
           "core problem, was disabled the whole time by a constructor incompatibility with the installed "
           "Albumentations version. It failed silently. An augmentation that is not asserted in a test is not "
           "running.")
    d.para("**Post processing can invent data.** Gap bridging originally drew straight roads along tile borders, "
           "joining road ends that simply left the tile. We now ignore ends within 16 pixels of the border. A "
           "step that adds pixels needs a rule for where it must not.")
    d.para("**Beyond the bugs.** Reading a paper is not the same as reproducing it, so cited numbers must be "
           "labelled as cited. A comparison is only fair if the protocol matches, which is why the mIoU and road "
           "IoU caution in Section 2 matters. And measuring cost is cheap and conclusive while measuring "
           "accuracy is neither, which is why we have a thorough cost story and a thin accuracy story.")

    d.h("8. Our advantages")
    d.para("Each of these is backed by a measurement or by code that runs.", after=4)
    d.bullets([
        f"**It runs where the work happens.** {ONNX['latency_ms'] / 1000:.2f} s per 1024² tile on an ordinary "
        f"laptop CPU, a {ONNX['file_mb_total']:.1f} MB payload, and no deep learning framework needed. "
        "TopoRF-Net at 56 M parameters is not deployable this way, and most connectivity aware papers report no "
        "speed at all.",
        f"**Connectivity is handled twice.** In training through clDice, and after inference through gap "
        f"bridging. The reviewed papers do one or the other. On tile {PS['tile']} the second stage alone turns "
        f"{PS['components_before']} road pieces into {PS['components_after']}.",
        f"**The export is verified, not assumed.** PyTorch and ONNX agree to "
        f"{PARITY['max_abs_diff'] * 1e7:.1f}×10⁻⁷ on square and rectangular inputs, so the deployed model is "
        "provably the trained model. Papers rarely report this check and teams often skip it.",
        "**Full tiles, not resized ones.** The model runs at native 0.5 m resolution, which matters for roads a "
        "few pixels wide. We learned the hard way what happens when you do not.",
        "**The failure modes are known and written down.** We can say where the model over predicts, where "
        "bridging invents links, and which numbers are untrustworthy. That is worth more than an unexamined "
        "headline metric.",
        "**Everything is reproducible.** Every number in our report and in this document comes from a script and "
        "a stored measurement file, not from notes.",
    ])

    d.h("9. Our disadvantages")
    d.para("Stated plainly, because a validation document that lists only strengths is not validation.", after=4)
    d.bullets([
        f"**Accuracy is unproven, and probably behind the field.** Our single figure, {pct(F['val_iou'], 2)} IoU, "
        "comes from a protocol we have shown to be wrong, and the papers report 58% to 70% road IoU on the same "
        "dataset. Even allowing for the protocol, we should expect to be behind until measured otherwise.",
        f"**Being small is not novel.** Road MobileSeg Tiny is 1.41 M parameters, smaller than our "
        f"{P['params'] / 1e6:.2f} M, on the same dataset, with 71.52% mIoU reported. Our contribution has to be "
        "connectivity at that size, which is precisely the untested part.",
        f"**We are not the cheapest option.** LR-ASPP with MobileNetV3 is {LR['gflops']:.1f} GFLOPs and "
        f"{LR['latency_ms'] / 1000:.2f} s against our {P['gflops']:.1f} and {ONNX['latency_ms'] / 1000:.2f}. If "
        "only speed mattered, that would be the model to beat, and we have not shown we beat it on anything.",
        "**No ablation.** Strip convolutions, channel shift, attention gates and clDice are all in the model at "
        "once. We cannot attribute any behaviour to any one of them.",
        "**Bridging is unvalidated and can invent roads.** It adds thousands of pixels per tile with no ground "
        "truth check, and the sample tiles already show one false link along a field boundary.",
        "**The current model carries known defects.** It was trained with the canopy augmentation disabled and "
        "selected at the wrong validation scale. It is the best we have, not the best the design can do.",
        "**Not tested on the target imagery.** Everything is DeepGlobe, which covers Thailand, Indonesia and "
        "India. We have not evaluated on Indian rural imagery specifically, and GLTDNet is a clear warning that "
        "domain shift is real.",
        "**No graph metrics yet.** APLS and TOPO code exists in the repository with unit tests, but has never "
        "been run on this model, so our connectivity claims rest on component counts and on looking at pictures.",
        "**Weak supervision was dropped.** OpenStreetMap centreline training was in the plan and is not in the "
        "final system, so the annotation cost argument from the proposal is currently unsupported.",
    ])

    d.h("10. Open gaps and what would close them")
    d.para("In rough order of how much each would strengthen the work.", after=4)
    d.bullets([
        "**Run the evaluation notebook.** It is written, it computes road IoU, precision, recall, F1, relaxed F1 "
        "at 3 pixels, clDice, component counts, recall under canopy and bootstrap intervals, and it has never "
        "been run. This single step turns our accuracy story from unproven into measured, whichever way it falls.",
        "**Retrain with full resolution validation.** The current model was selected at the wrong scale. "
        "Retraining with the fixed validation, and with the canopy augmentation actually running, is the most "
        "likely source of a real accuracy gain.",
        "**Report bridging on and off.** Because bridging adds thousands of pixels, every metric should be "
        "reported both ways. Otherwise we cannot tell whether the network or the geometry is doing the work.",
        "**Run APLS and TOPO.** The code and tests exist. Graph metrics are how the literature judges "
        "connectivity, and without them our connectivity claim is qualitative.",
        "**Train one baseline ourselves.** LR-ASPP on our split, with our protocol, would give the only fair "
        "accuracy comparison we can afford. Cited numbers from papers cannot fill this gap.",
        "**Ablate the four novelties.** One run each with strip convolutions, channel shift, attention gates and "
        "clDice removed, so contributions can be attributed.",
        "**Test on Indian rural imagery.** The stated goal is PMGSY style auditing. Until we test there, domain "
        "shift is an open risk, and GLTDNet suggests it is a large one.",
    ], numbered=True)
    d.para("**Where this leaves the project.** The engineering is sound and verified: a small model that runs on "
           "a CPU, exports faithfully, and produces thin connected roads on real tiles. The science is "
           "incomplete: we have not shown that it is accurate, and we have not isolated which of our ideas "
           "matters. Items 1 to 3 are days of work, not months, and they are what would let us make claims "
           "rather than observations.")

    d.h("Sources for cited figures")
    d.bullets([
        "Road MobileSeg: Sensors 24(2):531, 2024. doi.org/10.3390/s24020531",
        "TopoRF-Net: Sensors 25(24):7428, 2025. doi.org/10.3390/s25247428",
        "clDice: Shit et al., CVPR 2021 (openaccess.thecvf.com), arXiv:2003.07311",
        "SAC loss: Shojaei et al., WACV Workshops 2025 (IEEE Xplore document 10972635)",
        "GLTDNet: Wang et al., IEEE Xplore document 11049897, 2025",
        "MViT-PCD: IEEE Geoscience and Remote Sensing Letters, document 10007802, 2023",
        "MobileViT v2: Mehta and Rastegari, arXiv:2206.02680, TMLR 2023",
    ], size=10)
    d.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
