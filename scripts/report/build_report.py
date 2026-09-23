"""
Fills the IEEE CS Bangalore Chapter internship report template
(docs/report/Internship Report Template.docx) and writes docs/report/Internship_Report.docx.

The template's cover page, logos, page borders, fonts and contents list are kept; only the
placeholders are filled and the report body is appended in the template's own typography
(Times New Roman, 12 pt body, 14 pt bold headings). Team, mentor and project ID come from the
team's progress report (docs/Progress Report - IAMPro-2026.docx).

Every number is read from measured data, never typed in:
  models/*.pth                                       checkpoint metadata
  figures/real/measurements/efficiency.json          scripts/paper_figures/benchmark_efficiency.py
  figures/real/measurements/local_figure_stats.json  scripts/paper_figures/local_figures.py
  figures/real/deck/overlay_stats.json               scripts/presentation/deck_assets.py
  figures/real/measurements/onnx_parity.json         scripts/report/measure_onnx_parity.py

Usage (repo root):
  python scripts/report/report_figures.py
  python scripts/report/build_report.py
"""

import copy
import json
import os

import torch
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG = os.path.join(REPO, "figures", "real")
TEMPLATE = os.path.join(REPO, "docs", "report", "Internship Report Template.docx")
OUT = os.environ.get("REPORT_OUT", os.path.join(REPO, "docs", "report", "Internship_Report.docx"))

FONT = "Times New Roman"
TEXT_W = 6.27                       # A4 minus 1-inch margins (in)

# ── cover details (from docs/Progress Report - IAMPro-2026.docx) ────────────
PROGRAM = "Internship and Mentorship Program 2026"
PROJECT = "P29: “Lightweight MobileViT-Graph Network for Topological Rural Road Extraction from Satellite Imagery”"
COLLEGE = "R.V. College of Engineering, Bengaluru"
STUDENTS = ["Dilraj Singh", "Arya Wadhwa", "Ishaan Snehal Parikh", "Hrithik Bhattacharya"]
MENTOR = "Dr Basavaraj Patil"
AFFILIATION = ["Assistant Professor, School of Computer Science and Engineering", "R V University (RVU)"]


# ── measured data ───────────────────────────────────────────────────────────
def load_json(path):
    return json.load(open(path))


EFF = load_json(os.path.join(FIG, "measurements", "efficiency.json"))
LOCAL = load_json(os.path.join(FIG, "measurements", "local_figure_stats.json"))
OVER = load_json(os.path.join(FIG, "deck", "overlay_stats.json"))
PARITY = load_json(os.path.join(FIG, "measurements", "onnx_parity.json"))
M = EFF["models"]
P, UNET, ONNX = M["Proposed"], M["U-Net"], EFF["onnx"]
CPU = EFF["cpu"].replace("Intel(R) Core(TM) ", "Intel Core ").split(" CPU")[0]


def ckpt_meta(rel):
    ck = torch.load(os.path.join(REPO, rel), map_location="cpu")
    return {k: v for k, v in ck.items() if k not in ("model_state_dict", "optimizer_state_dict")}


CK = {
    "baseline": ckpt_meta("models/best_model_new.pth"),
    "collapsed": ckpt_meta("models/archive/best_model_v2_collapsed_epoch46.pth"),
    "run1": ckpt_meta("models/archive/best_model_v2_epoch18_iou0159.pth"),
    "final": ckpt_meta("models/best_model_v2.pth"),
}
F = CK["final"]


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# ── low-level document helpers ──────────────────────────────────────────────
class Report:
    def __init__(self, doc):
        self.doc = doc
        self.fig_n = 0
        self.tab_n = 0
        self.bullet_num = add_bullet_numbering(doc)
        self.h1 = ensure_heading_style(doc, "Heading 1", 14, 1, before=12, after=6)
        self.h2 = ensure_heading_style(doc, "Heading 2", 12, 2, before=9, after=3)

    # text -------------------------------------------------------------------
    def _runs(self, p, content, size=12, bold=False, italic=False):
        runs = content if isinstance(content, list) else [(content, {})]
        for t, o in runs:
            r = p.add_run(t)
            f = r.font
            f.name = FONT
            r._r.get_or_add_rPr().get_or_add_rFonts().set(qn("w:cs"), FONT)
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic)
            f.subscript = o.get("sub", None)
            f.superscript = o.get("sup", None)
        return p

    def para(self, content, size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=6, before=0, bold=False,
             italic=False, keep_next=False, indent=None):
        p = self.doc.add_paragraph()
        pf = p.paragraph_format
        pf.alignment = align
        pf.space_after = Pt(after)
        pf.space_before = Pt(before)
        pf.line_spacing = 1.15
        pf.keep_with_next = keep_next
        if indent is not None:
            pf.left_indent = Inches(indent)
        return self._runs(p, content, size, bold, italic)

    def heading(self, text, level=1):
        p = self.doc.add_paragraph(style=self.h1 if level == 1 else self.h2)
        p.paragraph_format.keep_with_next = True
        return self._runs(p, text, 14 if level == 1 else 12, bold=True)

    def bullets(self, items, size=12, after=3):
        for i, item in enumerate(items):
            p = self.doc.add_paragraph(style="List Paragraph")
            pPr = p._p.get_or_add_pPr()
            numPr = OxmlElement("w:numPr")
            for tag, val in (("w:ilvl", "0"), ("w:numId", str(self.bullet_num))):
                el = OxmlElement(tag); el.set(qn("w:val"), val); numPr.append(el)
            pPr.insert(0, numPr)
            pf = p.paragraph_format
            pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            pf.space_after = Pt(after if i < len(items) - 1 else 6)
            pf.line_spacing = 1.15
            self._runs(p, item, size)

    def equation(self, content, number):
        p = self.doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_before, pf.space_after = Pt(3), Pt(6)
        pf.keep_with_next = True                   # keep an equation with the text that follows
        pf.tab_stops.add_tab_stop(Inches(TEXT_W / 2), WD_TAB_ALIGNMENT.CENTER)
        pf.tab_stops.add_tab_stop(Inches(TEXT_W), WD_TAB_ALIGNMENT.RIGHT)
        runs = [("\t", {})] + (content if isinstance(content, list) else [(content, {})]) + [(f"\t({number})", {})]
        return self._runs(p, runs, 12, italic=False)

    # floats -----------------------------------------------------------------
    def figure(self, path, width, caption):
        self.fig_n += 1
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        p.add_run().add_picture(path, width=Inches(width))
        self.para([(f"Figure {self.fig_n}: ", {"bold": True}), (caption, {})], size=11,
                  align=WD_ALIGN_PARAGRAPH.CENTER, after=10)
        return self.fig_n

    def table(self, rows, widths, caption, size=10.5, align_right_from=None, bold_last=False):
        self.tab_n += 1
        self.para([(f"Table {self.tab_n}: ", {"bold": True}), (caption, {})], size=11,
                  align=WD_ALIGN_PARAGRAPH.CENTER, after=3, before=4, keep_next=True)
        t = self.doc.add_table(rows=len(rows), cols=len(rows[0]))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        for i, row in enumerate(rows):
            tr = t.rows[i]._tr
            trPr = tr.get_or_add_trPr()
            cant = OxmlElement("w:cantSplit"); trPr.append(cant)
            if i == 0:
                hdr = OxmlElement("w:tblHeader"); trPr.append(hdr)
            for j, val in enumerate(row):
                cell = t.cell(i, j)
                cell.width = Inches(widths[j])
                p = cell.paragraphs[0]
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.keep_with_next = i < len(rows) - 1
                right = align_right_from is not None and j >= align_right_from and i > 0
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if right else (
                    WD_ALIGN_PARAGRAPH.CENTER if (i == 0 and align_right_from is not None and j >= align_right_from)
                    else WD_ALIGN_PARAGRAPH.LEFT)
                self._runs(p, str(val), size, bold=(i == 0) or (bold_last and i == len(rows) - 1))
        # fixed column widths (Word honours tblGrid + tcW)
        grid = t._tbl.tblGrid
        for j, gc in enumerate(grid.findall(qn("w:gridCol"))):
            gc.set(qn("w:w"), str(int(widths[j] * 1440)))
        tblPr = t._tbl.tblPr
        layout = OxmlElement("w:tblLayout"); layout.set(qn("w:type"), "fixed"); tblPr.append(layout)
        cm = OxmlElement("w:tblCellMar")
        for side, v in (("top", 30), ("bottom", 30), ("left", 90), ("right", 90)):
            e = OxmlElement(f"w:{side}"); e.set(qn("w:w"), str(v)); e.set(qn("w:type"), "dxa"); cm.append(e)
        tblPr.append(cm)
        self.para("", size=6, after=4)
        return self.tab_n

    def page_break(self):
        p = self.doc.add_paragraph()
        p.add_run().add_break(WD_BREAK.PAGE)


def ensure_heading_style(doc, name, size, level, before, after):
    styles = doc.styles
    try:
        st = styles[name]
    except KeyError:
        st = styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    st.base_style = styles["Normal"]
    st.quick_style = True
    f = st.font
    f.name, f.size, f.bold, f.italic = FONT, Pt(size), True, False
    f.color.rgb = RGBColor(0, 0, 0)
    st.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:cs"), FONT)
    pf = st.paragraph_format
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.keep_with_next = True
    pPr = st.element.get_or_add_pPr()
    ol = pPr.find(qn("w:outlineLvl"))
    if ol is None:
        ol = OxmlElement("w:outlineLvl"); pPr.append(ol)
    ol.set(qn("w:val"), str(level - 1))
    return st


def add_bullet_numbering(doc):
    """Adds a plain bullet list definition to the template's numbering part; returns its numId."""
    numbering = doc.part.numbering_part.element
    abs_ids = [int(a.get(qn("w:abstractNumId"))) for a in numbering.findall(qn("w:abstractNum"))]
    num_ids = [int(n.get(qn("w:numId"))) for n in numbering.findall(qn("w:num"))]
    aid, nid = max(abs_ids) + 1, max(num_ids) + 1
    a = OxmlElement("w:abstractNum"); a.set(qn("w:abstractNumId"), str(aid))
    mlt = OxmlElement("w:multiLevelType"); mlt.set(qn("w:val"), "hybridMultilevel"); a.append(mlt)
    lvl = OxmlElement("w:lvl"); lvl.set(qn("w:ilvl"), "0")
    for tag, val in (("w:start", "1"), ("w:numFmt", "bullet"), ("w:lvlText", "•"), ("w:lvlJc", "left")):
        e = OxmlElement(tag); e.set(qn("w:val"), val); lvl.append(e)
    pPr = OxmlElement("w:pPr"); ind = OxmlElement("w:ind")
    ind.set(qn("w:left"), "360"); ind.set(qn("w:hanging"), "360"); pPr.append(ind); lvl.append(pPr)
    rPr = OxmlElement("w:rPr"); rf = OxmlElement("w:rFonts")
    for k in ("w:ascii", "w:hAnsi", "w:cs"):
        rf.set(qn(k), FONT)
    rPr.append(rf); lvl.append(rPr)
    a.append(lvl)
    first_num = numbering.find(qn("w:num"))
    first_num.addprevious(a)                       # every abstractNum precedes every num
    n = OxmlElement("w:num"); n.set(qn("w:numId"), str(nid))
    ref = OxmlElement("w:abstractNumId"); ref.set(qn("w:val"), str(aid)); n.append(ref)
    numbering.append(n)
    return nid


def set_text(paragraph, text):
    """Replaces a paragraph's text, keeping the formatting of its first run."""
    runs = paragraph.runs
    runs[0].text = text
    for r in runs[1:]:
        r._r.getparent().remove(r._r)


def delete(paragraph):
    paragraph._p.getparent().remove(paragraph._p)


# ── cover and contents (template pages 1-2) ─────────────────────────────────
def fill_template(doc):
    ps = doc.paragraphs
    set_text(ps[2], PROGRAM)
    set_text(ps[7], PROJECT)

    # NAME / COLLEGE rows: same paragraphs and run formatting, with one tab stop so the two
    # columns line up (the template's default tab stops only work for equal-length placeholders).
    name_w = 2.45
    left = (TEXT_W - name_w - 3.55) / 2
    for p, (name, college) in zip(ps[12:17], [("NAME", "COLLEGE")] + [(s, COLLEGE) for s in STUDENTS]):
        rpr = copy.deepcopy(p.runs[0]._r.rPr)
        for r in list(p.runs):
            r._r.getparent().remove(r._r)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.left_indent = Inches(left)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(left + name_w))
        r = p.add_run(f"{name}\t{college}")
        r._r.insert(0, rpr)

    set_text(ps[20], MENTOR)
    set_text(ps[21], AFFILIATION[0])
    ps[21].runs[0].add_break()
    ps[21].add_run(AFFILIATION[1])._r.insert(0, copy.deepcopy(ps[21].runs[0]._r.rPr))

    # Contents page: keep the template's title; replace its plain list and instructions with a
    # Word table of contents (sections and subsections, dot leaders, page numbers). The page
    # numbers are filled in by Word: scripts/report/export_pdf.ps1 updates the field and saves.
    # Item 9 of the template list (publication details) is omitted: the work is not published.
    for p in ps[27:]:
        delete(p)
    ps[26]._p.addnext(toc_field())
    add_toc_styles(doc)


def toc_field():
    p = OxmlElement("w:p")

    def run(child):
        r = OxmlElement("w:r")
        rpr = OxmlElement("w:rPr"); rf = OxmlElement("w:rFonts")
        for k in ("w:ascii", "w:hAnsi", "w:cs"):
            rf.set(qn(k), FONT)
        rpr.append(rf); r.append(rpr); r.append(child)
        p.append(r)

    begin = OxmlElement("w:fldChar"); begin.set(qn("w:fldCharType"), "begin"); begin.set(qn("w:dirty"), "true")
    run(begin)
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve")
    instr.text = ' TOC \\o "1-2" \\h \\z \\u '
    run(instr)
    sep = OxmlElement("w:fldChar"); sep.set(qn("w:fldCharType"), "separate")
    run(sep)
    t = OxmlElement("w:t"); t.text = "Open in Word and update fields to show the contents."
    run(t)
    end = OxmlElement("w:fldChar"); end.set(qn("w:fldCharType"), "end")
    run(end)
    return p


def add_toc_styles(doc):
    """Word's built-in contents styles, set to the report's typography. They must carry Word's own
    style IDs (TOC1, TOC2); under any other ID Word ignores them and regenerates its Calibri defaults.
    Spacing is compact so the whole contents list fits on the template's single contents page."""
    styles = doc.styles.element
    for sid, name, half_pts, bold, indent_tw, before_tw in (("TOC1", "toc 1", 23, True, 0, 90),
                                                            ("TOC2", "toc 2", 21, False, 360, 0)):
        for old in styles.findall(qn("w:style")):
            if old.get(qn("w:styleId")) == sid:
                styles.remove(old)
        xml = (
            f'<w:style xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
            f'w:type="paragraph" w:styleId="{sid}"><w:name w:val="{name}"/><w:basedOn w:val="Normal"/>'
            f'<w:next w:val="Normal"/><w:uiPriority w:val="39"/><w:unhideWhenUsed/>'
            f'<w:pPr><w:tabs><w:tab w:val="right" w:leader="dot" w:pos="{int(TEXT_W * 1440)}"/></w:tabs>'
            f'<w:spacing w:before="{before_tw}" w:after="0" w:line="288" w:lineRule="auto"/>'
            f'<w:ind w:left="{indent_tw}"/></w:pPr>'
            f'<w:rPr><w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}" w:cs="{FONT}"/>{"<w:b/>" if bold else ""}'
            f'<w:sz w:val="{half_pts}"/><w:szCs w:val="{half_pts}"/></w:rPr></w:style>'
        )
        styles.append(parse_xml(xml))


def start_numbered_section(doc):
    """Ends the cover and contents section here; the report body becomes a section of its own whose
    footer carries the page number, starting at 1 on the first page of the Introduction."""
    body_sect = doc.element.body.sectPr
    p = doc.add_paragraph()
    p._p.get_or_add_pPr().append(copy.deepcopy(body_sect))   # same page size, margins and border
    num = OxmlElement("w:pgNumType"); num.set(qn("w:start"), "1")
    body_sect.append(num)


def add_page_numbers(doc):
    sect = doc.sections[-1]
    sect.footer.is_linked_to_previous = False
    p = sect.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for kind, text in (("begin", None), (None, " PAGE "), ("end", None)):
        r = p.add_run()
        r.font.name, r.font.size = FONT, Pt(11)
        if kind:
            fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), kind); r._r.append(fc)
        else:
            it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = text; r._r.append(it)


# ── report body ─────────────────────────────────────────────────────────────
def body(rp):
    lr = M["LR-ASPP-MBv3"]
    ps = LOCAL["pipeline_stages"]
    col = LOCAL["collapse"]
    res = LOCAL["resolution"]
    n_zero = sum(1 for v in res.values() if v["resized_pos_frac"] == 0)

    start_numbered_section(rp.doc)

    # 1 ────────────────────────────────────────────────────────────────────
    # Background and problem statement follow the Problem Definition and Existing System sections
    # of the team's progress report (docs/Progress Report - IAMPro-2026.docx).
    rp.heading("1. Introduction")
    rp.heading("1.1 Background", 2)
    rp.para("In India, the Pradhan Mantri Gram Sadak Yojana (PMGSY) connects rural habitations with all weather "
            "roads, and the rural road networks built under it must be mapped and audited. Today this extraction "
            "and auditing relies mainly on manual GIS mapping, which is slow and labour intensive. High resolution "
            "satellite images, at about 0.5 m per pixel, cover large areas at regular intervals, so automatic road "
            "extraction from these images can make the auditing of rural roads faster and cheaper.")
    rp.heading("1.2 Problem Statement", 2)
    rp.para("Rural road extraction from satellite imagery is difficult. Rural roads are thin and irregular, are "
            "often covered by tree canopy or shadow, and change their appearance from one region to another. "
            "Manual auditing of these roads is slow and labour intensive, while models trained on urban highways "
            "suffer from domain shift when applied to rural areas. Standard training with BCE or IoU style losses "
            "optimises pixel overlap instead of road connectivity, so the output is fragmented and the road "
            "topology is broken. Models that try to repair connectivity are usually too large for edge devices.")
    rp.para("This project therefore aims to build an ultra lightweight road extraction system for Indian rural "
            "imagery that preserves road connectivity and is suitable for edge deployment.")
    rp.heading("1.3 Objectives", 2)
    rp.bullets([
        "Design a small encoder decoder segmentation network based on MobileViT v2 with fewer than two million "
        "parameters.",
        "Train it with a loss that rewards connected road centrelines (clDice) together with pixel level losses.",
        "Build a postprocessing step that reconnects road pieces separated by trees or shadow.",
        "Export the model so that it runs without a deep learning framework, and measure its cost against known "
        "models on the same hardware.",
        "Evaluate the system with a repeatable method and build an interactive demonstration.",
    ])

    # 2 ────────────────────────────────────────────────────────────────────
    rp.heading("2. Existing System")
    rp.heading("2.1 Manual Mapping", 2)
    rp.para("Rural road records are kept up to date through field surveys with GPS receivers, and by operators who "
            "trace roads over satellite or aerial images in GIS software. Both methods are accurate, but they need "
            "a lot of manual work, are slow to update, and are hard to scale to the length of road that national "
            "programmes must monitor.")
    rp.heading("2.2 Automated Road Extraction", 2)
    rp.para("Automatic methods treat road extraction as binary semantic segmentation: every pixel is labelled as "
            "road or background. Convolutional encoder decoder networks such as U-Net [1] and D-LinkNet [2] give "
            "good results on clearly visible roads; D-LinkNet won the road track of the DeepGlobe 2018 challenge "
            "[3]. These networks are large, and they are trained with pixel level losses that do not reward "
            "connectivity. Mobile networks such as MobileNetV3 with an LR-ASPP head [4], DeepLabV3 [5] and Road "
            "MobileSeg [13] cost much less, but they have no built in preference for thin, connected shapes. "
            "Vision Transformers [6] capture global context, but the cost of their self attention grows with the "
            "square of the number of image patches. MobileViT [7] and MobileViT v2 [8] reduce this cost by "
            "combining convolutions with a cheaper form of attention, and MViT-PCD [17] applies a small "
            "transformer to remote sensing images. For connectivity, the clDice loss [9] measures overlap on road "
            "centrelines, TopoRF-Net [14] and the structure aware connectivity (SAC) loss [15] penalise broken "
            "segments, and RoadTracer [10] and Sat2Graph [11] predict the road graph directly. Table 1 summarises "
            "these methods.")
    rp.table([
        ["Work", "Approach", "Limitation for rural roads on edge devices"],
        ["U-Net [1]", "Encoder and decoder with skip connections", "Large model; pixel level loss ignores connectivity"],
        ["D-LinkNet [2]", "ResNet-34 encoder with a dilated centre block", "Large model; no connectivity objective"],
        ["LR-ASPP, DeepLabV3 [4], [5]", "Segmentation heads for mobile networks",
         "No preference for thin, connected shapes"],
        ["Road MobileSeg [13]", "Mobile transformer backbone with coordinate attention",
         "No connectivity learning; roads come out broken"],
        ["MobileViT, MobileViT v2 [7], [8]", "Convolutions with low cost self attention",
         "General backbone, not designed for roads"],
        ["MViT-PCD [17]", "Small vision transformer for remote sensing", "Weaker on fine pixel level edges"],
        ["clDice [9]", "Overlap measured on centrelines", "Only a loss; needs a segmentation network"],
        ["TopoRF-Net [14]", "Multiple receptive field blocks and a continuity loss",
         "High computing cost; sensitive to noisy labels"],
        ["SAC loss [15]", "Penalty based on the distance between road parts",
         "Depends on image resolution; needs manual tuning"],
        ["GLTDNet [16]", "Global and local features for transfer across regions",
         "Needs much tuning for heavily covered roads"],
        ["RoadTracer [10], Sat2Graph [11]", "Predict the road graph directly",
         "Inference in many steps; hard to deploy"],
    ], [1.75, 2.2, 2.32], "Summary of existing road extraction methods", size=10)
    rp.heading("2.3 Limitations of the Existing System", 2)
    rp.bullets([
        "Manual mapping is accurate but too slow and costly for frequent audits of large areas.",
        "Accurate segmentation networks are too large for field hardware that has only a CPU.",
        "Pixel level losses do not penalise gaps, so roads under trees come out broken.",
        "Small models do not model road connectivity, while connectivity methods are costly or hard to deploy. "
        "Few methods handle both.",
    ])

    # 3 ────────────────────────────────────────────────────────────────────
    rp.heading("3. Proposed System")
    rp.para("The proposed system is a complete pipeline that turns a satellite tile into a connected road mask on a "
            "CPU. It has five parts: a small network that combines convolutions and attention, a training loss "
            "that rewards connected roads, a rule for choosing the saved model that rejects failed training runs, "
            "a postprocessing step that bridges gaps caused by trees, and a deployment path that does not need a "
            "deep learning framework. The main features are:")
    rp.bullets([
        [("Small network. ", {"bold": True}),
         (f"A MobileViT v2 encoder decoder with {P['params']:,} parameters, "
          f"{100 * (1 - P['params'] / UNET['params']):.1f}% fewer than U-Net ({UNET['params']:,}).", {})],
        [("Road shaped filters. ", {"bold": True}),
         ("Strip convolutions (1×3 followed by 3×1) respond to thin, long structures such as roads.", {})],
        [("Connectivity aware training. ", {"bold": True}),
         ("The loss adds clDice to binary cross entropy and Dice, and gives clDice more weight as training goes "
          "on.", {})],
        [("Safe model selection. ", {"bold": True}),
         ("The saved model is chosen by validation IoU, and any epoch that marks more than 20% of pixels as road "
          "is rejected. This stops a failed model from being saved as the best one.", {})],
        [("Canopy gap bridging. ", {"bold": True}),
         ("Open road ends that face each other across a short gap are joined, which reconnects roads broken by "
          "trees or shadow.", {})],
        [("Edge deployment. ", {"bold": True}),
         ("The network is exported to ONNX and runs with ONNX Runtime and OpenCV only. A Streamlit application "
          "is provided for demonstration.", {})],
    ])
    rp.para("Two parts of the original proposal were not included in the final system. Weak labels from "
            "OpenStreetMap were planned, but the final model was trained only on the DeepGlobe road masks. Graph "
            "metrics such as APLS were planned for evaluation; the evaluation notebook uses pixel, relaxed and "
            "centreline (clDice) metrics, and APLS is left for future work.")

    # 4 ────────────────────────────────────────────────────────────────────
    rp.heading("4. Knowledge Gained: Tools, Technology, Courses etc.")
    rp.heading("4.1 Tools and Technologies", 2)
    rp.table([
        ["Tool or technology", "Use in the project"],
        ["Python, PyTorch", "Building the model, training with mixed precision, saving checkpoints"],
        ["Albumentations", "Random crops, flips, brightness and colour changes"],
        ["OpenCV, scikit-image, SciPy", "Thresholding, morphology, skeletons and gap bridging"],
        ["ONNX, ONNX Runtime", "Model export and CPU inference without PyTorch"],
        ["Kaggle GPU notebooks", "Model training"],
        ["Streamlit", "Interactive demonstration application"],
        ["Matplotlib, NumPy", "Measurement scripts and the figures in this report"],
        ["Git, GitHub", "Version control and team work"],
    ], [2.2, 4.07], "Tools and technologies used", size=10.5)
    rp.heading("4.2 Technical Concepts", 2)
    rp.bullets([
        [("Efficient vision transformers: ", {"bold": True}),
         ("how MobileViT v2 gets global context with attention whose cost grows linearly with image size.", {})],
        [("Connectivity aware learning: ", {"bold": True}),
         ("soft skeletons, the clDice loss, and how a loss value can look good for a bad prediction.", {})],
        [("Class imbalance: ", {"bold": True}),
         ("weighting the loss when roads cover only a few percent of the pixels.", {})],
        [("Morphological image processing: ", {"bold": True}),
         ("hysteresis thresholding, closing, skeletons and detection of road ends.", {})],
        [("Model deployment: ", {"bold": True}),
         ("ONNX export with a variable input size, output checks against PyTorch, and execution providers.", {})],
        [("Experimental method: ", {"bold": True}),
         ("fixed data splits, testing at the resolution used in deployment, bootstrap confidence intervals, and "
          "keeping measured results separate from expectations.", {})],
        [("Research practice: ", {"bold": True}),
         ("literature survey, step by step debugging of a failed training run, and technical writing. The most "
          "useful findings came from looking at real predictions, not only at logged numbers.", {})],
    ])

    # 5 ────────────────────────────────────────────────────────────────────
    rp.heading("5. Architectural Framework")
    rp.heading("5.1 System Architecture", 2)
    rp.para("Figure 1 shows the two stages of the system. During training, DeepGlobe tiles are augmented and "
            "cropped, passed through the network, and scored by the combined loss. After each epoch the model is "
            "validated, and the selection rule decides whether to save it. The selected model is exported to ONNX. "
            "During inference, a full tile is passed through the network with four flip test time augmentation "
            "(TTA). The averaged probability map is then turned into a connected road mask by hysteresis "
            "thresholding, morphological closing and canopy gap bridging.")
    rp.figure(os.path.join(FIG, "report", "fig_r_pipeline.png"), 6.2, "Training and inference pipeline")
    rp.heading("5.2 Network Architecture", 2)
    rp.para("The network (Figure 2, Table 3) has a U shaped encoder and decoder. The encoder mixes low cost "
            "convolution stages with MobileViT v2 blocks and reduces the image size by 16 times. The decoder "
            "upsamples back to full size and adds encoder features through skip connections.")
    rp.figure(os.path.join(FIG, "report", "fig_r_network.png"), 6.2, "MobileViT v2 encoder and decoder")
    rp.table([
        ["Stage", "Operation", "Output"],
        ["Stem", "Strip convolution (1×3 then 3×1), stride 2", "32 × H/2"],
        ["Encoder 1", "MobileNetV2 inverted residual block, stride 2", "64 × H/4"],
        ["Encoder 2", "MobileViT v2 block (2 transformer layers), then inverted residual, stride 2", "96 × H/8"],
        ["Encoder 3", "MobileViT v2 block (2 layers), then inverted residual, stride 2", "128 × H/16"],
        ["Bottleneck", "MobileViT v2 block (3 layers)", "128 × H/16"],
        ["Decoders 3 to 1", "Upsample ×2, gated skip connection, strip convolution", "96, 64, 32 channels"],
        ["Head", "Upsample ×2, 1×1 convolution (sigmoid added in the exported model)", "1 × H"],
    ], [1.25, 3.75, 1.27], f"Layer configuration ({P['params']:,} parameters in total; H is the input height)",
             size=10.5)
    rp.para("Four parts adapt the network to roads:")
    rp.bullets([
        [("Strip convolution. ", {"bold": True}),
         ("A 3×3 convolution is replaced by a horizontal 1×3 convolution followed by a vertical 3×1 convolution, "
          "each with batch normalisation and SiLU. This uses about one third fewer weights and favours long, thin "
          "shapes.", {})],
        [("Channel shift. ", {"bold": True}),
         ("Before each transformer, 25% of the channels are moved by 2 pixels up, down, left or right. This "
          "widens the area that each position can see and adds no parameters.", {})],
        [("Separable self attention. ", {"bold": True}),
         ("MobileViT v2 blocks split the feature map into patches and apply separable self attention [8]. Its "
          "cost grows linearly with the number of patches, so the network gets global context at low cost.", {})],
        [("Attention gates. ", {"bold": True}),
         ("Each skip connection passes through an attention gate [12]. The gate uses the decoder signal to "
          "suppress encoder features that are not roads, such as roof edges and field boundaries.", {})],
    ])
    rp.heading("5.3 Loss Function", 2)
    rp.para("The network is trained with a weighted sum of three losses. The weights change with the epoch e:",
            keep_next=True)
    rp.equation([("L(e) = α(e)·L", {}), ("BCE", {"sub": True}), (" + 0.35·L", {}), ("Dice", {"sub": True}),
                 (" + (0.65 − α(e))·L", {}), ("clDice", {"sub": True})], 1)
    rp.equation([("α(e) = 0.50 − 0.35·min(e / 40, 1)", {}), ("0.5", {"sup": True})], 2)
    rp.para([("L", {}), ("BCE", {"sub": True}),
             (" is binary cross entropy with road pixels weighted twice as much as background, and L", {}),
             ("Dice", {"sub": True}),
             (" is the soft Dice loss. Both reward correct road pixels. The clDice loss [9] compares centrelines. "
              "A soft skeleton S(·) is computed with repeated min pooling and max pooling (10 iterations). "
              "Topology precision is the share of the predicted skeleton that lies inside the true road mask, and "
              "topology sensitivity is the share of the true skeleton that lies inside the prediction. clDice is "
              "their harmonic mean, and the loss is 1 − clDice. A missing piece of road removes part of the "
              "skeleton, so it is penalised even when it covers few pixels. At the start of training α = 0.50, so "
              "the pixel losses dominate while the network learns where roads are. α then falls to 0.15 by epoch "
              "40, which moves weight to the clDice term.", {})])
    rp.heading("5.4 Inference and Postprocessing", 2)
    rp.para("At inference the tile is processed at full resolution, with its size made a multiple of 32. The "
            "network runs on the tile and on its horizontal, vertical and double flips; the four outputs are "
            "flipped back and averaged. Hysteresis thresholding keeps pixels above 0.35, plus weaker pixels (above "
            "0.12) that touch them, so a faint, partly hidden road is kept when it joins a confident road. A 5×5 "
            "closing fills very thin breaks. Canopy gap bridging then reduces the mask to a one pixel skeleton, "
            "finds open road ends and their directions, and draws a 6 pixel wide road between two ends that are "
            "within 220 pixels of each other and point towards each other within 65°. Open ends that point at a "
            "nearby road are joined to it to complete T junctions. Ends within 16 pixels of the tile border are "
            "ignored, because a road that leaves the tile is not a gap.")

    # 6 ────────────────────────────────────────────────────────────────────
    rp.heading("6. Implementation Details")
    rp.heading("6.1 Dataset", 2)
    rp.para("The DeepGlobe Road Extraction dataset [3] has 6,226 labelled RGB tiles of 1024 × 1024 pixels at "
            "0.5 m per pixel, taken over Thailand, Indonesia and India, each with a binary road mask. The official "
            "validation and test masks were not released, so 10% of the labelled tiles were held out for "
            "validation using a fixed random seed (42). This gives 5,604 training tiles and 622 validation tiles.")
    rp.heading("6.2 Preprocessing and Augmentation", 2)
    rp.para("Each training sample is a random 256 × 256 crop at full resolution. The crop is flipped horizontally "
            "and vertically (probability 0.5 each), and its brightness, contrast and colour are changed at random "
            "(probability 0.2 and 0.3). Pixel values are then normalised with ImageNet statistics. A canopy shadow "
            "augmentation, which darkens random patches to imitate trees, was also written. Because of a "
            "constructor bug with the installed Albumentations version it was switched off during training, so "
            "the current model was trained without it. The bug has since been fixed in the notebook.")
    rp.heading("6.3 Training Configuration", 2)
    rp.table([
        ["Setting", "Value"],
        ["Framework and hardware", "PyTorch with mixed precision, Kaggle GPU notebook"],
        ["Optimiser", "AdamW, learning rate 1×10⁻³, weight decay 1×10⁻⁴"],
        ["Learning rate schedule", "Cosine annealing to 1×10⁻⁵ over at most 100 epochs"],
        ["Batch", "16 crops of 256 × 256 pixels"],
        ["Gradient clipping", "Maximum norm 1.0"],
        ["Loss", "Equations (1) and (2); road class weight 2 in BCE"],
        ["Initialisation", "Warm start from earlier trained weights"],
        ["Model selection", "Highest validation IoU; epochs that predict more than 20% road are rejected"],
        ["Early stopping", "Stop after 35 epochs without a better validation IoU"],
        ["Selected checkpoint", f"Epoch {F['epoch']} (counted from 0)"],
    ], [2.1, 4.17], "Training configuration of the current model", size=10.5)
    rp.heading("6.4 Model Selection and the Training Collapse", 2)
    rp.para("An earlier training run learned to mark almost every pixel as road. Two causes worked together. "
            "First, a road class weight of 3 made “everything is road” an easy way to lower the loss for a "
            "network that started from random weights. Second, the saved model was chosen by the soft clDice "
            "validation loss. This loss gives a good score to a prediction that covers the whole image, because "
            "its sensitivity term reaches its maximum. As a result, the failed epoch was saved as the best model. "
            "The fix lowered the class weight to 2, chose the saved model by IoU instead, added the 20% road limit, "
            "and started training from earlier trained weights instead of random weights (Section 7.3).")
    rp.heading("6.5 Deployment and Demonstration", 2)
    rp.para(f"The selected model is exported to ONNX with the sigmoid inside the graph and a variable input height "
            f"and width. On the current model, the outputs of ONNX Runtime and PyTorch differ by at most "
            f"{PARITY['max_abs_diff'] * 1e7:.1f}×10⁻⁷ on square and rectangular test inputs. The inference script "
            "needs only ONNX Runtime, OpenCV, NumPy, SciPy and scikit-image. It applies the same postprocessing "
            "and picks the best available execution provider (CUDA, TensorRT, CoreML, NNAPI or CPU). A Streamlit "
            "application runs the ONNX model: the user selects a sample tile or uploads one, sets the thresholds "
            "and the bridging distance, and switches TTA and bridging on or off. The application shows the input, "
            "a road overlay, the probability map and the road centreline, together with the inference time.")
    rp.heading("6.6 Evaluation Protocol", 2)
    rp.para("An evaluation notebook has been written to measure accuracy against ground truth on the 622 "
            "validation tiles at full resolution. It computes dataset level IoU, precision, recall and F1; relaxed "
            "F1 with a tolerance of 3 pixels; clDice on hard skeletons; the number of connected road components; "
            "recall on road under tree cover, found with the Excess Green index; 95% bootstrap confidence "
            "intervals; and Wilcoxon signed rank tests between models. Before reporting, it checks the data split "
            "by reproducing the validation metrics stored in the checkpoints. The notebook has not been run yet, "
            "so its results are not part of this report.")
    rp.heading("6.7 Issues Identified and Resolved", 2)
    rp.table([
        ["Issue", "Effect", "Resolution"],
        ["Training collapse", f"Model marked most pixels as road ({pct(col['collapsed_all_tiles_pos_frac'])} "
                              "on four sample tiles)", "Class weight lowered from 3 to 2; warm start"],
        ["Selection metric could be fooled", "Collapsed epoch saved as the best model",
         "Select by IoU with a 20% road limit"],
        ["Validation on tiles resized to 256²", "Roads 4 times thinner than in training; metrics likely understated",
         "Notebook now validates at full resolution"],
        ["Augmentation switched off", "Canopy shadow augmentation never ran", "Constructor fixed"],
        ["Bridging at tile borders", "False roads drawn along tile edges", "Road ends near the border are ignored"],
        ["Checkpoint loading", "Model without attention gates loaded into a model with random gates",
         "Architecture detected from the checkpoint"],
    ], [1.75, 2.35, 2.17], "Issues found during the internship and their resolution", size=10)

    # 7 ────────────────────────────────────────────────────────────────────
    rp.heading("7. Results")
    rp.para(f"All results in this section were measured on the current model. Computing cost was measured on one "
            f"laptop CPU ({CPU}, batch size 1, 1024 × 1024 tile). The reference models were not trained in this "
            "project, so their numbers show computing cost only.")
    rp.heading("7.1 Model Size and Speed", 2)
    order = ["Proposed", "LR-ASPP-MBv3", "DeepLabV3-MBv3", "D-LinkNet34", "U-Net"]
    names = {"Proposed": "Proposed (MobileViT v2)", "LR-ASPP-MBv3": "LR-ASPP MobileNetV3 [4]",
             "DeepLabV3-MBv3": "DeepLabV3 MobileNetV3 [5]", "D-LinkNet34": "D-LinkNet34 [2]", "U-Net": "U-Net [1]"}
    rows = [["Model", "Parameters (M)", "GFLOPs", "Weights (MB)", "CPU latency (s)"]]
    for k in order:
        m = M[k]
        lat = f"{m['latency_ms'] / 1000:.2f}" + ("†" if "latency_note" in m else "")
        rows.append([names[k], f"{m['params'] / 1e6:.2f}", f"{m['gflops']:,.1f}", f"{m['weights_mb_fp32']:.1f}", lat])
    rows.append(["Proposed, ONNX Runtime", f"{P['params'] / 1e6:.2f}", f"{P['gflops']:,.1f}",
                 f"{ONNX['file_mb_total']:.1f}", f"{ONNX['latency_ms'] / 1000:.2f}"])
    rp.table(rows, [2.2, 1.1, 0.85, 1.0, 1.12],
             "Computing cost per 1024 × 1024 tile (PyTorch unless stated; † timed as four 512² crops)",
             size=10.5, align_right_from=1)
    rp.figure(os.path.join(FIG, "fig_efficiency.png"), 5.8, "Parameters, operations and CPU latency per 1024² tile")
    rp.para(f"The proposed network has {UNET['params'] / P['params']:.1f} times fewer parameters and "
            f"{UNET['gflops'] / P['gflops']:.0f} times fewer operations than U-Net, and runs "
            f"{UNET['latency_ms'] / P['latency_ms']:.1f} times faster on the same CPU. With ONNX Runtime a tile "
            f"takes {ONNX['latency_ms'] / 1000:.2f} s, {P['latency_ms'] / ONNX['latency_ms']:.1f} times faster than "
            f"PyTorch. The deployable model is {ONNX['file_mb_total']:.1f} MB in total ({ONNX['graph_file_mb']:.2f} MB "
            f"graph and {ONNX['file_mb_total'] - ONNX['graph_file_mb']:.1f} MB weights). LR-ASPP with MobileNetV3 is "
            f"cheaper still ({lr['gflops']:.1f} GFLOPs, {lr['latency_ms'] / 1000:.2f} s), so the proposed model is "
            "much cheaper than U-Net class models but not cheaper than every mobile network (Table 6, Figure 3).")

    rp.heading("7.2 Validation During Training", 2)
    lab = {"baseline": "Baseline (no attention gates)", "collapsed": "Collapsed run", "run1": "Run 1",
           "final": "Final (current model)"}
    rows = [["Checkpoint", "Epoch", "IoU", "Precision", "Road pixels", "clDice loss"]]
    for k in ("baseline", "collapsed", "run1", "final"):
        m = CK[k]
        rows.append([lab[k], str(m["epoch"]), pct(m["val_iou"]) if "val_iou" in m else "n/a",
                     pct(m["val_precision"]) if "val_precision" in m else "n/a",
                     pct(m["val_positive_frac"]) if "val_positive_frac" in m else "n/a", f"{m['val_cldice']:.3f}"])
    rp.table(rows, [2.2, 0.65, 0.8, 0.9, 0.95, 0.77],
             "Validation metrics stored in each checkpoint (622 tiles resized to 256²; n/a: not logged)",
             size=10.5, align_right_from=1, bold_last=True)
    rp.para(f"The current model reached a validation IoU of {pct(F['val_iou'], 2)} and a precision of "
            f"{pct(F['val_precision'])}, while marking {pct(F['val_positive_frac'])} of pixels as road. This is "
            f"better than run 1 ({pct(CK['run1']['val_iou'], 2)}). These values come from the validation method "
            "used during training, which resized each tile to 256 × 256. Section 7.4 shows why this method likely "
            "understates the accuracy of the model at full resolution.")

    # Figure numbers in the text are taken from the running counter (rp.fig_n + 1 is the next figure).
    rp.heading("7.3 Training Collapse", 2)
    rp.para(f"Figure {rp.fig_n + 1} compares the collapsed checkpoint with the current model on the same tiles. On the four "
            f"sample tiles, the collapsed model marked {pct(col['collapsed_all_tiles_pos_frac'])} of pixels as "
            f"road, compared with {pct(col['final_all_tiles_pos_frac'])} for the current model. Yet its validation "
            f"clDice loss was {CK['collapsed']['val_cldice']:.3f}, lower, and so better looking, than the "
            f"{F['val_cldice']:.3f} of the current model. This is why selection by clDice kept the failed model. "
            f"With the changes in Section 6.4, the selected checkpoint marks {pct(F['val_positive_frac'])} of "
            "validation pixels as road.")
    rp.figure(os.path.join(FIG, "report", "fig_r_collapse_panels.png"), 3.6,
              "Collapsed checkpoint (epoch 46) and current model on two sample tiles")

    rp.heading("7.4 Effect of Validation Resolution", 2)
    rp.para(f"During training, validation resized each 1024² tile to 256², which made roads four times thinner "
            f"than in the training crops. Figure {rp.fig_n + 1} shows the result: on {n_zero} of the {len(res)} tiles, the "
            "current model finds no road at all after resizing, while at full resolution it traces the roads. The "
            "IoU logged during training therefore likely understates the deployed model, and the best epoch was "
            "chosen at the wrong scale. The training notebook now validates at full resolution; the current model "
            "has not yet been retrained with this change.")
    rp.figure(os.path.join(FIG, "fig_resolution.png"), 3.0,
              "Current model at full resolution and with the 256² validation method")

    rp.heading("7.5 Postprocessing", 2)
    rp.para(f"Figure {rp.fig_n + 1} follows tile {ps['tile']} through the inference pipeline. After hysteresis "
            f"thresholding and closing, the mask has {ps['components_before']} separate road pieces. Canopy gap "
            f"bridging adds {ps['bridged_pixels']:,} road pixels and reduces this to {ps['components_after']} pieces.")
    fig_stages = rp.figure(os.path.join(FIG, "fig_pipeline_stages.png"), 5.1,
              f"Postprocessing stages on tile {ps['tile']}: probability map, thresholded mask and bridged mask")

    rp.heading("7.6 Model Output", 2)
    road = ", ".join(pct(OVER[t]["road_frac"]) for t in ("100034", "117991", "115714"))
    road += f" and {pct(OVER['102408']['road_frac'])}"
    rp.para(f"Figure {rp.fig_n + 1} shows input tiles and the road masks produced by the full deployed pipeline "
            "(TTA, thresholding, closing and bridging). The forest road in tile 100034 and the farm road in tile 115714 "
            "are each traced as one continuous line, and the village and town tiles give connected street "
            f"networks. Predicted road covers {road} of the four tiles. The short branch on tile 115714 follows a "
            "field boundary and is probably a false positive.")
    fig_outputs = rp.figure(os.path.join(FIG, "report", "fig_r_outputs.png"), 5.2,
                            "Input tiles and model output (road shown in white)")

    rp.heading("7.7 Discussion", 2)
    rp.para(f"The model meets the efficiency goal: {P['params'] / 1e6:.1f} M parameters and under one second per "
            "tile on a laptop CPU with ONNX Runtime. It produces thin, continuous roads, and bridging reduces "
            "broken pieces. The results also have clear limits. The sample tiles have no ground truth, so Figures "
            f"{fig_stages} and {fig_outputs} show behaviour, not accuracy. Bridging can join features that are not roads. The reference "
            "models were compared on cost, not accuracy. Accuracy at full resolution on the held out tiles will "
            "come from the evaluation notebook (Section 6.6) and should replace the IoU logged during training "
            "once it has been run.")

    # 8 ────────────────────────────────────────────────────────────────────
    rp.heading("8. Conclusion")
    rp.para(f"This internship produced a small, connectivity aware pipeline for extracting rural roads from "
            f"satellite images. A MobileViT v2 encoder decoder with strip convolutions, channel shift and attention "
            f"gates, trained with a combined BCE, Dice and clDice loss, has {P['params']:,} parameters, needs "
            f"{P['gflops']:.1f} GFLOPs per 1024² tile, and runs in {ONNX['latency_ms'] / 1000:.2f} s on a laptop "
            "CPU with ONNX Runtime. Canopy gap bridging reconnects broken predictions, and the system comes with an "
            "ONNX inference script and a Streamlit demonstration. The work also found and fixed several problems: "
            "a training collapse hidden by a misleading selection metric, validation at the wrong scale, an "
            "augmentation that never ran, and bridging that drew roads along tile borders.")
    rp.para("Future work:", keep_next=True, after=3)
    rp.bullets([
        "Retrain with full resolution validation, so that the best epoch is chosen at the deployment scale.",
        "Run the evaluation notebook and report accuracy on the held out tiles, including road under trees.",
        "Tune canopy gap bridging against ground truth to reduce false links.",
        "Train a mobile CNN baseline (LR-ASPP) on the same data for a fair comparison of accuracy and cost, and "
        "add the APLS graph metric [18].",
        "Add OpenStreetMap weak labels, test on Indian rural images, and apply quantisation for low power devices.",
    ])

    # References ─────────────────────────────────────────────────────────────
    # Paper titles are given exactly as published, so hyphens inside titles are kept.
    rp.heading("References")
    refs = [
        "O. Ronneberger, P. Fischer and T. Brox, “U-Net: Convolutional networks for biomedical image "
        "segmentation,” in Proc. MICCAI, 2015.",
        "L. Zhou, C. Zhang and M. Wu, “D-LinkNet: LinkNet with pretrained encoder and dilated convolution for "
        "high resolution satellite imagery road extraction,” in Proc. CVPR Workshops, 2018.",
        "I. Demir et al., “DeepGlobe 2018: A challenge to parse the Earth through satellite images,” in "
        "Proc. CVPR Workshops, 2018.",
        "A. Howard et al., “Searching for MobileNetV3,” in Proc. ICCV, 2019.",
        "L. C. Chen, G. Papandreou, F. Schroff and H. Adam, “Rethinking atrous convolution for semantic image "
        "segmentation,” arXiv:1706.05587, 2017.",
        "A. Dosovitskiy et al., “An image is worth 16x16 words: Transformers for image recognition at "
        "scale,” in Proc. ICLR, 2021.",
        "S. Mehta and M. Rastegari, “MobileViT: Light-weight, general-purpose, and mobile-friendly vision "
        "transformer,” in Proc. ICLR, 2022.",
        "S. Mehta and M. Rastegari, “Separable self-attention for mobile vision transformers,” "
        "Transactions on Machine Learning Research, 2023.",
        "S. Shit et al., “clDice: A novel topology-preserving loss function for tubular structure "
        "segmentation,” in Proc. CVPR, 2021.",
        "F. Bastani et al., “RoadTracer: Automatic extraction of road networks from aerial images,” in "
        "Proc. CVPR, 2018.",
        "S. He et al., “Sat2Graph: Road graph extraction through graph-tensor encoding,” in Proc. ECCV, 2020.",
        "O. Oktay et al., “Attention U-Net: Learning where to look for the pancreas,” in Proc. MIDL, 2018.",
        "Qu et al., “Road MobileSeg: Lightweight and accurate road extraction model from remote sensing images "
        "for mobile devices,” 2024.",
        "Fu et al., “TopoRF-Net: Topology-aware road segmentation in multi-resolution remote sensing,” 2025.",
        "Shojaei et al., “Adaptive structure-aware connectivity preserving loss for improved road segmentation "
        "in remote sensing images,” 2025.",
        "Wang et al., “GLTDNet: Cross-domain road extraction through collaborative optimization,” 2025.",
        "Liu et al., “MViT-PCD: A lightweight ViT-based network for surface topographic detection,” 2023.",
        "A. Van Etten, D. Lindenbaum and T. M. Bacastow, “SpaceNet: A remote sensing dataset and challenge "
        "series,” arXiv:1807.01232, 2018.",
    ]
    for i, r in enumerate(refs, 1):
        p = rp.para(f"[{i}]\t{r}", size=10, after=1, align=WD_ALIGN_PARAGRAPH.LEFT)
        pf = p.paragraph_format
        pf.left_indent, pf.first_line_indent = Inches(0.4), Inches(-0.4)
        pf.tab_stops.add_tab_stop(Inches(0.4))


def main():
    doc = Document(TEMPLATE)
    fill_template(doc)
    body(Report(doc))
    add_page_numbers(doc)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
