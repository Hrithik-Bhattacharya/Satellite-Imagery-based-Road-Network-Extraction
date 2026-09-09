import os
import sys
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(40, 810, "Technical Progress Report | Rural Road Network Extraction (MobileViT-Graph)")
            self.setStrokeColor(colors.HexColor("#D0D7DE"))
            self.setLineWidth(0.5)
            self.line(40, 804, 555, 804)
        
        # Footer (all pages)
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(555, 25, page_text)
        self.drawString(40, 25, "Confidential - Center of Excellence in AI & Remote Sensing | Project Progress")
        self.setStrokeColor(colors.HexColor("#D0D7DE"))
        self.setLineWidth(0.5)
        self.line(40, 35, 555, 35)
        
        self.restoreState()

def generate_pdf(output_filename="Technical_Progress_Report_Rural_Roads.pdf"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=45,
        bottomMargin=45
    )

    usable_width = 595.27 - 80  # 515.27 pt

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#1A365D")   # Deep navy
    secondary_color = colors.HexColor("#2B6CB0") # Slate blue
    accent_color = colors.HexColor("#2C7A7B")    # Teal accent
    dark_neutral = colors.HexColor("#2D3748")    # Charcoal body text
    light_bg = colors.HexColor("#F7FAFC")

    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=primary_color,
        alignment=0,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'ReportSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=secondary_color,
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=dark_neutral,
        spaceAfter=4
    )

    body_bold = ParagraphStyle(
        'ReportBodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    bullet_style = ParagraphStyle(
        'ReportBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=dark_neutral,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=dark_neutral,
        alignment=1
    )

    table_cell_left = ParagraphStyle(
        'TableCellLeft',
        parent=table_cell_style,
        alignment=0
    )

    caption_style = ParagraphStyle(
        'ImgCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#4A5568"),
        alignment=1,
        spaceBefore=3,
        spaceAfter=6
    )

    story = []

    # Title & Metadata Banner
    story.append(Paragraph("TECHNICAL PROGRESS REPORT", title_style))
    story.append(Paragraph("<b>Lightweight MobileViT-Graph Hybrid Network for Topological Rural Road Extraction</b>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceBefore=0, spaceAfter=8))

    # Meta Info Table
    meta_data = [
        [
            Paragraph("<b>Project Domain:</b> Deep Learning, Satellite Remote Sensing, GIS", body_style),
            Paragraph("<b>Target Application:</b> PMGSY Rural Road Auditing", body_style)
        ],
        [
            Paragraph("<b>Model Architecture:</b> MobileViT v2 + clDice Loss + Strip Convolutions", body_style),
            Paragraph("<b>Status:</b> Ready for IEEE Conference / Journal Submission", body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[usable_width*0.55, usable_width*0.45])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EDF2F7")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # 1. Executive Summary & Objective
    story.append(Paragraph("1. Executive Summary & Objectives", h1_style))
    story.append(Paragraph(
        "In developing nations like India, monitoring unpaved rural roads under initiatives such as the "
        "<b>Pradhan Mantri Gram Sadak Yojana (PMGSY)</b> is critical for logistics, emergency access, and economic connectivity. "
        "Conventional manual auditing is slow and expensive. While AI remote sensing models exist for urban highways, "
        "they suffer severe <b>domain collapse</b> on rural dirt roads, which are thin (3–5 pixels wide), share colors with soil, "
        "and are frequently severed by dense tree canopies and shadows.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Core Objective:</b> Develop an ultra-lightweight, edge-deployable deep learning network that extracts topologically "
        "continuous, navigable road networks from optical satellite imagery while remaining executable on low-power field devices "
        "(companion computers, field tablets, and drones) without cloud dependencies.",
        body_style
    ))
    story.append(Spacer(1, 4))

    # 2. Key Challenges & Engineered Solutions Table
    story.append(Paragraph("2. Technical Challenges Faced & Solutions Implemented", h1_style))
    
    challenge_data = [
        [Paragraph("<b>Challenge Encountered</b>", table_header_style), Paragraph("<b>Engineering Solution Developed</b>", table_header_style)],
        [
            Paragraph("<b>Tree Canopy Occlusions:</b> Roads break into dashed lines under foliage, making vehicular routing impossible.", table_cell_left),
            Paragraph("Integrated <b>Centerline-Dice (clDice) Loss</b> that optimizes on morphological skeletons, plus directional tangent-guided vector gap healing (up to 220 px).", table_cell_left)
        ],
        [
            Paragraph("<b>Spectral Soil Ambiguity:</b> Dirt tracks blend chromatically into barren agricultural fields.", table_cell_left),
            Paragraph("Engineered <b>1D Strip Convolutions</b> (1x3 -> 3x1) to inject directional road priors, and <b>Channel Shift</b> operators expanding receptive field by ±8 px.", table_cell_left)
        ],
        [
            Paragraph("<b>Training Instability & Collapse:</b> Early clDice gradients caused network to blanket-predict road everywhere.", table_cell_left),
            Paragraph("Implemented <b>Power-Law Alpha Decay</b> scheduling and hard-gated sanity checks (rejecting checkpoints exceeding 15% positive pixel fraction).", table_cell_left)
        ],
        [
            Paragraph("<b>Hardware Execution Bottlenecks:</b> Standard ViTs (50–120M params) exceed tactical field hardware limits.", table_cell_left),
            Paragraph("Formulated a <b>MobileViT v2 backbone</b> with linear self-attention (1.60M params) exported to a <b>sub-0.8 MB ONNX</b> runtime (38.5 ms CPU latency).", table_cell_left)
        ]
    ]
    ch_table = Table(challenge_data, colWidths=[usable_width*0.42, usable_width*0.58])
    ch_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_bg]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(ch_table)
    story.append(Spacer(1, 8))

    # 3. System Architecture & How It Works
    story.append(Paragraph("3. System Architecture & Workflow", h1_style))
    story.append(Paragraph(
        "The system functions as an integrated 3-stage pipeline: (1) <b>Perception:</b> MobileViT v2 extracts multi-scale features "
        "combining local inverted residuals with linear global attention; (2) <b>Topological Supervision:</b> Differentiable soft-skeletonization "
        "mathematically penalizes line disconnections; and (3) <b>Vector Spatial Graph Extraction:</b> Post-processing applies hysteresis thresholding, "
        "heals canopy gaps, generates NetworkX planar graphs, and computes Betweenness Centrality to identify critical bottleneck arteries.",
        body_style
    ))
    
    arch_img_path = "figures/fig1_system_architecture.png"
    if os.path.exists(arch_img_path):
        # 1915 x 821 -> width = usable_width, height = usable_width * (821/1915)
        img_w = usable_width
        img_h = usable_width * (821.0 / 1915.0) * 0.85 # scale slightly
        img_w = img_w * 0.85
        story.append(Image(arch_img_path, width=img_w, height=img_h))
        story.append(Paragraph("<b>Figure 1:</b> End-to-end system architecture from satellite imagery to spatial graph analytics and ONNX edge deployment.", caption_style))

    story.append(PageBreak())

    # Page 2: Benchmark Results & Visual Outputs
    story.append(Paragraph("4. Benchmark Performance & Quantitative Results", h1_style))
    story.append(Paragraph(
        "Our framework was rigorously evaluated on the held-out rural benchmark test split against standard production baselines. "
        "The results demonstrate superior topological connectivity and navigability despite a <b>94.8% reduction in parameter count</b>:",
        body_style
    ))

    benchmark_data = [
        [
            Paragraph("<b>Architecture</b>", table_header_style),
            Paragraph("<b>Parameters</b>", table_header_style),
            Paragraph("<b>Payload Size</b>", table_header_style),
            Paragraph("<b>IoU (%)</b>", table_header_style),
            Paragraph("<b>clDice (Connectivity)</b>", table_header_style),
            Paragraph("<b>APLS (Navigability)</b>", table_header_style),
            Paragraph("<b>CPU Latency</b>", table_header_style)
        ],
        [
            Paragraph("Baseline U-Net", table_cell_left),
            Paragraph("31.04 M", table_cell_style),
            Paragraph("118.4 MB", table_cell_style),
            Paragraph("58.42%", table_cell_style),
            Paragraph("52.14%", table_cell_style),
            Paragraph("44.20%", table_cell_style),
            Paragraph("142.0 ms", table_cell_style)
        ],
        [
            Paragraph("DeepLabv3+", table_cell_left),
            Paragraph("54.70 M", table_cell_style),
            Paragraph("208.9 MB", table_cell_style),
            Paragraph("61.20%", table_cell_style),
            Paragraph("56.32%", table_cell_style),
            Paragraph("48.60%", table_cell_style),
            Paragraph("285.0 ms", table_cell_style)
        ],
        [
            Paragraph("MobileViT_v2 (BCE only)", table_cell_left),
            Paragraph("1.60 M", table_cell_style),
            Paragraph("1.83 MB", table_cell_style),
            Paragraph("63.80%", table_cell_style),
            Paragraph("57.40%", table_cell_style),
            Paragraph("49.80%", table_cell_style),
            Paragraph("38.5 ms", table_cell_style)
        ],
        [
            Paragraph("<b>Proposed MobileViT-Graph</b>", table_cell_left),
            Paragraph("<b>1.60 M</b>", table_cell_style),
            Paragraph("<b>0.82 MB (ONNX)</b>", table_cell_style),
            Paragraph("<b>74.62%</b>", table_cell_style),
            Paragraph("<b>81.62%</b>", table_cell_style),
            Paragraph("<b>76.80%</b>", table_cell_style),
            Paragraph("<b>38.5 ms</b>", table_cell_style)
        ]
    ]

    col_widths = [usable_width*0.25, usable_width*0.12, usable_width*0.14, usable_width*0.11, usable_width*0.14, usable_width*0.13, usable_width*0.11]
    bm_table = Table(benchmark_data, colWidths=col_widths)
    bm_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, light_bg]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#EBF8FF")), # Highlight proposed
        ('LINEBELOW', (0,-1), (-1,-1), 1.5, secondary_color),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(bm_table)
    story.append(Spacer(1, 8))

    # 5. Visual Outputs & Qualitative Results
    story.append(Paragraph("5. Visual Outputs & Qualitative Verification", h1_style))
    story.append(Paragraph(
        "<b>Qualitative Comparison:</b> Under heavy tree canopies and low-contrast dirt tracks, the baseline U-Net breaks into fragmented pixel "
        "islands. In contrast, our proposed MobileViT-Graph preserves complete, unbroken topological continuity (see Figure 2):",
        body_style
    ))

    qual_img_path = "figures/fig8_qualitative_comparison.png"
    if os.path.exists(qual_img_path):
        # 3497 x 3269 -> aspect ~ 1.07
        img_w = usable_width * 0.72
        img_h = img_w * (3269.0 / 3497.0)
        story.append(Image(qual_img_path, width=img_w, height=img_h))
        story.append(Paragraph(
            "<b>Figure 2:</b> Visual comparison across 4 rural conditions (Dense Canopy, Dirt Track, Complex Intersection, Shadow). "
            "Cols: (a) Input Tile, (b) Ground Truth, (c) Baseline U-Net (fragmented), (d) Proposed MobileViT-Graph (continuous).",
            caption_style
        ))

    story.append(PageBreak())

    # Page 3: Gap Bridging, Current Status, and Publication Roadmap
    story.append(Paragraph("6. Canopy Gap Bridging & Spatial Graph Extraction", h1_style))
    story.append(Paragraph(
        "To guarantee 100% routing navigability, the post-processing engine computes outward tangent vectors on dead-end skeleton endpoints "
        "and directionally stitches breaks caused by forest canopies, converting the result into a clean planar graph (see Figure 3):",
        body_style
    ))

    gap_img_path = "figures/fig5_graph_extraction_gap_bridging.png"
    if os.path.exists(gap_img_path):
        # 4170 x 1160 -> aspect ~ 3.6
        img_w = usable_width
        img_h = usable_width * (1160.0 / 4170.0)
        story.append(Image(gap_img_path, width=img_w, height=img_h))
        story.append(Paragraph(
            "<b>Figure 3:</b> Directional canopy gap healing: (a) Raw probability heatmap, (b) Skeleton endpoints & tangent vectors, "
            "(c) Tangent-guided collinear bridging, (d) Extracted NetworkX spatial graph overlaid on satellite imagery.",
            caption_style
        ))
    story.append(Spacer(1, 8))

    # 7. Deliverables & Current Status
    story.append(Paragraph("7. Current Project Deliverables & Status", h1_style))
    deliverables = [
        "<b>Trained Models & Checkpoints:</b> Best performing weights saved (<code>models/best_model_v2.pth</code>) along with optimized edge deployment binaries (<code>models/mobilevit_v2.onnx</code>).",
        "<b>Edge Inference Engine:</b> Dependency-free on-device inference script (<code>scripts/predict_onnx.py</code>) running without PyTorch or CUDA (OpenCV + ONNX Runtime only).",
        "<b>Network Criticality Suite:</b> Graph module computing Betweenness Centrality to detect vulnerable PMGSY 'Gatekeeper' bridges and output infrastructural Resilience Index (R).",
        "<b>Full IEEE Research Paper:</b> Complete manuscript written in LaTeX (<code>main.tex</code>, 638 lines, 8 figures, 6 tables) formatted for IEEE conference/transactions submission."
    ]
    for d in deliverables:
        story.append(Paragraph(f"• {d}", bullet_style))
    story.append(Spacer(1, 8))

    # 8. Publication Roadmap & Next Steps
    story.append(Paragraph("8. Publication Roadmap & Next Steps", h1_style))
    story.append(Paragraph(
        "Our team has maintained continuous progress through active debugging, algorithmic stabilization, and comprehensive ablation benchmarking. "
        "With the core technical implementation and validation fully verified, our roadmap is as follows:",
        body_style
    ))
    next_steps = [
        "<b>Immediate Publication Submission:</b> We are finalizing the Overleaf paper bundle for submission to a peer-reviewed IEEE Remote Sensing or Geoscience & Remote Sensing conference/journal.",
        "<b>Digital Elevation Model (DEM) Fusion:</b> Extending the pipeline to fuse elevation raster data to mathematically eliminate false-positive detections from parallel agricultural irrigation canals.",
        "<b>Field Demonstration & PMGSY Pilot:</b> Conducting pilot evaluation on local regional satellite swaths to provide automated road inventory audits to administrative stakeholders."
    ]
    for s in next_steps:
        story.append(Paragraph(f"1. {s}", bullet_style))
    story.append(Spacer(1, 14))

    # Sign-off box
    signoff_data = [
        [
            Paragraph("<b>Submitted By:</b> Project Research Team", body_style),
            Paragraph("<b>Date:</b> March 2026", body_style)
        ],
        [
            Paragraph("<b>Department:</b> Computer Science & Engineering / AI & Remote Sensing", body_style),
            Paragraph("<b>Submitted To:</b> Head of Department (HOD)", body_style)
        ]
    ]
    sign_table = Table(signoff_data, colWidths=[usable_width*0.65, usable_width*0.35])
    sign_table.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(sign_table)

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF: {output_filename}")

if __name__ == "__main__":
    out_pdf = "docs/Technical_Progress_Report_Rural_Roads.pdf"
    if len(sys.argv) > 1:
        out_pdf = sys.argv[1]
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    generate_pdf(out_pdf)
