import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'Nimbus Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

# ==============================================================================
# FIG 1: Minimal, High-Grade Academic Pipeline Diagram
# ==============================================================================
def create_academic_fig1():
    # 17 x 6.2 inches, 300 DPI
    fig, ax = plt.subplots(figsize=(17, 6.2), dpi=300)
    ax.set_xlim(0, 170)
    ax.set_ylim(0, 62)
    ax.axis('off')

    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    # Color Scheme: 3 Cohesive Functional Disciplines
    # 1. Data & Preprocessing: Charcoal Slate (#334155)
    # 2. Hybrid Model & Loss: Oxford Navy (#1E3A8A) / Crimson (#991B1B)
    # 3. Vector Graph & Deployment: Dark Spruce (#0F766E)

    def draw_academic_card(x, y, w, h, stage_id, title, bullets, top_color='#334155'):
        # Crisp white card with clean thin slate border
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6",
                                     facecolor='#FFFFFF', edgecolor='#94A3B8', linewidth=1.1, zorder=2)
        ax.add_patch(box)

        # Top accent rule (2.0 pt)
        top_rule = patches.Rectangle((x, y + h - 1.8), w, 1.8, facecolor=top_color, edgecolor='none', zorder=3)
        ax.add_patch(top_rule)

        # Stage ID tag
        tag = patches.FancyBboxPatch((x + 2.5, y + h - 7.0), 9.0, 3.8, boxstyle="round,pad=0,rounding_size=0.4",
                                     facecolor='#F1F5F9', edgecolor='#CBD5E1', linewidth=0.7, zorder=3)
        ax.add_patch(tag)
        ax.text(x + 7.0, y + h - 5.1, stage_id, ha='center', va='center',
                fontsize=7.5, fontweight='bold', color='#1E293B', zorder=4)

        # Card Title
        ax.text(x + 13.5, y + h - 5.1, title, ha='left', va='center',
                fontsize=8.8, fontweight='bold', color='#0F172A', zorder=4)

        # Horizontal thin separator
        ax.plot([x + 2.0, x + w - 2.0], [y + h - 8.8, y + h - 8.8], color='#E2E8F0', lw=0.7, zorder=3)

        # Bullets
        n_b = len(bullets)
        avail_h = h - 11.5
        step_y = avail_h / max(n_b, 1)
        start_y = y + h - 11.8

        for idx, (b_name, b_val) in enumerate(bullets):
            curr_y = start_y - idx * step_y
            # Bullet dot
            ax.scatter([x + 3.8], [curr_y], s=6, color=top_color, zorder=4)
            # Text
            full_text = f"{b_name} {b_val}"
            ax.text(x + 5.6, curr_y, full_text, ha='left', va='center',
                    fontsize=7.3, color='#334155', zorder=4)

    def draw_path_arrow(x1, y1, x2, y2, label=None, label_pos='top', color='#475569', lw=1.2):
        arrow = patches.FancyArrowPatch((x1, y1), (x2, y2),
                                       arrowstyle='-|>',
                                       mutation_scale=10,
                                       linewidth=lw,
                                       color=color,
                                       zorder=5)
        ax.add_patch(arrow)
        if label:
            lx = (x1 + x2) / 2
            ly = (y1 + y2) / 2
            offset_y = 2.2 if label_pos == 'top' else (-2.2 if label_pos == 'bottom' else 0)
            offset_x = 2.4 if label_pos == 'right' else (-2.4 if label_pos == 'left' else 0)
            ax.text(lx + offset_x, ly + offset_y, label,
                    ha='center' if label_pos in ['top', 'bottom'] else ('left' if label_pos == 'right' else 'right'),
                    va='center',
                    fontsize=7.0, fontweight='bold', color='#0F172A',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFFFFF', edgecolor='#CBD5E1', lw=0.6),
                    zorder=6)

    # Layout coordinates
    w_card = 35
    h_card = 24
    y_row1 = 34
    y_row2 = 4

    # ------------------ TOP STREAM: VISION & SEGMENTATION ------------------
    # Card 1.1: Optical Satellite Input
    draw_academic_card(4, y_row1, w_card, h_card, "1.1", "Optical Satellite Ingestion", [
        ("Resolution:", "0.5m - 2.0m spatial GSD"),
        ("Spectral Bands:", "RGB visible multi-spectral"),
        ("Corridor Focus:", "PMGSY rural village networks"),
        ("Terrain Class:", "Forested, unpaved dirt tracks")
    ], top_color='#334155')

    draw_path_arrow(39, y_row1 + h_card/2, 47, y_row1 + h_card/2)

    # Card 1.2: Resilience Prior & Augmentation
    draw_academic_card(47, y_row1, w_card, h_card, "1.2", "Canopy-Resilient Prior", [
        ("CanopyDropout:", "Simulates foliage occlusions"),
        ("ColorJitter:", "Monsoonal radiometric shifts"),
        ("Tiling Grid:", "Non-overlapping 256x256 tiles"),
        ("Tensor Format:", "Normalized torch.FloatTensor")
    ], top_color='#334155')

    draw_path_arrow(82, y_row1 + h_card/2, 90, y_row1 + h_card/2, label="Augmented X", label_pos='top')

    # Card 1.3: MobileViT_v2 Backbone
    draw_academic_card(90, y_row1, 38, h_card, "1.3", "MobileViT_v2 Hybrid Network", [
        ("Capacity:", "1.60M params (4.8% of standard U-Net)"),
        ("Novelty 1 (StripConv):", "Factorized 1x3 + 3x1 tubular filters"),
        ("Novelty 2 (ChannelShift):", "+/-8 px receptive field at 0 params"),
        ("Linear Attention:", "O(Nd) self-attention token mixing"),
        ("Decoder Hierarchy:", "Multi-scale attention-gated skip fusion")
    ], top_color='#1E3A8A')

    draw_path_arrow(128, y_row1 + h_card/2, 136, y_row1 + h_card/2, label="Prob Map P", label_pos='top')

    # Card 1.4: Topological Gap Healing
    draw_academic_card(136, y_row1, 30, h_card, "1.4", "Topological Gap Healing", [
        ("Dual Threshold:", "Hysteresis (0.35 / 0.12)"),
        ("Skeletonization:", "Medial axis thin centerline"),
        ("Tangent Bridging:", "Collinear endpoints <=220 px"),
        ("Contiguity Gain:", "Reduces fragments to 1.3/tile")
    ], top_color='#0F766E')

    # ------------------ BOTTOM STREAM: SUPERVISION & GRAPH ANALYSIS ------------------
    # Card 2.1: Weak OSM Supervision
    draw_academic_card(4, y_row2, w_card, h_card, "2.1", "Weak OSM Supervision", [
        ("Vector Source:", "OpenStreetMap spatial ways"),
        ("Rasterization:", "1-px differential centerline target"),
        ("Label Noise:", "Uncurated gaps & misalignments"),
        ("Geometric Prior:", "Tubular continuity constraint")
    ], top_color='#334155')

    draw_path_arrow(39, y_row2 + h_card/2, 47, y_row2 + h_card/2)

    # Card 2.2: Differential Morphological Engine
    draw_academic_card(47, y_row2, w_card, h_card, "2.2", "Differential Skeleton Engine", [
        ("soft_skel(Y):", "Continuous morphological operator"),
        ("Formulation:", "Iterative differentiable min-pooling"),
        ("Backpropagation:", "Exact end-to-end autograd flow"),
        ("Target Tensor:", "Topological 1D skeleton prior")
    ], top_color='#334155')

    draw_path_arrow(82, y_row2 + h_card/2, 90, y_row2 + h_card/2, label="Target Y_skel", label_pos='top')

    # Card 2.3: Topology Loss Engine
    draw_academic_card(90, y_row2, 38, h_card, "2.3", "Topology Loss Engine", [
        ("Loss Objective:", "L = alpha*L_BCE + beta*L_Dice + gamma*L_clDice"),
        ("clDice Metric:", "clDice = 2(Prec * Rec) / (Prec + Rec)"),
        ("Alpha Decay:", "Dynamic scheduling: 0.50 -> 0.40 (40 ep)"),
        ("Stability Gate:", "Prevents trivial all-background collapse")
    ], top_color='#991B1B')

    # Feedback Gradient Arrow (2.3 -> 1.3)
    ax.annotate('', xy=(109, y_row1), xytext=(109, y_row2 + h_card),
                arrowprops=dict(facecolor='#991B1B', edgecolor='#991B1B', width=1.4, headwidth=6, headlength=7, linestyle='--'))
    ax.text(111, (y_row1 + y_row2 + h_card)/2, "Topological Gradient Penalty", ha='left', va='center',
            fontsize=7.0, color='#991B1B', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF1F2', edgecolor='#FECDD3', lw=0.6), zorder=6)

    # Vertical Arrow from Gap Healing down to Vector Graph (1.4 -> 2.4)
    draw_path_arrow(151, y_row1, 151, y_row2 + h_card, label="Connected Mask M", label_pos='right', color='#0F766E')

    # Card 2.4: Vector Spatial Graph & Tactical Edge
    draw_academic_card(136, y_row2, 30, h_card, "2.4", "Vector Graph & Resilience", [
        ("Spatial Topology:", "NetworkX graph G(V, E)"),
        ("Criticality C_B(v):", "Betweenness pinch-point analysis"),
        ("Resilience Index R:", "Quantifies isolation vulnerability"),
        ("Edge Runtime:", "FP16 ONNX engine (<0.8MB payload)")
    ], top_color='#0F766E')

    plt.tight_layout()
    out1 = "figures/fig1_system_architecture.png"
    out2 = "docs/figures/fig1_system_architecture.png"
    fig.savefig(out1, dpi=300, bbox_inches='tight')
    fig.savefig(out2, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved academic Fig 1: {out1}")

# ==============================================================================
# FIG 2: Minimal, High-Grade Academic Architecture Diagram
# ==============================================================================
def create_academic_fig2():
    fig, ax = plt.subplots(figsize=(17, 9.5), dpi=300)
    ax.set_xlim(0, 170)
    ax.set_ylim(0, 95)
    ax.axis('off')

    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    # --------------------------------------------------------------------------
    # TOP SECTION: TECHNICAL SCHEMATICS FOR NOVELTY 1 & NOVELTY 2
    # --------------------------------------------------------------------------
    # Panel 1 Container
    p1 = patches.FancyBboxPatch((4, 63), 79, 28, boxstyle="round,pad=0,rounding_size=0.8",
                               facecolor='#FFFFFF', edgecolor='#94A3B8', linewidth=1.1, zorder=2)
    ax.add_patch(p1)
    ax.add_patch(patches.Rectangle((4, 89.2), 79, 1.8, facecolor='#1E3A8A', edgecolor='none', zorder=3))

    ax.text(6.5, 87.0, "Novelty 1: Factorized Directional Strip Convolution (StripConv)",
            ha='left', va='center', fontsize=8.8, fontweight='bold', color='#0F172A', zorder=4)
    ax.text(6.5, 84.0, "Decomposed 1D Scanning Kernels for Elongated Tubular Rural Corridors",
            ha='left', va='center', fontsize=7.6, color='#475569', zorder=4)
    ax.plot([5.5, 81.5], [82.0, 82.0], color='#E2E8F0', lw=0.7, zorder=3)

    # Technical schematic of kernel factorization inside Panel 1
    # 3x3 Standard Kernel
    ax.text(12, 77.5, "Standard 2D Conv", ha='center', fontsize=7.2, fontweight='bold', color='#334155')
    for r in range(3):
        for c in range(3):
            sq = patches.Rectangle((8 + c*2.6, 68.5 + (2-r)*2.6), 2.2, 2.2,
                                   facecolor='#F1F5F9', edgecolor='#94A3B8', lw=0.8, zorder=4)
            ax.add_patch(sq)
    ax.text(12, 66.5, "3 x 3 Kernel (9 params)", ha='center', fontsize=6.8, color='#64748B')

    # Arrow: Factorize ->
    ax.annotate('', xy=(24.5, 72.5), xytext=(19.0, 72.5),
                arrowprops=dict(facecolor='#1E3A8A', edgecolor='#1E3A8A', width=1.2, headwidth=5, headlength=6))
    ax.text(21.75, 74.0, "Decompose", ha='center', fontsize=6.5, fontweight='bold', color='#1E3A8A')

    # 1x3 Horizontal Conv
    ax.text(34, 77.5, "Horizontal Strip (1x3)", ha='center', fontsize=7.2, fontweight='bold', color='#1E3A8A')
    for c in range(3):
        sq = patches.Rectangle((29.5 + c*3.0, 71.2), 2.6, 2.6,
                               facecolor='#EFF6FF', edgecolor='#2563EB', lw=1.0, zorder=4)
        ax.add_patch(sq)
    ax.text(34, 66.5, "X_h = SiLU(BN(Conv_1x3(X)))", ha='center', fontsize=6.8, color='#1E3A8A')

    # Plus / Followed by ->
    ax.annotate('', xy=(44.5, 72.5), xytext=(40.5, 72.5),
                arrowprops=dict(facecolor='#1E3A8A', edgecolor='#1E3A8A', width=1.2, headwidth=5, headlength=6))

    # 3x1 Vertical Conv
    ax.text(53, 77.5, "Vertical Strip (3x1)", ha='center', fontsize=7.2, fontweight='bold', color='#1E3A8A')
    for r in range(3):
        sq = patches.Rectangle((51.7, 67.5 + (2-r)*3.0), 2.6, 2.6,
                               facecolor='#EFF6FF', edgecolor='#2563EB', lw=1.0, zorder=4)
        ax.add_patch(sq)
    ax.text(53, 64.5, "X_out = SiLU(BN(Conv_3x1(X_h)))", ha='center', fontsize=6.8, color='#1E3A8A')

    # Key Quantitative Advantage
    summary_box1 = patches.FancyBboxPatch((62, 65.5), 19.5, 14.5, boxstyle="round,pad=0,rounding_size=0.4",
                                          facecolor='#F8FAFC', edgecolor='#CBD5E1', lw=0.8, zorder=3)
    ax.add_patch(summary_box1)
    ax.text(71.75, 77.0, "Theoretical Gains", ha='center', fontsize=7.2, fontweight='bold', color='#0F172A')
    ax.text(63.5, 73.5, "• 33.3% parameter reduction", fontsize=6.8, color='#1E293B')
    ax.text(63.5, 70.5, "• Directional prior for roads", fontsize=6.8, color='#1E293B')
    ax.text(63.5, 67.5, "• Eliminates diagonal blur", fontsize=6.8, color='#1E293B')


    # Panel 2 Container: Novelty 2 (Channel Shift)
    p2 = patches.FancyBboxPatch((87, 63), 79, 28, boxstyle="round,pad=0,rounding_size=0.8",
                               facecolor='#FFFFFF', edgecolor='#94A3B8', linewidth=1.1, zorder=2)
    ax.add_patch(p2)
    ax.add_patch(patches.Rectangle((87, 89.2), 79, 1.8, facecolor='#0F766E', edgecolor='none', zorder=3))

    ax.text(89.5, 87.0, "Novelty 2: Zero-Parameter Channel Shift Operator",
            ha='left', va='center', fontsize=8.8, fontweight='bold', color='#0F172A', zorder=4)
    ax.text(89.5, 84.0, "Spatial Receptive Field Expansion via Zero-Cost Memory Slicing",
            ha='left', va='center', fontsize=7.6, color='#475569', zorder=4)
    ax.plot([88.5, 164.5], [82.0, 82.0], color='#E2E8F0', lw=0.7, zorder=3)

    # Technical schematic of 4-cardinal channel displacement inside Panel 2
    ax.text(99, 77.5, "Tensor Partitioning", ha='center', fontsize=7.2, fontweight='bold', color='#334155')
    # Sliced channels
    c_labels = [r"$C_1 \uparrow$", r"$C_2 \downarrow$", r"$C_3 \leftarrow$", r"$C_4 \rightarrow$", r"$C_{\mathrm{rest}}$"]
    c_colors = ['#CCFBF1', '#CCFBF1', '#CCFBF1', '#CCFBF1', '#F1F5F9']
    for idx, (clab, ccol) in enumerate(zip(c_labels, c_colors)):
        sq = patches.Rectangle((92 + idx*3.2, 69.5), 3.0, 6.5,
                               facecolor=ccol, edgecolor='#0F766E', lw=0.8, zorder=4)
        ax.add_patch(sq)
        ax.text(92 + idx*3.2 + 1.5, 72.75, clab, ha='center', va='center', fontsize=6.2, fontweight='bold', color='#0F766E')
    ax.text(99, 66.5, "Split into 4x 6.25% + 75% rest", ha='center', fontsize=6.8, color='#64748B')

    # Arrow -> Spatial Shift
    ax.annotate('', xy=(114.0, 72.5), xytext=(109.5, 72.5),
                arrowprops=dict(facecolor='#0F766E', edgecolor='#0F766E', width=1.2, headwidth=5, headlength=6))

    # Cardinal Shift Grid
    ax.text(125, 77.5, "Spatial Displacements", ha='center', fontsize=7.2, fontweight='bold', color='#0F766E')
    grid_center_x, grid_center_y = 125, 71.5
    # draw 4 cardinal arrows from center
    ax.annotate('', xy=(grid_center_x, grid_center_y + 4.5), xytext=(grid_center_x, grid_center_y),
                arrowprops=dict(facecolor='#0F766E', edgecolor='#0F766E', width=1.2, headwidth=4, headlength=5))
    ax.annotate('', xy=(grid_center_x, grid_center_y - 4.5), xytext=(grid_center_x, grid_center_y),
                arrowprops=dict(facecolor='#0F766E', edgecolor='#0F766E', width=1.2, headwidth=4, headlength=5))
    ax.annotate('', xy=(grid_center_x - 5.0, grid_center_y), xytext=(grid_center_x, grid_center_y),
                arrowprops=dict(facecolor='#0F766E', edgecolor='#0F766E', width=1.2, headwidth=4, headlength=5))
    ax.annotate('', xy=(grid_center_x + 5.0, grid_center_y), xytext=(grid_center_x, grid_center_y),
                arrowprops=dict(facecolor='#0F766E', edgecolor='#0F766E', width=1.2, headwidth=4, headlength=5))
    ax.scatter([grid_center_x], [grid_center_y], color='#0F766E', s=16, zorder=5)
    ax.text(125, 65.0, r"$\pm 8\,$px Receptive Field", ha='center', fontsize=6.8, fontweight='bold', color='#0F766E')

    # Summary box 2
    summary_box2 = patches.FancyBboxPatch((136.5, 65.5), 28.0, 14.5, boxstyle="round,pad=0,rounding_size=0.4",
                                          facecolor='#F8FAFC', edgecolor='#CBD5E1', lw=0.8, zorder=3)
    ax.add_patch(summary_box2)
    ax.text(150.5, 77.0, "Receptive Field Expansion", ha='center', fontsize=7.2, fontweight='bold', color='#0F172A')
    ax.text(138.0, 73.5, "• 0 Trainable Parameters Added", fontsize=6.8, color='#1E293B')
    ax.text(138.0, 70.5, "• 0 FLOP Computational Overhead", fontsize=6.8, color='#1E293B')
    ax.text(138.0, 67.5, "• Bridges tree canopy interruptions", fontsize=6.8, color='#1E293B')


    # --------------------------------------------------------------------------
    # MAIN ARCHITECTURE: ENCODER - BOTTLENECK - DECODER
    # --------------------------------------------------------------------------
    def draw_pipeline_block(x, y, w, h, stage_name, tensor_shape, specs, top_color='#1E3A8A'):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6",
                                     facecolor='#FFFFFF', edgecolor='#94A3B8', linewidth=1.1, zorder=2)
        ax.add_patch(box)
        ax.add_patch(patches.Rectangle((x, y + h - 1.8), w, 1.8, facecolor=top_color, edgecolor='none', zorder=3))

        ax.text(x + w/2, y + h - 4.6, stage_name, ha='center', va='center',
                fontsize=8.0, fontweight='bold', color='#0F172A', zorder=4)

        # Tensor shape badge
        tb = patches.FancyBboxPatch((x + 2.0, y + h - 9.4), w - 4.0, 3.6, boxstyle="round,pad=0,rounding_size=0.4",
                                    facecolor='#F1F5F9', edgecolor='#CBD5E1', lw=0.6, zorder=3)
        ax.add_patch(tb)
        ax.text(x + w/2, y + h - 7.6, tensor_shape, ha='center', va='center',
                fontsize=7.0, fontweight='bold', color=top_color, zorder=4)

        # Specs lines
        n_s = len(specs)
        avail_h = h - 12.0
        step_y = avail_h / max(n_s, 1)
        start_y = y + h - 12.2

        for i, s in enumerate(specs):
            curr_y = start_y - i * step_y
            ax.text(x + w/2, curr_y, s, ha='center', va='center',
                    fontsize=6.8, color='#334155', zorder=4)

    def draw_orthogonal_arrow(x1, y1, x2, y2, color='#334155', lw=1.2):
        arrow = patches.FancyArrowPatch((x1, y1), (x2, y2),
                                       arrowstyle='-|>',
                                       mutation_scale=9,
                                       linewidth=lw,
                                       color=color,
                                       zorder=5)
        ax.add_patch(arrow)

    y_enc = 35
    y_dec = 5
    bw = 22
    bh = 22

    # ENCODER ROW: Unified Academic Navy (#1E3A8A)
    # Input
    draw_pipeline_block(4, y_enc, 18, bh, "Input Tile", "3 x 256 x 256", [
        "Optical Satellite",
        "RGB Multi-Spectral",
        "0.5m - 2.0m GSD"
    ], top_color='#475569')

    draw_orthogonal_arrow(22, y_enc + bh/2, 26, y_enc + bh/2)

    # Stem
    draw_pipeline_block(26, y_enc, bw, bh, "Stem (StripConv)", "32 x 128 x 128", [
        "1x3 -> 3x1 Factorized",
        "Stride = 2",
        "SiLU + BatchNorm"
    ], top_color='#1E3A8A')

    draw_orthogonal_arrow(48, y_enc + bh/2, 52, y_enc + bh/2)

    # Stage 1
    draw_pipeline_block(52, y_enc, bw, bh, "Stage 1 (MV2)", "64 x 64 x 64", [
        "Inverted Residual",
        "Expand = 2, Stride = 2",
        "Depthwise 3x3 Conv"
    ], top_color='#1E3A8A')

    draw_orthogonal_arrow(74, y_enc + bh/2, 78, y_enc + bh/2)

    # Stage 2
    draw_pipeline_block(78, y_enc, bw, bh, "Stage 2 (MViT-v2)", "96 x 32 x 32", [
        "ChannelShift (25%)",
        "Linear Self-Attention",
        "O(Nd) Complexity"
    ], top_color='#1E3A8A')

    draw_orthogonal_arrow(100, y_enc + bh/2, 104, y_enc + bh/2)

    # Stage 3
    draw_pipeline_block(104, y_enc, bw, bh, "Stage 3 (MViT-v2)", "128 x 16 x 16", [
        "ChannelShift (25%)",
        "Linear Self-Attention",
        "Local-to-Global Fusion"
    ], top_color='#1E3A8A')

    draw_orthogonal_arrow(126, y_enc + bh/2, 130, y_enc + bh/2)

    # Bottleneck
    draw_pipeline_block(130, y_enc, 26, bh, "Bottleneck", "128 x 16 x 16", [
        "3x MobileViT Blocks",
        "Separable Attention",
        "Global Context Prior"
    ], top_color='#0F172A')

    # Connection: Bottleneck -> Decoder Stage 3
    ax.plot([143, 143], [y_enc, y_dec + bh/2], color='#0F766E', lw=1.5, zorder=3)
    draw_orthogonal_arrow(143, y_dec + bh/2, 126, y_dec + bh/2, color='#0F766E', lw=1.5)

    # DECODER ROW: Unified Academic Teal / Spruce (#0F766E)
    # Decoder Stage 3
    draw_pipeline_block(104, y_dec, bw, bh, "Decoder Stage 3", "96 x 32 x 32", [
        "Bilinear Upsample 2x",
        "+ Attention Gate 3",
        "StripConv Fusion"
    ], top_color='#0F766E')

    draw_orthogonal_arrow(104, y_dec + bh/2, 100, y_dec + bh/2, color='#0F766E')

    # Decoder Stage 2
    draw_pipeline_block(78, y_dec, bw, bh, "Decoder Stage 2", "64 x 64 x 64", [
        "Bilinear Upsample 2x",
        "+ Attention Gate 2",
        "StripConv Fusion"
    ], top_color='#0F766E')

    draw_orthogonal_arrow(78, y_dec + bh/2, 74, y_dec + bh/2, color='#0F766E')

    # Decoder Stage 1
    draw_pipeline_block(52, y_dec, bw, bh, "Decoder Stage 1", "32 x 128 x 128", [
        "Bilinear Upsample 2x",
        "+ Attention Gate 1",
        "StripConv Fusion"
    ], top_color='#0F766E')

    draw_orthogonal_arrow(52, y_dec + bh/2, 48, y_dec + bh/2, color='#0F766E')

    # Output Head
    draw_pipeline_block(24, y_dec, 24, bh, "Segmentation Head", "1 x 256 x 256", [
        "Bilinear Upsample 2x",
        "1x1 Conv + Sigmoid",
        "Continuous Prob Map"
    ], top_color='#334155')

    # ATTENTION GATES & SKIPS: Muted Ochre / Amber (#B45309)
    def draw_ag_badge(x, y, label):
        b = patches.FancyBboxPatch((x - 5.5, y - 2.0), 11.0, 4.0, boxstyle="round,pad=0,rounding_size=0.4",
                                   facecolor='#FFFBEB', edgecolor='#D97706', lw=0.8, zorder=6)
        ax.add_patch(b)
        ax.text(x, y, label, ha='center', va='center', fontsize=6.8, fontweight='bold', color='#92400E', zorder=7)

    # Gate 3
    ax.plot([89, 89], [y_enc, 30], color='#D97706', lw=1.2, zorder=3)
    ax.plot([89, 115], [30, 30], color='#D97706', lw=1.2, zorder=3)
    draw_ag_badge(102, 30, "AG 3")
    draw_orthogonal_arrow(115, 30, 115, y_dec + bh, color='#D97706', lw=1.2)

    # Gate 2
    ax.plot([63, 63], [y_enc, 30], color='#D97706', lw=1.2, zorder=3)
    ax.plot([63, 89], [30, 30], color='#D97706', lw=1.2, zorder=3)
    draw_ag_badge(76, 30, "AG 2")
    draw_orthogonal_arrow(89, 30, 89, y_dec + bh, color='#D97706', lw=1.2)

    # Gate 1
    ax.plot([37, 37], [y_enc, 30], color='#D97706', lw=1.2, zorder=3)
    ax.plot([37, 63], [30, 30], color='#D97706', lw=1.2, zorder=3)
    draw_ag_badge(50, 30, "AG 1")
    draw_orthogonal_arrow(63, 30, 63, y_dec + bh, color='#D97706', lw=1.2)

    plt.tight_layout()
    out1 = "figures/fig2_network_architecture.png"
    out2 = "docs/figures/fig2_network_architecture.png"
    fig.savefig(out1, dpi=300, bbox_inches='tight')
    fig.savefig(out2, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved academic Fig 2: {out1}")

if __name__ == '__main__':
    create_academic_fig1()
    create_academic_fig2()
