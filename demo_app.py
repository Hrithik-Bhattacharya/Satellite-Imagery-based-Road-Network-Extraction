"""
demo_app.py -- Research Demonstration Platform
Lightweight MobileViT-Graph Network for Topological Rural Road Extraction
Center of Excellence in Artificial Intelligence and Remote Sensing
"""

import os
import sys
import time
import numpy as np
import cv2
import streamlit as st

# Set project root
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(REPO_ROOT)

from backend.src.utils.graph_postprocess import connect_canopy_gaps, hysteresis_threshold

try:
    from skimage.morphology import skeletonize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

import onnxruntime as ort

# Streamlit Page Configuration - Professional Academic Presentation
st.set_page_config(
    page_title="Rural Road Network Extraction | Research Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Professional CSS
st.markdown("""
<style>
    /* Global Typography and Palette */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Top Institutional Header */
    .institute-bar {
        border-bottom: 1px solid #334155;
        padding-bottom: 8px;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .institute-name {
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94A3B8;
    }
    .project-status {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 4px;
        background-color: #0F172A;
        border: 1px solid #3B82F6;
        color: #60A5FA;
        letter-spacing: 0.04em;
    }

    /* Title Block */
    .title-text {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        margin-bottom: 4px;
    }
    .subtitle-text {
        font-size: 0.95rem;
        font-weight: 400;
        color: #94A3B8;
        margin-bottom: 18px;
        line-height: 1.5;
    }

    /* Professional Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 18px;
    }
    .metric-box {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-left: 3px solid #2563EB;
        border-radius: 6px;
        padding: 12px 16px;
    }
    .metric-box.accent {
        border-left-color: #059669;
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 1.55rem;
        font-weight: 700;
        color: #F1F5F9;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.74rem;
        font-weight: 500;
        color: #10B981;
        margin-top: 4px;
    }
    .metric-sub.neutral {
        color: #94A3B8;
    }

    /* Status Bar */
    .status-strip {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 10px 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
    }
    .status-info {
        font-size: 0.85rem;
        color: #CBD5E1;
    }
    .status-badge {
        font-size: 0.75rem;
        font-weight: 600;
        background: #064E3B;
        color: #34D399;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #059669;
    }

    /* Section Subheadings */
    .panel-header {
        font-size: 0.95rem;
        font-weight: 600;
        color: #E2E8F0;
        margin-bottom: 8px;
        padding-bottom: 4px;
        border-bottom: 1px solid #1E293B;
    }
    
    /* Table Styling */
    .academic-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
        margin-top: 10px;
    }
    .academic-table th {
        background-color: #1E293B;
        color: #E2E8F0;
        text-align: left;
        padding: 10px 14px;
        font-weight: 600;
        border-bottom: 2px solid #334155;
    }
    .academic-table td {
        padding: 10px 14px;
        border-bottom: 1px solid #1E293B;
        color: #CBD5E1;
    }
    .academic-table tr:hover {
        background-color: #0F172A;
    }
    .highlight-row {
        background-color: rgba(37, 99, 235, 0.08);
        font-weight: 600;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# Image Normalization Constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Session Cache Initialization
if 'prediction_cache' not in st.session_state:
    st.session_state['prediction_cache'] = {}

@st.cache_resource
def load_onnx_model():
    model_path = os.path.join(REPO_ROOT, "models", "mobilevit_v2.onnx")
    if not os.path.exists(model_path):
        return None, None, None
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    return session, in_name, out_name

def preprocess_image(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    new_h = (h // 32) * 32
    new_w = (w // 32) * 32
    resized = cv2.resize(image_rgb, (new_w, new_h))
    normed = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    chw = normed.transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(chw), resized, new_h, new_w

def run_inference(session, in_name, out_name, tensor, use_tta=True):
    start = time.perf_counter()
    if use_tta:
        # 4-way orthogonal invariance testing
        variants = [
            (lambda x: x, lambda p: p),
            (lambda x: x[:, :, :, ::-1], lambda p: p[:, :, :, ::-1]),
            (lambda x: x[:, :, ::-1, :], lambda p: p[:, :, ::-1, :]),
            (lambda x: x[:, :, ::-1, ::-1], lambda p: p[:, :, ::-1, ::-1]),
        ]
        sum_probs = None
        for f_fn, inv_fn in variants:
            t = np.ascontiguousarray(f_fn(tensor))
            raw = session.run([out_name], {in_name: t})[0]
            p = inv_fn(raw)
            sum_probs = p if sum_probs is None else sum_probs + p
        probs = sum_probs / len(variants)
    else:
        probs = session.run([out_name], {in_name: tensor})[0]
        
    duration = time.perf_counter() - start
    prob_map = probs[0, 0] if probs.ndim == 4 else probs.squeeze()
    return prob_map, duration

def generate_overlay(rgb, mask, color=(0, 230, 118), alpha=0.55):
    overlay = rgb.copy()
    idx = mask > 0
    color_arr = np.array(color, dtype=np.uint8)
    overlay[idx] = ((1 - alpha) * overlay[idx] + alpha * color_arr).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    return overlay

# --- SIDEBAR CONFIGURATION ---
st.sidebar.markdown("### System Configuration")
st.sidebar.caption("MobileViT v2 + clDice Loss Inference Engine")

sample_dir = os.path.join(REPO_ROOT, "data", "samples")
sample_catalog = {
    "100034_sat.jpg": "Sample 1: Rural Unpaved Arterial (ID: 100034)",
    "102408_sat.jpg": "Sample 2: Agricultural Terrain with Foliage (ID: 102408)",
    "115714_sat.jpg": "Sample 3: Dense Forest Canopy Corridor (ID: 115714)",
    "117991_sat.jpg": "Sample 4: Complex Road Junction (ID: 117991)",
}

input_mode = st.sidebar.radio("Data Acquisition Mode", ["Benchmark Satellite Tiles", "Upload Custom Tile"])

current_image_key = None
active_image_bgr = None

if input_mode == "Benchmark Satellite Tiles":
    selected_name = st.sidebar.selectbox(
        "Select Target Scene",
        options=list(sample_catalog.keys()),
        format_func=lambda k: sample_catalog[k],
        index=0
    )
    current_image_key = selected_name
    img_path = os.path.join(sample_dir, selected_name)
    if os.path.exists(img_path):
        active_image_bgr = cv2.imread(img_path)
    else:
        st.sidebar.error(f"File not found: {img_path}")
else:
    uploaded = st.sidebar.file_uploader("Upload Optical Raster (PNG, JPG, TIFF)", type=["png", "jpg", "jpeg", "tif"])
    if uploaded is not None:
        raw_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        active_image_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        current_image_key = f"upload_{uploaded.name}_{uploaded.size}"

st.sidebar.markdown("---")
st.sidebar.markdown("### Pipeline Controls")
enable_gap_bridging = st.sidebar.checkbox("Centerline Gap Bridging (Morphological Graph)", value=True)
enable_tta = st.sidebar.checkbox("Test-Time Augmentation (4-Way Invariance)", value=True)

with st.sidebar.expander("Threshold Configuration"):
    high_threshold = st.slider("High Confidence Threshold", 0.10, 0.80, 0.35, 0.05)
    low_threshold = st.slider("Hysteresis Low Threshold", 0.05, 0.40, 0.12, 0.01)
    max_gap_distance = st.slider("Max Bridge Distance (px)", 50, 300, 220, 10)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Institutional Metadata**  
Center of Excellence in AI & Remote Sensing  
Target: National Rural Road Audit (PMGSY)  
Compilation: ONNX Dynamic Runtime
""")

# --- MAIN WORKSPACE ---
# Header
st.markdown("""
<div class="institute-bar">
    <div class="institute-name">Center of Excellence in Artificial Intelligence & Remote Sensing</div>
    <div class="project-status">IEEE Research Evaluation Platform</div>
</div>
<div class="title-text">Topological Rural Road Network Extraction</div>
<div class="subtitle-text">
    High-resolution optical satellite road vectorization using MobileViT v2 linear attention, Strip Convolutions, and centerline-Dice (clDice) loss.
</div>
""", unsafe_allow_html=True)

# Tabs
tab_extraction, tab_metrics, tab_architecture, tab_application = st.tabs([
    "Live Inference & Segmentation",
    "Benchmark Evaluation",
    "Model Architecture & Novelty",
    "PMGSY Infrastructure Auditing"
])

with tab_extraction:
    if active_image_bgr is None:
        st.warning("Select or upload an optical satellite raster from the left sidebar to begin analysis.")
    else:
        # Preprocess current selected image
        input_tensor, current_resized_rgb, img_h, img_w = preprocess_image(active_image_bgr)
        
        # Metric Strip
        st.markdown("""
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-label">Model Parameters</div>
                <div class="metric-val">1.60 M</div>
                <div class="metric-sub">94.8% reduction vs U-Net (31.0M)</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">ONNX Edge Payload</div>
                <div class="metric-val">0.87 MB</div>
                <div class="metric-sub neutral">Drone & Field Laptop Ready</div>
            </div>
            <div class="metric-box accent">
                <div class="metric-label">Topological clDice</div>
                <div class="metric-val">81.62%</div>
                <div class="metric-sub">+8.42% over baseline</div>
            </div>
            <div class="metric-box accent">
                <div class="metric-label">Routing Navigability (APLS)</div>
                <div class="metric-val">76.80%</div>
                <div class="metric-sub">+9.30% over baseline</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Caching & Inference Execution
        cache_key = f"{current_image_key}_tta_{enable_tta}"
        
        if cache_key not in st.session_state['prediction_cache']:
            with st.spinner("Executing MobileViT v2 Linear Attention ONNX Inference..."):
                session, in_name, out_name = load_onnx_model()
                if session is None:
                    st.error("Model checkpoint models/mobilevit_v2.onnx could not be loaded.")
                    prob_map, duration = np.zeros((img_h, img_w), dtype=np.float32), 0.0
                else:
                    prob_map, duration = run_inference(
                        session, in_name, out_name, input_tensor, use_tta=enable_tta
                    )
                st.session_state['prediction_cache'][cache_key] = (prob_map, duration)

        prob_map, duration = st.session_state['prediction_cache'][cache_key]

        # Post-Processing
        mask_hyst = hysteresis_threshold(prob_map, high_thresh=high_threshold, low_thresh=low_threshold)
        kernel_close = np.ones((5, 5), np.uint8)
        mask_closed = cv2.morphologyEx(mask_hyst, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        
        if enable_gap_bridging:
            final_mask = connect_canopy_gaps(
                mask_closed, max_gap_dist=float(max_gap_distance), max_angle_deg=65.0, road_width=6
            )
            bridge_status = "Active (Tangent-Guided Skeleton Bridging)"
        else:
            final_mask = mask_closed
            bridge_status = "Disabled (Raw Thresholded Output)"

        # Skeleton Generation
        if HAS_SKIMAGE:
            skel = (skeletonize(final_mask > 0) * 255).astype(np.uint8)
        else:
            skel = final_mask

        # Overlay Generation
        overlay_image = generate_overlay(current_resized_rgb, final_mask, color=(0, 230, 118))

        # Status Strip
        st.markdown(f"""
        <div class="status-strip">
            <div class="status-info">
                Target: <strong>{current_image_key}</strong> &nbsp;|&nbsp; 
                Resolution: <strong>{img_w} × {img_h} px</strong> &nbsp;|&nbsp; 
                Inference Latency: <strong>{duration*1000:.1f} ms</strong> (Edge CPU) &nbsp;|&nbsp; 
                Gap Healing: <strong>{bridge_status}</strong>
            </div>
            <div class="status-badge">EXECUTION COMPLETE</div>
        </div>
        """, unsafe_allow_html=True)

        # 4-Quadrant Display
        q1, q2 = st.columns(2)
        with q1:
            st.markdown('<div class="panel-header">1. Input Optical Satellite Tile</div>', unsafe_allow_html=True)
            st.image(current_resized_rgb, caption=f"Active Scene: {current_image_key} (0.5m GSD)", use_container_width=True)
        
        with q2:
            st.markdown('<div class="panel-header">2. Extracted Road Network Overlay</div>', unsafe_allow_html=True)
            st.image(overlay_image, caption="Vectorized Road Boundaries (Green) on Optical Imagery", use_container_width=True)

        q3, q4 = st.columns(2)
        with q3:
            st.markdown('<div class="panel-header">3. Soft Probability Surface</div>', unsafe_allow_html=True)
            prob_vis = (prob_map * 255).astype(np.uint8)
            prob_colored = cv2.applyColorMap(prob_vis, cv2.COLORMAP_VIRIDIS)
            prob_colored_rgb = cv2.cvtColor(prob_colored, cv2.COLOR_BGR2RGB)
            st.image(prob_colored_rgb, caption="Continuous Confidence Map (Demonstrates corridor continuity under vegetation)", use_container_width=True)

        with q4:
            st.markdown('<div class="panel-header">4. Topological Centerline Skeleton</div>', unsafe_allow_html=True)
            st.image(skel, caption="1-Pixel Centerline Graph (Preserves Network Topology for Pathfinding)", use_container_width=True)

with tab_metrics:
    st.markdown('<div class="panel-header">Architectural and Quantitative Benchmark Comparison</div>', unsafe_allow_html=True)
    st.markdown("""
    Standard segmentation models (U-Net, DeepLabv3+) optimize pixel-wise cross-entropy, causing 
    catastrophic topological disconnection on thin rural roads. The proposed MobileViT v2 + clDice 
    architecture enforces spatial contiguity with a 94.8% parameter reduction.
    """)

    # Comparison Table
    st.markdown("""
    <table class="academic-table">
        <thead>
            <tr>
                <th>Model Architecture</th>
                <th>Backbone Type</th>
                <th>Parameters</th>
                <th>Model Size</th>
                <th>clDice (%)</th>
                <th>APLS (%)</th>
                <th>IoU (%)</th>
                <th>Latency (CPU)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Baseline U-Net (Ronneberger et al.)</td>
                <td>Standard CNN</td>
                <td>31.04 M</td>
                <td>~118.0 MB</td>
                <td>73.20%</td>
                <td>67.50%</td>
                <td>68.10%</td>
                <td>~1800 ms</td>
            </tr>
            <tr>
                <td>DeepLabv3+ (Chen et al.)</td>
                <td>ResNet-50</td>
                <td>40.20 M</td>
                <td>~155.0 MB</td>
                <td>74.80%</td>
                <td>69.10%</td>
                <td>70.40%</td>
                <td>~2400 ms</td>
            </tr>
            <tr>
                <td>D-LinkNet (Zhou et al.)</td>
                <td>ResNet-34 + Dilated</td>
                <td>21.20 M</td>
                <td>~81.0 MB</td>
                <td>78.40%</td>
                <td>73.30%</td>
                <td>72.80%</td>
                <td>~1450 ms</td>
            </tr>
            <tr class="highlight-row">
                <td><strong>MobileViT v2 + clDice (Proposed)</strong></td>
                <td><strong>Linear Attention + Strip Conv</strong></td>
                <td><strong>1.60 M</strong></td>
                <td><strong>0.87 MB</strong></td>
                <td><strong>81.62%</strong></td>
                <td><strong>76.80%</strong></td>
                <td><strong>74.60%</strong></td>
                <td><strong>~550-700 ms</strong></td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_comp1, col_comp2 = st.columns(2)
    with col_comp1:
        if os.path.exists("figures/fig8_qualitative_comparison.png"):
            st.image("figures/fig8_qualitative_comparison.png", caption="Fig 8: Qualitative Comparison across Baseline Networks", use_container_width=True)
    with col_comp2:
        if os.path.exists("figures/fig7_ablation_study.png"):
            st.image("figures/fig7_ablation_study.png", caption="Fig 7: Ablation Analysis (Strip Convolutions, Channel Shift, clDice Loss)", use_container_width=True)

with tab_architecture:
    st.markdown('<div class="panel-header">Methodology and Novel Technical Contributions</div>', unsafe_allow_html=True)
    
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.markdown("""
        **1. Separable Linear Self-Attention Backbone**  
        Standard Vision Transformers require quadratic complexity $O(N^2)$, making high-resolution satellite tiles computationally intractable. MobileViT v2 computes linear self-attention $O(N \cdot d)$ by encoding spatial patches into a scalar projection vector, maintaining global context to bridge tree occlusions.

        **2. Factorized 1D Strip Convolutions ($1\\times3 \\to 3\\times1$)**  
        Standard square convolutions ($3\\times3$) are isotropic and parameter-heavy. We factorize them into sequential 1D directional strip convolutions, introducing a strong inductive bias for elongated, tubular road corridors while reducing convolutional parameters by **33%**.
        """)
    with c_m2:
        st.markdown("""
        **3. Zero-Parameter Channel Shift Operator**  
        We displace 25% of feature channels by $\pm8$ pixels across cardinal directions prior to attention blocks. This expands the effective spatial receptive field without adding a single multiply-accumulate operation or parameter.

        **4. Differentiable clDice Loss Formulation**  
        Centerline-Dice calculates overlap directly on the soft-skeletonized prediction and ground-truth masks:
        $$\\text{clDice}(X, Y) = 2 \\times \\frac{\\text{Prec}_{cl}(X, Y) \\times \\text{Rec}_{cl}(X, Y)}{\\text{Prec}_{cl}(X, Y) + \\text{Rec}_{cl}(X, Y)}$$
        This mathematically penalizes topological breaks and fragmentation.
        """)

    st.markdown("<br>", unsafe_allow_html=True)
    if os.path.exists("figures/fig2_network_architecture.png"):
        st.image("figures/fig2_network_architecture.png", caption="Fig 2: Complete MobileViT v2 Hybrid Architecture with Strip Convolutions and Attention Gates", use_container_width=True)
        
    if os.path.exists("figures/fig5_graph_extraction_gap_bridging.png"):
        st.image("figures/fig5_graph_extraction_gap_bridging.png", caption="Fig 5: Multi-Strategy Topological Gap Healing Mechanism", use_container_width=True)

with tab_application:
    st.markdown('<div class="panel-header">PMGSY Infrastructure Auditing & Spatial Network Criticality</div>', unsafe_allow_html=True)
    st.markdown("""
    Under India's **Pradhan Mantri Gram Sadak Yojana (PMGSY)**, over 750,000 km of rural roads connect habitations to economic centers.
    Automated satellite monitoring provides three concrete capabilities:
    
    1. **Monsoon Washout and Encroachment Auditing:** Rapid comparison against baseline vector traces detects severed links and deterioration.
    2. **Autonomous Drone (UAV) Deployment:** The compiled sub-1MB ONNX payload operates locally aboard companion computers without internet access.
    3. **Spatial Graph Resilience (Betweenness Centrality):** Road vectors are converted into NetworkX planar graphs $G = (V, E)$. Nodes with extreme Betweenness Centrality are identified as critical infrastructural bottlenecks for disaster relief.
    """)

    if os.path.exists("figures/fig6_centrality_resilience.png"):
        st.image("figures/fig6_centrality_resilience.png", caption="Fig 6: Betweenness Centrality and Infrastructure Resilience Analysis", use_container_width=True)

    st.info("Technical Progress Report available at: Technical_Progress_Report_Rural_Roads.pdf (IEEE Conference Standard)")
