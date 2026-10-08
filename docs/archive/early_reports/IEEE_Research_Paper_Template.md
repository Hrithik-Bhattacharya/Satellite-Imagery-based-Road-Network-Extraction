# [PAPER TITLE — e.g., "A Lightweight MobileViT-Graph Hybrid Network for Topology-Preserving Rural Road Extraction from Satellite Imagery"]

**[Author Name(s)]¹, [Co-Author Name(s)]²**
¹*[Department, Institution/COE Name, City, Country]*
²*[Department, Institution/COE Name, City, Country]*
*[email1@domain.com, email2@domain.com]*

---

## Abstract

> *[150–250 words. Structure: (1) Problem statement — manual PMGSY rural road auditing is labor-intensive; standard segmentation models fail under occlusion/domain shift. (2) Gap — pixel-wise models ignore topology; heavyweight topology-aware models are not edge-deployable. (3) Proposed solution — one-sentence summary of MobileViT_v2 + StripConv + ChannelShift + SoftClDice + graph post-processing. (4) Key quantitative results — parameter count, IoU, clDice score, APLS/TOPO scores, inference latency. (5) One-line significance/impact statement.]*

**Keywords** — *Rural Road Extraction, MobileViT, Semantic Segmentation, clDice Loss, Topological Graph Extraction, Edge Deployment, ONNX, PMGSY, Remote Sensing, Vision Transformer*

---

## I. Introduction

### A. Background and Motivation
*[Context: importance of rural connectivity, PMGSY scale, cost of manual surveys.]*

### B. Problem Statement
*[Formal statement of the road-extraction-under-occlusion problem; define domain shift, slender geometry, occlusion vulnerability, compute bottleneck as four sub-problems.]*

### C. Limitations of Existing Approaches
*[Brief 1-paragraph preview of the three failure classes: pixel-level baselines (disconnected output), topology-preserving heavyweights (compute bottleneck), lightweight edge models (topological failure) — to be expanded in Section II.]*

### D. Proposed Contributions
*[Bulleted list, e.g.:]*
- A sub-2M-parameter hybrid CNN–Transformer encoder-decoder (MobileViT_v2) with two novel zero/low-cost inductive-prior modules: Strip Convolutions and Channel Shift.
- A differentiable topology-preserving loss (SoftClDice) with dynamic alpha scheduling for stable convergence.
- A canopy-resilient augmentation strategy and graph-based post-processing pipeline for gap bridging.
- A graph-theoretic criticality/resilience analysis layer (Betweenness Centrality, Resilience Index) with an interactive dashboard.
- Quantized ONNX edge deployment validated across multiple execution providers.

### E. Paper Organization
*[One paragraph mapping Sections II–VIII.]*

---

## II. Related Work / Literature Review

### A. Pixel-Level Semantic Segmentation Baselines
*[U-Net (Ronneberger et al., 2015), DeepLabv3 (Chen et al., 2017) — strengths, BCE/Dice limitations, fragmentation under occlusion.]*

### B. Topology-Aware Heavyweight Networks
*[TopoRF-Net and similar — connectivity enforcement mechanisms, compute/memory cost, edge-infeasibility.]*

### C. Lightweight Edge-Oriented Networks
*[HPLNet (2025) and similar — parameter compression strategies, loss of global connectivity reasoning.]*

### D. Vision Transformers for Mobile/Edge Vision
*[MobileViT (Mehta & Rastegari, 2022) — hybrid CNN-ViT rationale, why selected as backbone.]*

### E. Topology-Preserving Loss Functions
*[clDice (Shit et al., 2021) — origin in vascular segmentation, analogy to road networks, differentiable skeletonization.]*

### F. Spatial Network Analysis & Graph Theory
*[Crucitti et al. (2006) — centrality measures, spatial network robustness, adaptation to road-network resilience.]*

### G. Research Gap Summary
*[Table I — see below]*

**Table I. Comparative Summary of Existing Approaches vs. Proposed Method**

| Model / Approach | Connectivity-Aware | Parameter Count | Edge-Deployable | Occlusion-Robust | Graph/Routing Output |
|---|---|---|---|---|---|
| U-Net (baseline) | ✗ | ~31 M | ✗ | ✗ | ✗ |
| DeepLabv3+ | ✗ | *[fill]* | ✗ | ✗ | ✗ |
| TopoRF-Net | ✓ | *[fill — high]* | ✗ | Partial | ✗ |
| HPLNet (2025) | ✗ | *[fill — low]* | ✓ | Partial | ✗ |
| **Proposed (MobileViT-Graph)** | **✓** | **~1.6 M** | **✓** | **✓** | **✓** |

---

## III. Proposed Methodology

### A. System Overview
*[High-level pipeline diagram description: Input imagery → Preprocessing/Augmentation → MobileViT_v2 Encoder-Decoder → SoftClDice-optimized mask → Graph Post-Processing → NetworkX Graph → Resilience Analysis Dashboard.]*

**Fig. 1.** *[Insert end-to-end system architecture / block diagram here.]*

### B. Network Architecture: MobileViT_v2
1. **Stem — Strip Convolution ("Road-Shaped Scanner")**
   *[Describe 1×3 → 3×1 factorization, parameter savings, inductive prior rationale.]*
2. **Encoder Stages 1–3 — Inverted Residual + MobileViTv2 Blocks**
   *[Describe MobileNetV2 bottlenecks, ChannelShift placement, Linear Self-Attention.]*
3. **Bottleneck — Deep MobileViTv2 Block**
   *[3 transformer layers, global context aggregation.]*
4. **Decoder — Attention-Gated Skip Connections + Strip Convolutions**
   *[Bilinear upsampling, AttentionGate suppression of background clutter.]*
5. **Output Head**
   *[Sigmoid road-probability mask, resolution.]*

**Fig. 2.** *[Insert detailed layer-by-layer architecture schematic — the ASCII diagram from your project description, redrawn as a formal figure.]*

### C. Novel Module 1: Strip Convolutions
*[Mathematical formulation, FLOP/parameter reduction (~33%), qualitative justification for tubular structures.]*

### D. Novel Module 2: Channel Shift
*[torch.roll formulation, 25% channel shift in 4 cardinal directions, ±8 px receptive field expansion, zero-parameter cost — include equation.]*

### E. Attention-Gated Skip Connections
*[Additive gating formulation $g + s$, purpose of clutter suppression.]*

### F. Linear Self-Attention Formulation
*[Equations: $\alpha = \text{Softmax}(K)$, $c = \sum \alpha_n V_n$, $O = \text{ReLU}(Q) \odot c$; complexity comparison $O(N^2)$ vs. $O(N \cdot d)$.]*

---

## IV. Topology-Preserving Loss Function

### A. Motivation: Why Pixel Accuracy Is Insufficient
*[The "95% accurate but broken every 50 m" argument.]*

### B. Differentiable Soft Skeletonization
*[soft_erode, soft_dilate, soft_skel — min/max pooling approximation; equation for $T_{prec}$ and $T_{sens}$.]*

### C. clDice Loss Formulation
$$\mathcal{L}_{\text{clDice}} = 1 - 2 \cdot \frac{T_{\text{prec}} \cdot T_{\text{sens}}}{T_{\text{prec}} + T_{\text{sens}}}$$

### D. Composite Loss and Dynamic Alpha Scheduling
$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{BCE}} + \beta \cdot \mathcal{L}_{\text{SoftDice}} + (1 - \alpha - \beta) \cdot \mathcal{L}_{\text{clDice}}$$
*[Describe power-curve decay of α from 0.50 → 0.15 (or revised floor 0.4), and the empirical justification for the revision — BCE collapse observation.]*

### E. Collapse-Prevention Safeguards
*[pos_weight=2.0 calibration; IoU checkpoint hard-gating with MAX_POS_FRAC=0.15; CosineAnnealingLR.]*

**Fig. 3.** *[Insert training curve — Loss vs. Epoch, showing BCE/clDice/IoU trajectories and the α decay schedule overlay.]*

---

## V. Data Engineering & Weak Supervision

### A. Dataset Description
*[DeepGlobe road dataset — source, tile size, split ratios. State resolution/sensor details for any additional imagery: Cartosat, LISS-IV, Sentinel-2.]*

### B. OSM-Based Weak Label Generation
*[Overpass API ingestion, rasterio/geopandas centerline burning, alignment to GeoTIFFs, skeletonization simulation.]*

### C. Canopy-Resilient Data Augmentation
*[CanopyShadowDropout — green-tint, Gaussian blur, luminance reduction; Albumentations pipeline for monsoon cloud cover, shadow, canopy occlusion simulation.]*

**Fig. 4.** *[Insert sample augmented tile grid — original vs. shadow-dropout-augmented vs. ground truth mask.]*

### D. Training Configuration
**Table II. Training Hyperparameters**

| Parameter | Value |
|---|---|
| Input resolution | 256 × 256 |
| Batch size | *[fill]* |
| Optimizer | AdamW |
| LR scheduler | CosineAnnealingLR |
| Epochs | 50 |
| pos_weight | 2.0 |
| α range | 0.50 → 0.15 (floor revised to 0.4) |
| MAX_POS_FRAC | 0.15 |
| Precision | AMP (mixed precision) |
| Hardware | *[fill — GPU model, Kaggle/local]* |

---

## VI. Graph Construction & Topological Post-Processing

### A. Hysteresis Thresholding
*[High=0.35 / Low=0.12 dual-threshold rationale for faint dirt-road preservation.]*

### B. Skeleton Endpoint Analysis
*[Degree-1 node detection, outward tangent vector computation.]*

### C. Canopy Gap Bridging
1. **Strategy A — Facing Endpoint Bridging** *[collinear pairs, ≤220 px gap, across-canopy connection.]*
2. **Strategy B — T-Junction Completion** *[trajectory extension to perpendicular trunk roads.]*

### D. Topological Healing via MST / Disjoint Sets
*[Minimum Spanning Tree + Disjoint Set safety net; Euclidean distance + angular alignment criteria for closed-loop enforcement.]*

### E. Vector Graph Construction
*[NetworkX node/edge schema: terminal endpoints (deg=1), junctions (deg≥3); edge attributes (pixel path, Euclidean length); simplify_graph pruning of spurious branches.]*

**Fig. 5.** *[Insert before/after graph overlay — raw predicted mask vs. gap-bridged, pruned topological graph.]*

---

## VII. Criticality & Resilience Analysis

### A. Betweenness Centrality Computation
*[Formal definition, application to identifying "Gatekeeper Nodes."]*

### B. Node Ablation Stress Testing
*[Simulated failure of high-centrality nodes; recomputation of global network efficiency.]*

### C. The Resilience Index (R)
*[Definition/formula for R; interpretation of high vs. low resilience.]*

### D. Interactive Dashboard (React.js)
*[UI description: node ablation controls, before/after efficiency visualization, use case for urban/rural planners.]*

**Fig. 6.** *[Insert dashboard screenshot showing pre- and post-ablation network state and Resilience Index readout.]*

---

## VIII. Edge Deployment Pipeline

### A. ONNX Export
*[Sigmoid baked into graph; dynamic axes for arbitrary tile size; PyTorch↔ONNX Runtime parity verification methodology.]*

### B. On-Device Inference Runtime
*[Dependency-free runtime — ONNX Runtime + OpenCV + NumPy + SciPy; supported execution providers: CUDA, TensorRT, CoreML, NNAPI, CPU.]*

### C. Test-Time Augmentation & Inference Optimization
*[4-way flip TTA, morphological gap-bridging at inference time.]*

**Table III. Model Footprint & Latency Comparison**

| Model | Parameters | Model Size (PyTorch) | Model Size (ONNX) | Inference Latency | FPS |
|---|---|---|---|---|---|
| Baseline U-Net | 31,037,633 | ~118.4 MB | *[fill]* | *[fill]* | *[fill]* |
| Proposed MobileViT_v2 (full) | 1,604,657 | ~1.8 MB | ~0.8 MB | *[fill]* | *[fill]* |
| Proposed MobileViT_v2 (width_mult=0.5) | ~0.48 M | *[fill]* | *[fill]* | *[fill]* | *[fill]* |

---

## IX. Experimental Results & Discussion

### A. Evaluation Protocol
*[Train/val/test split, hardware used for benchmarking, number of runs/seeds if applicable.]*

### B. Pixel-Level Segmentation Metrics
**Table IV. Pixel-Level Performance Comparison**

| Model | IoU | Precision | Recall | F1 | Dice |
|---|---|---|---|---|---|
| U-Net (baseline) | *[fill]* | *[fill]* | *[fill]* | *[fill]* | *[fill]* |
| DeepLabv3+ | *[fill]* | *[fill]* | *[fill]* | *[fill]* | *[fill]* |
| Proposed (BCE-only ablation) | *[fill]* | *[fill]* | *[fill]* | *[fill]* | *[fill]* |
| Proposed (Full RoadExtractionLoss) | *[fill]* | *[fill]* | *[fill]* | *[fill]* | **0.8162 (clDice)** |

### C. Topological & Graph-Routing Metrics
**Table V. APLS & TOPO Comparison**

| Model | APLS | TOPO Precision | TOPO Recall | TOPO F1 |
|---|---|---|---|---|
| U-Net (baseline) | *[fill]* | *[fill]* | *[fill]* | *[fill]* |
| Proposed Method | *[fill]* | *[fill]* | *[fill]* | *[fill]* |

### D. Ablation Study
**Table VI. 4-Stage Ablation Results**

| Configuration | IoU | clDice | Params | Notes |
|---|---|---|---|---|
| (1) Baseline U-Net | *[fill]* | *[fill]* | 31 M | No topology loss |
| (2) MobileViT_v2 + BCE only | *[fill]* | *[fill]* | 1.6 M | No topology loss |
| (3) MobileViT_v2 + clDice | *[fill]* | *[fill]* | 1.6 M | No OSM weak labels |
| (4) MobileViT_v2 + clDice + OSM weak labels (Full) | *[fill]* | 0.8162 | 1.6 M | Complete pipeline |

**Fig. 7.** *[Insert bar chart comparing ablation stages across IoU and clDice.]*

### E. Qualitative Results
**Fig. 8.** *[Insert side-by-side qualitative panel: (a) input satellite tile, (b) ground truth, (c) U-Net prediction (fragmented), (d) proposed method prediction (connected), across canopy-occluded and clear samples.]*

### F. Failure Case Analysis
*[Discuss the observed BCE collapse (Val Loss 0.7018) under aggressive α decay, and how the fix (α floor 0.4, spatial dropout, CosineAnnealingLR) resolved it. Discuss remaining failure modes — e.g., very wide gaps beyond 220 px, ambiguous junctions.]*

### G. Discussion
*[Interpret results in light of Section II's research gap: does the method close the "Speed vs. Topology" bottleneck? Compare footprint/latency vs. connectivity trade-off against TopoRF-Net and HPLNet qualitatively if quantitative numbers are unavailable.]*

---

## X. Real-World Application: PMGSY Auditing Use Case
*[Optional but recommended section — describe a concrete workflow: drone/satellite tile ingestion → inference → graph → resilience dashboard → auditor decision support. Include a case study region if available.]*

---

## XI. Conclusion and Future Work

### A. Summary of Contributions
*[Recap architecture, loss, post-processing, deployment, and resilience-analysis contributions with headline numbers.]*

### B. Limitations
*[Dataset generalizability, weak-label noise, gap-bridging distance limits, dependency on OSM coverage in remote areas.]*

### C. Future Work
*[E.g., multi-temporal change detection, transformer-based graph refinement (GNNs), semi-supervised domain adaptation, larger-scale PMGSY field validation, mobile app integration.]*

---

## Acknowledgment
*[Funding body, COE, collaborators, dataset providers to be credited.]*

---

## References

*[IEEE numbered format, cited in-text as [1], [2], etc. Populate with full details; examples below show correct IEEE style — replace with verified bibliographic data.]*

[1] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in *Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Intervent. (MICCAI)*, 2015, pp. 234–241.

[2] L.-C. Chen, G. Papandreou, F. Schroff, and H. Adam, "Rethinking atrous convolution for semantic image segmentation," *arXiv preprint arXiv:1706.05587*, 2017.

[3] S. Mehta and M. Rastegari, "MobileViT: Light-weight, general-purpose, and mobile-friendly vision transformer," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2022.

[4] S. Shit *et al.*, "clDice — A novel topology-preserving loss function for tubular structure segmentation," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2021, pp. 16560–16569.

[5] P. Crucitti, V. Latora, and S. Porta, "Centrality measures in spatial networks of urban streets," *Phys. Rev. E*, vol. 73, no. 3, p. 036125, 2006.

[6] *[TopoRF-Net citation — fill]*

[7] *[HPLNet (2025) citation — fill]*

[8] *[DeepGlobe Road Extraction Dataset citation — fill]*

[9] *[PMGSY / Government of India rural roads program reference — fill]*

[10] *[Additional references — fill]*

---

## Appendix (Optional)

### A. Full Repository / Code Availability Statement
*[Link to GitHub repo, license, reproducibility statement.]*

### B. Extended Hyperparameter Tables
*[Any additional configuration details not shown in Table II.]*

### C. Additional Qualitative Samples
*[Extra figure panel if journal allows supplementary material.]*

---

### Notes for Filling This Template
- Replace every *[fill]* / *[Insert...]* placeholder with your measured data, figures, and citations.
- IEEE conference papers are typically double-column, 10pt Times New Roman, using the official IEEE conference LaTeX/Word template (`IEEEtran.cls` or `IEEEtran.dotx`) — this Markdown file mirrors that section structure so you can paste content directly into the official template.
- Figures should be numbered sequentially and referenced in-text (e.g., "as shown in Fig. 3").
- Tables should be numbered with Roman numerals per IEEE convention (Table I, II, III...) and referenced in-text.
- Keep the Abstract self-contained — a reader should understand your contribution and results without reading further.
