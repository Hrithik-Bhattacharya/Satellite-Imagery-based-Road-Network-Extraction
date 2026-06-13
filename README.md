# Lightweight MobileViT-Graph Network for Topological Rural Road Extraction

> **COE Research Project** — Satellite Imagery-based Rural Road Network Extraction using MobileViT v2 + clDice Loss

---

## Overview

In developing nations like India, mapping millions of kilometers of rural, unpaved roads under initiatives such as the **Pradhan Mantri Gram Sadak Yojana (PMGSY)** is critical — yet current manual auditing workflows are prohibitively slow and labor-intensive.

Standard deep learning segmentation models trained on urban datasets exhibit severe **Domain Shift** when deployed in rural environments, where roads are slender, irregular, and frequently occluded by tree canopies, shadows, and complex terrain.

### Core Objective

Construct an ultra-lightweight **encoder-decoder architecture** using **MobileViT v2** as a fast, low-parameter backbone, synthesized with graph-theoretic **clDice (Centerline-Dice) Loss** to enforce contiguous, navigable road predictions on Indian rural topographies — optimized for edge-device deployment.

---

## Project Structure

```
.
├── backend/                        # All ML source code and scripts
│   ├── src/
│   │   ├── models/                 # Member 1: Model architectures
│   │   │   ├── __init__.py
│   │   │   ├── unet_baseline.py    # Phase 1 baseline (31M params)
│   │   │   └── mobilevit_v2.py     # Phase 2 core model (478K params)
│   │   ├── data/                   # Member 3: Data pipeline
│   │   │   ├── dataset.py          # DeepGlobe dataset loader
│   │   │   ├── augmentations.py    # Shadow, canopy, jitter augmentations
│   │   │   ├── weak_labels.py      # OSM centerline → raster pipeline
│   │   │   └── test_run.py
│   │   └── utils/                  # Shared utilities (loss, metrics — Member 2 & 4)
│   ├── scripts/
│   │   ├── test_model.py           # Forward pass + parameter benchmark
│   │   └── export_onnx.py          # Edge-device ONNX export
│   ├── outputs/
│   │   └── mobilevit_v2.onnx       # Compiled edge model (0.38 MB)
│   └── requirements.txt
│
├── frontend/                       # Visualization / demo UI (in progress)
│
├── docs/
│   ├── papers/                     # Research papers, literature review
│   ├── planning/                   # Architecture diagrams
│   └── proposal/                   # Project proposal & literature survey
│
├── tools/                          # AI research tooling (AI-Researcher-AV1)
├── .gitignore
└── README.md
```

---

## Technical Approach

### 1. Backbone Encoder — MobileViT v2
Replaces the $O(N^2)$ computational complexity of standard Vision Transformers with **localized linear self-attention**, maintaining a global receptive field to "see" past occlusions where CNNs fail — while keeping the parameter footprint ultra-low.

| Model | Parameters | Size |
|---|---|---|
| Baseline U-Net | 31,037,633 | Heavy |
| **MobileViT v2 (ours)** | **478,849** | **Ultra-lightweight** |

### 2. Loss Function — clDice (Centerline-Dice)
Replaces standard pixel-wise IoU/BCE losses with a **graph-theoretic topology-preserving loss** that computes overlap explicitly on the morphological skeleton of predicted and ground-truth road centerlines:

```
clDice(X, Y) = 2 × ½(Prec_cl(X, Y) + Rec_cl(X, Y))
```

This penalizes topological disconnections — enforcing that predicted roads form navigable, continuous paths rather than fragmented pixel blobs.

### 3. Weak Supervision — OSM Centerlines
To eliminate manual annotation bottlenecks, training uses **1-pixel-wide vector line-strings** extracted from OpenStreetMap (OSM) as weak labels, which clDice iteratively expands into precise road boundary masks.

---

## Team Roles

| Member | Role | Responsibilities | Status |
|---|---|---|---|
| **Member 1** | Lead Architect | MobileViT v2 encoder-decoder, ONNX export, parameter optimization | ✅ Complete |
| **Member 2** | Graph & Loss Engineer | clDice loss, morphological skeletonization, topology constraints | 🔄 In progress |
| **Member 3** | Data & Pipeline Engineer | DeepGlobe ingestion, augmentations, OSM weak supervision pipeline | ✅ Complete |
| **Member 4** | Validation & MLOps | APLS metrics, ablation studies, W&B tracking, benchmarking | 🔄 In progress |

---

## 6-Week Execution Roadmap

| Phase | Weeks | Focus |
|---|---|---|
| **Phase 1** | Week 1 | Environment setup, baseline U-Net, DeepGlobe pipeline, W&B dashboard |
| **Phase 2** | Weeks 2–3 | MobileViT v2 encoder, clDice loss, OSM weak annotation, APLS library |
| **Phase 3** | Week 4 | Full pipeline integration, training sweeps, advanced augmentations |
| **Phase 4** | Weeks 5–6 | ONNX export, ablation studies, domain adaptation validation, final docs |

---

## Quickstart

### Prerequisites
```bash
cd backend
pip install -r requirements.txt
```

### Run Model Verification
```bash
cd backend
python scripts/test_model.py
```

Expected output:
```
Detected Device: mps / cuda / cpu
U-Net Parameters:        31,037,633
MobileViT v2 Parameters: 478,849
Forward pass successful!
Output Shape: torch.Size([4, 1, 256, 256])
✅ SUCCESS
```

### Export to ONNX (Edge Deployment)
```bash
cd backend
python scripts/export_onnx.py
# → outputs/mobilevit_v2.onnx (0.38 MB)
```

---

## Dataset

- **Primary:** [DeepGlobe Road Extraction Dataset](http://deepglobe.org/)
- **Weak Labels:** OpenStreetMap (OSM) vector centerlines rasterized to 1-pixel line strings
- **Domain Adaptation Target:** Indian rural satellite imagery (PMGSY corridors)

---

## Evaluation Metrics

| Metric | Type | Purpose |
|---|---|---|
| IoU / F1 | Pixel-level | Baseline spatial accuracy |
| **APLS** | Graph / routing | Navigability & path-length similarity |
| **TOPO** | Topological | Connectivity and loop preservation |
| FPS / Latency | Deployment | Edge-device real-world performance |

---

## Branches

| Branch | Owner | Description |
|---|---|---|
| `main` | Team | Stable, merged releases |
| `arya` | Member 1 | Lead architecture — MobileViT v2, ONNX export |
| `dilraj` | Member 3 | Data pipeline — DeepGlobe, OSM, augmentations |

---

## References

- MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer (Mehta & Rastegari, 2021)
- clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation (Shit et al., 2021)
- DeepGlobe Road Extraction Challenge (Demir et al., 2018)
- PMGSY — Pradhan Mantri Gram Sadak Yojana, Government of India
