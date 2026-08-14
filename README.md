# Lightweight MobileViT-Graph Network for Topological Rural Road Extraction

> **COE Research Project** — Satellite Imagery-based Rural Road Network Extraction using MobileViT v2 + clDice Loss

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)
![Deployment](https://img.shields.io/badge/Deployment-Edge_Ready-blue)

---

## Overview

In developing nations like India, mapping millions of kilometers of rural, unpaved roads under initiatives such as the **Pradhan Mantri Gram Sadak Yojana (PMGSY)** is critical. However, current manual auditing workflows are prohibitively slow and labor-intensive.

Standard deep learning segmentation models trained on urban datasets exhibit severe **Domain Shift** when deployed in rural environments. Rural roads are slender, irregular, and frequently occluded by **tree canopies, shadows, and complex terrain**.

### Core Objective
Construct an ultra-lightweight **encoder-decoder architecture** using **MobileViT v2** as a fast, low-parameter backbone. By synthesizing this with the graph-theoretic **clDice (Centerline-Dice) Loss**, we enforce contiguous, navigable road predictions on Indian rural topographies — all while remaining optimized for edge-device deployment.

---

## Key Features & Technical Approach

### 1. Backbone Encoder — MobileViT v2
Replaces the $O(N^2)$ computational complexity of standard Vision Transformers with **localized linear self-attention**. It maintains a global receptive field to "see" past occlusions where CNNs fail, while keeping the parameter footprint ultra-low.

| Model | Parameters | Size |
|---|---|---|
| Baseline U-Net | 31,037,633 | Heavy |
| **MobileViT v2 (ours)** | **1,604,657** | **Ultra-lightweight (1.6M)** |

### 2. Topology-Preserving Loss — clDice
Replaces standard pixel-wise IoU/BCE losses with a **graph-theoretic topology-preserving loss**. It computes overlap explicitly on the morphological skeleton of predicted and ground-truth road centerlines:
```math
clDice(X, Y) = 2 \times \frac{Prec_{cl}(X, Y) \times Rec_{cl}(X, Y)}{Prec_{cl}(X, Y) + Rec_{cl}(X, Y)}
```
This mathematically penalizes topological disconnections, enforcing that predicted roads form continuous, navigable paths rather than fragmented pixel blobs.

### 3. Advanced Canopy Resilience & Post-Processing
To combat the severe domain shift of rural environments:
- **Simulated Occlusions:** The training pipeline utilizes `CoarseDropout`, `ColorJitter`, and shadow augmentations to mathematically force the model to bridge hidden roads underneath thick tree canopies.
- **Aggressive Gap Bridging:** Our inference scripts utilize computer vision morphology (`cv2.MORPH_CLOSE` and `cv2.MORPH_OPEN`) to dynamically heal broken segments and remove isolated noise blobs during real-time extraction.

---

## Project Structure

```text
.
├── notebooks/                      # Jupyter Notebooks for training and exploration
│   ├── model_training_v2.ipynb     # Latest Kaggle training pipeline (Canopy resilient)
│   └── kaggle_runner.ipynb         # Cloud execution configurations
│
├── scripts/                        # Standalone execution scripts
│   ├── predict_single_image.py     # PyTorch high-res inference with topological post-processing
│   └── predict_onnx.py             # On-device inference via ONNX Runtime (no PyTorch/CUDA needed)
│
├── backend/                        # ML source code, training, and deployment tooling
│   ├── requirements.txt            # Full training/dev environment
│   ├── requirements-edge.txt       # Minimal on-device inference environment (for predict_onnx.py)
│   ├── scripts/
│   │   ├── train.py                # Training loop (IoU-based, collapse-gated checkpointing)
│   │   ├── export_onnx.py          # Trained checkpoint -> ONNX, with parity verification
│   │   ├── evaluate.py             # Held-out evaluation
│   │   └── test_model.py           # Forward pass + parameter benchmark
│   └── src/
│       ├── models/                 # MobileViT v2 and U-Net architectures
│       ├── data/                   # Data ingestion and weak label processing
│       └── utils/                  # Loss functions (clDice, BCE), metrics, postprocessing
│
├── models/                         # Trained checkpoints (best_model_new.pth, best_model_v2.pth)
│   └── archive/                    # Retired/collapsed checkpoints kept for reference
├── data/samples/                   # High-resolution satellite testing images
├── results/                        # Generated output visualizations
├── docs/                           # Research papers, progress reports, proposals
└── README.md
```

---

## Quickstart

### 1. Prerequisites
Clone the repository and install the required dependencies:
```bash
pip install -r backend/requirements.txt
```

### 2. Run Inference on a Satellite Image
You can test the topology-aware extraction on any satellite image. The script automatically handles scale mismatches and applies aggressive morphological gap-closing.
```bash
python scripts/predict_single_image.py data/samples/100034_sat.jpg \
       --model models/best_model_new.pth \
       --output results/prediction_100034.png
```

### 3. Export to ONNX (Edge Deployment)
Compile a trained checkpoint into an ONNX graph for deployment on edge devices like drones or
mobile mappers. Sigmoid is baked into the graph, so ONNX Runtime output is already a `[0, 1]`
road-probability map, and height/width are dynamic (the model isn't limited to 256x256 tiles):
```bash
python backend/scripts/export_onnx.py \
       --checkpoint models/best_model_new.pth \
       --output models/mobilevit_v2.onnx
# → models/mobilevit_v2.onnx (~0.8 MB), with a PyTorch <-> ONNX Runtime parity check printed at the end
```

### 4. Run Inference On-Device (no PyTorch required)
`scripts/predict_onnx.py` is the on-device reference implementation: ONNX Runtime + OpenCV +
NumPy + SciPy only (see `backend/requirements-edge.txt`) -- no PyTorch/CUDA/albumentations, so it's
what actually ships to a drone companion computer, Jetson-class board, or field laptop. Same
hysteresis-threshold + canopy-gap-bridging postprocessing as `predict_single_image.py`:
```bash
pip install -r backend/requirements-edge.txt   # on the target device
python scripts/predict_onnx.py data/samples/100034_sat.jpg \
       --model models/mobilevit_v2.onnx \
       --output results/prediction_100034_onnx.png
```
It auto-picks the best available ONNX Runtime execution provider (CUDA/TensorRT on a Jetson,
CoreML on Apple hardware, NNAPI on Android, else CPU) and prints inference latency and the
positive-pixel-fraction canary so a collapsed/miscalibrated export is obvious immediately rather
than silently shipping. Measured on a 1024x1024 tile on CPU: ~550-700 ms/tile. For a native
Android/iOS app rather than a Python-capable edge board, port `preprocess()`/`postprocess()` from
this script to Kotlin/Swift against the platform's ONNX Runtime Mobile package -- the `.onnx` file
itself is unchanged either way.

---

## Evaluation Metrics

| Metric | Type | Purpose |
|---|---|---|
| **clDice** | Topological | Skeleton intersection & centerline connectivity |
| **APLS** | Graph / Routing | Navigability & path-length similarity |
| **IoU / F1** | Pixel-level | Baseline spatial segmentation accuracy |
| **FPS / Latency** | Deployment | Edge-device real-world inference speed |

---

## References

- **MobileViT:** Light-weight, General-purpose, and Mobile-friendly Vision Transformer (Mehta & Rastegari, 2021)
- **clDice:** A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation (Shit et al., 2021)
- **DeepGlobe:** Road Extraction Challenge (Demir et al., 2018)
- **PMGSY:** Pradhan Mantri Gram Sadak Yojana, Government of India
