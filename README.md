# Lightweight MobileViT-Graph Network for Topological Rural Road Extraction

> **IAMPro-2026 Internship Project** — IEEE CS Bangalore Chapter  
> Satellite Imagery-based Rural Road Network Extraction using MobileViT v2 + SoftClDice Loss

![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)
![Deployment](https://img.shields.io/badge/Deployment-ONNX_Edge_Ready-blue)
![Parameters](https://img.shields.io/badge/Parameters-1.6M-green)

---

## Overview

India's Pradhan Mantri Gram Sadak Yojana (PMGSY) has built over 700,000 km of rural roads since 2000. Auditing connectivity and maintenance status across this network manually is prohibitively slow. Satellite imagery offers nationwide coverage, but standard segmentation models trained on urban datasets fail on rural roads — which are narrow, unpaved, and frequently hidden under tree canopies and shadows.

This project builds a 1.6 M parameter encoder-decoder that extracts rural road networks from satellite tiles with full topological post-processing (canopy gap bridging, skeletonisation, graph construction). The model runs on CPU-only edge hardware via ONNX Runtime with no PyTorch dependency.

---

## Architecture

| Component | Design choice | Reason |
|---|---|---|
| Encoder | MobileViT v2 (width mult 1.0) | Linear-cost separable self-attention; global context at 1.6 M parameters |
| Strip conv stem | 1×3 then 3×1 factorised convolutions | Road-shaped directional filters, fewer FLOPs than 3×3 |
| Channel shift | Zero-parameter 2-pixel spatial displacement | Cheap spatial context for thin structures |
| Skip connections | Attention gates | Suppresses background noise on decoder skip paths |
| Loss | SoftClDice + Dice + BCE composite | Topology precision and sensitivity via soft skeleton; annealed weights over training |
| Post-processing | Hysteresis threshold → morphological close → canopy gap bridge | Recovers connectivity broken by shadow occlusions |
| Export | ONNX with dynamic H×W, sigmoid baked in | Runs on CPU, drone boards, or mobile via ONNX Runtime Mobile |

**Parameter count (verified):** 1,604,657  
**ONNX vs PyTorch output difference (verified):** max absolute diff = 1.76 × 10⁻⁷ at 256², 256×384, and 512²

---

## Project Structure

```text
.
├── backend/
│   ├── requirements.txt            # Full training environment
│   ├── requirements-edge.txt       # Inference only (no PyTorch)
│   ├── scripts/
│   │   ├── train.py                # Training loop with collapse-gated checkpointing
│   │   ├── export_onnx.py          # Checkpoint to ONNX with parity check
│   │   ├── evaluate.py             # Held-out evaluation script
│   │   └── test_model.py           # Parameter count and forward-pass benchmark
│   └── src/
│       ├── models/mobilevit_v2.py  # MobileViT v2 encoder-decoder
│       ├── data/                   # Dataset loader and augmentations
│       └── utils/                  # Loss (clDice), metrics, graph postprocessing
│
├── scripts/
│   ├── predict_single_image.py     # PyTorch inference with full postprocessing
│   ├── predict_onnx.py             # ONNX Runtime inference (edge deployment)
│   ├── paper_figures/              # Scripts that generate all measured figures
│   └── report/                     # Scripts that build the internship report and literature doc
│
├── notebooks/
│   ├── model_training_v2.ipynb     # Kaggle training notebook
│   └── evaluate_for_paper.ipynb    # Full held-out evaluation (run on Kaggle)
│
├── models/
│   ├── best_model_v2.pth           # Current best checkpoint (epoch 53)
│   ├── mobilevit_v2.onnx           # Exported ONNX graph
│   └── archive/                    # Earlier and collapsed checkpoints
│
├── figures/real/
│   ├── deck/                       # Tile images, masks, and overlays for 4 sample tiles
│   ├── measurements/               # JSON files with all measured numbers used in the report
│   └── report/                     # Figures embedded in the internship report
│
├── data/samples/                   # 4 sample satellite tiles (1024×1024, RGB)
├── docs/
│   ├── report/                     # Internship_Report.pdf, Literature_Validation.pdf
│   ├── papers/                     # Reference papers and literature review
│   └── main.tex                    # IEEE paper draft (LaTeX)
└── demo_app.py                     # Streamlit research demonstration
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 2. Run inference on a sample tile

```bash
python scripts/predict_single_image.py data/samples/100034_sat.jpg \
       --model models/best_model_v2.pth \
       --output prediction_100034.png
```

Run on all 4 sample tiles:

```bash
for tile in data/samples/*.jpg; do
    python scripts/predict_single_image.py "$tile" \
           --model models/best_model_v2.pth \
           --output "prediction_$(basename $tile .jpg).png"
done
```

### 3. Run inference with ONNX Runtime (no PyTorch required)

```bash
pip install -r backend/requirements-edge.txt
python scripts/predict_onnx.py data/samples/100034_sat.jpg \
       --model models/mobilevit_v2.onnx \
       --output prediction_100034_onnx.png
```

This prints inference latency and road pixel fraction. Expected output on the 4 sample tiles: road fractions of 1.8%, 2.7%, 1.4%, and 11.4%.

### 4. Verify parameter count

```bash
python backend/scripts/test_model.py
# Expected: 1,604,657 parameters
```

### 5. Re-export ONNX from checkpoint

```bash
python backend/scripts/export_onnx.py \
       --checkpoint models/best_model_v2.pth \
       --output models/mobilevit_v2.onnx
# Prints PyTorch vs ONNX Runtime max absolute difference — expected ~1.8e-7
```

---

## Reproducing Report Results

All numbers in the internship report (`docs/report/Internship_Report.pdf`) are measured and traceable. The table below maps each claim to the script or file that produces it.

| Report claim | How to verify |
|---|---|
| 1,604,657 parameters | `python backend/scripts/test_model.py` |
| ONNX parity 1.76 × 10⁻⁷ | `python scripts/report/measure_onnx_parity.py` |
| Inference time per tile (CPU) | `python scripts/paper_figures/benchmark_efficiency.py` — writes `figures/real/measurements/efficiency.json` |
| Pipeline stage comparison (9→3 components, 5,076 bridged pixels on tile 117991) | `python scripts/paper_figures/local_figures.py` — writes `figures/real/measurements/local_figure_stats.json` |
| Road fraction on 4 sample tiles | `python scripts/predict_onnx.py` on each tile in `data/samples/` — reference values in `figures/real/deck/overlay_stats.json` |
| Qualitative prediction figures | `python scripts/paper_figures/local_figures.py` — produces `figures/real/fig_pipeline_stages.png`, `fig_collapse.png`, `fig_resolution.png`, `fig_efficiency.png` |
| Report figures (architecture, outputs) | `python scripts/report/report_figures.py` — produces `figures/real/report/` |

Pre-measured JSON results are already committed in `figures/real/measurements/` so the report can be rebuilt without re-running the benchmarks.

**Note on accuracy metrics (IoU, clDice, APLS):** Full accuracy evaluation against ground truth requires the DeepGlobe held-out validation set and a GPU. The evaluation notebook is `notebooks/evaluate_for_paper.ipynb` — upload it to Kaggle, attach `models/best_model_v2.pth`, and run all cells. This has not been run yet; no accuracy numbers are claimed in the current report.

---

## Rebuild the Report

Requires Microsoft Word (for PDF export via COM automation).

```bash
python scripts/report/build_report.py
powershell -ExecutionPolicy Bypass -File scripts/report/export_pdf.ps1
# Output: docs/report/Internship_Report.docx and .pdf
```

```bash
python scripts/report/build_validation_doc.py
powershell -ExecutionPolicy Bypass -File scripts/report/export_pdf.ps1 \
    -Docx docs/report/Literature_Validation.docx \
    -Pdf  docs/report/Literature_Validation.pdf
```

---

## Training

Training was run on Kaggle (GPU P100, 16 GB). The committed notebook is `notebooks/model_training_v2.ipynb`.

Key hyperparameters: Adam, lr=3×10⁻⁴, batch size 8, 53 epochs, composite loss (annealed BCE + Dice + SoftClDice), 4-flip test-time augmentation, patience 35.

To retrain locally (requires GPU and the DeepGlobe dataset):

```bash
python backend/scripts/train.py \
       --data /path/to/deepglobe \
       --checkpoint models/best_model_v2.pth \
       --output models/retrained.pth
```

---

## Evaluation Metrics

| Metric | Type | Purpose |
|---|---|---|
| clDice | Topological | Skeleton intersection and centerline connectivity |
| APLS | Graph / Routing | Path-length similarity on the road graph |
| IoU / F1 | Pixel-level | Spatial segmentation accuracy |
| Inference latency | Deployment | Edge-device real-world speed |

---

## References

- MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer (Mehta & Rastegari, 2021)
- clDice: A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation (Shit et al., 2021)
- DeepGlobe Road Extraction Challenge (Demir et al., 2018)
- Pradhan Mantri Gram Sadak Yojana — Ministry of Rural Development, Government of India
