# Lightweight Topology Aware MobileViT Network for Rural Road Extraction

A road extraction network with **1.60 million parameters** that finds rural roads in 0.5 m satellite
images, keeps them connected, and runs on an ordinary laptop processor through ONNX Runtime
without PyTorch or a graphics card.

![Parameters](https://img.shields.io/badge/Parameters-1.60M-green)
![Test IoU](https://img.shields.io/badge/Test%20IoU-61.1%25-blue)
![Deployment](https://img.shields.io/badge/Deployment-ONNX%20Runtime-lightgrey)

---

## Results (current model: v4)

Measured on **623 DeepGlobe test tiles** at full 1024 x 1024 resolution. These tiles were never
used for training or for choosing the best epoch, and the evaluation notebook checks this before
reporting (it prints `SPLIT VERIFIED`).

| Inference setting | IoU | F1 | Relaxed F1 | clDice | Road parts per tile |
|---|---|---|---|---|---|
| Single pass, threshold 0.5 | 59.8 | 74.8 | 85.9 | 84.2 | 17 |
| **Four flip TTA, threshold 0.5** | **61.1** | **75.8** | **86.7** | **84.5** | 13 |
| Four flip TTA + hysteresis + gap bridging | 52.6 | 69.0 | 80.7 | 75.8 | **2** (ground truth: 2) |

All values in %. IoU 95% bootstrap interval for the best setting: 60.0 to 62.1.

* **Roads under tree cover:** recall 60.3% (open road 80.3%); the full postprocessing raises it to 75.1%.
* **Speed:** 1.27 s per 1024 x 1024 tile with ONNX Runtime on an Intel Core i5-1035G4 (4 threads).
* **Size:** 6.2 MB ONNX file. ONNX and PyTorch outputs differ by at most 1.2 x 10^-7.
* **Training recipe matters:** v4 beats the earlier v3 model on 90% of the test tiles
  (IoU 61.1% against 52.7%, Wilcoxon p ~ 10^-87).

Every number above comes from a committed file: `figures/real/v4/paper_results/results.json`,
`figures/real/measurements/onnx_latency_v4.json`, and `figures/real/v4/paper_results/onnx/onnx_export.json`.

---

## Method in short

| Part | Choice |
|---|---|
| Encoder | MobileViT v2 blocks (separable self attention, cost linear in image size) |
| Road friendly blocks | Strip convolutions (1x3 then 3x1) and channel shift, both taken from HPLNet (Cui et al., 2025) |
| Skip connections | Attention gates |
| Loss | Binary cross entropy + Dice + clDice; the clDice weight grows during training |
| Collapse guard | Epochs that mark more than 20% of pixels as road are never saved |
| Postprocessing | Hysteresis threshold (0.35 / 0.12), 5x5 closing, gap bridging of road ends up to 220 px apart |
| Graph analysis | Skeleton to graph, betweenness centrality, resilience index |

v4 training: 160 epochs, 512 x 512 crops, flips, 90 degree rotations, small affine changes, colour
changes, synthetic tree shadows, EMA of the weights, best epoch chosen on 280 tiles held out from
the training set. See `notebooks/train_v4.ipynb` and `models/v4_training/history.csv`.

---

## Quick start

```bash
pip install -r backend/requirements-edge.txt          # inference only, no PyTorch
python scripts/predict_onnx.py data/samples/117991_sat.jpg --output prediction_117991.png
```

This uses `models/mobilevit_v2.onnx` (the v4 model), prints the time and the road fraction, and
saves the road mask. Eight sample tiles are in `data/samples/`.

Other entry points (shortcuts for most of them are in the `Makefile`):

| Task | Command |
|---|---|
| Full environment (training, evaluation, figures) | `pip install -r backend/requirements.txt` |
| Inference with PyTorch | `python scripts/predict_single_image.py data/samples/100034_sat.jpg` |
| Count parameters | `python backend/scripts/test_model.py` (1,604,657) |
| Export ONNX from a checkpoint | `python backend/scripts/export_onnx.py --checkpoint models/best_model_v4.pth` |
| Streamlit demo | `python -m streamlit run demo_app.py` |
| REST API | `uvicorn backend.api:app --reload` |
| Tests | `python -m pytest backend/tests -q` |

---

## Reproducing the results

Training and accuracy evaluation need the DeepGlobe dataset and a GPU, so they run on Kaggle.

**1. Train v4** (`notebooks/train_v4.ipynb`, about 12.5 GPU hours)
1. Open the Kaggle dataset `balraj98/deepglobe-road-extraction-dataset` and create a new notebook.
2. File, Import Notebook, choose `notebooks/train_v4.ipynb`.
3. Settings: GPU T4, Internet on. Save Version, Save & Run All.
4. If it prints `RESUME NEEDED` (12 hour limit), run a new version with the previous output attached.
5. Download `v4/best_model_v4.pth` and `v4/history.csv`.

**2. Evaluate** (`notebooks/evaluate_for_paper.ipynb`, about 20 minutes)
1. New notebook from the same dataset, import `notebooks/evaluate_for_paper.ipynb`.
2. Add the checkpoints `best_model_v4.pth` and `best_model_v3_native.pth` as an input (a Kaggle model or dataset).
3. GPU T4, Internet on, Run All. Check for `SPLIT VERIFIED`.
4. Download `paper_results.zip`; it contains `results.json`, per tile metrics, figures, and the ONNX export.

**3. Paper figures that run locally**

```bash
make figures      # architecture diagrams, prediction grid, postprocessing stages, graph analysis, training curve
```

The notebooks are generated from Python sources; after editing those, run `make notebooks`.

---

## Repository layout

```text
.
├── backend/
│   ├── src/models/mobilevit_v2.py      network
│   ├── src/utils/                      loss (clDice), postprocessing, graph tools, metrics
│   ├── src/data/                       dataset and augmentations
│   ├── scripts/                        train.py, export_onnx.py, evaluate.py, benchmarks
│   ├── tests/                          unit tests
│   └── api.py                          FastAPI endpoint
├── models/
│   ├── best_model_v4.pth               current model (epoch 158)
│   ├── mobilevit_v2.onnx               v4 exported to ONNX
│   ├── best_model_v3_native.pth        earlier model, used for comparison
│   ├── v4_training/                    training history, split, curves
│   └── archive/                        older checkpoints
├── notebooks/
│   ├── train_v4.ipynb                  Kaggle training notebook (v4)
│   ├── evaluate_for_paper.ipynb        Kaggle evaluation notebook (verified split)
│   └── archive/                        earlier training notebooks (v2, v3)
├── scripts/
│   ├── predict_onnx.py                 ONNX Runtime inference
│   ├── predict_single_image.py         PyTorch inference
│   ├── build_train_v4_notebook.py      source of notebooks/train_v4.ipynb
│   ├── paper_figures/                  figure scripts and the evaluation notebook source
│   ├── report/                         project report builders
│   └── presentation/                   slide deck builder
├── figures/real/
│   ├── v4/paper_results/               evaluation output for v4 and v3 (current)
│   ├── v3/paper_results/               evaluation output for v3 alone
│   ├── measurements/                   speed, parity, graph analysis JSON files
│   └── deck/, report/, fig_*           figures of the project report (August v2 model)
├── data/samples/                       eight 1024 x 1024 sample tiles
├── docs/
│   ├── paper/                          IEEE Access paper (main.tex, figures, class files)
│   ├── report/                         project report, execution guide
│   ├── literature/                     reference papers and literature review
│   ├── proposal/, planning/, presentation/
│   └── archive/                        superseded drafts and outputs (see note below)
└── demo_app.py                         Streamlit demonstration
```

**Archived material.** `docs/archive/old_paper/` holds an early paper draft whose accuracy numbers
were never measured; do not cite them. `docs/archive/early_reports/` and
`docs/archive/backend_outputs/` hold early progress reports and generated outputs of that period.
The project report in `docs/report/` describes the August v2 model, not v4.

---

## Model history

| Model | Training | Test IoU (623 tiles, TTA) | File |
|---|---|---|---|
| v2 (August) | 256 crops, validation on resized tiles | not on this split | archived |
| v3 (October) | 50 epochs, 256 crops, flips and colour, canopy augmentation inactive | 52.7% | `models/best_model_v3_native.pth` |
| **v4** | 160 epochs, 512 crops, full augmentation, EMA, separate selection set | **61.1%** | `models/best_model_v4.pth` |

---

## References

* S. Mehta and M. Rastegari, Separable self attention for mobile vision transformers, TMLR 2023.
* S. Cui et al., HPLNet: a hierarchical perception lightweight network for road extraction, Front. Remote Sens. 2025.
* S. Shit et al., clDice: a novel topology preserving loss function for tubular structure segmentation, CVPR 2021.
* I. Demir et al., DeepGlobe 2018: a challenge to parse the Earth through satellite images, CVPRW 2018.
* O. Oktay et al., Attention U-Net, MIDL 2018.
