# Measured figures for the paper

Every figure in this folder is produced from real data — trained checkpoints, real
imagery, real logs, or hardware measurements. Nothing is simulated. Each has a 300-DPI
PNG (slides, Word) and a vector PDF (LaTeX).

## Produced locally (`scripts/paper_figures/`)

| Figure | What it shows | Data source |
|---|---|---|
| `fig_efficiency` | Parameters, GFLOPs and CPU latency per 1024² tile vs U-Net, D-LinkNet34, DeepLabV3-MBv3, LR-ASPP-MBv3 | `benchmark_efficiency.py` → `measurements/efficiency.json` (Intel i5-1035G4). Reference models are untrained: compute cost only |
| `fig_baseline_training` | Train vs validation BCE / soft-clDice for the June baseline run | `wandb_local.json` (repo root) — the June run's own log |
| `fig_pipeline_stages` | One tile through probability → hysteresis + closing → canopy-gap bridging | Final checkpoint on `data/samples/117991_sat.jpg` (the sample with the most bridging) |
| `fig_collapse` | Collapsed (ep. 46) vs final checkpoint on real tiles + probability histograms | `models/archive/…collapsed_epoch46.pth`, `models/best_model_v2.pth` |
| `fig_resolution` | Native-resolution prediction vs the 256²-resize protocol used for validation during training | Final checkpoint on sample tiles |

Regenerate: `python scripts/paper_figures/benchmark_efficiency.py` (once), then
`python scripts/paper_figures/local_figures.py`.

## Produced on Kaggle (`notebooks/evaluate_for_paper.ipynb`)

Accuracy against ground truth needs the DeepGlobe masks (3.8 GB, does not fit locally).
Run the notebook on Kaggle, download `paper_results.zip`, and unzip it here as `kaggle/`.
It first **verifies the split** by reproducing the metrics stored in the checkpoints; its
results are only held-out results if it prints `SPLIT VERIFIED`.

| Figure | What it shows |
|---|---|
| `fig_eval_protocol` | IoU per checkpoint: training-time 256² protocol vs native resolution |
| `fig_eval_main` | IoU, F1, relaxed F1 (ρ = 3 px), clDice with 95% bootstrap CIs |
| `fig_eval_pr` | Precision–recall curves and IoU vs threshold |
| `fig_eval_ablation` | Contribution of hysteresis, closing, gap bridging (original vs border fix), TTA |
| `fig_eval_canopy` | Recall on canopy-covered vs open road; clDice by canopy tertile |
| `fig_eval_per_tile` | Per-tile IoU distributions, paired difference, Wilcoxon test |
| `fig_eval_collapse` | Why checkpoint selection picked the collapsed epoch |
| `fig_eval_qualitative`, `…_canopy` | Error maps on percentile-selected / random high-canopy tiles |
| `fig_training_curves_N` | v2 training log(s), if attached as notebook inputs |

Also written: `results.json` (every number), `per_tile_metrics.csv`, `results_table.tex`.

## Claims in `main.tex` that the measurements contradict

These must be replaced before submission — none of them came from an experiment:

- **Abstract (l. 44), results (l. 465–537), conclusion (l. 573):** IoU 74.62%, F1 85.46%,
  clDice 81.62%, APLS 76.8%, and every number in the pixel-metric, topology and ablation
  tables. No U-Net or ablation model was trained; `fig7_ablation_study` is hard-coded
  (`generate_paper_figures.py` l. 660–662). The best measured validation IoU is 16.85%
  (256² protocol); native-resolution numbers come from the Kaggle notebook.
- **`fig3_training_dynamics`:** synthetic curves (`generate_paper_figures.py` l. 396–415).
  The "real" version (`update_fig3_real.py`) uses the June run's loss log with an invented
  IoU curve and a sine-wave fragmentation curve.
- **`fig6_centrality_resilience` and l. 392–400:** random graph and hard-coded resilience
  indices; the 78% efficiency-collapse figure was not measured.
- **l. 246:** "α_end = 0.40 … across all 50 epochs" — the final run used α_end = 0.15, best
  epoch 53, early-stopping patience 35.
- **l. 426 and abstract:** "0.82 MB ONNX payload", "144× compression", "Intel Core i7,
  580 ms". Measured: 6.94 MB (0.83 MB graph + 6.1 MB weights file), 17× smaller than
  U-Net's 118.4 MB fp32 weights, 0.91 s per tile on an Intel i5-1035G4.
- **l. 276:** training on centerline weak labels — the training notebook loads only the
  DeepGlobe `*_mask.png` masks; no weak or OSM labels were used.
- **Canopy-shadow augmentation:** its constructor call left it disabled (p = 0) on
  albumentations 2.x. The Kaggle notebook reports whether it was active in that environment;
  do not claim it contributed until confirmed.
- **`demo_app.py` (l. 349–505)** displays the same fabricated metrics and baseline table.
  (The 94.8% parameter reduction vs U-Net is correct: 1.60 M vs 31.04 M.)
