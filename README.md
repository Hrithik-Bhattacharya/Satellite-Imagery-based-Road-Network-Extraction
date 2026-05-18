# Satellite-Imagery-based-Road-Network-Extraction
## 📋 Executive Project Summary & WBS Overview

This plan synthesizes the foundational strategies from your initial planning documents with the modern technical approaches highlighted in the rural roads project proposal. It specifically addresses the **Domain Shift** seen when applying standard models to Indian rural topographies where dirt roads are slender, highly irregular, and hidden by tree canopies or shadows.

By combining an ultra-lightweight **MobileViT v2** backbone with **Centerline-Dice (clDice) Loss** , the team can maintain the global structural context of a heavy transformer while matching the fast inference speeds of edge-friendly networks , relying on **DeepGlobe** and **OpenStreetMap (OSM) weak labels**  for training.

Below is the **4-Member Project Structure** embedded within the generated layout:

#### 👥 Roles and Key Responsibilities

1. **Member 1: Lead Architect (Pixel/Segmentation Focus)**
* 
**Core Focus:** Building the lightweight encoder-decoder architecture using **MobileViT v2** blocks.
* 
**Responsibilities:** Optimizing parameter count and computational layers; managing attention mechanics to capture long, continuous road structures ; and ensuring high inference speeds (Frames Per Second / FPS).

2. **Member 2: Graph & Loss Engineer (Topology/Connectivity Focus)**
* 
**Core Focus:** Implementing graph-theoretic modeling and topology-preserving losses.
* 
**Responsibilities:** Coding the **clDice function** ; creating the morphological skeletonization modules to ensure the network bridges occluded road gaps instead of just calculating pixel count ; and penalizing topological disconnections during backpropagation.

3. **Member 3: Data & Pipeline Engineer (Data/Weak Supervision Focus)**
* 
**Core Focus:** Dataset curation, pipeline optimization, and weak supervision constraints.
* 
**Responsibilities:** Ingesting and preprocessing the **DeepGlobe Dataset** ; configuring the **weakly supervised framework** by transforming OpenStreetMap (OSM) vector line strings into one-pixel training paths ; and writing data augmentations for shadows, cloud cover, and canopy occlusions.

4. **Member 4: Validation & MLOps (Evaluation/Metrics Focus)**
* 
**Core Focus:** Rigorous model validation and tracking performance metrics.


* 
**Responsibilities:** Building evaluation frameworks for non-pixel metrics like **APLS (Average Path Length Similarity)** and TOPO metrics ; running structural ablation studies ; and benchmarking performance against 100M+ parameter baseline models to prove scalable real-world auditing value.

---

### 📅 Phased High-Level Roadmap (6-Week Timeline)

* **Week 1: Environment Baseline & Data Pipeline**
* Repository initialization (M1); Prototype differentiable morphological skeleton layers (M2); DeepGlobe dataset tiling and data loader engineering (M3); Setup tracking dashboards (e.g., Weights & Biases) with basic IoU metrics (M4).

* **Weeks 2–3: Core Model Formulation & Loss Fusion**
* Construct MobileViT v2 encoder blocks (M1); Implement full clDice loss routines (M2); Set up OSM vector-to-raster weak annotation pipeline (M3); Package the network APLS calculation library (M4).

* **Week 4: Integration, Training & Tuning**
* Core pipeline integration (M1 + M2); Execute advanced data augmentations for canopy and shadow occlusions (M3); Generate graph alignment loss convergence profiles (M4).

* **Weeks 5–6: Optimization, Ablation & Final Delivery**
* Model optimization and export formats (ONNX/TensorRT) for real-world edge deployment testing (M1); Multi-loss optimization balancing cross-entropy and clDice coefficients (M2); Validate domain adaptation metrics (M3); Compile ablation studies matching parameter count vs. topological scores (M4).
