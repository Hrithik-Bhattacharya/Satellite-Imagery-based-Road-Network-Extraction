# Evaluating Topology Preserving Loss on Lightweight Vision Transformers for Rural Road Extraction

## 1. Introduction

In developing nations like India, the Pradhan Mantri Gram Sadak Yojana (PMGSY) drives the imperative to map and audit millions of kilometers of rural, unpaved roads. Currently, mapping relies heavily on labor intensive manual auditing. While deep learning has revolutionized remote sensing, semantic segmentation models suffer from severe Domain Shift. They achieve high accuracy on paved urban highways but fail entirely in rural environments where dirt roads are slender, highly irregular, and frequently hidden by dense tree canopies or topographical shadows.

Addressing this issue requires models that prioritize spatial connectivity over raw pixel count. However, current topological models are prohibitively expensive computationally, requiring massive architectures that are unfeasible for edge deployment or rapid auditing by low resource rural agencies. This research proposes an ultra lightweight, state of the art solution.

## 2. Literature Survey

Recent developments (2024 to 2026) in remote sensing and topological extraction highlight two major, but disconnected, trends:

* Lightweight Architectures (The MobileViT and CNN Hybrid Paradigm): Models like BIR Net (2024) and LiteSeger (2025) demonstrate that integrating mobile friendly self attention blocks drastically reduces parameters while using spatial features to capture global context. However, these papers focus primarily on standard urban datasets (such as DeepGlobe) and rely on generic spatial losses.
* Topological Connectivity and State Space Modeling: Networks such as TF RoadNet (2026) and CP SDUNet (2025) have proven that incorporating tubular state space tracking or connectivity preserving losses (such as clDice or Skeleton Recall Loss) is superior for extracting Irregular structures by mathematically penalizing disconnections.
  **The Gap:** Despite these advancements, no existing research has successfully synthesized an ultra lightweight MobileViT V2 or Mamba hybrid backbone with a memory efficient connectivity preserving loss to overcome the severe domain shift and extreme data imbalance of weakly supervised Indian rural road networks.

For a detailed, tabular evaluation of 20 state of the art papers in this field, see the [Literature Survey Table](Literature_Survey_Table.md).

## 3. Problem Statement

Current state of the art architectures designed to maintain road topological connectivity rely on massive, computationally heavy Vision Transformers (ViTs) operating in O(N^2) complexity. Conversely, lightweight CNNs are fast but suffer from fragmented outputs (disconnected roads) under rural occlusions. There is an unsolved need for a highly optimized network that operates with the inference speed of a mobile friendly CNN but achieves the topological connectivity of a heavy Transformer.

## 4. Objectives

1. To construct a state of the art, ultra lightweight encoder decoder architecture utilizing MobileViT V2 to ensure minimal computational footprint.
2. To replace standard spatial loss functions with the graph theoretic Centerline Dice Loss (clDice), mathematically forcing the lightweight model to bridge occluded road gaps.
3. To validate this novel integration on a dataset of rural Indian satellite imagery cross referenced with sparse OpenStreetMap (OSM) centerlines.
4. To demonstrate that this specific architecture produces equivalent or superior Average Path Length Similarity (APLS) topological scores compared to 100M parameter baseline models.

## 5. Methodology

### 5.1 Architecture (MobileViT V2 Encoder)

Rather than writing heavy attention mechanisms from scratch, this research leverages the MobileViT V2 architecture to process optical satellite imagery. MobileViT replaces local processing in convolutions with global processing using localized Transformers, keeping the model lightweight while providing a sufficient global receptive field to "see" where a road continues after an occlusion.

### 5.2 Topological Preservation (clDice Function)

Standard Binary Cross Entropy (BCE) treats all pixels equally, resulting in fragmented roads when dirt paths are obscured. This model will optimize using the clDice loss function. Instead of measuring pure volumetric pixel intersection, clDice extracts the morphological skeleton of both the ground truth centerline and the predicted road mask, calculating the Intersection over Union (IoU) exclusively on the topological graph.

### 5.3 Weakly Supervised Training

To solve the annotation bottleneck, the model will be trained using weak supervision. Instead of expensive dense pixel masks, training ground truths will be one pixel wide line strings extracted from open source maps, relying entirely on the clDice function to iteratively expand and define the road boundaries during training.

## 6. Outcomes

We anticipate that the integration of clDice with a MobileViT U Net will produce an unbroken, navigable road topology matrix on complex rural terrain. The final research paper will feature comprehensive ablation studies proving the Frames Per Second (FPS) inference speed gain and parameter reduction compared to standard Transformer models, making this a highly scalable, real world auditing tool.

## 7. References

1. BIR Net: Lightweight and Efficient Bilateral Interaction Road Extraction Network. (2024). Explores the use of MobileViT and MobileNet modules to drastically reduce network parameters for remote sensing road extraction.
2. SDUNet: Road Extraction via Spatial Enhanced and Densely Connected UNet. (2024). Remote Sensing / Journal of Supercomputing. Demonstrates the critical effectiveness of combining Centerline preserving Dice Loss (clDice) with standard dice loss to significantly improve road topological connectivity.
3. HPLNet: Hierarchical Perception Lightweight Network for Road Extraction. (2025). Frontiers in Remote Sensing. Profiles how lightweight MobileViT modules achieve an optimal balance between execution speed and segmentation precision globally.
4. TopoRF Net: Topology Aware Road Segmentation in Multi Resolution Remote Sensing via Multi Receptive Field Adaptation. (2025). Highlights clDice as a fundamental topology preserving loss function necessary for tubular structure segmentation and road extraction to prevent fragmentation under occlusions.
5. FDMamba and LiteSeger. (2025). IEEE Transactions on Geoscience and Remote Sensing. Validates MobileViT as a mobile friendly vision transformer uniquely suited for real time land cover mapping from high resolution imagery.
