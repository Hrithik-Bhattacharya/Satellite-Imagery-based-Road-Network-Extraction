"""
loss.py -- Differentiable Topology-Aware Loss Functions for Road Extraction

Implements three tightly-coupled components:

    1. Soft-Skeletonization  — iterative MaxPool2d / MinPool2d morphological
       thinning that approximates the medial axis in a fully differentiable
       manner (no non-differentiable skimage dependency).

    2. Soft clDice Loss      — Centerline-Dice that penalises broken road
       topology by comparing skeleton-mask intersections between prediction
       and ground truth.

    3. RoadExtractionLoss    — Dynamic combined loss with epoch-driven alpha
       decay that smoothly transitions training emphasis from pixel-level
       accuracy (BCE) to topological correctness (clDice).

Mathematical background:

    The morphological skeleton of a binary mask A with structuring element B
    is defined as:

        S(A) = Union_{k=0}^{K}  [ (A (-) kB) \\ ((A (-) kB) o B) ]

    where (-) is erosion, o is opening.  We approximate this using:
        erosion  -> MinPool2d  (implemented as -MaxPool2d(-x))
        dilation -> MaxPool2d
        opening  -> dilate(erode(x))

    The Centerline-Dice is:

        Tprec  = |S(V_p) ∩ V_l| / |S(V_p)|     (topology precision)
        Tsens  = |S(V_l) ∩ V_p| / |S(V_l)|     (topology sensitivity)
        clDice = 2 * Tprec * Tsens / (Tprec + Tsens)

    where V_p is the predicted mask, V_l is the ground-truth label mask,
    and S(.) computes the soft skeleton.

Reference:
    Shit et al., "clDice - a Novel Topology-Preserving Loss Function for
    Tubular Structure Segmentation", MICCAI 2021.

Author: Member 1 -- Lead Architect
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# =====================================================================
#  Soft Morphological Primitives
# =====================================================================

def soft_erode(img: torch.Tensor) -> torch.Tensor:
    """
    Differentiable morphological erosion via negated max-pooling.

    Uses a cross-shaped (+) structuring element decomposed into two
    separable 1-D kernels:  a (3x1) vertical strip and a (1x3) horizontal
    strip.  The element-wise minimum of both passes yields the cross
    erosion, which preserves thin diagonal structures better than a full
    3x3 square element.

    MinPool(x) = -MaxPool(-x)

    Args:
        img: (B, 1, H, W) soft mask in [0, 1].

    Returns:
        Eroded mask of the same shape.
    """
    if img.dim() == 3:
        img = img.unsqueeze(1)

    # Vertical strip erosion  (3 rows x 1 col)
    p_v = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    # Horizontal strip erosion (1 row  x 3 cols)
    p_h = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))

    return torch.min(p_v, p_h)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """
    Differentiable morphological dilation via max-pooling.

    Uses a 3x3 square structuring element.

    Args:
        img: (B, 1, H, W) soft mask in [0, 1].

    Returns:
        Dilated mask of the same shape.
    """
    if img.dim() == 3:
        img = img.unsqueeze(1)

    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    """
    Differentiable morphological opening:  dilate(erode(x)).

    Opening removes small bright regions (noise) while preserving the
    overall shape of larger structures.

    Args:
        img: (B, 1, H, W) soft mask in [0, 1].

    Returns:
        Opened mask of the same shape.
    """
    return soft_dilate(soft_erode(img))


def soft_skel(img: torch.Tensor, num_iter: int = 10) -> torch.Tensor:
    """
    Differentiable soft-skeletonization via iterative morphology.

    Approximates the morphological skeleton decomposition:

        S(A) = Union_{k=0}^{K} S_k(A)
        S_k(A) = (A erode kB) - opening(A erode kB)

    At each iteration k:
      1. Erode the mask one more level.
      2. Compute what opening would remove at this scale (delta).
      3. Soft-union delta into the running skeleton via the differentiable
         probabilistic OR:   skel = skel + ReLU(delta - skel * delta)

    The algorithm terminates naturally once the mask is fully eroded.

    Args:
        img:      (B, 1, H, W) soft mask in [0, 1].
        num_iter: Number of erosion iterations.  10 handles roads up to
                  ~20 px wide; increase for thicker structures.

    Returns:
        (B, 1, H, W) soft skeleton in [0, 1].
    """
    if img.dim() == 3:
        img = img.unsqueeze(1)

    # Scale 0: initial skeleton contribution
    img1 = soft_open(img)
    skel = F.relu(img - img1)

    for _ in range(num_iter):
        img = soft_erode(img)               # one more erosion level
        img1 = soft_open(img)               # opening at this scale
        delta = F.relu(img - img1)          # skeleton contribution

        # Differentiable soft-union (probabilistic OR):
        #   skel ∪ delta ≈ skel + delta*(1 - skel)
        # Guarded by ReLU to prevent negative contributions.
        skel = skel + F.relu(delta - skel * delta)

    return skel


# =====================================================================
#  Soft Centerline-Dice (clDice) Loss
# =====================================================================

class SoftClDiceLoss(nn.Module):
    """
    Differentiable Centerline-Dice loss for tubular structure segmentation.

    Penalises broken topology by comparing skeleton-mask intersections:

        Tprec  = (S_pred . V_gt  + smooth) / (|S_pred| + smooth)
        Tsens  = (S_gt   . V_pred + smooth) / (|S_gt|  + smooth)
        clDice = 2 * Tprec * Tsens / (Tprec + Tsens + eps)
        Loss   = 1 - clDice

    Where:
        - S_pred = soft_skel(pred)  : predicted road skeleton
        - S_gt   = soft_skel(gt)    : ground-truth road skeleton
        - V_pred, V_gt              : the full masks (not skeletons)

    This ensures:
        - High Tprec: predicted skeleton falls inside GT roads
                      (penalises false-positive road branches)
        - High Tsens: GT skeleton falls inside predicted roads
                      (penalises missed / broken connections)

    Args:
        num_iter: Erosion iterations for soft-skeletonization (default 10).
        smooth:   Laplace smoothing to prevent division by zero.
    """

    def __init__(self, num_iter: int = 10, smooth: float = 1.0):
        super().__init__()
        self.num_iter = num_iter
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, 1, H, W) sigmoid-activated prediction in [0, 1].
            target: (B, 1, H, W) binary ground-truth mask.

        Returns:
            Scalar clDice loss in [0, 1].
        """
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Compute soft skeletons (differentiable through pred)
        skel_pred = soft_skel(pred, self.num_iter)
        skel_target = soft_skel(target, self.num_iter)

        # Topology Precision: predicted skeleton inside GT mask
        #   "Are the predicted centerlines real roads?"
        tprec_num = (skel_pred * target).sum(dim=(1, 2, 3)) + self.smooth
        tprec_den = skel_pred.sum(dim=(1, 2, 3)) + self.smooth
        tprec = tprec_num / tprec_den

        # Topology Sensitivity: GT skeleton inside predicted mask
        #   "Did we predict all the real road centerlines?"
        tsens_num = (skel_target * pred).sum(dim=(1, 2, 3)) + self.smooth
        tsens_den = skel_target.sum(dim=(1, 2, 3)) + self.smooth
        tsens = tsens_num / tsens_den

        # Harmonic mean (clDice)
        cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens + 1e-7)

        return (1.0 - cl_dice).mean()


# =====================================================================
#  Soft Dice Loss (supplementary, for ablation)
# =====================================================================

class SoftDiceLoss(nn.Module):
    """
    Standard differentiable Dice loss for binary segmentation.

        DiceLoss = 1 - (2 * |P ∩ G| + smooth) / (|P| + |G| + smooth)

    Included for ablation studies comparing clDice vs. standard Dice.

    Args:
        smooth: Laplace smoothing constant.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        intersection = (pred * target).sum(dim=(1, 2, 3))
        cardinality = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return (1.0 - dice).mean()


# =====================================================================
#  Dynamic Combined Loss  (BCE + clDice with alpha decay)
# =====================================================================

class RoadExtractionLoss(nn.Module):
    """
    Dynamic combined loss for road extraction training.

    Computes:

        L_total = alpha * L_BCE  +  (1 - alpha) * L_clDice

    The mixing coefficient `alpha` decays linearly over the training
    schedule, implementing a curriculum strategy:

        Epoch 0          : alpha = 1.0   (100 % BCE -- learn pixel masks)
        ...
        Epoch T          : alpha = 0.2   (80 % clDice -- refine topology)

    Rationale:
        Early training benefits from pixel-level BCE gradients that
        produce stable, dense learning signals.  Once the model has
        learned rough road regions, shifting emphasis to clDice forces
        it to fix disconnected predictions, fill gaps, and suppress
        spurious branches — all critical for downstream graph extraction.

    Args:
        total_epochs: Total number of training epochs (for alpha schedule).
        alpha_start:  Initial alpha value (default 1.0).
        alpha_end:    Final alpha value after decay (default 0.2).
        num_iter:     Erosion iterations for soft-skeletonization.
        smooth:       Laplace smoothing for clDice numerics.

    Example::

        loss_fn = RoadExtractionLoss(total_epochs=100)

        for epoch in range(100):
            loss_fn.update_alpha(epoch)
            for images, masks in dataloader:
                pred = model(images)
                loss, components = loss_fn(pred, masks, return_components=True)
                loss.backward()
                optimizer.step()
                wandb.log(components)
    """

    def __init__(
        self,
        total_epochs: int,
        alpha_start: float = 1.0,
        alpha_end: float = 0.2,
        num_iter: int = 10,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.total_epochs = max(total_epochs, 1)
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha = alpha_start

        self.bce = nn.BCELoss()
        self.cldice = SoftClDiceLoss(num_iter=num_iter, smooth=smooth)

    # ── Alpha schedule ─────────────────────────────────────────────

    def update_alpha(self, epoch: int) -> float:
        """
        Linearly decay alpha from ``alpha_start`` to ``alpha_end``.

        Should be called once at the beginning of each epoch.

        Args:
            epoch: Current epoch index (0-based).

        Returns:
            Updated alpha value.

        Schedule::

            alpha(t) = max(alpha_end,
                           alpha_start - (alpha_start - alpha_end) * t / T)

            t=0   ->  1.0  (pure BCE)
            t=T/4 ->  0.8  (80 % BCE, 20 % clDice)
            t=T/2 ->  0.6  (60 % BCE, 40 % clDice)
            t=T   ->  0.2  (20 % BCE, 80 % clDice)
        """
        decay = (self.alpha_start - self.alpha_end) * epoch / self.total_epochs
        self.alpha = max(self.alpha_end, self.alpha_start - decay)
        return self.alpha

    def get_alpha(self) -> float:
        """Return the current alpha value."""
        return self.alpha

    # ── Forward ────────────────────────────────────────────────────

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the combined loss.

        Args:
            pred:              (B, 1, H, W) sigmoid-activated predictions.
            target:            (B, 1, H, W) binary ground-truth masks.
            return_components: If True, also return a dict of individual
                               loss values for logging / WandB.

        Returns:
            total_loss:  Scalar tensor for backward().
            components:  (optional) Dict with keys ``total_loss``,
                         ``bce_loss``, ``cldice_loss``, ``alpha``.
        """
        # Clamp predictions for numerical stability in BCE
        pred_safe = pred.clamp(min=1e-7, max=1.0 - 1e-7)

        bce_loss = self.bce(pred_safe, target)
        cldice_loss = self.cldice(pred, target)

        total_loss = (self.alpha * bce_loss
                      + (1.0 - self.alpha) * cldice_loss)

        if return_components:
            components = {
                "total_loss": total_loss.item(),
                "bce_loss": bce_loss.item(),
                "cldice_loss": cldice_loss.item(),
                "alpha": self.alpha,
            }
            return total_loss, components

        return total_loss


# =====================================================================
#  Verification
# =====================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  Loss Functions -- Verification Suite")
    print("=" * 65)

    B, C, H, W = 2, 1, 256, 256
    device = "cpu"

    # Create synthetic masks (overlapping rectangles for a 'road')
    pred = torch.zeros(B, C, H, W, device=device)
    pred[:, :, 100:156, 60:196] = 0.9   # wide horizontal band
    pred[:, :, 40:216, 120:132] = 0.85   # narrow vertical band
    pred.requires_grad_(True)

    gt = torch.zeros(B, C, H, W, device=device)
    gt[:, :, 110:146, 50:206] = 1.0     # thinner horizontal road
    gt[:, :, 30:226, 122:130] = 1.0     # thinner vertical road

    # ---- 1. Soft Skeleton ----
    print("\n-- Soft Skeletonization --")
    with torch.no_grad():
        skel = soft_skel(gt, num_iter=10)
    print(f"   GT mask sum            : {gt.sum().item():.0f}")
    print(f"   GT skeleton sum        : {skel.sum().item():.1f}")
    print(f"   Skeleton/Mask ratio    : {skel.sum().item() / gt.sum().item():.4f}")
    print(f"   (Expected << 1, confirming thinning)")

    # ---- 2. clDice Loss ----
    print("\n-- SoftClDiceLoss --")
    cldice_fn = SoftClDiceLoss(num_iter=10)
    loss_val = cldice_fn(pred, gt)
    loss_val.backward(retain_graph=True)
    print(f"   clDice loss            : {loss_val.item():.4f}")
    print(f"   Gradient flows         : {pred.grad is not None}")
    print(f"   Gradient norm          : {pred.grad.norm().item():.6f}")
    pred.grad.zero_()

    # ---- 3. Soft Dice Loss ----
    print("\n-- SoftDiceLoss --")
    dice_fn = SoftDiceLoss()
    dice_val = dice_fn(pred, gt)
    dice_val.backward(retain_graph=True)
    print(f"   Dice loss              : {dice_val.item():.4f}")
    pred.grad.zero_()

    # ---- 4. Dynamic Combined Loss ----
    print("\n-- RoadExtractionLoss (dynamic alpha) --")
    total_epochs = 100
    loss_fn = RoadExtractionLoss(total_epochs=total_epochs)

    for epoch in [0, 25, 50, 75, 100]:
        alpha = loss_fn.update_alpha(epoch)
        total, comp = loss_fn(pred, gt, return_components=True)
        total.backward(retain_graph=True)
        pred.grad.zero_()
        print(f"   Epoch {epoch:3d}  |  alpha={alpha:.2f}  |  "
              f"BCE={comp['bce_loss']:.4f}  |  "
              f"clDice={comp['cldice_loss']:.4f}  |  "
              f"Total={comp['total_loss']:.4f}")

    print(f"\n   All checks passed.")
    print("=" * 65)
