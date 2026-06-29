import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

# =====================================================================
#  Soft Morphological Primitives
# =====================================================================

def soft_erode(img: torch.Tensor) -> torch.Tensor:
    if img.dim() == 3:
        img = img.unsqueeze(1)
    p_v = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p_h = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p_v, p_h)

def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    if img.dim() == 3:
        img = img.unsqueeze(1)
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)

def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))

def soft_skel(img: torch.Tensor, num_iter: int = 10) -> torch.Tensor:
    if img.dim() == 3:
        img = img.unsqueeze(1)
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(num_iter):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel

# =====================================================================
#  Soft Centerline-Dice (clDice) Loss
# =====================================================================

class SoftClDiceLoss(nn.Module):
    def __init__(self, num_iter: int = 10, smooth: float = 1.0):
        super().__init__()
        self.num_iter = num_iter
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        skel_pred = soft_skel(torch.sigmoid(pred), self.num_iter)
        skel_target = soft_skel(target, self.num_iter)

        tprec_num = (skel_pred * target).sum(dim=(1, 2, 3)) + self.smooth
        tprec_den = skel_pred.sum(dim=(1, 2, 3)) + self.smooth
        tprec = tprec_num / tprec_den

        tsens_num = (skel_target * torch.sigmoid(pred)).sum(dim=(1, 2, 3)) + self.smooth
        tsens_den = skel_target.sum(dim=(1, 2, 3)) + self.smooth
        tsens = tsens_num / tsens_den

        cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens + 1e-7)
        return (1.0 - cl_dice).mean()

# =====================================================================
#  Soft Dice Loss
# =====================================================================

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)  # ensure probabilities
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        intersection = (pred * target).sum(dim=(1, 2, 3))
        cardinality = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return (1.0 - dice).mean()

# =====================================================================
#  RoadExtractionLoss (BCEWithLogits + clDice)
# =====================================================================

class RoadExtractionLoss(nn.Module):
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

        # FIX: Use BCEWithLogitsLoss instead of BCELoss
        self.bce = nn.BCEWithLogitsLoss()
        self.cldice = SoftClDiceLoss(num_iter=num_iter, smooth=smooth)

    def update_alpha(self, epoch: int) -> float:
        decay = (self.alpha_start - self.alpha_end) * epoch / self.total_epochs
        self.alpha = max(self.alpha_end, self.alpha_start - decay)
        return self.alpha

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        # logits: raw model outputs (no sigmoid)
        bce_loss = self.bce(logits, target)
        cldice_loss = self.cldice(logits, target)

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
