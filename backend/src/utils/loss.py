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
#  RoadExtractionLoss (BCEWithLogits + clDice with pos_weight & decay)
# =====================================================================

class RoadExtractionLoss(nn.Module):
    def __init__(
        self,
        total_epochs: int,
        alpha_start: float = 0.5,
        alpha_end: float = 0.15,
        dice_weight: float = 0.35,
        num_iter: int = 10,
        smooth: float = 1.0,
        pos_weight: float = 2.0,
        decay_power: float = 0.5,
        alpha_decay_epochs: int | None = None,
    ):
        super().__init__()
        self.total_epochs = max(total_epochs, 1)
        self.decay_epochs = max(alpha_decay_epochs or total_epochs, 1)
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha = alpha_start
        self.dice_weight = dice_weight
        self.decay_power = decay_power

        pw_tensor = torch.tensor([pos_weight]) if isinstance(pos_weight, (int, float)) else pos_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
        self.dice = SoftDiceLoss(smooth=smooth)
        self.cldice = SoftClDiceLoss(num_iter=num_iter, smooth=smooth)

    def get_alpha(self) -> float:
        return self.alpha

    def update_alpha(self, epoch: int) -> float:
        # Non-linear decay decoupled from total epochs: reaches alpha_end by decay_epochs
        frac = min(epoch / self.decay_epochs, 1.0) ** self.decay_power
        self.alpha = self.alpha_start - (self.alpha_start - self.alpha_end) * frac
        self.alpha = max(self.alpha_end, self.alpha)
        return self.alpha

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        # Ensure pos_weight is on the same device as logits
        if self.bce.pos_weight is not None and self.bce.pos_weight.device != logits.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logits.device)

        # Force float32 computation for numerical stability and to prevent autocast issues.
        with torch.amp.autocast(device_type=logits.device.type, enabled=False):
            logits_f32 = logits.float()
            target_f32 = target.float()
            bce_loss = self.bce(logits_f32, target_f32)
            dice_loss = self.dice(logits_f32, target_f32)
            cldice_loss = self.cldice(logits_f32, target_f32)

        cldice_w = max(0.0, 1.0 - self.alpha - self.dice_weight)
        total_loss = self.alpha * bce_loss + self.dice_weight * dice_loss + cldice_w * cldice_loss

        if return_components:
            components = {
                "total_loss": total_loss.item(),
                "bce_loss": bce_loss.item(),
                "dice_loss": dice_loss.item(),
                "cldice_loss": cldice_loss.item(),
                "alpha": self.alpha,
            }
            return total_loss, components

        return total_loss


# =====================================================================
#  Fragmentation Metric (Connected Component Ratio)
# =====================================================================

def avg_component_count(pred_masks: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Computes average number of connected components in predictions.
    Lower count indicates continuous road networks; high count indicates fragmentation.
    """
    from scipy import ndimage
    counts = []
    preds_np = (torch.sigmoid(pred_masks) > threshold).cpu().numpy().astype("uint8")
    for m in preds_np:
        if m.ndim == 3:
            m = m[0]
        _, n = ndimage.label(m)
        counts.append(n)
    return float(sum(counts) / max(len(counts), 1))

