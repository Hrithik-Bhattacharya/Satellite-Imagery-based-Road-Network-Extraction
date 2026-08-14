import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DeepGlobeDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # DeepGlobe typically uses IDs (e.g., 100014_sat.jpg and 100014_mask.png)
        # We extract just the IDs to safely pair them.
        self.ids = [f.split("_")[0] for f in os.listdir(image_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]

        # Construct exact file paths
        img_path = os.path.join(self.image_dir, f"{img_id}_sat.jpg")
        mask_path = os.path.join(self.mask_dir, f"{img_id}_mask.png")

        # Load image (OpenCV loads in BGR, convert to RGB)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask in Grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Apply the tiling and transformation pipeline
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        # Neural networks need masks to be strictly 0.0 or 1.0 (Floats)
        # DeepGlobe masks are 0 (background) and 255 (road)
        if torch.is_tensor(mask):
            mask = mask.to(dtype=torch.float32) / 255.0
        else:
            mask = torch.tensor(mask, dtype=torch.float32) / 255.0

        # Add a channel dimension to the mask (from 256x256 to 1x256x256)
        mask = mask.unsqueeze(0)

        return image, mask


import random

def estimate_pos_weight(dataset: Dataset, max_samples: int = 500, max_pos_weight: float = 3.0) -> float:
    """
    Measures class imbalance over dataset masks.
    Returns (negative_pixels / positive_pixels) ratio for BCEWithLogitsLoss pos_weight,
    clamped to ``max_pos_weight`` since the raw neg/pos ratio for road masks (often 15-30x)
    is aggressive enough to push a freshly-initialized decoder into predicting
    "mostly road" everywhere before the topology/Dice terms can correct it.
    """
    pos, total = 0, 0
    num_samples = min(len(dataset), max_samples)
    print(f"Estimating pos_weight across {num_samples} dataset samples...")
    for i in range(num_samples):
        _, mask = dataset[i]
        pos += float(mask.sum().item())
        total += float(mask.numel())
    neg = total - pos
    raw_ratio = neg / max(pos, 1.0)
    ratio = min(raw_ratio, max_pos_weight)
    print(f"Estimated dataset pos_weight ratio: {raw_ratio:.2f} (clamped to {ratio:.2f}, cap={max_pos_weight})")
    return ratio


# ==========================================
# TRANSFORMATION & TILING PIPELINE
# ==========================================

class CanopyShadowDropout(A.ImageOnlyTransform):
    """
    Simulates tree-canopy occlusion: darkened, desaturated, soft-edged green-tinted patches
    instead of unrealistic solid-black holes. Mask is left untouched.
    """
    def __init__(
        self,
        max_holes: int = 6,
        max_size: int = 48,
        min_size: int = 16,
        darken_range: tuple = (0.25, 0.55),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.max_holes = max_holes
        self.max_size = max_size
        self.min_size = min_size
        self.darken_range = darken_range

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img = img.copy()
        h, w = img.shape[:2]
        n_holes = random.randint(1, self.max_holes)
        for _ in range(n_holes):
            hh = random.randint(self.min_size, self.max_size)
            ww = random.randint(self.min_size, self.max_size)
            y = random.randint(0, max(h - hh, 1))
            x = random.randint(0, max(w - ww, 1))
            patch = img[y:y + hh, x:x + ww].astype(np.float32)

            # Darken + push toward green (canopy tint), keep texture
            darken = random.uniform(*self.darken_range)
            green_tint = np.array([0.85, 1.0, 0.8], dtype=np.float32)
            patch = patch * darken * green_tint

            # Mild blur so it reads as soft shadow, not a hard-edged artifact
            patch = cv2.GaussianBlur(patch, (5, 5), 0)
            img[y:y + hh, x:x + ww] = np.clip(patch, 0, 255).astype(img.dtype)
        return img


def get_train_transforms():
    """
    Crops satellite images into 256x256 tiles, applies geometric flips,
    canopy shadow occlusion, color jitter, and normalizes pixel values.
    """
    return A.Compose(
        [
            A.RandomCrop(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            CanopyShadowDropout(max_holes=6, max_size=48, min_size=16, p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )


def get_val_transforms():
    """Validation pipeline only normalizes and converts to tensor. No random crops."""
    return A.Compose(
        [
            A.Resize(height=256, width=256),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

