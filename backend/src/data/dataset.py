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
        mask = mask / 255.0
        mask = torch.tensor(mask, dtype=torch.float32)

        # Add a channel dimension to the mask (from 256x256 to 1x256x256)
        mask = mask.unsqueeze(0)

        return image, mask


# ==========================================
# TRANSFORMATION & TILING PIPELINE
# ==========================================


def get_train_transforms():
    """
    Crops massive satellite images into 256x256 tiles, applies
    geometric flips, and normalizes pixel values.
    """
    return A.Compose(
        [
            A.RandomCrop(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                # Standard ImageNet means
                std=[0.229, 0.224, 0.225],
                # Standard ImageNet stds
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
            # Resize strictly for validation
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )
