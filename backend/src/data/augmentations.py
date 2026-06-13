import albumentations as A


def get_indian_rural_transforms():
    """
    Simulation of challenging Indian rural environments:
    Shadows, tree canopies, and varied material color jitters.
    """
    return A.Compose(
        [
            # 1. Random Brightness & Contrast: Simulates different lighting
            # (sunny vs. monsoon overcast skies)
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            # 2. Color Jitter: Roads look different based on material
            # (e.g., wet soil vs. dry bituminous blacktop)
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
            ),
            # 3. CoarseDropout (Simulating Tree Canopy/Occlusions):
            # This randomly cuts out squares of the image to force the AI to
            # learn to infer roads even when they are blocked by trees or buildings.
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                fill_value=0,
                p=0.5,
            ),
            # 4. Blur & Noise: Simulates low-quality sensor data
            # (common in LISS-4 imagery)
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                ],
                p=0.3,
            ),
        ]
    )
