import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from dataset import DeepGlobeDataset, get_train_transforms


def test_pipeline():
    # Update these paths once you download the Kaggle data
    IMAGE_DIR = "../data/deepglobe/train"
    MASK_DIR = "../data/deepglobe/train"

    # 1. Initialize your custom dataset
    train_dataset = DeepGlobeDataset(
        image_dir=IMAGE_DIR, mask_dir=MASK_DIR, transform=get_train_transforms()
    )

    # 2. Load it into a PyTorch DataLoader (batches of 4)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # 3. Pull one batch
    images, masks = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    # Should be [4, 3, 256, 256]
    print(f"Batch Mask Shape: {masks.shape}")
    # Should be [4, 1, 256, 256]

    # 4. Visualize the first tile in the batch
    image = images[0].permute(1, 2, 0).numpy()
    # Convert PyTorch tensor back to image format
    mask = masks[0].squeeze().numpy()

    # Un-normalize the image so it looks normal to the human eye
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title("256x256 Satellite Tile")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Aligned Road Mask")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Wrap in a try-except block just in case the data folder is empty
    try:
        test_pipeline()
    except FileNotFoundError:
        print(
            "Please ensure you have downloaded the DeepGlobe dataset into the data/deepglobe/train directory."
        )
