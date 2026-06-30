import argparse
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

# Add the parent directory to sys.path so we can import 'backend'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the model from the local backend package
from backend.src.models.mobilevit_v2 import MobileViT_v2

def predict_single_image(image_path, model_path="best_model.pth", output_path="prediction_result.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Model and Load Weights
    print(f"Loading model weights from {model_path}...")
    model = MobileViT_v2(num_classes=1, width_mult=1.0)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        # Check if it's a full checkpoint dict or just weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print(f"Error: Could not find {model_path}. Make sure it is in the same directory.")
        sys.exit(1)
        
    model.to(device)
    model.eval()

    # 2. Load and Preprocess Image
    print(f"Loading image {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}. Check the path.")
        sys.exit(1)
    
    # Convert BGR (OpenCV default) to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Ensure dimensions are divisible by 32 for the MobileViT encoder-decoder
    h, w = image.shape[:2]
    new_h = (h // 32) * 32
    new_w = (w // 32) * 32
    
    transform = A.Compose([
        A.Resize(height=new_h, width=new_w),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    transformed = transform(image=image)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    # 3. Run Inference
    print("Running inference...")
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            logits = model(input_tensor)
            # The model outputs raw logits, so we apply sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(logits)
            # Drop threshold aggressively to pick up very faint signals under dense canopies
            preds = (probs > 0.15).float()

    # Move to CPU and remove batch/channel dimensions
    mask_np = preds.squeeze().cpu().numpy()

    # --- Aggressive Post-Processing to Improve Connectivity ---
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    
    # 0. Morphological Opening (Erosion followed by Dilation) to remove tiny isolated noise blobs
    kernel_open = np.ones((3, 3), np.uint8)
    mask_opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # 1. Heavy Morphological Closing to bridge massive gaps across tree canopies
    kernel_close = np.ones((15, 15), np.uint8)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # 2. Light Dilation to make the roads visibly continuous and smooth
    kernel_dilate = np.ones((3, 3), np.uint8)
    mask_final = cv2.dilate(mask_closed, kernel_dilate, iterations=1)
    
    mask_plot = mask_final / 255.0

    # 4. Plot and Save Side-by-Side
    print("Generating visualization plot...")
    plt.figure(figsize=(10, 5))
    
    # Original Image
    plt.subplot(1, 2, 1)
    orig_resized = cv2.resize(image, (new_w, new_h))
    plt.imshow(orig_resized)
    plt.title("Original Satellite Image")
    plt.axis("off")

    # Predicted Mask
    plt.subplot(1, 2, 2)
    plt.imshow(mask_plot, cmap="gray")
    plt.title("Predicted Road Network")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Success! Result saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MobileViT_v2 on a single image")
    parser.add_argument("image_path", type=str, help="Path to the input satellite image (e.g. test_image.jpg)")
    parser.add_argument("--model", type=str, default="best_model.pth", help="Path to best_model.pth")
    parser.add_argument("--output", type=str, default="prediction_result.png", help="Filename to save the result plot")
    args = parser.parse_args()
    
    predict_single_image(args.image_path, args.model, args.output)
