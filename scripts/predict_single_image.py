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


def predict_tta(model, input_tensor, device):
    """
    Test-time augmentation: averages predictions over the original tile plus its
    horizontal, vertical, and both-axis flips (un-flipped back before averaging).
    Cheap (4x forward passes, no retraining) and generally reduces spurious
    single-orientation false positives (e.g. field-boundary lines that only look
    road-like from one angle) since a true road stays road-shaped under any flip.
    """
    variants = [
        (lambda x: x, lambda p: p),                                                    # identity
        (lambda x: torch.flip(x, dims=[3]), lambda p: torch.flip(p, dims=[3])),          # horizontal
        (lambda x: torch.flip(x, dims=[2]), lambda p: torch.flip(p, dims=[2])),          # vertical
        (lambda x: torch.flip(x, dims=[2, 3]), lambda p: torch.flip(p, dims=[2, 3])),    # both
    ]
    probs_sum = None
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            for forward_fn, inverse_fn in variants:
                logits = model(forward_fn(input_tensor))
                probs = inverse_fn(torch.sigmoid(logits))
                probs_sum = probs if probs_sum is None else probs_sum + probs
    return (probs_sum / len(variants)).squeeze().cpu().numpy()


def predict_single_image(image_path, model_path=None, output_path="prediction_result.png", use_tta=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine default model path if not explicitly provided.
    # Preference order: a freshly (correctly) retrained model in models/, then the
    # last known-good canopy-resilient checkpoint, then legacy root-level locations
    # for backward compatibility with older checkouts.
    if model_path is None or model_path == "best_model.pth":
        candidates = [
            os.path.join("models", "best_model_v2.pth"),
            os.path.join("models", "best_model_new.pth"),
            os.path.join("models", "best_model.pth"),
            "best_model_new.pth",
            "best_model_v2.pth",
            "best_model.pth",
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                model_path = candidate
                break

    # 1. Initialize Model and Load Weights
    print(f"Loading model weights from {model_path}...")
    model = MobileViT_v2(num_classes=1, width_mult=1.0)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if (isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint) else checkpoint
        model.load_state_dict(state_dict, strict=False)
    except FileNotFoundError:
        print(f"Error: Could not find {model_path}. Make sure it is in the project directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading checkpoint {model_path}: {e}")
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

    # 3. Run Inference (with optional test-time augmentation)
    if use_tta:
        print("Running inference (4-way flip TTA)...")
        probs = predict_tta(model, input_tensor, device)
    else:
        print("Running inference...")
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(input_tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    # --- Hysteresis Thresholding & Enhanced Post-Processing ---
    from backend.src.utils.graph_postprocess import connect_canopy_gaps, hysteresis_threshold
    
    # 1. Hysteresis thresholding to recover weak road probabilities beneath tree foliage
    mask_hyst = hysteresis_threshold(probs, high_thresh=0.35, low_thresh=0.12)

    # 2. Refined Morphological Closing (bridge minor breaks before noise filtering)
    kernel_close = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(mask_hyst, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # 3. Multi-Strategy Graph-Based Canopy Gap & T-Junction Completion
    mask_connected = connect_canopy_gaps(mask_closed, max_gap_dist=220.0, max_angle_deg=65.0, road_width=6)
    
    mask_plot = mask_connected / 255.0

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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Success! Result saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MobileViT_v2 on a single image")
    parser.add_argument("image_path", type=str, help="Path to the input satellite image (e.g. test_image.jpg)")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint (e.g. best_model_v2.pth)")
    parser.add_argument("--output", type=str, default="prediction_result.png", help="Filename to save the result plot")
    parser.add_argument("--no-tta", dest="use_tta", action="store_false",
                         help="Disable 4-way flip test-time augmentation (on by default; costs 4x forward passes)")
    args = parser.parse_args()

    predict_single_image(args.image_path, args.model, args.output, args.use_tta)

