import torch
import torch.nn as nn
from src.models import UNet, MobileViT_v2
import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device():
    """Detect appropriate device: CUDA, MPS (Apple Silicon), or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def test_model():
    print("Initializing Phase 2 MobileViT v2 Test...")
    device = get_device()
    print(f"Detected Device: {device}")

    try:
        # Initialize U-Net for comparison
        unet = UNet(n_channels=3, n_classes=1)
        unet_params = count_parameters(unet)

        # Initialize MobileViT v2
        model = MobileViT_v2(num_classes=1, width_mult=0.5).to(device)
        mv2_params = count_parameters(model)
        
        print(f"U-Net Parameters:        {unet_params:,}")
        print(f"MobileViT v2 Parameters: {mv2_params:,}")
        print("MobileViT v2 successfully instantiated and moved to device.")

        # Create dummy batch of 4 images (e.g. 256x256 RGB)
        dummy_input = torch.randn(4, 3, 256, 256).to(device)
        
        # Test Forward Pass
        start_time = time.time()
        output = model(dummy_input)
        end_time = time.time()

        print(f"Forward pass successful!")
        print(f"Input Shape:  {dummy_input.shape}")
        print(f"Output Shape: {output.shape} (Expected: [4, 1, 256, 256])")
        print(f"Time Taken:   {end_time - start_time:.4f} seconds")
        
        if output.shape == (4, 1, 256, 256):
            print("\n✅ SUCCESS: Architectural Baseline is ready for Phase 1.")
        else:
            print("\n❌ FAILED: Output shape mismatch.")

    except Exception as e:
        print(f"\n❌ FAILED: Exception occurred during initialization or forward pass:")
        print(e)

if __name__ == "__main__":
    test_model()
