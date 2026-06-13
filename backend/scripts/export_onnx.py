import torch
from src.models import MobileViT_v2
import os

def export_to_onnx():
    print("Initializing ONNX Export for Edge Deployment...")
    
    # 1. Initialize Model
    # In a real scenario, you would load trained weights here using model.load_state_dict()
    model = MobileViT_v2(num_classes=1, width_mult=0.5)
    model.eval() # MUST be in eval mode for ONNX export
    
    # 2. Create Dummy Input
    # Typical satellite image tile size is 256x256 or 512x512
    # Batch size is 1 for edge inference
    dummy_input = torch.randn(1, 3, 256, 256)
    
    export_path = "mobilevit_v2.onnx"
    
    # 3. Export
    try:
        print(f"Exporting model to {export_path}...")
        torch.onnx.export(
            model,
            dummy_input,
            export_path,
            export_params=True,
            opset_version=14,          # Opset 14 supports many modern operations
            do_constant_folding=True,  # Optimize constant operations
            input_names=['input_image'],
            output_names=['segmentation_mask'],
            dynamic_axes={
                'input_image': {0: 'batch_size'},    # Allow variable batch size
                'segmentation_mask': {0: 'batch_size'}
            }
        )
        print("✅ SUCCESS: Model successfully exported to ONNX.")
        print(f"File size: {os.path.getsize(export_path) / (1024*1024):.2f} MB")
        
    except Exception as e:
        print("❌ FAILED: Error during ONNX export.")
        print(e)

if __name__ == "__main__":
    export_to_onnx()
