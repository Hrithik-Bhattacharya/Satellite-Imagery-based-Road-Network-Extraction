"""
train.py -- MLOps training script for Rural Road Extraction

Designed for GPU environments (Kaggle/Colab).
Features:
  - Automatic Mixed Precision (AMP)
  - Dynamic alpha decay for topology-aware loss
  - W&B integration with graceful fallback
  - Best-checkpoint saving based on validation clDice loss
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# --- Path Fix ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

# --- Dataset Paths (Default CLI options) ---
TRAIN_IMG_DIR = '/kaggle/working/dataset/train'
TRAIN_MASK_DIR = '/kaggle/working/dataset/train'
VAL_IMG_DIR   = '/kaggle/working/dataset/valid'
VAL_MASK_DIR  = '/kaggle/working/dataset/valid'
OUTPUT_DIR    = '/kaggle/working/models'

from src.data.dataset import DeepGlobeDataset, get_train_transforms, get_val_transforms
from src.models.mobilevit_v2 import MobileViT_v2
from src.utils.loss import RoadExtractionLoss
from src.utils.wandb_logger import WandbLogger

def train_one_epoch(epoch, model, dataloader, optimizer, scaler, loss_fn, logger, device):
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_cldice = 0.0
    
    current_alpha = loss_fn.update_alpha(epoch)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        # AMP Forward Pass
        with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            logits = model(images)  # raw outputs (no sigmoid)
            loss, components = loss_fn(logits, masks, return_components=True)
            
        # AMP Backward Pass & Optimizer Step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_bce += components['bce_loss']
        total_cldice += components['cldice_loss']
        
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Alpha": f"{current_alpha:.2f}"})
        
        logger.log_metrics({
            "train/step_loss": loss.item(),
            "train/step_bce": components['bce_loss'],
            "train/step_cldice": components['cldice_loss'],
            "lr": optimizer.param_groups[0]['lr']
        })
        
    num_batches = len(dataloader)
    return total_loss / num_batches, total_bce / num_batches, total_cldice / num_batches

@torch.no_grad()
def validate(epoch, model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_bce = 0.0
    total_cldice = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            preds = model(images)
            loss, components = loss_fn(preds, masks, return_components=True)
            
        total_loss += loss.item()
        total_bce += components['bce_loss']
        total_cldice += components['cldice_loss']
        
        pbar.set_postfix({"Val Loss": f"{loss.item():.4f}"})
        
    num_batches = len(dataloader)
    return total_loss / num_batches, total_bce / num_batches, total_cldice / num_batches

def main(args):
    # Setup paths
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Path validation
    print("--- Validating Paths ---")
    for name, path in [
        ('Train Images', args.train_image_dir),
        ('Train Masks', args.train_mask_dir),
        ('Val Images', args.val_image_dir),
        ('Val Masks', args.val_mask_dir)
    ]:
        if not os.path.exists(path):
            print(f"⚠️ Warning: Path not found for {name}: {path}")
        else:
            print(f"✅ {name} path exists: {path}")
            
    # Initialize W&B Logger
    logger = WandbLogger(
        project="rural-road-extraction", 
        run_name=args.run_name, 
        config=vars(args),
        output_dir=args.output_dir
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Datasets & DataLoaders
    train_dataset = DeepGlobeDataset(
        image_dir=args.train_image_dir,
        mask_dir=args.train_mask_dir,
        transform=get_train_transforms()
    )
    val_dataset = DeepGlobeDataset(
        image_dir=args.val_image_dir,
        mask_dir=args.val_mask_dir,
        transform=get_val_transforms()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # 2. Model
    model = MobileViT_v2(num_classes=1, width_mult=args.width_mult).to(device)
    logger.log_config({"num_parameters": model.num_parameters})
    
    # 3. Loss & Optimizer
    loss_fn = RoadExtractionLoss(total_epochs=args.epochs, alpha_start=1.0, alpha_end=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    best_val_cldice = float('inf')
    
    # 4. Training Loop
    for epoch in range(args.epochs):
        train_loss, train_bce, train_cldice = train_one_epoch(
            epoch, model, train_loader, optimizer, scaler, loss_fn, logger, device
        )
        
        val_loss, val_bce, val_cldice = validate(epoch, model, val_loader, loss_fn, device)
        
        # Epoch-level logging
        logger.log_metrics({
            "epoch": epoch,
            "train/epoch_loss": train_loss,
            "train/epoch_bce": train_bce,
            "train/epoch_cldice": train_cldice,
            "val/epoch_loss": val_loss,
            "val/epoch_bce": val_bce,
            "val/epoch_cldice": val_cldice, # This is the key metric for checkpointing
            "alpha": loss_fn.get_alpha()
        }, step=epoch) # Use epoch as step for epoch-level logs
        
        print(f"Epoch [{epoch}/{args.epochs-1}] - "
              f"Train Loss: {train_loss:.4f} (clDice: {train_cldice:.4f}) | "
              f"Val Loss: {val_loss:.4f} (clDice: {val_cldice:.4f})")
        
        # 5. Checkpointing based on lowest Validation clDice Loss
        if val_cldice < best_val_cldice:
            best_val_cldice = val_cldice
            save_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cldice': best_val_cldice,
            }, save_path)
            print(f"--> Saved new best model to {save_path} (Val clDice: {best_val_cldice:.4f})")
            
            # Optionally log the artifact to W&B
            logger.log_artifact(save_path, artifact_type="model", name="best_model")

    logger.log_summary({"best_val_cldice": best_val_cldice})
    logger.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileViT v2 for Road Extraction")
    parser.add_argument("--train_image_dir", type=str, required=True, help="Directory containing training images")
    parser.add_argument("--train_mask_dir", type=str, required=True, help="Directory containing training masks")
    parser.add_argument("--val_image_dir", type=str, required=True, help="Directory containing validation images")
    parser.add_argument("--val_mask_dir", type=str, required=True, help="Directory containing validation masks")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save checkpoints and logs")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--width_mult", type=float, default=1.0, help="Model width multiplier")
    parser.add_argument("--num_workers", type=int, default=min(4, os.cpu_count() or 2), help="DataLoader workers")
    
    args = parser.parse_args()
    main(args)
