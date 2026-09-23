"""
train.py -- MLOps training script for Rural Road Extraction

Designed for GPU environments (Kaggle/Colab).
Features:
  - Automatic Mixed Precision (AMP)
  - Dynamic alpha decay for topology-aware loss
  - W&B integration with graceful fallback
  - Best-checkpoint saving based on validation IoU, hard-gated against
    the "predict everything as road" collapse mode (see validate()).
"""

import os
import sys
import argparse
import random
import numpy as np
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

from src.data.dataset import DeepGlobeDataset, get_train_transforms, get_val_transforms, estimate_pos_weight
from src.models.mobilevit_v2 import MobileViT_v2
from src.utils.loss import RoadExtractionLoss, avg_component_count
from src.utils.wandb_logger import WandbLogger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_warm_start(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    """
    Partially loads weights from a prior checkpoint (e.g. an older model without
    AttentionGate layers). Missing/unexpected keys are reported but not fatal --
    this lets a new architecture start from an encoder/decoder that already knows
    where roads are, instead of a fully random init that pos_weight can push
    into a blanket-positive collapse before the topology losses can correct it.
    """
    print(f"--- Warm-starting from {checkpoint_path} ---")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if (isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint) else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"    Loaded {len(state_dict) - len(unexpected)} matching tensors.")
    if missing:
        print(f"    Missing (randomly initialized): {missing}")
    if unexpected:
        print(f"    Unexpected (ignored): {unexpected}")


def train_one_epoch(epoch, model, dataloader, optimizer, scaler, loss_fn, logger, device, grad_clip_norm):
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
        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
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
    total_comp_count = 0.0
    total_iou = 0.0
    total_prec = 0.0
    total_pos_frac = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)

        with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            preds = model(images)
            loss, components = loss_fn(preds, masks, return_components=True)

        total_loss += loss.item()
        total_bce += components['bce_loss']
        total_cldice += components['cldice_loss']
        total_comp_count += avg_component_count(preds)

        # Calculate IoU, Precision, and Positive Fraction Canary Metrics
        probs = torch.sigmoid(preds)
        pred_binary = (probs > 0.5).float()
        target_binary = (masks > 0.5).float()

        pos_frac = pred_binary.mean().item()
        total_pos_frac += pos_frac

        intersection = (pred_binary * target_binary).sum(dim=(1, 2, 3))
        union = (pred_binary + target_binary).clamp(0, 1).sum(dim=(1, 2, 3))
        iou = (intersection + 1e-7) / (union + 1e-7)
        total_iou += iou.mean().item()

        prec = (intersection + 1e-7) / (pred_binary.sum(dim=(1, 2, 3)) + 1e-7)
        total_prec += prec.mean().item()

        pbar.set_postfix({"Val Loss": f"{loss.item():.4f}"})

    num_batches = len(dataloader)
    avg_pos_frac = total_pos_frac / num_batches
    if avg_pos_frac > 0.10:
        print(f"⚠️ CANARY WARNING: High predicted positive pixel fraction ({avg_pos_frac*100:.1f}% > 10%). Check model precision!")

    return (
        total_loss / num_batches,
        total_bce / num_batches,
        total_cldice / num_batches,
        total_comp_count / num_batches,
        total_iou / num_batches,
        total_prec / num_batches,
        avg_pos_frac
    )

def main(args):
    seed_everything(args.seed)

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

    # Estimate pos_weight dynamically if requested (clamped -- see estimate_pos_weight)
    pos_weight = args.pos_weight
    if args.auto_pos_weight and os.path.exists(args.train_image_dir):
        pos_weight = estimate_pos_weight(train_dataset, max_pos_weight=args.max_pos_weight)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    # Validation runs on full native-resolution tiles (16x the pixels of a training
    # crop), so it uses a proportionally smaller batch to stay within GPU memory.
    val_loader = DataLoader(val_dataset, batch_size=max(1, args.batch_size // 4), shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 2. Model
    model = MobileViT_v2(num_classes=1, width_mult=args.width_mult).to(device)
    if args.init_from:
        load_warm_start(model, args.init_from, device)
    logger.log_config({"num_parameters": model.num_parameters})

    # 3. Loss, Optimizer, & LR Scheduler
    loss_fn = RoadExtractionLoss(
        total_epochs=args.epochs,
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        pos_weight=pos_weight,
        decay_power=args.decay_power,
        alpha_decay_epochs=args.alpha_decay_epochs
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # Checkpointing is driven by val_iou (bounded [0, 1], directly punished by both
    # false positives and false negatives) rather than raw val_cldice, which we've
    # verified can score a "predict ~90% of every tile as road" collapse as *better*
    # than a correct sparse prediction (its sensitivity term saturates near 1.0 once
    # the prediction blankets the image). val_pos_frac is used as a hard gate on top:
    # regardless of any metric, a checkpoint predicting an implausible fraction of the
    # tile as road is never saved as "best".
    best_val_iou = -1.0
    patience_counter = 0

    # 4. Training Loop
    for epoch in range(args.epochs):
        train_loss, train_bce, train_cldice = train_one_epoch(
            epoch, model, train_loader, optimizer, scaler, loss_fn, logger, device, args.grad_clip_norm
        )

        val_loss, val_bce, val_cldice, val_components, val_iou, val_prec, val_pos_frac = validate(
            epoch, model, val_loader, loss_fn, device
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Epoch-level logging
        logger.log_metrics({
            "epoch": epoch,
            "train/epoch_loss": train_loss,
            "train/epoch_bce": train_bce,
            "train/epoch_cldice": train_cldice,
            "val/epoch_loss": val_loss,
            "val/epoch_bce": val_bce,
            "val/epoch_cldice": val_cldice, # Logged for visibility -- NOT used for checkpointing
            "val/component_count": val_components,
            "val/iou": val_iou,
            "val/precision": val_prec,
            "val/positive_frac": val_pos_frac,
            "alpha": loss_fn.get_alpha(),
            "lr": current_lr
        }, step=epoch)

        print(f"Epoch [{epoch}/{args.epochs-1}] - "
              f"Train Loss: {train_loss:.4f} (clDice: {train_cldice:.4f}) | "
              f"Val Loss: {val_loss:.4f} (clDice: {val_cldice:.4f}, IoU: {val_iou:.4f}, Prec: {val_prec:.4f}, PosFrac: {val_pos_frac*100:.1f}%) | "
              f"LR: {current_lr:.6f}")

        # 5. Checkpointing & Early Stopping based on Validation IoU, hard-gated
        #    against blanket-positive collapse (see comment above best_val_iou).
        collapsed = val_pos_frac > args.max_pos_frac
        if collapsed:
            print(f"🚫 REJECTED checkpoint at epoch {epoch}: predicted positive fraction "
                  f"({val_pos_frac*100:.1f}%) exceeds --max_pos_frac ({args.max_pos_frac*100:.1f}%). "
                  f"This looks like blob collapse, not a real road prediction -- not saving.")

        if (not collapsed) and val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            save_path = os.path.join(args.output_dir, "best_model_v2.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cldice': val_cldice,
                'val_iou': best_val_iou,
                'val_precision': val_prec,
                'val_positive_frac': val_pos_frac,
            }, save_path)
            print(f"--> Saved new best model to {save_path} (Val IoU: {best_val_iou:.4f}, clDice: {val_cldice:.4f}, PosFrac: {val_pos_frac*100:.1f}%)")

            logger.log_artifact(save_path, artifact_type="model", name="best_model_v2")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"🛑 Early stopping triggered after {patience_counter} epochs without val_iou improvement at epoch {epoch}.")
                break

    logger.log_summary({"best_val_iou": best_val_iou})
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
    parser.add_argument("--pos_weight", type=float, default=2.0, help="BCE loss positive class weight for imbalance")
    parser.add_argument("--auto_pos_weight", action="store_true", help="Dynamically compute pos_weight from training dataset")
    parser.add_argument("--max_pos_weight", type=float, default=3.0, help="Upper clamp for --auto_pos_weight (prevents runaway class-imbalance weighting)")
    parser.add_argument("--alpha_start", type=float, default=0.5, help="Starting alpha for BCE vs clDice loss weighting")
    parser.add_argument("--alpha_end", type=float, default=0.15, help="Ending alpha floor for loss weighting")
    parser.add_argument("--decay_power", type=float, default=0.5, help="Exponent for front-loaded alpha decay")
    parser.add_argument("--alpha_decay_epochs", type=int, default=40, help="Fixed epoch count for alpha decay curve")
    parser.add_argument("--patience", type=int, default=35, help="Patience epochs for early stopping on val_iou")
    parser.add_argument("--max_pos_frac", type=float, default=0.20, help="Hard collapse gate: never checkpoint if predicted positive pixel fraction exceeds this")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Max gradient norm (0 disables clipping)")
    parser.add_argument("--init_from", type=str, default=None, help="Optional checkpoint path to warm-start weights from (partial load, strict=False)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    main(args)
