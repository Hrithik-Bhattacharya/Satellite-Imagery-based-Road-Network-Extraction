"""Builds notebooks/train_v4.ipynb: a full, resumable training run for the v4 model.

Recipe (changes from v3):
  * 160 epochs, linear warmup then cosine decay, no early stopping;
  * 512 x 512 crops; flips, 90 degree rotations, small affine, colour, canopy shadow (active);
  * best epoch chosen on a separate validation set carved from the training tiles, so the
    623 test tiles used in the paper are never seen during training or model selection;
  * exponential moving average (EMA) of the weights;
  * full training state saved every epoch; a second Kaggle session resumes automatically.

Usage (repo root): python scripts/build_train_v4_notebook.py
"""
import json
import os

CELLS = [
("markdown", """# Train v4: long, resumable training with a held out test set

**Inputs:** the DeepGlobe road extraction dataset (`balraj98/deepglobe-road-extraction-dataset`).
To resume after a 12 hour limit, also add the previous run's output (Add Input, Your Work, this notebook).

**Settings:** GPU T4, Internet on. Run with *Save Version, Save & Run All (Commit)*.

**Outputs (in `/kaggle/working/v4/`):** `best_model_v4.pth`, `history.csv`, `training_curves.png`, `split.json`, `train_state.pth` (for resuming)."""),

("code", r'''# 1. Settings
EPOCHS = 160              # maximum epochs
PATIENCE = 40             # early stopping: stop if the best selection IoU has not improved for 40 epochs,
MIN_EPOCHS = 80           # but never before epoch 80 (cosine decay gives most of its gain late)
WARMUP_EPOCHS = 5
ALPHA_DECAY_EPOCHS = 60   # clDice weight rises over the first 60 epochs
CROP = 512
BATCH = 8                 # 8 x 512^2 = same pixels per step as 32 x 256^2
LR = 1e-3
MIN_LR = 1e-5
WEIGHT_DECAY = 1e-4
EMA_DECAY = 0.999
SELECT_VAL_TILES = 280    # model selection set, taken from the training tiles
MAX_POS_FRAC = 0.20       # collapse gate
TIME_LIMIT_H = 11.0       # stop and save before Kaggle's 12 hour limit
NUM_WORKERS = 4
SEED = 42
TEST_SHA1 = "2811fca6c9e44e6cdf713455a422502e4e5e4431"   # the 623 verified test tiles'''),

("code", r'''# 2. Code from the repository (model, loss, canopy augmentation)
import os, sys, glob, random, hashlib, subprocess, json, shutil, time, math, copy
REPO = "/kaggle/working/repo"
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "-q", "--depth", "1",
                    "https://github.com/Hrithik-Bhattacharya/Satellite-Imagery-based-Road-Network-Extraction.git", REPO],
                   check=True)
sys.path.insert(0, os.path.join(REPO, "backend"))
print(subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%h %s"], capture_output=True, text=True).stdout)

import numpy as np, cv2, torch, pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from src.models.mobilevit_v2 import MobileViT_v2
from src.utils.loss import RoadExtractionLoss
from src.data.dataset import CanopyShadowDropout

assert float(CanopyShadowDropout(p=0.5).p) == 0.5, "canopy augmentation would be inactive"
DEVICE = torch.device("cuda")
print("torch", torch.__version__, "| albumentations", A.__version__, "|", torch.cuda.get_device_name(0))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)'''),

("code", r'''# 3. Split: the same 623 test tiles as v3, plus a separate selection set from the training tiles
DG = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "train" in dirs:
        try:
            if any("sat.jpg" in f for f in os.listdir(os.path.join(root, "train"))):
                DG = os.path.join(root, "train"); break
        except OSError:
            pass
assert DG, "Attach the DeepGlobe road extraction dataset."

ids = [os.path.basename(f).replace("_sat.jpg", "") for f in glob.glob(os.path.join(DG, "*_sat.jpg"))]
ids = [i for i in ids if os.path.exists(os.path.join(DG, f"{i}_mask.png"))]
random.seed(42); random.shuffle(ids)            # identical to the v3 notebook
cut = int(len(ids) * 0.9)
pool, TEST_IDS = ids[:cut], ids[cut:]
assert hashlib.sha1("\n".join(TEST_IDS).encode()).hexdigest() == TEST_SHA1, "test split differs from v3"
rng = random.Random(7)
VAL_IDS = sorted(rng.sample(pool, SELECT_VAL_TILES))
TRAIN_IDS = sorted(set(pool) - set(VAL_IDS))
assert not (set(TRAIN_IDS) | set(VAL_IDS)) & set(TEST_IDS)
OUT = "/kaggle/working/v4"; os.makedirs(OUT, exist_ok=True)
sha = lambda l: hashlib.sha1("\n".join(l).encode()).hexdigest()
SPLIT = {"train": len(TRAIN_IDS), "select_val": len(VAL_IDS), "test": len(TEST_IDS),
         "train_sha1": sha(TRAIN_IDS), "select_val_sha1": sha(VAL_IDS), "test_sha1": sha(TEST_IDS)}
json.dump({**SPLIT, "select_val_ids": VAL_IDS, "test_ids": TEST_IDS}, open(f"{OUT}/split.json", "w"), indent=1)
print(SPLIT)'''),

("code", r'''# 4. Data
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
train_tf = A.Compose([
    A.RandomCrop(CROP, CROP),
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
    A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), border_mode=cv2.BORDER_REFLECT_101, p=0.3),
    A.RandomBrightnessContrast(0.2, 0.2, p=0.3),
    A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.3),
    CanopyShadowDropout(p=0.5),
    A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0), ToTensorV2()])
val_tf = A.Compose([A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0), ToTensorV2()])


class Tiles(Dataset):
    def __init__(self, ids, tf):
        self.ids, self.tf = ids, tf

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        tid = self.ids[i]
        img = cv2.cvtColor(cv2.imread(f"{DG}/{tid}_sat.jpg"), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(f"{DG}/{tid}_mask.png", cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        out = self.tf(image=img, mask=mask)
        return out["image"], out["mask"].float().unsqueeze(0)


train_loader = DataLoader(Tiles(TRAIN_IDS, train_tf), batch_size=BATCH, shuffle=True, drop_last=True,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(Tiles(VAL_IDS, val_tf), batch_size=4, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
print(len(train_loader), "steps per epoch")'''),

("code", r'''# 5. Model, loss, optimiser, schedule, EMA, resume
model = MobileViT_v2(num_classes=1, width_mult=1.0, attention_gates=True).to(DEVICE)
ema = copy.deepcopy(model).eval()
for p in ema.parameters():
    p.requires_grad_(False)
loss_fn = RoadExtractionLoss(total_epochs=EPOCHS, alpha_start=0.5, alpha_end=0.15, dice_weight=0.35,
                             pos_weight=2.0, decay_power=0.5, alpha_decay_epochs=ALPHA_DECAY_EPOCHS)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
TOTAL = EPOCHS * len(train_loader); WARM = WARMUP_EPOCHS * len(train_loader)


def lr_lambda(step):
    if step < WARM:
        return (step + 1) / WARM
    t = (step - WARM) / max(1, TOTAL - WARM)
    return MIN_LR / LR + (1 - MIN_LR / LR) * 0.5 * (1 + math.cos(math.pi * t))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
scaler = torch.amp.GradScaler("cuda")
start_epoch, best, history, ema_steps = 0, {"iou": -1.0, "epoch": 0}, [], 0

prev = sorted(glob.glob("/kaggle/input/**/train_state.pth", recursive=True)) + \
       ([f"{OUT}/train_state.pth"] if os.path.exists(f"{OUT}/train_state.pth") else [])
if prev:
    st = torch.load(prev[-1], map_location=DEVICE, weights_only=False)
    model.load_state_dict(st["model"]); ema.load_state_dict(st["ema"])
    opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
    start_epoch, best, history, ema_steps = st["epoch"] + 1, st["best"], st["history"], st["ema_steps"]
    for f in ("best_model_v4.pth",):
        src = os.path.join(os.path.dirname(prev[-1]), f)
        if os.path.exists(src) and not os.path.exists(f"{OUT}/{f}"):
            shutil.copy(src, f"{OUT}/{f}")
    print(f"RESUMED from {prev[-1]} at epoch {start_epoch}, best so far {best}")
print("parameters:", sum(p.numel() for p in model.parameters()))'''),

("code", r'''# 6. Validation (native 1024 x 1024 tiles, pooled pixel counts like the paper)
from src.utils.loss import soft_skel


@torch.no_grad()
def validate(net):
    net.eval()
    tp = fp = fn = 0.0
    pos, cl, n = 0.0, 0.0, 0
    for x, y in val_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.autocast("cuda"):
            p = torch.sigmoid(net(x)).float()
        b = (p > 0.5).float()
        tp += (b * y).sum().item(); fp += (b * (1 - y)).sum().item(); fn += ((1 - b) * y).sum().item()
        pos += b.mean((1, 2, 3)).sum().item()
        sp, sy = soft_skel(p, 10), soft_skel(y, 10)
        tprec = ((sp * y).sum((1, 2, 3)) + 1) / (sp.sum((1, 2, 3)) + 1)
        tsens = ((sy * p).sum((1, 2, 3)) + 1) / (sy.sum((1, 2, 3)) + 1)
        cl += (2 * tprec * tsens / (tprec + tsens + 1e-7)).sum().item()
        n += x.shape[0]
    return {"iou": tp / (tp + fp + fn + 1e-9), "precision": tp / (tp + fp + 1e-9),
            "recall": tp / (tp + fn + 1e-9), "positive_frac": pos / n, "soft_cldice": cl / n}'''),

("code", r'''# 7. Train
t_start = time.time()
stopped_for_time = False
for epoch in range(start_epoch, EPOCHS):
    t0 = time.time()
    model.train()
    alpha = loss_fn.update_alpha(epoch)
    run_loss, steps = 0.0, 0
    for x, y in train_loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda"):
            logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        ema_steps += 1
        d = min(EMA_DECAY, (1 + ema_steps) / (10 + ema_steps))
        with torch.no_grad():
            for (k, ve), vm in zip(ema.state_dict().items(), model.state_dict().values()):
                if ve.dtype.is_floating_point:
                    ve.mul_(d).add_(vm.detach(), alpha=1 - d)
                else:
                    ve.copy_(vm)
        run_loss += loss.item(); steps += 1

    v_raw, v_ema = validate(model), validate(ema)
    row = {"epoch": epoch, "lr": sched.get_last_lr()[0], "alpha": alpha, "train_loss": run_loss / steps,
           **{f"val_{k}": v for k, v in v_raw.items()}, **{f"ema_{k}": v for k, v in v_ema.items()},
           "epoch_min": (time.time() - t0) / 60}
    history.append(row)
    for tag, net, v in (("raw", model, v_raw), ("ema", ema, v_ema)):
        if v["positive_frac"] <= MAX_POS_FRAC and v["iou"] > best["iou"]:
            best = {"iou": v["iou"], "epoch": epoch, "weights": tag}
            torch.save({"model_state_dict": net.state_dict(), "epoch": epoch, "weights": tag,
                        "val_iou": v["iou"], "val_precision": v["precision"], "val_recall": v["recall"],
                        "val_positive_frac": v["positive_frac"], "val_soft_cldice": v["soft_cldice"],
                        "selection_set": "280 tiles from the training pool (split.json)",
                        "test_ids_sha1": TEST_SHA1}, f"{OUT}/best_model_v4.pth")
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                "best": best, "history": history, "ema_steps": ema_steps}, f"{OUT}/train_state.pth")
    pd.DataFrame(history).to_csv(f"{OUT}/history.csv", index=False)
    print(f"epoch {epoch:3d} | {row['epoch_min']:.1f} min | loss {row['train_loss']:.4f} | "
          f"val IoU raw {v_raw['iou']:.4f} ema {v_ema['iou']:.4f} | pos {v_ema['positive_frac']:.3f} | best {best}")
    if epoch + 1 >= MIN_EPOCHS and epoch - best["epoch"] >= PATIENCE:
        print(f"\nEARLY STOP at epoch {epoch}: no improvement for {PATIENCE} epochs.")
        break
    elapsed_h = (time.time() - t_start) / 3600
    if elapsed_h + 1.3 * row["epoch_min"] / 60 > TIME_LIMIT_H:
        stopped_for_time = True
        print(f"\nTIME LIMIT: stopped after epoch {epoch}. Start a new version with this output attached to resume.")
        break
print("DONE" if not stopped_for_time else "RESUME NEEDED", "| best:", best)'''),

("code", r'''# 8. Training curves
import matplotlib.pyplot as plt
h = pd.DataFrame(history)
fig, a = plt.subplots(1, 2, figsize=(10, 3.5))
a[0].plot(h.epoch, h.train_loss, label="training loss"); a[0].set_xlabel("epoch"); a[0].legend()
a[1].plot(h.epoch, 100 * h.val_iou, label="validation IoU"); a[1].plot(h.epoch, 100 * h.ema_iou, label="validation IoU (EMA)")
a[1].set_xlabel("epoch"); a[1].set_ylabel("%"); a[1].legend()
fig.tight_layout(); fig.savefig(f"{OUT}/training_curves.png", dpi=200)
print(sorted(os.listdir(OUT)))'''),
]


def main():
    nb = {"cells": [], "nbformat": 4, "nbformat_minor": 4,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}}
    for kind, src in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(True)}
        if kind == "code":
            compile(src, "<cell>", "exec")
            cell.update({"execution_count": None, "outputs": []})
        nb["cells"].append(cell)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks", "train_v4.ipynb")
    json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
