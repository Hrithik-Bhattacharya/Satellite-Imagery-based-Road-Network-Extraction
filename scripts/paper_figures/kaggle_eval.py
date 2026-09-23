# %% [markdown]
# # Road Extraction on DeepGlobe — Measured Evaluation & Paper Figures
#
# Every number and figure this notebook produces is **measured** on the DeepGlobe
# validation split with the trained checkpoints — nothing is simulated or hand-entered.
#
# **What it does**
# 1. Rebuilds the dataset exactly the way every training run did (same archive, same
#    extraction, same 90/10 split with seed 42).
# 2. **Verifies the split**: re-runs the training-time validation protocol and checks
#    the result against the metrics stored inside each checkpoint. A match to ~3
#    decimals means these are the same held-out tiles the checkpoints never trained on.
# 3. Evaluates every checkpoint at **native resolution** (the scale the model trains
#    and deploys at): IoU, precision, recall, F1, relaxed F1 (ρ = 3 px), clDice,
#    fragmentation, canopy-occluded recall; plus post-processing / TTA ablations,
#    bootstrap 95% CIs and a paired Wilcoxon test.
# 4. Writes 300-DPI PNG + vector PDF figures, `results.json`, `per_tile_metrics.csv`
#    and a LaTeX table into `/kaggle/working/paper_results/`, zipped for download.
#
# **Before running**
# * *Settings → Accelerator*: GPU (T4 or P100). *Internet*: on.
# * *Add Input* → your checkpoints dataset. Any of these filenames are recognized:
#   `best_model_v2.pth` (final model), `best_model_v2_epoch18_iou0159.pth`,
#   `best_model_v2_collapsed_epoch46.pth`, `best_model_new.pth` (June baseline).
# * *(Optional)* *Add Input* → *Notebooks* → your training notebook, so its
#   `wandb_local.json` log is picked up for the training-curve figure.
# * *Run All* (≈45–75 min, mostly CPU post-processing), then download
#   `paper_results.zip` from the link printed by the last cell.

# %%
# ── 1. Setup: dependencies and the dataset (identical source to every training run) ──
import os, sys, json, time, math, random, hashlib, subprocess, glob, zipfile, types, warnings
warnings.filterwarnings("ignore")

IN_KAGGLE = os.path.isdir("/kaggle/input")
WORK = os.environ.get("EVAL_WORK", "/kaggle/working")
DATA_DIR = os.environ.get("EVAL_DATA", os.path.join(WORK, "dataset"))
INPUT_ROOTS = os.environ.get("EVAL_INPUT", "/kaggle/input").split(os.pathsep)
NUM_WORKERS = 2 if IN_KAGGLE else 0   # DataLoader workers (0 for Windows smoke tests: no fork)
OUT_DIR = os.environ.get("EVAL_OUT", os.path.join(WORK, "paper_results"))
MAX_TILES = int(os.environ.get("EVAL_MAX_TILES", "0"))   # 0 = the full split. >0 is for smoke tests only.
FILE_ID = "1OI0XJ1-ejxd0JS45hBzJbAYe9_2hJJwN"               # the archive every training run downloaded
SEED = 42
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

if IN_KAGGLE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown", "albumentations", "scikit-image"], check=True)

if not os.path.isdir(os.path.join(DATA_DIR, "train")):
    import gdown
    zip_path = os.path.join(WORK, "archive.zip")
    if not os.path.exists(zip_path):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", zip_path, quiet=False)
    # Same extraction command as the training notebooks: the directory listing order it
    # produces is what the seed-42 shuffle was applied to.
    subprocess.run(["unzip", "-q", zip_path, "-d", DATA_DIR], check=True)
TRAIN_DIR = os.path.join(DATA_DIR, "train")
print("train dir:", TRAIN_DIR, "| files:", len(os.listdir(TRAIN_DIR)))

# %%
# ── 2. Imports and environment record ──
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
from scipy import stats as sstats
import matplotlib
if not IN_KAGGLE:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False


class _OriginalCanopyShadowDropout(A.ImageOnlyTransform):
    """The exact constructor call the training runs used, to check whether the canopy
    augmentation was actually active in this albumentations version."""
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)


try:
    _orig_p = float(_OriginalCanopyShadowDropout(p=0.5).p)
except Exception as e:  # albumentations versions that reject the extra positional arg
    _orig_p = f"constructor raised {type(e).__name__}: {e}"

ENV = {
    "device": str(DEVICE),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "torch": torch.__version__, "albumentations": A.__version__, "opencv": cv2.__version__,
    "canopy_aug_effective_p_with_training_code": _orig_p,
}
print(json.dumps(ENV, indent=2))
if isinstance(_orig_p, float) and _orig_p == 0.0:
    print("NOTE: with this albumentations version the training notebooks' CanopyShadowDropout "
          "had p = 0.0, i.e. canopy augmentation was NOT applied during those runs.")

# %%
# ── 3. Model, loss and post-processing code (inlined verbatim from the repository) ──
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # <<LOCAL-ONLY>>
sys.path[:0] = [_REPO, os.path.join(_REPO, "scripts", "paper_figures")]  # <<LOCAL-ONLY>>
# <<INLINE:backend/src/models/mobilevit_v2.py>>
from backend.src.models.mobilevit_v2 import MobileViT_v2, load_mobilevit_checkpoint  # <<LOCAL-ONLY>>

# <<INLINE:backend/src/utils/graph_postprocess.py>>
from backend.src.utils.graph_postprocess import hysteresis_threshold, connect_canopy_gaps  # <<LOCAL-ONLY>>

# <<INLINE:backend/src/utils/loss.py>>
from backend.src.utils.loss import SoftClDiceLoss, soft_skel  # <<LOCAL-ONLY>>

import style  # <<MODULE:scripts/paper_figures/style.py>>

# %%
# ── 4. Checkpoints and training logs attached as inputs ──
ROLE_BY_FILENAME = {
    "best_model_v2.pth": "final",
    "best_model_v2_new.pth": "final",
    "best_model_v2_epoch18_iou0159.pth": "run1",
    "best_model_v2_collapsed_epoch46.pth": "collapsed",
    "best_model_new.pth": "baseline",
}
ROLE_ORDER = ["baseline", "collapsed", "run1", "final"]


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


CHECKPOINTS = {}
for path in sorted(p for root in INPUT_ROOTS for p in glob.glob(os.path.join(root, "**", "*.pth"), recursive=True)):
    role = ROLE_BY_FILENAME.get(os.path.basename(path))
    if role is None:
        print("ignored (unrecognized name):", path)
        continue
    digest = md5(path)
    if role in CHECKPOINTS:
        if CHECKPOINTS[role]["md5"] != digest:
            print(f"WARNING: two different files for role '{role}'; keeping {CHECKPOINTS[role]['path']}")
        continue
    _, meta = load_mobilevit_checkpoint(path, device="cpu")
    CHECKPOINTS[role] = {"path": path, "md5": digest,
                         "meta": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in meta.items()}}
ROLES = [r for r in ROLE_ORDER if r in CHECKPOINTS]
assert "final" in ROLES, "Attach a dataset containing best_model_v2.pth (the final checkpoint)."
for r in ROLES:
    print(f"{r:10s} {CHECKPOINTS[r]['path']}  stored={CHECKPOINTS[r]['meta']}")

TRAINING_LOGS = []
for path in sorted(p for root in INPUT_ROOTS
                   for p in glob.glob(os.path.join(root, "**", "wandb_local*.json"), recursive=True)):
    try:
        records = json.load(open(path))
        epochs = sorted([r for r in records if "val/epoch_loss" in r], key=lambda r: r["epoch"])
        if epochs:
            TRAINING_LOGS.append({"path": path, "epochs": epochs, "has_iou": "val/iou" in epochs[0]})
            print(f"training log: {path} ({len(epochs)} epochs, val IoU logged: {'val/iou' in epochs[0]})")
    except Exception as e:
        print("could not read", path, e)

# %%
# ── 5. Reproduce the training split exactly ──
def list_ids(img_dir, mask_dir):
    """Identical to the training notebooks' DeepGlobeDataset id listing (directory order)."""
    ids = []
    for f in os.listdir(img_dir):
        if f.endswith(".jpg"):
            tid = f.split("_")[0]
            if os.path.exists(os.path.join(mask_dir, f"{tid}_mask.png")):
                ids.append(tid)
    return ids


ALL_IDS = list_ids(TRAIN_DIR, TRAIN_DIR)
_shuffled = list(ALL_IDS)
random.Random(SEED).shuffle(_shuffled)
VAL_IDS = _shuffled[: max(1, int(len(_shuffled) * 0.1))]
EVAL_IDS = VAL_IDS[:MAX_TILES] if MAX_TILES else VAL_IDS
SPLIT = {
    "n_labeled": len(ALL_IDS), "n_val": len(VAL_IDS), "n_eval": len(EVAL_IDS),
    "listing_order_sha1": hashlib.sha1("\n".join(ALL_IDS).encode()).hexdigest(),
    "val_ids_sha1": hashlib.sha1("\n".join(VAL_IDS).encode()).hexdigest(),
}
json.dump({"val_ids": VAL_IDS, **SPLIT}, open(os.path.join(OUT_DIR, "val_ids.json"), "w"), indent=1)
print(SPLIT, "| first val ids:", VAL_IDS[:5])

# %%
# ── 6. Verify the split: re-run the training-time validation protocol ──
# Training validated on tiles RESIZED to 256x256, batch 16, AMP, and stored the resulting
# metrics in each checkpoint. Reproducing them to within TOL on these tiles is strong
# evidence the split is the same, i.e. no validation tile was ever trained on.
TOL = 2e-3
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class TrainingProtocolDS(Dataset):
    tf = A.Compose([A.Resize(height=256, width=256),
                    A.Normalize(mean=IMNET_MEAN, std=IMNET_STD, max_pixel_value=255.0), ToTensorV2()])

    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        tid = self.ids[i]
        image = cv2.cvtColor(cv2.imread(os.path.join(TRAIN_DIR, f"{tid}_sat.jpg")), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(TRAIN_DIR, f"{tid}_mask.png"), cv2.IMREAD_GRAYSCALE)
        out = self.tf(image=image, mask=mask)
        m = out["mask"]
        m = (m.to(torch.float32) if torch.is_tensor(m) else torch.tensor(m, dtype=torch.float32)) / 255.0
        return out["image"], m.unsqueeze(0)


def soft_cldice_per_image(logits_f32, target, num_iter=10, smooth=1.0):
    """SoftClDiceLoss's score, returned per image instead of batch-averaged."""
    p = torch.sigmoid(logits_f32)
    sp, st = soft_skel(p, num_iter), soft_skel(target, num_iter)
    tprec = ((sp * target).sum((1, 2, 3)) + smooth) / (sp.sum((1, 2, 3)) + smooth)
    tsens = ((st * p).sum((1, 2, 3)) + smooth) / (st.sum((1, 2, 3)) + smooth)
    return 2.0 * tprec * tsens / (tprec + tsens + 1e-7)


@torch.no_grad()
def training_protocol_eval(model, ids):
    loader = DataLoader(TrainingProtocolDS(ids), batch_size=16, shuffle=False, num_workers=NUM_WORKERS)
    cl = SoftClDiceLoss(num_iter=10, smooth=1.0)
    sums = {"iou": 0.0, "precision": 0.0, "positive_frac": 0.0, "cldice": 0.0}
    per_img_iou, per_img_cl, n_batches = [], [], 0
    for images, masks in loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            preds = model(images)
        sums["cldice"] += cl(preds.float(), masks.float()).item()     # training computed this in fp32
        per_img_cl.append(soft_cldice_per_image(preds.float(), masks.float()).cpu())
        pb = (torch.sigmoid(preds) > 0.5).float()                     # training thresholded the AMP output
        tb = (masks > 0.5).float()
        inter = (pb * tb).sum((1, 2, 3))
        union = (pb + tb).clamp(0, 1).sum((1, 2, 3))
        iou = (inter + 1e-7) / (union + 1e-7)
        sums["iou"] += iou.mean().item()
        sums["precision"] += ((inter + 1e-7) / (pb.sum((1, 2, 3)) + 1e-7)).mean().item()
        sums["positive_frac"] += pb.mean().item()
        per_img_iou.append(iou.cpu())
        n_batches += 1
    res = {k: v / n_batches for k, v in sums.items()}
    return res, torch.cat(per_img_iou).numpy(), torch.cat(per_img_cl).numpy()


VERIFY, TRAIN_PROTOCOL, TRAIN_PROTOCOL_PER_TILE = {}, {}, {}
STORED_KEYS = {"val_iou": "iou", "val_precision": "precision", "val_positive_frac": "positive_frac", "val_cldice": "cldice"}
for role in ROLES:
    model, _ = load_mobilevit_checkpoint(CHECKPOINTS[role]["path"], device=DEVICE)
    res, per_iou, per_cl = training_protocol_eval(model, VAL_IDS)   # always the full split: stored values cover it
    res["cldice_score"] = 1.0 - res["cldice"]
    TRAIN_PROTOCOL[role] = res
    TRAIN_PROTOCOL_PER_TILE[role] = {"iou": per_iou, "soft_cldice": per_cl}
    checks = {}
    for stored_key, ours in STORED_KEYS.items():
        if stored_key in CHECKPOINTS[role]["meta"]:
            stored = CHECKPOINTS[role]["meta"][stored_key]
            checks[stored_key] = {"stored": stored, "recomputed": res[ours], "abs_diff": abs(stored - res[ours]),
                                  "match": abs(stored - res[ours]) <= TOL}
    VERIFY[role] = checks
    del model
    torch.cuda.empty_cache()
    print(f"\n{role}: recomputed training-protocol metrics {({k: round(v, 5) for k, v in res.items()})}")
    for k, c in checks.items():
        print(f"   {k:18s} stored={c['stored']:.5f} recomputed={c['recomputed']:.5f} "
              f"diff={c['abs_diff']:.5f} {'MATCH' if c['match'] else 'MISMATCH'}")

_v2_checks = [c for r in ("final", "run1") if r in VERIFY for c in VERIFY[r].values()]
SPLIT_VERIFIED = bool(_v2_checks) and all(c["match"] for c in _v2_checks)
print("\nSPLIT VERIFIED" if SPLIT_VERIFIED else
      "\n*** SPLIT NOT VERIFIED — stored metrics were not reproduced; these tiles may include training "
      "tiles, so the metrics below would be optimistic. Do not report them as held-out results. ***")

# %%
# ── 7. Canopy (vegetation) mask: Excess-Green index + Otsu threshold ──
# Road pixels whose image colour is vegetation-like are treated as canopy-occluded.
# ExG = 2g - r - b on chromatic coordinates (Woebbecke et al., 1995); the threshold is
# Otsu's on pixels sampled from the evaluated tiles -- data-driven, not hand-tuned.
def exg(rgb):
    rgb = rgb.astype(np.float32)
    s = rgb.sum(axis=2) + 1e-6
    return (2 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]) / s


def load_tile(tid):
    rgb = cv2.cvtColor(cv2.imread(os.path.join(TRAIN_DIR, f"{tid}_sat.jpg")), cv2.COLOR_BGR2RGB)
    gt = cv2.imread(os.path.join(TRAIN_DIR, f"{tid}_mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    return rgb, gt


_rng = np.random.default_rng(0)
_samples = []
for tid in EVAL_IDS:
    e = exg(load_tile(tid)[0]).ravel()
    _samples.append(e[_rng.integers(0, e.size, 4000)])
VEG_T = float(threshold_otsu(np.concatenate(_samples)))
print(f"ExG Otsu threshold = {VEG_T:.4f}")

# %%
# ── 8. Native-resolution evaluation ──
RHO = 3   # relaxed-matching tolerance in pixels (1.5 m at 0.5 m/px), after Mnih & Hinton (2010)
# Which binarizations to evaluate, per checkpoint and probability source.
#   raw        : probability > 0.5
#   hyst       : hysteresis thresholding (0.35 / 0.12)
#   hyst_close : + 5x5 morphological closing
#   full_orig  : + canopy-gap bridging as originally written (also bridges tile-border exits)
#   full       : + canopy-gap bridging with the border fix  (the deployed post-processing)
# Sources: "single" = one forward pass, "tta" = mean of 4 flips (the deployed inference).
VARIANTS = {
    "final":     {"single": ["raw", "hyst", "hyst_close", "full_orig", "full"], "tta": ["raw", "full_orig", "full"]},
    "baseline":  {"single": ["raw", "full"], "tta": ["full"]},
    "run1":      {"single": ["raw"], "tta": ["full"]},
    "collapsed": {"single": ["raw"], "tta": []},
}
COUNT_KEYS = ["tp", "fp", "fn", "npred", "rel_tp_pred", "rel_tp_gt", "sk_p", "sk_p_in_g", "sk_g", "sk_g_in_p",
              "can_gt", "can_hit", "can_hit_rel", "open_gt", "open_hit", "open_hit_rel"]


class NativeDS(Dataset):
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        rgb = cv2.cvtColor(cv2.imread(os.path.join(TRAIN_DIR, f"{self.ids[i]}_sat.jpg")), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - np.array(IMNET_MEAN, np.float32)) / np.array(IMNET_STD, np.float32)
        return i, torch.from_numpy(x.transpose(2, 0, 1))


@torch.no_grad()
def forward_probs(model, x, tta):
    with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
        p = torch.sigmoid(model(x)).float()
        if not tta:
            return p, None
        acc = p.clone()
        for dims in ([3], [2], [2, 3]):
            acc += torch.flip(torch.sigmoid(model(torch.flip(x, dims))).float(), dims)
    return p, acc / 4.0


def binarize_all(probs, variants):
    out = {}
    if "raw" in variants:
        out["raw"] = probs > 0.5
    if any(v in variants for v in ("hyst", "hyst_close", "full_orig", "full")):
        hyst = hysteresis_threshold(probs, high_thresh=0.35, low_thresh=0.12)
        if "hyst" in variants:
            out["hyst"] = hyst > 0
        closed = cv2.morphologyEx(hyst, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        if "hyst_close" in variants:
            out["hyst_close"] = closed > 0
        if "full_orig" in variants:
            out["full_orig"] = connect_canopy_gaps(closed, max_gap_dist=220.0, max_angle_deg=65.0,
                                                   road_width=6, border_margin=0) > 0
        if "full" in variants:
            out["full"] = connect_canopy_gaps(closed, max_gap_dist=220.0, max_angle_deg=65.0,
                                              road_width=6, border_margin=16) > 0
    return out


def count_metrics(P, G, V, near_g, sk_g):
    if P.any():
        near_p = cv2.distanceTransform((~P).astype(np.uint8), cv2.DIST_L2, 3) <= RHO
    else:
        near_p = np.zeros_like(P)
    sk_p = skeletonize(P)
    gv, go = G & V, G & ~V
    return {
        "tp": int((P & G).sum()), "fp": int((P & ~G).sum()), "fn": int((~P & G).sum()), "npred": int(P.sum()),
        "rel_tp_pred": int((P & near_g).sum()), "rel_tp_gt": int((G & near_p).sum()),
        "sk_p": int(sk_p.sum()), "sk_p_in_g": int((sk_p & G).sum()),
        "sk_g": int(sk_g.sum()), "sk_g_in_p": int((sk_g & P).sum()),
        "cc_pred": int(cv2.connectedComponents(P.astype(np.uint8), connectivity=8)[0] - 1),
        "can_gt": int(gv.sum()), "can_hit": int((gv & P).sum()), "can_hit_rel": int((gv & near_p).sum()),
        "open_gt": int(go.sum()), "open_hit": int((go & P).sum()), "open_hit_rel": int((go & near_p).sum()),
    }


def tile_worker(job):
    tid, probs_by_src, variants = job
    rgb, G = load_tile(tid)
    V = exg(rgb) > VEG_T
    near_g = (cv2.distanceTransform((~G).astype(np.uint8), cv2.DIST_L2, 3) <= RHO) if G.any() else np.zeros_like(G)
    sk_g = skeletonize(G)
    rec = {"tile": tid, "gt_pixels": int(G.sum()), "canopy_ratio": float((G & V).sum() / max(int(G.sum()), 1)),
           "cc_gt": int(cv2.connectedComponents(G.astype(np.uint8), connectivity=8)[0] - 1),
           "pixels": int(G.size), "variants": {}, "hist": {}}
    for src, vlist in variants.items():
        if not vlist and src != "single":
            continue
        probs = probs_by_src[src].astype(np.float32)
        q = np.clip(np.rint(probs * 255), 0, 255).astype(np.int64)
        rec["hist"][src] = (np.bincount(q[G], minlength=256), np.bincount(q[~G], minlength=256))
        for v, P in binarize_all(probs, vlist).items():
            rec["variants"][f"{src}_{v}"] = count_metrics(P, G, V, near_g, sk_g)
    return rec


def _pool():
    import multiprocessing as mp
    if "fork" in mp.get_all_start_methods():
        return mp.get_context("fork").Pool(os.cpu_count())
    return None   # e.g. Windows smoke tests: run serially


RECORDS = {}        # role -> list of per-tile records
HIST = {}           # role -> src -> (pos_hist, neg_hist)
t_start = time.time()
pool = _pool()
for role in ROLES:
    model, _ = load_mobilevit_checkpoint(CHECKPOINTS[role]["path"], device=DEVICE)
    need_tta = bool(VARIANTS[role]["tta"])
    loader = DataLoader(NativeDS(EVAL_IDS), batch_size=4 if DEVICE.type == "cuda" else 1,
                        shuffle=False, num_workers=NUM_WORKERS)
    RECORDS[role], HIST[role], jobs = [], {}, []

    def _flush(jobs):
        results = pool.map(tile_worker, jobs) if pool else [tile_worker(j) for j in jobs]
        for rec in results:
            for src, (hp, hn) in rec.pop("hist").items():
                a = HIST[role].setdefault(src, [np.zeros(256, np.int64), np.zeros(256, np.int64)])
                a[0] += hp; a[1] += hn
            RECORDS[role].append(rec)

    for idx, x in loader:
        p, pt = forward_probs(model, x.to(DEVICE), need_tta)
        for k, i in enumerate(idx.tolist()):
            srcs = {"single": p[k, 0].cpu().numpy().astype(np.float16)}
            if pt is not None:
                srcs["tta"] = pt[k, 0].cpu().numpy().astype(np.float16)
            jobs.append((EVAL_IDS[i], srcs, VARIANTS[role]))
        if len(jobs) >= 32:
            _flush(jobs); jobs = []
    if jobs:
        _flush(jobs)
    del model
    torch.cuda.empty_cache()
    print(f"{role}: {len(RECORDS[role])} tiles evaluated ({(time.time() - t_start) / 60:.1f} min elapsed)")
if pool:
    pool.close()

# %%
# ── 9. Aggregate: dataset-level metrics, bootstrap CIs, paired tests ──
def derive(s):
    div = lambda a, b: a / b if b else float("nan")
    gt = s["tp"] + s["fn"]
    iou = div(s["tp"], s["tp"] + s["fp"] + s["fn"])
    p, r = div(s["tp"], s["tp"] + s["fp"]), div(s["tp"], gt)
    rp, rr = div(s["rel_tp_pred"], s["npred"]), div(s["rel_tp_gt"], gt)
    tprec, tsens = div(s["sk_p_in_g"], s["sk_p"]), div(s["sk_g_in_p"], s["sk_g"])
    f = lambda a, b: div(2 * a * b, a + b)
    return {"iou": iou, "precision": p, "recall": r, "f1": f(p, r),
            "relaxed_precision": rp, "relaxed_recall": rr, "relaxed_f1": f(rp, rr),
            "cldice": f(tprec, tsens), "topo_precision": tprec, "topo_sensitivity": tsens,
            "canopy_recall": div(s["can_hit"], s["can_gt"]), "canopy_recall_relaxed": div(s["can_hit_rel"], s["can_gt"]),
            "open_recall": div(s["open_hit"], s["open_gt"]), "open_recall_relaxed": div(s["open_hit_rel"], s["open_gt"])}


def count_matrix(role, variant):
    recs = RECORDS[role]
    return np.array([[r["variants"][variant][k] for k in COUNT_KEYS] for r in recs], dtype=np.float64)


def summarize(mat):
    return derive(dict(zip(COUNT_KEYS, mat.sum(0))))


def bootstrap_ci(mat, keys=("iou", "f1", "relaxed_f1", "cldice"), reps=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = mat.shape[0]
    draws = {k: [] for k in keys}
    for _ in range(reps):
        d = summarize(mat[rng.integers(0, n, n)])
        for k in keys:
            draws[k].append(d[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in draws.items()}


def per_tile_table(role, variant):
    rows = []
    for r in RECORDS[role]:
        c = r["variants"][variant]
        d = derive(c)
        union = c["tp"] + c["fp"] + c["fn"]
        rows.append({"tile": r["tile"], "model": role, "variant": variant,
                     "iou": d["iou"] if union else np.nan, "precision": d["precision"], "recall": d["recall"],
                     "f1": d["f1"], "relaxed_f1": d["relaxed_f1"], "cldice": d["cldice"],
                     "components_pred": c["cc_pred"], "components_gt": r["cc_gt"],
                     "positive_frac": c["npred"] / r["pixels"], "gt_frac": r["gt_pixels"] / r["pixels"],
                     "canopy_ratio": r["canopy_ratio"], "gt_pixels": r["gt_pixels"]})
    return pd.DataFrame(rows)


CONFIGS = []   # (role, variant) pairs actually evaluated, in display order
for role in ROLES:
    for src in ("single", "tta"):
        for v in VARIANTS[role][src]:
            CONFIGS.append((role, f"{src}_{v}"))

RESULTS = {}
PER_TILE = pd.concat([per_tile_table(r, v) for r, v in CONFIGS], ignore_index=True)
for role, variant in CONFIGS:
    mat = count_matrix(role, variant)
    pt = PER_TILE[(PER_TILE.model == role) & (PER_TILE.variant == variant)]
    RESULTS.setdefault(role, {})[variant] = {
        **summarize(mat), "ci95": bootstrap_ci(mat),
        "mean_tile_iou": float(np.nanmean(pt.iou)), "median_components_per_tile": float(np.median(pt.components_pred)),
        "median_gt_components_per_tile": float(np.median(pt.components_gt)),
        "mean_positive_frac": float(pt.positive_frac.mean()), "mean_gt_frac": float(pt.gt_frac.mean()),
    }

DEPLOYED = "tta_full"


def paired(role_a, var_a, role_b, var_b, metric="iou"):
    a = PER_TILE[(PER_TILE.model == role_a) & (PER_TILE.variant == var_a)].set_index("tile")[metric]
    b = PER_TILE[(PER_TILE.model == role_b) & (PER_TILE.variant == var_b)].set_index("tile")[metric]
    df = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    diff = df.a - df.b
    w = sstats.wilcoxon(df.a, df.b) if len(df) > 10 and (diff != 0).any() else None
    return {"n_tiles": int(len(df)), "mean_diff": float(diff.mean()), "median_diff": float(diff.median()),
            "frac_tiles_improved": float((diff > 0).mean()), "wilcoxon_p": float(w.pvalue) if w else None}


TESTS = {}
if "baseline" in ROLES:
    TESTS["final_vs_baseline_deployed_iou"] = paired("final", DEPLOYED, "baseline", DEPLOYED)
    TESTS["final_vs_baseline_single_raw_iou"] = paired("final", "single_raw", "baseline", "single_raw")
    TESTS["final_vs_baseline_deployed_cldice"] = paired("final", DEPLOYED, "baseline", DEPLOYED, "cldice")
TESTS["final_tta_vs_single_full_iou"] = paired("final", "tta_full", "final", "single_full")

# Canopy tertiles over tiles that actually contain road.
_road = PER_TILE[(PER_TILE.model == "final") & (PER_TILE.variant == DEPLOYED) & (PER_TILE.gt_pixels >= 1000)]
TERTILE_EDGES = np.quantile(_road.canopy_ratio, [0, 1 / 3, 2 / 3, 1]).tolist()
TILE_TERTILE = {t: int(np.clip(np.searchsorted(TERTILE_EDGES[1:-1], c, side="right"), 0, 2))
                for t, c in zip(_road.tile, _road.canopy_ratio)}
CANOPY_TERTILES = {}
for role, variant in CONFIGS:
    if variant not in ("single_raw", DEPLOYED):
        continue
    ids = [r["tile"] for r in RECORDS[role]]
    mat = count_matrix(role, variant)
    for t in range(3):
        sel = np.array([TILE_TERTILE.get(i) == t for i in ids])
        if sel.sum() >= 5:
            CANOPY_TERTILES.setdefault(f"{role}/{variant}", {})[t] = {
                **{k: v for k, v in summarize(mat[sel]).items() if k in ("iou", "relaxed_f1", "cldice", "recall")},
                "ci95": bootstrap_ci(mat[sel], keys=("relaxed_f1", "cldice")), "n_tiles": int(sel.sum())}

# Threshold sweeps from the pooled probability histograms (exact at 1/255 resolution).
SWEEPS = {}
for role in ROLES:
    for src, (hp, hn) in HIST[role].items():
        tp = hp[::-1].cumsum()[::-1].astype(np.float64)   # pixels with q >= t
        fp = hn[::-1].cumsum()[::-1].astype(np.float64)
        pos = float(hp.sum())
        th = np.arange(256) / 255.0
        with np.errstate(divide="ignore", invalid="ignore"):
            prec = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
            rec = tp / pos
            iou = tp / (fp + pos)
            f1 = 2 * prec * rec / (prec + rec)
        SWEEPS[f"{role}/{src}"] = {"threshold": th.tolist(), "precision": prec.tolist(), "recall": rec.tolist(),
                                   "iou": iou.tolist(), "f1": f1.tolist()}

print(json.dumps({r: {v: {k: round(x, 4) for k, x in m.items() if isinstance(x, float)}
                      for v, m in d.items()} for r, d in RESULTS.items()}, indent=1)[:4000])
print(json.dumps(TESTS, indent=1))

# %%
# ── 10. Save results: JSON, per-tile CSV, LaTeX table ──
NAMES = {"baseline": "Baseline (June, no attention gates)", "collapsed": "v2, collapsed run (ep. 46)",
         "run1": "v2, run 1 (ep. 18)", "final": "Proposed v2 (ep. 53)"}
VARIANT_NAMES = {"single_raw": "single pass, p > 0.5", "single_hyst": "hysteresis",
                 "single_hyst_close": "hysteresis + closing", "single_full_orig": "+ gap bridging (original)",
                 "single_full": "+ gap bridging (border fix)", "tta_raw": "TTA, p > 0.5",
                 "tta_full_orig": "TTA + post-proc. (original bridging)",
                 "tta_full": "TTA + post-proc. (deployed)"}

OUT = {
    "env": ENV, "split": SPLIT, "split_verified": SPLIT_VERIFIED, "verification": VERIFY,
    "training_protocol_metrics": TRAIN_PROTOCOL, "checkpoints": CHECKPOINTS,
    "native_resolution": RESULTS, "paired_tests": TESTS,
    "canopy": {"exg_otsu_threshold": VEG_T, "tertile_edges": TERTILE_EDGES, "by_tertile": CANOPY_TERTILES},
    "threshold_sweeps": SWEEPS, "relaxed_tolerance_px": RHO,
    "protocol": {"resolution": "native 1024x1024 tiles", "threshold": 0.5,
                 "iou": "dataset-level (pixels pooled over all tiles), road class",
                 "clDice": "hard, skimage skeletonize, pooled counts",
                 "ci": "95% percentile bootstrap over tiles, 1000 resamples"},
}
json.dump(OUT, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=1, default=float)
PER_TILE.to_csv(os.path.join(OUT_DIR, "per_tile_metrics.csv"), index=False)

pct = lambda v: f"{100 * v:.1f}" if v == v else "--"
rows = []
for role, variant in CONFIGS:
    if variant not in ("single_raw", DEPLOYED):
        continue
    m = RESULTS[role][variant]
    ci = m["ci95"]["iou"]
    rows.append(f"{NAMES[role]} & {VARIANT_NAMES[variant]} & {pct(m['iou'])} [{pct(ci[0])}, {pct(ci[1])}] & "
                f"{pct(m['precision'])} & {pct(m['recall'])} & {pct(m['f1'])} & {pct(m['relaxed_f1'])} & "
                f"{pct(m['cldice'])} & {m['median_components_per_tile']:.0f} \\\\")
latex = (
    "% Generated by notebooks/evaluate_for_paper.ipynb -- measured values, do not edit by hand.\n"
    f"% Split verified against checkpoint-stored metrics: {SPLIT_VERIFIED}. Tiles: {SPLIT['n_eval']}.\n"
    "\\begin{table*}[t]\n\\centering\n"
    f"\\caption{{Road extraction on the DeepGlobe validation split ({SPLIT['n_eval']} tiles, native "
    "$1024\\times1024$ resolution). Dataset-level pixel metrics (\\%); IoU with 95\\% bootstrap CI; "
    f"relaxed F1 uses a $\\rho={RHO}$\\,px tolerance; clDice on hard skeletons.}}\n"
    "\\label{tab:deepglobe_results}\n\\small\n"
    "\\begin{tabular}{llccccccc}\n\\toprule\n"
    "Model & Inference & IoU & Prec. & Rec. & F1 & Relaxed F1 & clDice & Comp./tile \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
open(os.path.join(OUT_DIR, "results_table.tex"), "w").write(latex)
print(latex)

# %%
# ── 11. Figures ──
style.apply()
C, L = style.MODEL_COLORS, style.MODEL_LABELS
SPLIT_NOTE = "" if SPLIT_VERIFIED else "  [split NOT verified]"
FOOT = f"DeepGlobe validation split, {SPLIT['n_eval']} tiles, native 1024\u00b2 resolution{SPLIT_NOTE}"


def footnote(fig, text=FOOT, y=None):
    """Placed below everything already drawn (tick labels, x-labels, legends), so it never collides."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bottom = min(a.get_tightbbox(renderer).y0 for a in fig.axes)
    for leg in fig.legends:
        bottom = min(bottom, leg.get_window_extent(renderer).y0)
    y_auto = bottom / fig.bbox.height - 0.025
    fig.text(0.0, min(y_auto, y) if y is not None else y_auto, text, fontsize=6.5, color=style.MUTED, va="top")


def hbars(ax, labels, values, colors, cis=None, fmt="{:.1f}", hatches=None, xmax=None):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.64, color=colors, edgecolor="white", linewidth=1.0,
            hatch=None if hatches is None else None)
    if hatches:
        for yi, v, h in zip(y, values, hatches):
            if h:
                ax.barh(yi, v, height=0.64, color="none", edgecolor="white", linewidth=0, hatch=h)
    if cis:
        for yi, (lo, hi) in zip(y, cis):
            ax.plot([lo, hi], [yi, yi], color=style.INK, lw=0.8, solid_capstyle="butt")
    reach = max((ci[1] if ci else v) for v, ci in zip(values, cis or [None] * len(values)) if v == v)
    top = xmax or reach * 1.22
    for yi, v, ci in zip(y, values, cis or [None] * len(values)):
        xpos = (ci[1] if ci else v) + top * 0.015
        ax.text(xpos, yi, fmt.format(v), va="center", fontsize=6.8, color=style.INK_2)
    ax.set_xlim(0, top)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(axis="y", visible=False)
    return y


# 11a. Validation protocol: 256-resize (training-time) vs native resolution
fig, ax = plt.subplots(figsize=(style.COL_W, 0.42 * len(ROLES) + 0.75))
y = np.arange(len(ROLES))[::-1]
for yi, role in zip(y, ROLES):
    a = 100 * TRAIN_PROTOCOL[role]["iou"]
    b = 100 * RESULTS[role]["single_raw"]["iou"]
    ax.plot([a, b], [yi, yi], color=style.AXIS, lw=1.5, zorder=1)
    ax.plot(a, yi, "o", ms=6, mfc="white", mec=C[role], mew=1.5, zorder=2)
    ax.plot(b, yi, "o", ms=6, color=C[role], mec="white", mew=1.0, zorder=3)
    ax.text(b + 1.2, yi, f"{b:.1f}", va="center", fontsize=6.8, color=style.INK_2)
    ax.text(a - 1.2, yi, f"{a:.1f}", va="center", ha="right", fontsize=6.8, color=style.MUTED)
ax.set_yticks(y); ax.set_yticklabels([L[r] for r in ROLES]); ax.grid(axis="y", visible=False)
ax.set_xlabel("IoU (%)")
ax.set_xlim(0, max(100 * RESULTS[r]["single_raw"]["iou"] for r in ROLES) * 1.25 + 4)
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=style.INK_2, label="tiles resized to 256\u00b2 (training-time validation)"),
                   Line2D([], [], marker="o", ls="", color=style.INK_2, label="native 1024\u00b2 (deployment)")],
          loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=1, fontsize=6.6)
footnote(fig, f"Single forward pass, p > 0.5.{SPLIT_NOTE}", y=-0.08)
style.save(fig, FIG_DIR, "fig_eval_protocol")

# 11b. Main comparison
main_cfgs = [(r, "single_raw") for r in ROLES] + [(r, DEPLOYED) for r in ("baseline", "final") if r in ROLES]
labels = [f"{L[r]}" + (" \u2014 deployed" if v == DEPLOYED else "") for r, v in main_cfgs]
colors = [C[r] for r, _ in main_cfgs]
hatches = ["////" if v == DEPLOYED else None for _, v in main_cfgs]
metrics = [("iou", "IoU"), ("f1", "F1"), ("relaxed_f1", f"Relaxed F1 (\u03c1={RHO}px)"), ("cldice", "clDice")]
fig, axes = plt.subplots(1, 4, figsize=(style.PAGE_W, 0.36 * len(main_cfgs) + 0.9), sharey=True)
xmax = min(118.0, 100 * max(RESULTS[r][v]["ci95"][k][1] for r, v in main_cfgs for k, _ in metrics) * 1.2)
for ax, (k, title), letter in zip(axes, metrics, "abcd"):
    vals = [100 * RESULTS[r][v][k] for r, v in main_cfgs]
    cis = [[100 * c for c in RESULTS[r][v]["ci95"][k]] for r, v in main_cfgs]
    hbars(ax, labels, vals, colors, cis=cis, hatches=hatches, xmax=xmax)
    ax.set_title(title, loc="left"); ax.set_xlabel("%")
    style.panel_label(ax, letter)
footnote(fig, FOOT + ". Hatched = deployed inference (4-flip TTA + hysteresis + closing + gap bridging); "
         "whiskers = 95% bootstrap CI.")
style.save(fig, FIG_DIR, "fig_eval_main")

# 11c. Precision-recall and IoU-vs-threshold
curves = [("baseline", "single", "-"), ("final", "single", "-"), ("final", "tta", "--")]
curves = [c for c in curves if f"{c[0]}/{c[1]}" in SWEEPS]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(style.PAGE_W, 2.45))
for role, src, ls in curves:
    s = SWEEPS[f"{role}/{src}"]
    lab = L[role] + (" + TTA" if src == "tta" else "")
    rec, prec, th = np.array(s["recall"]), np.array(s["precision"]), np.array(s["threshold"])
    ok = ~np.isnan(prec)
    a1.plot(100 * rec[ok], 100 * prec[ok], ls=ls, color=C[role], lw=1.5, label=lab)
    i5 = int(np.argmin(np.abs(th - 0.5)))
    a1.plot(100 * rec[i5], 100 * prec[i5], "o", ms=5, color=C[role], mec="white", mew=1.0)
    iou = 100 * np.array(s["iou"])
    ib = int(np.nanargmax(iou))
    a2.plot(th, iou, ls=ls, color=C[role], lw=1.5, label=f"{lab}: best {iou[ib]:.1f} at {th[ib]:.2f}")
    a2.plot(th[ib], iou[ib], "o", ms=5, color=C[role], mec="white", mew=1.0)
a1.set_xlabel("Recall (%)"); a1.set_ylabel("Precision (%)"); a1.set_xlim(0, 100); a1.set_ylim(0, 100)
a1.set_title("Precision\u2013recall (dot = threshold 0.5)", loc="left")
a2.set_xlabel("Probability threshold"); a2.set_ylabel("IoU (%)"); a2.set_xlim(0, 1)
a2.set_title("IoU vs. threshold (dot = best)", loc="left")
a1.legend(loc="lower left")
a2.legend(loc="upper right", fontsize=6.3)
style.panel_label(a1, "a"); style.panel_label(a2, "b")
footnote(fig, FOOT + ". Pixel-level, pooled over tiles.")
style.save(fig, FIG_DIR, "fig_eval_pr")

# 11d. Post-processing / TTA ablation on the final model
abl = [v for v in ["single_raw", "single_hyst", "single_hyst_close", "single_full_orig", "single_full",
                   "tta_raw", "tta_full_orig", "tta_full"] if v in RESULTS["final"]]
abl_labels = [VARIANT_NAMES[v] for v in abl]
fig, axes = plt.subplots(1, 4, figsize=(style.PAGE_W, 0.34 * len(abl) + 0.9), sharey=True)
panels = [("iou", "IoU (%)", 100, "{:.1f}"), ("relaxed_f1", "Relaxed F1 (%)", 100, "{:.1f}"),
          ("cldice", "clDice (%)", 100, "{:.1f}"), ("median_components_per_tile", "Components / tile (median)", 1, "{:.0f}")]
for ax, (k, title, sc, fmt), letter in zip(axes, panels, "abcd"):
    vals = [sc * RESULTS["final"][v][k] for v in abl]
    cols = [C["final"] if v == DEPLOYED else "#86b6ef" for v in abl]
    cis = [[sc * c for c in RESULTS["final"][v]["ci95"][k]] for v in abl] if k in RESULTS["final"][abl[0]]["ci95"] else None
    hbars(ax, abl_labels, vals, cols, cis=cis, fmt=fmt)
    ax.set_title(title, loc="left")
    style.panel_label(ax, letter)
gt_cc = RESULTS["final"]["single_raw"]["median_gt_components_per_tile"]
axes[3].axvline(gt_cc, color=style.INK_2, lw=0.8, ls=":")
axes[3].text(gt_cc, -0.75, f" ground truth: {gt_cc:.0f}", fontsize=6.5, color=style.INK_2, va="top")
footnote(fig, FOOT + ". Proposed model; dark bar = deployed configuration.")
style.save(fig, FIG_DIR, "fig_eval_ablation")

# 11e. Canopy analysis
can_cfgs = [c for c in [("baseline", "single_raw"), ("final", "single_raw"), ("baseline", DEPLOYED), ("final", DEPLOYED)]
            if c[0] in ROLES]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(style.PAGE_W, 2.5), gridspec_kw={"width_ratios": [1, 1.15]})
groups = [("open_recall_relaxed", "Unoccluded road"), ("canopy_recall_relaxed", "Canopy-covered road")]
w = 0.8 / len(can_cfgs)
for gi, (k, glab) in enumerate(groups):
    for ci_, (role, v) in enumerate(can_cfgs):
        val = 100 * RESULTS[role][v][k]
        x = gi + (ci_ - (len(can_cfgs) - 1) / 2) * w
        a1.bar(x, val, width=w * 0.92, color=C[role], edgecolor="white", linewidth=1.0,
               hatch="////" if v == DEPLOYED else None,
               label=(L[role] + (" \u2014 deployed" if v == DEPLOYED else "")) if gi == 0 else None)
        a1.text(x, val + 1, f"{val:.0f}", ha="center", fontsize=6.3, color=style.INK_2)
a1.set_xticks([0, 1]); a1.set_xticklabels([g for _, g in groups]); a1.grid(axis="x", visible=False)
a1.set_ylabel(f"Relaxed recall (%, \u03c1={RHO}px)"); a1.set_ylim(0, 105)
a1.set_title("Recall of ground-truth road pixels", loc="left")
fig.legend(*a1.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=6.4)
tert_labels = ["low", "medium", "high"]
for role in ("baseline", "final"):
    key = f"{role}/{DEPLOYED}"
    if key not in CANOPY_TERTILES:
        continue
    d = CANOPY_TERTILES[key]
    ts = sorted(d)
    vals = [100 * d[t]["cldice"] for t in ts]
    lo = [100 * d[t]["ci95"]["cldice"][0] for t in ts]
    hi = [100 * d[t]["ci95"]["cldice"][1] for t in ts]
    a2.fill_between(ts, lo, hi, color=C[role], alpha=0.12, lw=0)
    a2.plot(ts, vals, "-o", color=C[role], ms=5, mec="white", mew=1.0, label=L[role] + " \u2014 deployed")
    a2.text(ts[-1] + 0.06, vals[-1], f"{vals[-1]:.1f}", va="center", fontsize=6.5, color=style.INK_2)
a2.set_xticks(range(3))
a2.set_xticklabels([f"{n}\n({100*TERTILE_EDGES[i]:.0f}\u2013{100*TERTILE_EDGES[i+1]:.0f}% of road)"
                    for i, n in enumerate(tert_labels)], fontsize=6.6)
a2.set_xlim(-0.3, 2.45); a2.set_ylabel("clDice (%)")
a2.set_title("Connectivity by canopy cover (tertiles)", loc="left")
if a2.lines:
    a2.legend(loc="lower left", fontsize=6.5)
else:
    a2.text(0.5, 0.5, "too few tiles per tertile (< 5)", transform=a2.transAxes, ha="center", color=style.MUTED)
style.panel_label(a1, "a"); style.panel_label(a2, "b")
footnote(fig, FOOT + f". Canopy = ExG > {VEG_T:.3f} (Otsu). Band = 95% bootstrap CI.", y=-0.12)
style.save(fig, FIG_DIR, "fig_eval_canopy")

# 11f. Per-tile distribution and paired difference
if "baseline" in ROLES:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(style.PAGE_W, 2.35), gridspec_kw={"width_ratios": [1, 1.2]})
    dist_cfgs = [("baseline", "single_raw"), ("final", "single_raw"), ("baseline", DEPLOYED), ("final", DEPLOYED)]
    data = [PER_TILE[(PER_TILE.model == r) & (PER_TILE.variant == v)].iou.dropna().values * 100 for r, v in dist_cfgs]
    bp = a1.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color=style.INK, lw=1.2), whiskerprops=dict(color=style.AXIS),
                    capprops=dict(color=style.AXIS))
    for patch, (r, v) in zip(bp["boxes"], dist_cfgs):
        patch.set_facecolor(C[r]); patch.set_alpha(0.85); patch.set_edgecolor("white")
        if v == DEPLOYED:
            patch.set_hatch("////")
    a1.set_yticks(range(1, len(dist_cfgs) + 1))
    a1.set_yticklabels([L[r] + (" \u2014 deployed" if v == DEPLOYED else "") for r, v in dist_cfgs])
    a1.set_xlabel("Per-tile IoU (%)"); a1.grid(axis="y", visible=False)
    a1.set_title("Per-tile IoU distribution", loc="left")
    pa = PER_TILE[(PER_TILE.model == "final") & (PER_TILE.variant == DEPLOYED)].set_index("tile").iou
    pb = PER_TILE[(PER_TILE.model == "baseline") & (PER_TILE.variant == DEPLOYED)].set_index("tile").iou
    diff = (pa - pb).dropna() * 100
    bins = np.linspace(-max(abs(diff.min()), abs(diff.max())), max(abs(diff.min()), abs(diff.max())), 41)
    a2.hist(diff[diff <= 0], bins=bins, color=C["baseline"], edgecolor="white", linewidth=0.5, label="baseline better")
    a2.hist(diff[diff > 0], bins=bins, color=C["final"], edgecolor="white", linewidth=0.5, label="proposed better")
    a2.axvline(0, color=style.INK_2, lw=0.8)
    t = TESTS["final_vs_baseline_deployed_iou"]
    ptxt = f"p = {t['wilcoxon_p']:.1e}" if t["wilcoxon_p"] is not None else "p n/a"
    a2.text(0.0, 1.015, f"proposed better on {100*t['frac_tiles_improved']:.0f}% of tiles \u00b7 "
                        f"median \u0394 = {t['median_diff']*100:+.1f} pts \u00b7 Wilcoxon {ptxt}",
            transform=a2.transAxes, ha="left", va="bottom", fontsize=6.6, color=style.INK_2)
    a2.set_xlabel("\u0394 IoU per tile, proposed \u2212 baseline (points)"); a2.set_ylabel("Tiles")
    a2.set_title("Paired per-tile difference (deployed)", loc="left", pad=13)
    a2.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    a2.legend(loc="upper left", fontsize=6.5)
    style.panel_label(a1, "a"); style.panel_label(a2, "b")
    footnote(fig, FOOT + ".")
    style.save(fig, FIG_DIR, "fig_eval_per_tile")

# 11g. Collapse diagnosis on real validation data
if "collapsed" in ROLES:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(style.PAGE_W, 2.3), gridspec_kw={"width_ratios": [1.25, 1]})
    bins = np.linspace(0, 100, 51)
    gt_frac = PER_TILE[(PER_TILE.model == "final") & (PER_TILE.variant == "single_raw")].gt_frac * 100
    a1.hist(gt_frac, bins=bins, histtype="step", color=style.INK_2, lw=1.2, ls=":", label="ground truth")
    for role in ("collapsed", "final"):
        pf = PER_TILE[(PER_TILE.model == role) & (PER_TILE.variant == "single_raw")].positive_frac * 100
        a1.hist(pf, bins=bins, histtype="step", color=C[role], lw=1.5, label=L[role])
    a1.axvline(20, color=style.MUTED, lw=0.8, ls="--")
    a1.text(21, a1.get_ylim()[1] * 0.9, "collapse gate (20%)", fontsize=6.5, color=style.MUTED)
    a1.set_xlabel("Pixels predicted as road per tile (%)"); a1.set_ylabel("Tiles")
    a1.set_title("Predicted road fraction per tile (native)", loc="left")
    a1.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    a1.legend(loc="upper right", fontsize=6.5)
    roles2 = [r for r in ("collapsed", "run1", "final") if r in ROLES]
    yy = np.arange(len(roles2))[::-1] * 1.0
    for yi, role in zip(yy, roles2):
        s = 100 * TRAIN_PROTOCOL[role]["cldice_score"]
        i = 100 * TRAIN_PROTOCOL[role]["iou"]
        a2.text(1, yi + 0.36, L[role], fontsize=6.8, color=style.INK, fontweight="bold", va="bottom")
        a2.plot(s, yi + 0.12, "s", ms=6, color=C[role], mec="white", mew=1.0)
        a2.plot(i, yi - 0.14, "o", ms=6, mfc="white", mec=C[role], mew=1.5)
        a2.text(s + 2.5, yi + 0.12, f"soft clDice {s:.0f}", va="center", fontsize=6.4, color=style.INK_2)
        a2.text(i + 2.5, yi - 0.14, f"IoU {i:.1f}", va="center", fontsize=6.4, color=style.INK_2)
    a2.set_yticks([]); a2.grid(axis="y", visible=False)
    a2.set_ylim(-0.45, yy.max() + 0.75)
    a2.set_xlim(0, 118); a2.set_xlabel("Training-protocol score (%)")
    a2.set_title("What checkpoint selection saw", loc="left")
    style.panel_label(a1, "a"); style.panel_label(a2, "b")
    footnote(fig, FOOT + ". (b): soft clDice = 1 \u2212 logged clDice loss; 256\u00b2-resized tiles as in training.")
    style.save(fig, FIG_DIR, "fig_eval_collapse")

# %%
# ── 12. Qualitative results: error maps on representative tiles ──
PROB_CMAP = LinearSegmentedColormap.from_list("pb", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0


def error_map(P, G):
    img = np.ones((*G.shape, 3))
    img[P & G] = hex_rgb(style.TP_COLOR)
    img[P & ~G] = hex_rgb(style.FP_COLOR)
    img[~P & G] = hex_rgb(style.FN_COLOR)
    return img


@torch.no_grad()
def deployed_mask(model, rgb):
    x = (rgb.astype(np.float32) / 255.0 - np.array(IMNET_MEAN, np.float32)) / np.array(IMNET_STD, np.float32)
    _, pt = forward_probs(model, torch.from_numpy(x.transpose(2, 0, 1))[None].to(DEVICE), True)
    return binarize_all(pt[0, 0].cpu().numpy(), ["full"])["full"]


def _render_qualitative(panels, heads, name, subtitle, wide):
    """panels: [(tile_id, [(image, iou_or_None, is_mask), ...])]. wide=False puts tiles in rows
    (paper, full page); wide=True puts tiles in columns (slides)."""
    n_t, n_k = len(panels), len(heads)
    rows, cols = (n_k, n_t) if wide else (n_t, n_k)
    fig_w = style.PAGE_W
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_w / cols * rows + 0.45))
    axes = np.atleast_2d(axes)
    for t_i, (tid, items) in enumerate(panels):
        for k_i, (img, iou, is_mask) in enumerate(items):
            ax = axes[k_i, t_i] if wide else axes[t_i, k_i]
            if is_mask:
                ax.imshow(img, cmap=ListedColormap(["#ffffff", style.INK]), interpolation="nearest")
            else:
                ax.imshow(img, interpolation="nearest")
            if iou is not None:
                ax.text(0.03, 0.03, f"IoU {iou:.1f}", transform=ax.transAxes, fontsize=6.5, color=style.INK,
                        va="bottom", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
            style.image_axes(ax)
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(style.GRID); s.set_linewidth(0.5)
    if wide:
        for t_i, (tid, _) in enumerate(panels):
            axes[0, t_i].set_title(tid, loc="left", fontsize=7, color=style.MUTED)
        for k_i, h in enumerate(heads):
            axes[k_i, 0].set_ylabel(h, fontsize=7.2, color=style.INK, rotation=90, labelpad=4)
    else:
        for t_i, (tid, _) in enumerate(panels):
            axes[t_i, 0].set_ylabel(tid, fontsize=6.5, color=style.MUTED)
        for ax, h in zip(axes[0], heads):
            ax.set_title(h, loc="left", fontsize=7.2)
    fig.subplots_adjust(wspace=0.03, hspace=0.05, bottom=0.06)
    fig.legend(handles=[Patch(color=style.TP_COLOR, label="correct road"), Patch(color=style.FP_COLOR, label="false road"),
                        Patch(color=style.FN_COLOR, label="missed road")],
               loc="upper center", bbox_to_anchor=(0.5, 0.055), ncol=3, fontsize=6.8)
    fig.text(0.02, 0.012, subtitle, fontsize=6.5, color=style.MUTED, va="top")
    style.save(fig, FIG_DIR, name + ("_wide" if wide else ""))


def qualitative(tiles, name, subtitle):
    roles_q = [r for r in ("baseline", "final") if r in ROLES]
    models = {r: load_mobilevit_checkpoint(CHECKPOINTS[r]["path"], device=DEVICE)[0] for r in roles_q}
    iou_lookup = PER_TILE.set_index(["tile", "model", "variant"]).iou
    panels = []
    for tid in tiles:
        rgb, G = load_tile(tid)
        items = [(rgb, None, False), (G, None, True)]
        for role in roles_q:
            items.append((error_map(deployed_mask(models[role], rgb), G),
                          100 * iou_lookup[(tid, role, DEPLOYED)], False))
        panels.append((tid, items))
    heads = ["Input", "Ground truth"] + [L[r] + "\n(deployed)" for r in roles_q]
    for wide in (False, True):
        _render_qualitative(panels, heads, name, subtitle, wide)
    del models
    torch.cuda.empty_cache()


_q = PER_TILE[(PER_TILE.model == "final") & (PER_TILE.variant == DEPLOYED) & (PER_TILE.gt_pixels >= 1000)].dropna(subset=["iou"])
_q = _q.sort_values("iou").reset_index(drop=True)
QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]
Q_TILES = []
for q in QUANTILES:   # nearest not-yet-used tile to each percentile (never shows a tile twice)
    order = sorted(range(len(_q)), key=lambda i: abs(i - q * (len(_q) - 1)))
    pick = next((i for i in order if _q.tile[i] not in Q_TILES), None)
    if pick is not None:
        Q_TILES.append(_q.tile[pick])
qualitative(Q_TILES, "fig_eval_qualitative",
            "Tiles at the 10/30/50/70/90th percentile of the proposed model's per-tile IoU (not hand-picked).")
_hi = [t for t, b in TILE_TERTILE.items() if b == 2]
CANOPY_TILES = sorted(random.Random(7).sample(_hi, min(3, len(_hi))))
qualitative(CANOPY_TILES, "fig_eval_qualitative_canopy",
            f"Three tiles drawn at random (seed 7) from the highest canopy-cover tertile.")

# %%
# ── 13. Training curves from attached training logs (if any) ──
for k, log in enumerate([l for l in TRAINING_LOGS if l["has_iou"]]):
    ep = log["epochs"]
    e = np.array([r["epoch"] for r in ep])
    iou = np.array([r["val/iou"] for r in ep]) * 100
    fig, axes = plt.subplots(1, 3, figsize=(style.PAGE_W, 2.2))
    a1, a2, a3 = axes
    a1.plot(e, [r["train/epoch_loss"] for r in ep], color=style.BLUE, label="train")
    a1.plot(e, [r["val/epoch_loss"] for r in ep], color=style.ORANGE, label="validation")
    a1.set_title("Composite loss", loc="left"); a1.set_xlabel("Epoch"); a1.legend()
    a2.plot(e, iou, color=style.BLUE, label="IoU")
    a2.plot(e, np.array([r["val/precision"] for r in ep]) * 100, color=style.AQUA, label="precision")
    b = int(np.argmax(iou))
    a2.plot(e[b], iou[b], "o", ms=5, color=style.BLUE, mec="white", mew=1.0)
    late = e[b] > (e[0] + e[-1]) / 2
    a2.annotate(f"best {iou[b]:.1f} (ep. {e[b]})", (e[b], iou[b]), xytext=(-5 if late else 5, 6),
                textcoords="offset points", ha="right" if late else "left", fontsize=6.5, color=style.INK_2)
    a2.set_title("Validation (256\u00b2 protocol)", loc="left"); a2.set_xlabel("Epoch"); a2.set_ylabel("%"); a2.legend(loc="center right")
    pf = np.array([r["val/positive_frac"] for r in ep]) * 100
    a3.plot(e, pf, color=style.BLUE)
    a3.axhline(20, color=style.MUTED, lw=0.8, ls="--"); a3.text(e[-1], 20.5, "collapse gate", ha="right", fontsize=6.5, color=style.MUTED)
    a3.set_title("Predicted road fraction", loc="left"); a3.set_xlabel("Epoch"); a3.set_ylabel("%")
    a3.set_ylim(0, max(25, pf.max() * 1.15))
    for ax, letter in zip(axes, "abc"):
        style.panel_label(ax, letter)
    fig.subplots_adjust(wspace=0.38)
    footnote(fig, f"Logged during training: {os.path.basename(os.path.dirname(log['path']))}/"
                  f"{os.path.basename(log['path'])}. Validation used 256\u00b2-resized tiles.")
    style.save(fig, FIG_DIR, f"fig_training_curves_{k + 1}")

# %%
# ── 14. Package everything for download ──
zip_out = os.path.join(WORK, "paper_results.zip")
with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(OUT_DIR):
        for f in files:
            full = os.path.join(root, f)
            zf.write(full, os.path.relpath(full, WORK))
print(f"{zip_out}: {os.path.getsize(zip_out) / 2**20:.1f} MB | split verified: {SPLIT_VERIFIED}")
print(sorted(os.listdir(FIG_DIR)))
if IN_KAGGLE:
    from IPython.display import FileLink, display
    display(FileLink(os.path.relpath(zip_out, WORK)))
