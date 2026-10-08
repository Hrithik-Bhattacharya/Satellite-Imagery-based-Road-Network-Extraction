import os
import subprocess
import shutil
import glob
import random

print("Files in /kaggle/input:")
for root, dirs, files in os.walk('/kaggle/input'):
    for d in dirs:
        print(f"DIR: {os.path.join(root, d)}")

print("Finding backend codebase...")
backend_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'requirements.txt' in files and 'src' in dirs and 'scripts' in dirs:
        backend_dir = root
        break

if not backend_dir:
    raise FileNotFoundError("Could not find the backend codebase directory.")

print(f"Found backend at: {backend_dir}")

working_backend = "/kaggle/working/backend"
if os.path.exists(working_backend):
    shutil.rmtree(working_backend)
shutil.copytree(backend_dir, working_backend)

print("Finding DeepGlobe dataset...")
deepglobe_train = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'train' in dirs and deepglobe_train is None:
        try:
            if any('sat.jpg' in f for f in os.listdir(os.path.join(root, 'train'))):
                deepglobe_train = os.path.join(root, 'train')
        except: pass

if not deepglobe_train:
    print("WARNING: Could not auto-detect DeepGlobe train directory. Falling back to default.")
    deepglobe_train = "/kaggle/input/deepglobe-road-extraction-dataset/train"

print(f"Original Train Dir: {deepglobe_train}")

# Create symlinked split in /kaggle/working/dataset
working_dataset = "/kaggle/working/dataset"
train_split = os.path.join(working_dataset, "train")
valid_split = os.path.join(working_dataset, "valid")

os.makedirs(train_split, exist_ok=True)
os.makedirs(valid_split, exist_ok=True)

all_sats = glob.glob(os.path.join(deepglobe_train, "*_sat.jpg"))
all_ids = [os.path.basename(f).replace("_sat.jpg", "") for f in all_sats]

# Ensure we only use IDs that also have a mask!
valid_ids = []
for i in all_ids:
    if os.path.exists(os.path.join(deepglobe_train, f"{i}_mask.png")):
        valid_ids.append(i)
all_ids = valid_ids

random.seed(42)
random.shuffle(all_ids)

# 90% train, 10% valid
split_idx = int(len(all_ids) * 0.9)
train_ids = all_ids[:split_idx]
valid_ids = all_ids[split_idx:]

print(f"Splitting {len(all_ids)} images -> {len(train_ids)} train, {len(valid_ids)} valid")

def create_links(id_list, dest_dir):
    for i in id_list:
        os.symlink(os.path.join(deepglobe_train, f"{i}_sat.jpg"), os.path.join(dest_dir, f"{i}_sat.jpg"))
        os.symlink(os.path.join(deepglobe_train, f"{i}_mask.png"), os.path.join(dest_dir, f"{i}_mask.png"))

create_links(train_ids, train_split)
create_links(valid_ids, valid_split)

print("Starting training...")
cmd = f"""
python {working_backend}/scripts/train.py \
    --train_image_dir {train_split} \
    --train_mask_dir {train_split} \
    --val_image_dir {valid_split} \
    --val_mask_dir {valid_split} \
    --epochs 50 \
    --batch_size 16 \
    --num_workers 2
"""

print(f"Running command: {cmd}")
subprocess.run(cmd, shell=True, check=True)
print("Training complete!")
