import os, random, shutil
from pathlib import Path

random.seed(42)

SRC_DIR = "DATASET"     # <-- your current folder
OUT_DIR = "dataset"     # <-- new output folder
train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1

os.makedirs(OUT_DIR, exist_ok=True)

classes = [d for d in os.listdir(SRC_DIR) if os.path.isdir(os.path.join(SRC_DIR, d))]
print("Found classes:", classes)

for split in ["train", "val", "test"]:
    for c in classes:
        os.makedirs(os.path.join(OUT_DIR, split, c), exist_ok=True)

for c in classes:
    class_path = os.path.join(SRC_DIR, c)
    images = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg",".jpeg",".png"))]
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_imgs = images[:n_train]
    val_imgs   = images[n_train:n_train+n_val]
    test_imgs  = images[n_train+n_val:]

    for f in train_imgs:
        shutil.copy2(os.path.join(class_path, f), os.path.join(OUT_DIR, "train", c, f))
    for f in val_imgs:
        shutil.copy2(os.path.join(class_path, f), os.path.join(OUT_DIR, "val", c, f))
    for f in test_imgs:
        shutil.copy2(os.path.join(class_path, f), os.path.join(OUT_DIR, "test", c, f))

    print(c, "=>", len(train_imgs), len(val_imgs), len(test_imgs))

print("Done. New dataset folder created:", OUT_DIR)
