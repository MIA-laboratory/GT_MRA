"""
DeepLabV3+ 5-Fold Cross-Validation for MRA Vessel Segmentation
Baseline: Yamada et al. Appl. Sci. 2025, 15, 3034
- Binary segmentation: intracranial vessel ROI vs background
- Data augmentation: rotation, scaling, horizontal flip
- 5-fold cross-validation at case level
"""

import os
import sys
import time
import json
import glob
import random
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# ============================================================
# Configuration (matching paper parameters)
# ============================================================
# All paths live in paths.py, resolved relative to the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import MRA_SEG_DIR, FOLD_MODEL_V1_DIR, SEG_RESULT_DIR, ensure_dirs  # noqa: E402

DATA_DIR = MRA_SEG_DIR            # rawJPEG / rawPNG / DICOMdata
OUTPUT_DIR = FOLD_MODEL_V1_DIR    # weights of this version, kept apart from v2
RESULT_DIR = SEG_RESULT_DIR       # evaluation results (JSON)
ensure_dirs(OUTPUT_DIR, RESULT_DIR)

NUM_FOLDS = 5
NUM_EPOCHS = 3
BATCH_SIZE = 8           # actual GPU batch
ACCUM_STEPS = 8          # gradient accumulation -> effective batch = 64
INITIAL_LR = 0.01
LR_DECAY_FACTOR = 0.3
NUM_CLASSES = 2  # background + vessel
IMG_SIZE = 512
NUM_WORKERS = 2
SEED = 42

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ============================================================
# Dataset
# ============================================================
class MRADataset(Dataset):
    """MRA vessel segmentation dataset with offline-style augmentation."""

    def __init__(self, case_ids, jpeg_dir, png_dir, augment=False):
        self.samples = []
        for cid in case_ids:
            jpeg_case = jpeg_dir / str(cid)
            png_case = png_dir / str(cid)
            jpegs = sorted(jpeg_case.glob("*.JPG"))
            for jp in jpegs:
                png_name = jp.stem + ".png"
                pp = png_case / png_name
                if pp.exists():
                    self.samples.append((jp, pp))

        self.augment = augment

        # Paper augmentation: rotation (-25 to +25, step 5), scale (0.7-1.0, step 0.1), hflip
        # We replicate this with albumentations
        if augment:
            self.transform = A.Compose([
                A.Rotate(limit=25, interpolation=1, border_mode=0, p=0.8),
                A.RandomScale(scale_limit=(-0.3, 0.0), p=0.8),  # 0.7x to 1.0x
                A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE,
                              border_mode=0, value=0, mask_value=0),
                A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        # Ensure mask is binary (0 or 1)
        mask = (mask > 0).astype(np.uint8)

        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]  # (3, H, W) float
        mask = transformed["mask"].long()  # (H, W) long

        return image, mask


# ============================================================
# Model
# ============================================================
def create_model():
    """Create DeepLabV3+ (ResNet-50 backbone) for binary segmentation.
    Use pretrained backbone for better convergence with few epochs."""
    model = deeplabv3_resnet50(weights="DEFAULT")
    # Replace classifier head for 2 classes
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    return model


# ============================================================
# Metrics
# ============================================================
def compute_metrics(pred, target, num_classes=2):
    """Compute Dice and IoU for the vessel class (class=1)."""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # Vessel class (1)
    pred_vessel = (pred_flat == 1).float()
    target_vessel = (target_flat == 1).float()

    intersection = (pred_vessel * target_vessel).sum()
    union = pred_vessel.sum() + target_vessel.sum() - intersection

    dice = (2.0 * intersection + 1e-7) / (pred_vessel.sum() + target_vessel.sum() + 1e-7)
    iou = (intersection + 1e-7) / (union + 1e-7)

    return dice.item(), iou.item()


# ============================================================
# Training
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)["out"]
        loss = criterion(outputs, masks) / ACCUM_STEPS
        loss.backward()

        if (batch_idx + 1) % ACCUM_STEPS == 0 or (batch_idx + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * ACCUM_STEPS * images.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_dice = []
    all_iou = []
    total_time = 0.0
    total_frames = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            start = time.time()
            outputs = model(images)["out"]
            torch.cuda.synchronize()
            elapsed = time.time() - start

            preds = outputs.argmax(dim=1)

            for i in range(images.size(0)):
                d, iou = compute_metrics(preds[i], masks[i])
                all_dice.append(d)
                all_iou.append(iou)

            total_time += elapsed
            total_frames += images.size(0)

    fps = total_frames / total_time if total_time > 0 else 0
    return np.mean(all_dice), np.std(all_dice), np.mean(all_iou), np.std(all_iou), fps


# ============================================================
# Main: 5-Fold Cross-Validation
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    jpeg_dir = DATA_DIR / "rawJPEG"
    png_dir = DATA_DIR / "rawPNG"

    # Get all case IDs
    case_ids = sorted([int(d.name) for d in jpeg_dir.iterdir() if d.is_dir()])
    print(f"Total cases: {len(case_ids)}, IDs: {case_ids}")

    # 5-fold at case level
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    case_ids_arr = np.array(case_ids)

    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(case_ids_arr)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{NUM_FOLDS}")
        print(f"{'='*60}")

        train_cases = case_ids_arr[train_idx].tolist()
        test_cases = case_ids_arr[test_idx].tolist()
        print(f"Train cases ({len(train_cases)}): {train_cases}")
        print(f"Test cases  ({len(test_cases)}): {test_cases}")

        # Create datasets
        train_dataset = MRADataset(train_cases, jpeg_dir, png_dir, augment=True)
        test_dataset = MRADataset(test_cases, jpeg_dir, png_dir, augment=False)
        print(f"Train images: {len(train_dataset)}, Test images: {len(test_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=NUM_WORKERS,
                                  pin_memory=True, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=NUM_WORKERS,
                                 pin_memory=True)

        # Create model
        model = create_model().to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=0.9,
                              weight_decay=1e-4)
        # LR schedule: reduce by factor 0.3 after each epoch
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1,
                                              gamma=LR_DECAY_FACTOR)

        # Training
        for epoch in range(NUM_EPOCHS):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion,
                                         optimizer, device)
            scheduler.step()
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {train_loss:.4f} "
                  f"- LR: {scheduler.get_last_lr()[0]:.6f} - Time: {elapsed:.1f}s")

        # Evaluate
        dice_mean, dice_std, iou_mean, iou_std, fps = evaluate(model, test_loader, device)
        print(f"\n  Results Fold {fold_idx+1}:")
        print(f"    DSC:  {dice_mean:.4f} +/- {dice_std:.4f}")
        print(f"    IoU:  {iou_mean:.4f} +/- {iou_std:.4f}")
        print(f"    FPS:  {fps:.2f}")

        fold_results.append({
            "fold": fold_idx + 1,
            "train_cases": train_cases,
            "test_cases": test_cases,
            "dice_mean": dice_mean,
            "dice_std": dice_std,
            "iou_mean": iou_mean,
            "iou_std": iou_std,
            "fps": fps,
        })

        # Save fold model
        model_path = OUTPUT_DIR / f"deeplabv3plus_fold{fold_idx+1}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"  Model saved: {model_path}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("OVERALL 5-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")

    all_dice = [r["dice_mean"] for r in fold_results]
    all_iou = [r["iou_mean"] for r in fold_results]
    all_fps = [r["fps"] for r in fold_results]

    print(f"DSC:  {np.mean(all_dice):.4f} +/- {np.std(all_dice):.4f}")
    print(f"IoU:  {np.mean(all_iou):.4f} +/- {np.std(all_iou):.4f}")
    print(f"FPS:  {np.mean(all_fps):.2f} +/- {np.std(all_fps):.2f}")


    # Save results
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": "DeepLabV3+ (ResNet-50)",
            "num_folds": NUM_FOLDS,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE * ACCUM_STEPS,
            "initial_lr": INITIAL_LR,
            "lr_decay_factor": LR_DECAY_FACTOR,
            "img_size": IMG_SIZE,
            "seed": SEED,
        },
        "fold_results": fold_results,
        "overall": {
            "dice_mean": float(np.mean(all_dice)),
            "dice_std": float(np.std(all_dice)),
            "iou_mean": float(np.mean(all_iou)),
            "iou_std": float(np.std(all_iou)),
            "fps_mean": float(np.mean(all_fps)),
            "fps_std": float(np.std(all_fps)),
        },
    }

    results_path = RESULT_DIR / "results_5fold.json"
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
