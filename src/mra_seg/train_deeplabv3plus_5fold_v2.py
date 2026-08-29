"""
DeepLabV3+ 5-Fold Cross-Validation for MRA Vessel Segmentation (v2)
Baseline: Yamada et al. Appl. Sci. 2025, 15, 3034

Improvements over v1:
- Multi-GPU (DataParallel over every detected GPU)
- Enhanced online augmentation
- More epochs to match the effective data volume of offline augmentation
- CE + Dice hybrid loss for class imbalance
- Cosine annealing LR
"""

import os
import sys
import time
import json
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
sys.stdout.reconfigure(line_buffering=True)

# ============================================================
# Configuration
# ============================================================
# All paths live in paths.py, resolved relative to the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import MRA_SEG_DIR, FOLD_MODEL_DIR, SEG_RESULT_DIR, ensure_dirs  # noqa: E402

DATA_DIR = MRA_SEG_DIR            # rawJPEG / rawPNG / DICOMdata
OUTPUT_DIR = FOLD_MODEL_DIR       # trained weights (.pth)
RESULT_DIR = SEG_RESULT_DIR       # evaluation results (JSON)
ensure_dirs(OUTPUT_DIR, RESULT_DIR)

NUM_FOLDS = 5
NUM_EPOCHS = 15
BATCH_SIZE_PER_GPU = 8
NUM_GPUS = torch.cuda.device_count()
BATCH_SIZE = BATCH_SIZE_PER_GPU * NUM_GPUS  # 24 total
INITIAL_LR = 0.007
NUM_CLASSES = 2
IMG_SIZE = 512
NUM_WORKERS = 4
SEED = 42

print(f"Config: {NUM_GPUS} GPUs, batch={BATCH_SIZE}, epochs={NUM_EPOCHS}, lr={INITIAL_LR}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# Dataset with enhanced augmentation
# ============================================================
class MRADataset(Dataset):
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

        if augment:
            # Enhanced augmentation matching paper's offline scheme:
            # rotation(-25..+25), scale(0.7..1.0), hflip
            # Plus additional online augmentation for better generalization
            self.transform = A.Compose([
                A.Rotate(limit=25, interpolation=1, border_mode=0, p=0.9),
                A.RandomScale(scale_limit=(-0.3, 0.0), p=0.9),
                A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE,
                              border_mode=0, value=0, mask_value=0),
                A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                # Additional augmentation
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                ], p=0.2),
                A.RandomBrightnessContrast(brightness_limit=0.15,
                                           contrast_limit=0.15, p=0.3),
                A.GaussNoise(std_range=(0.01, 0.03), p=0.2),
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
        mask = (mask > 0).astype(np.uint8)

        transformed = self.transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"].long()


# ============================================================
# Dice Loss + CE hybrid
# ============================================================
class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)

        # Dice loss for vessel class
        pred_soft = torch.softmax(pred, dim=1)[:, 1]  # vessel prob
        target_f = (target == 1).float()

        intersection = (pred_soft * target_f).sum(dim=(1, 2))
        cardinality = pred_soft.sum(dim=(1, 2)) + target_f.sum(dim=(1, 2))
        dice = (2.0 * intersection + 1.0) / (cardinality + 1.0)
        dice_loss = 1.0 - dice.mean()

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


# ============================================================
# Model
# ============================================================
def create_model():
    model = deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    return model


# ============================================================
# Metrics
# ============================================================
def compute_metrics(pred, target):
    pred_vessel = (pred == 1).float()
    target_vessel = (target == 1).float()

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
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output["out"], masks)
        # Aux loss
        if "aux" in output:
            loss += 0.4 * criterion(output["aux"], masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_dice = []
    all_iou = []
    total_time = 0.0
    total_frames = 0

    # Use single GPU for consistent FPS measurement
    eval_model = model.module if isinstance(model, nn.DataParallel) else model

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            start = time.time()
            outputs = eval_model(images)["out"]
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
# Main
# ============================================================
def main():
    device = torch.device("cuda:0")
    print(f"Using {NUM_GPUS} GPUs: ", end="")
    for i in range(NUM_GPUS):
        print(f"{torch.cuda.get_device_name(i)}", end=", " if i < NUM_GPUS - 1 else "\n")

    jpeg_dir = DATA_DIR / "rawJPEG"
    png_dir = DATA_DIR / "rawPNG"

    case_ids = sorted([int(d.name) for d in jpeg_dir.iterdir() if d.is_dir()])
    print(f"Total cases: {len(case_ids)}, IDs: {case_ids}")

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

        train_dataset = MRADataset(train_cases, jpeg_dir, png_dir, augment=True)
        test_dataset = MRADataset(test_cases, jpeg_dir, png_dir, augment=False)
        print(f"Train images: {len(train_dataset)}, Test images: {len(test_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=NUM_WORKERS,
                                  pin_memory=True, drop_last=True,
                                  persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=NUM_WORKERS,
                                 pin_memory=True, persistent_workers=True)

        # Create model with DataParallel across 3 GPUs
        model = create_model()
        model = nn.DataParallel(model, device_ids=list(range(NUM_GPUS)))
        model = model.to(device)

        criterion = DiceCELoss(dice_weight=0.5, ce_weight=0.5)
        optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS,
                                                          eta_min=1e-6)

        best_dice = 0.0
        for epoch in range(NUM_EPOCHS):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion,
                                         optimizer, device)
            scheduler.step()
            elapsed = time.time() - t0

            # Evaluate every 5 epochs or last epoch
            if (epoch + 1) % 5 == 0 or (epoch + 1) == NUM_EPOCHS:
                dice_mean, dice_std, iou_mean, iou_std, fps = evaluate(
                    model, test_loader, device)
                print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} - Loss: {train_loss:.4f} "
                      f"- LR: {scheduler.get_last_lr()[0]:.6f} - Time: {elapsed:.1f}s "
                      f"- DSC: {dice_mean:.4f} - IoU: {iou_mean:.4f}")

                if dice_mean > best_dice:
                    best_dice = dice_mean
                    best_state = {k: v.cpu().clone() for k, v in model.module.state_dict().items()}
            else:
                print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} - Loss: {train_loss:.4f} "
                      f"- LR: {scheduler.get_last_lr()[0]:.6f} - Time: {elapsed:.1f}s")

        # Final evaluation with best model
        model.module.load_state_dict(best_state)
        dice_mean, dice_std, iou_mean, iou_std, fps = evaluate(model, test_loader, device)
        print(f"\n  Best Results Fold {fold_idx+1}:")
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

        model_path = OUTPUT_DIR / f"deeplabv3plus_fold{fold_idx+1}.pth"
        torch.save(best_state, model_path)
        print(f"  Model saved: {model_path}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("OVERALL 5-FOLD CROSS-VALIDATION RESULTS (v2)")
    print(f"{'='*60}")

    all_dice = [r["dice_mean"] for r in fold_results]
    all_iou = [r["iou_mean"] for r in fold_results]
    all_fps = [r["fps"] for r in fold_results]

    print(f"DSC:  {np.mean(all_dice):.4f} +/- {np.std(all_dice):.4f}")
    print(f"IoU:  {np.mean(all_iou):.4f} +/- {np.std(all_iou):.4f}")
    print(f"FPS:  {np.mean(all_fps):.2f} +/- {np.std(all_fps):.2f}")


    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": "DeepLabV3+ (ResNet-50) + DataParallel",
            "num_gpus": NUM_GPUS,
            "num_folds": NUM_FOLDS,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "initial_lr": INITIAL_LR,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "loss": "DiceCE (0.5/0.5)",
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

    results_path = RESULT_DIR / "results_5fold_v2.json"
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
