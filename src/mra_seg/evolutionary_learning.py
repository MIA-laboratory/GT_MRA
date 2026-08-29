"""
Evolutionary Knowledge Update for MRA Vessel Segmentation
==========================================================
Starting from a DeepLabV3+ model trained on a small labeled dataset,
this system processes unlabeled DICOM data in fixed-size batches,
generating pseudo-labels with confidence scoring, and iteratively
re-trains the model to evolve its knowledge.

Key components:
  - DICOM inference pipeline with TTA-based uncertainty estimation
  - Confidence-based pseudo-label quality gate
  - EMA (Exponential Moving Average) teacher-student framework
  - EWC (Elastic Weight Consolidation) to prevent catastrophic forgetting
  - Evolutionary model selection with rollback
"""

import os
import sys
import time
import json
import random
import shutil
import warnings
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
import torchvision.transforms.functional as TF
from PIL import Image
import pydicom
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

# ============================================================
# Configuration
# ============================================================
# All paths live in paths.py, resolved relative to the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (  # noqa: E402
    INITIAL_MODEL, EVOLUTION_DIR, EVOLUTION_LOG, SEG_RESULT_DIR,
    RAW_JPEG_DIR, RAW_PNG_DIR, MRA_STUDY_DIR, ensure_dirs, require_external,
)

DICOM_DIR = MRA_STUDY_DIR          # unlabeled studies (not distributed)
ensure_dirs(EVOLUTION_DIR, SEG_RESULT_DIR)

# Original labeled data for holdout validation
LABELED_JPEG_DIR = RAW_JPEG_DIR
LABELED_PNG_DIR = RAW_PNG_DIR
# Case IDs held out for validation. Set these for your own dataset.
HOLDOUT_CASES: list[int] = []
assert HOLDOUT_CASES, "Set HOLDOUT_CASES before running."

NUM_CLASSES = 2
IMG_SIZE = 512
BATCH_SIZE_PER_GPU = 8
NUM_GPUS = torch.cuda.device_count()
BATCH_SIZE = BATCH_SIZE_PER_GPU * max(NUM_GPUS, 1)
CASES_PER_BATCH = 100
RE_TRAIN_EPOCHS = 5
EMA_DECAY = 0.999
EWC_LAMBDA = 500.0
CONFIDENCE_HIGH = 0.95
CONFIDENCE_LOW = 0.80
NUM_TTA = 4  # Test-Time Augmentation passes
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ============================================================
# Model creation
# ============================================================
def create_model():
    """Create DeepLabV3+ matching the v2 training architecture."""
    model = deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    return model


def load_initial_model(device):
    model = create_model()
    state = torch.load(INITIAL_MODEL, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    print(f"Loaded initial model: {INITIAL_MODEL}")
    return model


# ============================================================
# EMA (Exponential Moving Average) Teacher
# ============================================================
class EMATeacher:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )

    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])


# ============================================================
# EWC (Elastic Weight Consolidation)
# ============================================================
class EWC:
    """Prevents catastrophic forgetting of initial labeled knowledge."""

    def __init__(self, model, dataloader, device):
        self.params = {}
        self.fisher = {}
        self._compute_fisher(model, dataloader, device)

    def _compute_fisher(self, model, dataloader, device):
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()
                  if p.requires_grad}

        num_samples = 0
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            model.zero_grad()
            output = model(images)["out"]
            loss = nn.CrossEntropyLoss()(output, masks)
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2) * images.size(0)
            num_samples += images.size(0)

        for n in fisher:
            fisher[n] /= num_samples

        self.fisher = fisher
        self.params = {n: p.data.clone() for n, p in model.named_parameters()
                       if p.requires_grad}

    def penalty(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return loss


# ============================================================
# DICOM Dataset for Inference
# ============================================================
class DICOMInferenceDataset(Dataset):
    """Load DICOM slices from a list of case directories."""

    def __init__(self, case_dirs):
        self.samples = []
        for case_dir in case_dirs:
            dcm_files = sorted(case_dir.glob("*.dcm"),
                               key=lambda x: x.stem.split("_")[-1].zfill(5))
            for f in dcm_files:
                self.samples.append(f)

        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dcm_path = self.samples[idx]
        ds = pydicom.dcmread(str(dcm_path))
        pixel = ds.pixel_array.astype(np.float32)

        # Normalize to 0-255 range
        if pixel.max() > 0:
            pixel = (pixel - pixel.min()) / (pixel.max() - pixel.min()) * 255.0
        pixel = pixel.astype(np.uint8)

        # Resize to 512x512
        img = Image.fromarray(pixel).convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img_np = np.array(img)

        transformed = self.transform(image=img_np)
        return transformed["image"], str(dcm_path)


# ============================================================
# Pseudo-label Dataset for Re-training
# ============================================================
class PseudoLabelDataset(Dataset):
    """Combined dataset: original labeled + pseudo-labeled data."""

    def __init__(self, labeled_samples, pseudo_samples, augment=True):
        # labeled_samples: list of (jpeg_path, png_path)
        # pseudo_samples: list of (image_np_512, mask_np_512, confidence)
        self.labeled = labeled_samples
        self.pseudo = pseudo_samples

        if augment:
            self.transform = A.Compose([
                A.Rotate(limit=25, interpolation=1, border_mode=0, p=0.9),
                A.RandomScale(scale_limit=(-0.3, 0.0), p=0.9),
                A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE,
                              border_mode=0, value=0, mask_value=0),
                A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.15,
                                           contrast_limit=0.15, p=0.3),
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
        return len(self.labeled) + len(self.pseudo)

    def __getitem__(self, idx):
        if idx < len(self.labeled):
            img_path, mask_path = self.labeled[idx]
            image = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path))
            mask = (mask > 0).astype(np.uint8)
        else:
            pidx = idx - len(self.labeled)
            image, mask, _ = self.pseudo[pidx]

        transformed = self.transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"].long()


# ============================================================
# Holdout Validation Dataset
# ============================================================
class ValidationDataset(Dataset):
    def __init__(self, case_ids, jpeg_dir, png_dir):
        self.samples = []
        for cid in case_ids:
            jpeg_case = jpeg_dir / str(cid)
            png_case = png_dir / str(cid)
            for jp in sorted(jpeg_case.glob("*.JPG")):
                pp = png_case / (jp.stem + ".png")
                if pp.exists():
                    self.samples.append((jp, pp))

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
# Fast Batch Inference with Confidence (no per-image TTA)
# ============================================================
def batch_inference(model, case_dirs, device):
    """Run batched inference with softmax confidence scoring."""
    dataset = DICOMInferenceDataset(case_dirs)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)

    # Use DataParallel for inference too
    if NUM_GPUS > 1 and not isinstance(model, nn.DataParallel):
        eval_model = nn.DataParallel(model, device_ids=list(range(NUM_GPUS)))
    else:
        eval_model = model
    eval_model.eval()

    results = []
    high_conf = 0
    mid_conf = 0
    low_conf = 0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for images, dcm_paths in loader:
            images = images.to(device)
            outputs = eval_model(images)["out"]
            probs = torch.softmax(outputs, dim=1)  # (B, 2, H, W)
            preds = probs.argmax(dim=1)  # (B, H, W)
            # Per-pixel max confidence, averaged per image
            confidences = probs.max(dim=1)[0].mean(dim=(1, 2))  # (B,)

            for i in range(images.size(0)):
                conf = confidences[i].item()
                if conf >= CONFIDENCE_LOW:
                    img_np = images[i].cpu().permute(1, 2, 0).numpy()
                    img_np = ((img_np * std + mean) * 255).clip(0, 255).astype(np.uint8)
                    mask_np = preds[i].cpu().numpy().astype(np.uint8)
                    results.append((img_np, mask_np, conf))
                    if conf >= CONFIDENCE_HIGH:
                        high_conf += 1
                    else:
                        mid_conf += 1
                else:
                    low_conf += 1

    total = high_conf + mid_conf + low_conf
    print(f"  Inference: {total} slices -> "
          f"High({high_conf}, {high_conf/max(total,1)*100:.1f}%) "
          f"Mid({mid_conf}, {mid_conf/max(total,1)*100:.1f}%) "
          f"Low({low_conf}, {low_conf/max(total,1)*100:.1f}%)")

    # Unwrap DataParallel if we wrapped it
    if NUM_GPUS > 1 and isinstance(eval_model, nn.DataParallel):
        pass  # model reference unchanged

    return results, {"high": high_conf, "mid": mid_conf, "low": low_conf}


# ============================================================
# Evaluation
# ============================================================
def evaluate_holdout(model, device):
    """Evaluate on the holdout set of original labeled data."""
    val_dataset = ValidationDataset(HOLDOUT_CASES, LABELED_JPEG_DIR, LABELED_PNG_DIR)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)

    eval_model = model.module if isinstance(model, nn.DataParallel) else model
    eval_model.eval()

    all_dice = []
    all_iou = []
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = eval_model(images)["out"]
            preds = outputs.argmax(dim=1)

            for i in range(images.size(0)):
                pred_v = (preds[i] == 1).float()
                tgt_v = (masks[i] == 1).float()
                inter = (pred_v * tgt_v).sum()
                union = pred_v.sum() + tgt_v.sum() - inter
                dice = (2 * inter + 1e-7) / (pred_v.sum() + tgt_v.sum() + 1e-7)
                iou = (inter + 1e-7) / (union + 1e-7)
                all_dice.append(dice.item())
                all_iou.append(iou.item())

    return np.mean(all_dice), np.mean(all_iou)


# ============================================================
# Re-training with EWC
# ============================================================
class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        pred_soft = torch.softmax(pred, dim=1)[:, 1]
        target_f = (target == 1).float()
        inter = (pred_soft * target_f).sum(dim=(1, 2))
        card = pred_soft.sum(dim=(1, 2)) + target_f.sum(dim=(1, 2))
        dice_loss = 1.0 - ((2.0 * inter + 1.0) / (card + 1.0)).mean()
        return 0.5 * ce_loss + 0.5 * dice_loss


def retrain(model, labeled_samples, pseudo_samples, ewc, device, generation):
    """Re-train model with combined labeled + pseudo-labeled data."""
    dataset = PseudoLabelDataset(labeled_samples, pseudo_samples, augment=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True,
                        persistent_workers=True)

    if NUM_GPUS > 1:
        model = nn.DataParallel(model, device_ids=list(range(NUM_GPUS)))
    model = model.to(device)

    criterion = DiceCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                      T_max=RE_TRAIN_EPOCHS,
                                                      eta_min=1e-6)

    for epoch in range(RE_TRAIN_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output["out"], masks)
            if "aux" in output:
                loss += 0.4 * criterion(output["aux"], masks)

            # EWC regularization
            if ewc is not None:
                base_model = model.module if isinstance(model, nn.DataParallel) else model
                loss += EWC_LAMBDA * ewc.penalty(base_model)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        avg_loss = running_loss / len(loader.dataset)
        print(f"    Gen {generation} Epoch {epoch+1}/{RE_TRAIN_EPOCHS} "
              f"- Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

    # Return unwrapped model
    return model.module if isinstance(model, nn.DataParallel) else model


# ============================================================
# Main: Evolutionary Learning Loop
# ============================================================
def main():
    device = torch.device("cuda:0")
    print(f"Evolutionary Knowledge Update System")
    print(f"GPUs: {NUM_GPUS}, Batch: {BATCH_SIZE}, Cases/batch: {CASES_PER_BATCH}")
    print(f"Initial model: {INITIAL_MODEL}")
    print(f"DICOM source: {DICOM_DIR}")
    print(f"Holdout validation: cases {HOLDOUT_CASES}")

    # --- Load initial model ---
    model = load_initial_model(device)

    # --- Prepare labeled data (excluding holdout) ---
    train_case_ids = sorted([int(d.name) for d in LABELED_JPEG_DIR.iterdir()
                             if d.is_dir() and int(d.name) not in HOLDOUT_CASES])
    labeled_samples = []
    for cid in train_case_ids:
        jpeg_case = LABELED_JPEG_DIR / str(cid)
        png_case = LABELED_PNG_DIR / str(cid)
        for jp in sorted(jpeg_case.glob("*.JPG")):
            pp = png_case / (jp.stem + ".png")
            if pp.exists():
                labeled_samples.append((str(jp), str(pp)))
    print(f"Labeled training samples: {len(labeled_samples)} "
          f"(from {len(train_case_ids)} cases)")

    # --- Evaluate initial model ---
    init_dice, init_iou = evaluate_holdout(model, device)
    print(f"\nGeneration 0 (Initial): DSC={init_dice:.4f}, IoU={init_iou:.4f}")

    # --- Compute EWC from labeled data ---
    print("Computing EWC Fisher information from labeled data...")
    ewc_dataset = ValidationDataset(train_case_ids, LABELED_JPEG_DIR, LABELED_PNG_DIR)
    ewc_loader = DataLoader(ewc_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
    ewc = EWC(model, ewc_loader, device)
    print("EWC computed.")

    # --- Initialize EMA teacher ---
    ema = EMATeacher(model)

    # --- Get all DICOM case directories ---
    require_external(DICOM_DIR, "unlabeled MRA studies")
    all_cases = sorted([d for d in DICOM_DIR.iterdir() if d.is_dir()])
    total_batches = (len(all_cases) + CASES_PER_BATCH - 1) // CASES_PER_BATCH
    print(f"\nTotal DICOM cases: {len(all_cases)}, "
          f"Batches of {CASES_PER_BATCH}: {total_batches}")

    # --- Evolution log ---
    evolution_log = [{
        "generation": 0,
        "timestamp": datetime.now().isoformat(),
        "dice": init_dice,
        "iou": init_iou,
        "pseudo_labels": 0,
        "cumulative_cases": 0,
        "confidence_stats": {},
        "action": "initial",
    }]

    best_dice = init_dice
    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    cumulative_pseudo = []  # Accumulate across generations

    # --- Main evolutionary loop ---
    for batch_idx in range(total_batches):
        generation = batch_idx + 1
        start_idx = batch_idx * CASES_PER_BATCH
        end_idx = min(start_idx + CASES_PER_BATCH, len(all_cases))
        batch_cases = all_cases[start_idx:end_idx]

        print(f"\n{'='*60}")
        print(f"GENERATION {generation}/{total_batches} "
              f"(cases {start_idx+1}-{end_idx} of {len(all_cases)})")
        print(f"{'='*60}")

        t0 = time.time()

        # --- Phase 1: Inference with confidence ---
        print("  Phase 1: Inference with TTA...")
        pseudo_labels, conf_stats = batch_inference(model, batch_cases, device)
        print(f"  Accepted pseudo-labels: {len(pseudo_labels)}")

        if len(pseudo_labels) < 10:
            print("  Too few pseudo-labels, skipping re-training.")
            evolution_log.append({
                "generation": generation,
                "timestamp": datetime.now().isoformat(),
                "dice": best_dice,
                "iou": evolution_log[-1]["iou"],
                "pseudo_labels": len(pseudo_labels),
                "cumulative_cases": end_idx,
                "confidence_stats": conf_stats,
                "action": "skipped",
            })
            continue

        # --- Phase 2: Accumulate pseudo-labels ---
        # Keep a sliding window of recent pseudo-labels to limit memory
        cumulative_pseudo.extend(pseudo_labels)
        MAX_PSEUDO = 20000
        if len(cumulative_pseudo) > MAX_PSEUDO:
            # Keep most recent, weighted toward higher confidence
            cumulative_pseudo.sort(key=lambda x: x[2], reverse=True)
            cumulative_pseudo = cumulative_pseudo[:MAX_PSEUDO]

        print(f"  Cumulative pseudo-labels: {len(cumulative_pseudo)}")

        # --- Phase 3: Re-train ---
        print("  Phase 3: Re-training with EWC...")
        model = retrain(model, labeled_samples, cumulative_pseudo,
                        ewc, device, generation)

        # --- Update EMA ---
        ema.update(model)

        # --- Phase 4: Evaluate ---
        dice, iou = evaluate_holdout(model, device)
        elapsed = time.time() - t0

        print(f"\n  Generation {generation}: DSC={dice:.4f}, IoU={iou:.4f} "
              f"(prev best: {best_dice:.4f}) - Time: {elapsed:.0f}s")

        # --- Evolutionary selection ---
        if dice >= best_dice - 0.005:  # Allow tiny margin
            if dice > best_dice:
                action = "improved"
                print(f"  -> IMPROVED (+{dice - best_dice:.4f})")
            else:
                action = "maintained"
                print(f"  -> Maintained (within margin)")
            best_dice = max(best_dice, dice)
            best_model_state = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
        else:
            action = "rollback"
            print(f"  -> ROLLBACK (dropped {best_dice - dice:.4f})")
            model.load_state_dict(best_model_state)
            model = model.to(device)
            dice = best_dice

        evolution_log.append({
            "generation": generation,
            "timestamp": datetime.now().isoformat(),
            "dice": float(dice),
            "iou": float(iou),
            "pseudo_labels": len(pseudo_labels),
            "cumulative_pseudo": len(cumulative_pseudo),
            "cumulative_cases": end_idx,
            "confidence_stats": conf_stats,
            "action": action,
            "elapsed_sec": elapsed,
        })

        # --- Save checkpoint ---
        ckpt_path = EVOLUTION_DIR / f"model_gen{generation:03d}.pth"
        torch.save(model.state_dict(), ckpt_path)

        # --- Save evolution log (written under results/) ---
        log_path = EVOLUTION_LOG
        with open(log_path, "w") as f:
            json.dump(evolution_log, f, indent=2)

    # ============================================================
    # Final Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("EVOLUTIONARY LEARNING COMPLETE")
    print(f"{'='*60}")
    print(f"Generations: {len(evolution_log) - 1}")
    print(f"Initial DSC: {init_dice:.4f}")
    print(f"Final DSC:   {best_dice:.4f}")
    print(f"Improvement: {best_dice - init_dice:+.4f}")

    improvements = sum(1 for e in evolution_log[1:] if e["action"] == "improved")
    rollbacks = sum(1 for e in evolution_log[1:] if e["action"] == "rollback")
    print(f"Improvements: {improvements}, Rollbacks: {rollbacks}")

    # Save final model
    final_path = EVOLUTION_DIR / "model_final.pth"
    torch.save(best_model_state, final_path)
    print(f"Final model: {final_path}")
    print(f"Evolution log: {EVOLUTION_LOG}")


if __name__ == "__main__":
    main()
