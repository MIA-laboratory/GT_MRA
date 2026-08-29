"""
Evolutionary Knowledge Update v2 - Improved
=============================================
Round 2 improvements over v1:
  1. Weighted loss: labeled x1.0, pseudo x0.3
  2. Lower learning rate: 0.0001 (vs 0.001)
  3. Pseudo-label ratio limit: max 2x labeled data (top confidence only)
  4. Backbone frozen for first 2 epochs, then unfreeze
  5. Separate DataLoaders for labeled vs pseudo (weighted sampling)
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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.models.segmentation import deeplabv3_resnet50
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
    INITIAL_MODEL, EVOLUTION_V2_DIR, EVOLUTION_LOG, EVOLUTION_LOG_V2,
    SEG_RESULT_DIR, RAW_JPEG_DIR, RAW_PNG_DIR, MRA_STUDY_DIR,
    ensure_dirs, require_external,
)

DICOM_DIR = MRA_STUDY_DIR          # unlabeled studies (not distributed)
EVOLUTION_DIR = EVOLUTION_V2_DIR
ensure_dirs(EVOLUTION_DIR, SEG_RESULT_DIR)

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
NUM_GENERATIONS = 10
RE_TRAIN_EPOCHS = 7
EWC_LAMBDA = 1000.0        # v2: stronger EWC
PSEUDO_WEIGHT = 0.3         # v2: pseudo-label loss weight
MAX_PSEUDO_RATIO = 2.0      # v2: max pseudo = 2x labeled
CONFIDENCE_THRESHOLD = 0.97 # v2: stricter threshold
LR_BACKBONE = 0.00005       # v2: very low backbone LR
LR_HEAD = 0.0005            # v2: moderate head LR
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print(f"=== Evolutionary Learning v2 (Improved) ===")
print(f"GPUs: {NUM_GPUS}, Batch: {BATCH_SIZE}")
print(f"Pseudo weight: {PSEUDO_WEIGHT}, Max ratio: {MAX_PSEUDO_RATIO}x")
print(f"EWC lambda: {EWC_LAMBDA}, Confidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"LR backbone: {LR_BACKBONE}, LR head: {LR_HEAD}")


# ============================================================
# Model
# ============================================================
def create_model():
    model = deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    return model


def load_initial_model(device):
    model = create_model()
    state = torch.load(INITIAL_MODEL, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    return model


# ============================================================
# EWC
# ============================================================
class EWC:
    def __init__(self, model, dataloader, device):
        self.params = {}
        self.fisher = {}
        self._compute_fisher(model, dataloader, device)

    def _compute_fisher(self, model, dataloader, device):
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()
                  if p.requires_grad}
        num_samples = 0
        for batch in dataloader:
            images, masks = batch[0], batch[1]
            images, masks = images.to(device), masks.to(device)
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
# Datasets
# ============================================================
class LabeledDataset(Dataset):
    def __init__(self, case_ids, jpeg_dir, png_dir, augment=True):
        self.samples = []
        for cid in case_ids:
            jpeg_case = jpeg_dir / str(cid)
            png_case = png_dir / str(cid)
            for jp in sorted(jpeg_case.glob("*.JPG")):
                pp = png_case / (jp.stem + ".png")
                if pp.exists():
                    self.samples.append((str(jp), str(pp)))

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
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        return transformed["image"], transformed["mask"].long(), 1.0  # weight=1.0


class PseudoDataset(Dataset):
    def __init__(self, pseudo_samples):
        # pseudo_samples: list of (image_np_512, mask_np_512, confidence)
        self.samples = pseudo_samples
        self.transform = A.Compose([
            A.Rotate(limit=15, interpolation=1, border_mode=0, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1,
                                       contrast_limit=0.1, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, mask, conf = self.samples[idx]
        transformed = self.transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"].long(), PSEUDO_WEIGHT  # weighted


class DICOMInferenceDataset(Dataset):
    def __init__(self, case_dirs):
        self.samples = []
        for case_dir in case_dirs:
            dcm_files = sorted(case_dir.glob("*.dcm"),
                               key=lambda x: x.stem.split("_")[-1].zfill(5))
            for f in dcm_files:
                self.samples.append(f)
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dcm_path = self.samples[idx]
        ds = pydicom.dcmread(str(dcm_path))
        pixel = ds.pixel_array.astype(np.float32)
        if pixel.max() > 0:
            pixel = (pixel - pixel.min()) / (pixel.max() - pixel.min()) * 255.0
        pixel = pixel.astype(np.uint8)
        img = Image.fromarray(pixel).convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img_np = np.array(img)
        transformed = self.transform(image=img_np)
        return transformed["image"], str(dcm_path)


class ValidationDataset(Dataset):
    def __init__(self, case_ids, jpeg_dir, png_dir):
        self.samples = []
        for cid in case_ids:
            jpeg_case = jpeg_dir / str(cid)
            png_case = png_dir / str(cid)
            for jp in sorted(jpeg_case.glob("*.JPG")):
                pp = png_case / (jp.stem + ".png")
                if pp.exists():
                    self.samples.append((str(jp), str(pp)))
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
# Loss
# ============================================================
class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target, weights=None):
        # Per-pixel CE
        ce_map = self.ce(pred, target)  # (B, H, W)
        if weights is not None:
            # Apply per-sample weight
            w = weights.view(-1, 1, 1)
            ce_loss = (ce_map * w).mean()
        else:
            ce_loss = ce_map.mean()

        # Dice loss (not weighted -- structural)
        pred_soft = torch.softmax(pred, dim=1)[:, 1]
        target_f = (target == 1).float()
        inter = (pred_soft * target_f).sum(dim=(1, 2))
        card = pred_soft.sum(dim=(1, 2)) + target_f.sum(dim=(1, 2))
        dice_loss = 1.0 - ((2.0 * inter + 1.0) / (card + 1.0)).mean()

        return 0.5 * ce_loss + 0.5 * dice_loss


# ============================================================
# Inference
# ============================================================
def batch_inference(model, case_dirs, device, max_pseudo):
    dataset = DICOMInferenceDataset(case_dirs)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)

    if NUM_GPUS > 1 and not isinstance(model, nn.DataParallel):
        eval_model = nn.DataParallel(model, device_ids=list(range(NUM_GPUS)))
    else:
        eval_model = model
    eval_model.eval()

    candidates = []  # (image_np, mask_np, confidence)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    total = 0
    high = 0

    with torch.no_grad():
        for images, dcm_paths in loader:
            images = images.to(device)
            outputs = eval_model(images)["out"]
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            confidences = probs.max(dim=1)[0].mean(dim=(1, 2))

            for i in range(images.size(0)):
                total += 1
                conf = confidences[i].item()
                if conf >= CONFIDENCE_THRESHOLD:
                    high += 1
                    img_np = images[i].cpu().permute(1, 2, 0).numpy()
                    img_np = ((img_np * std + mean) * 255).clip(0, 255).astype(np.uint8)
                    mask_np = preds[i].cpu().numpy().astype(np.uint8)
                    candidates.append((img_np, mask_np, conf))

    # Select top confidence up to max_pseudo
    candidates.sort(key=lambda x: x[2], reverse=True)
    selected = candidates[:max_pseudo]

    print(f"  Inference: {total} slices, "
          f"above threshold: {high} ({high/max(total,1)*100:.1f}%), "
          f"selected: {len(selected)}/{max_pseudo}")

    return selected, {"total": total, "above_threshold": high, "selected": len(selected)}


# ============================================================
# Evaluation
# ============================================================
def evaluate_holdout(model, device):
    val_dataset = ValidationDataset(HOLDOUT_CASES, LABELED_JPEG_DIR, LABELED_PNG_DIR)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    eval_model = model.module if isinstance(model, nn.DataParallel) else model
    eval_model.eval()

    all_dice = []
    all_iou = []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
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
# Training with weighted loss + differential LR
# ============================================================
def retrain(model, labeled_dataset, pseudo_samples, ewc, device, generation):
    pseudo_dataset = PseudoDataset(pseudo_samples)

    # Interleave: repeat labeled data to balance ratio
    # labeled gets sampled more frequently
    from torch.utils.data import WeightedRandomSampler

    combined = ConcatDataset([labeled_dataset, pseudo_dataset])
    n_labeled = len(labeled_dataset)
    n_pseudo = len(pseudo_dataset)
    n_total = n_labeled + n_pseudo

    # Weight labeled samples higher so they appear ~equally often
    sample_weights = [1.0 / n_labeled] * n_labeled + [1.0 / max(n_pseudo, 1)] * n_pseudo
    sampler = WeightedRandomSampler(sample_weights, num_samples=n_total, replacement=True)

    loader = DataLoader(combined, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=4, pin_memory=True, drop_last=True,
                        persistent_workers=True)

    if NUM_GPUS > 1:
        model = nn.DataParallel(model, device_ids=list(range(NUM_GPUS)))
    model = model.to(device)

    base_model = model.module if isinstance(model, nn.DataParallel) else model

    # Differential LR: backbone low, head higher
    backbone_params = []
    head_params = []
    for name, param in base_model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    criterion = DiceCELoss()
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LR_BACKBONE},
        {'params': head_params, 'lr': LR_HEAD},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=RE_TRAIN_EPOCHS, eta_min=1e-7)

    for epoch in range(RE_TRAIN_EPOCHS):
        model.train()

        # Freeze backbone for first 2 epochs
        freeze_backbone = epoch < 2
        for param in backbone_params:
            param.requires_grad = not freeze_backbone

        running_loss = 0.0
        for images, masks, weights in loader:
            images = images.to(device)
            masks = masks.to(device)
            weights = weights.float().to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output["out"], masks, weights)
            if "aux" in output:
                loss += 0.4 * criterion(output["aux"], masks, weights)

            # EWC penalty
            if ewc is not None:
                bm = model.module if isinstance(model, nn.DataParallel) else model
                loss += EWC_LAMBDA * ewc.penalty(bm)

            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        avg_loss = running_loss / len(loader.dataset)
        bb_lr = optimizer.param_groups[0]['lr']
        hd_lr = optimizer.param_groups[1]['lr']
        frozen_str = " [backbone frozen]" if freeze_backbone else ""
        print(f"    Gen {generation} Epoch {epoch+1}/{RE_TRAIN_EPOCHS} "
              f"- Loss: {avg_loss:.4f} - LR bb={bb_lr:.6f} hd={hd_lr:.6f}{frozen_str}")

    # Unfreeze all before returning
    for param in base_model.parameters():
        param.requires_grad = True

    return model.module if isinstance(model, nn.DataParallel) else model


# ============================================================
# Main
# ============================================================
def main():
    device = torch.device("cuda:0")

    # Load initial model
    print(f"\nLoading initial model: {INITIAL_MODEL.name}")
    model = load_initial_model(device)

    # Prepare labeled data
    train_case_ids = sorted([int(d.name) for d in LABELED_JPEG_DIR.iterdir()
                             if d.is_dir() and int(d.name) not in HOLDOUT_CASES])
    labeled_dataset = LabeledDataset(train_case_ids, LABELED_JPEG_DIR, LABELED_PNG_DIR, augment=True)
    labeled_val = LabeledDataset(train_case_ids, LABELED_JPEG_DIR, LABELED_PNG_DIR, augment=False)
    max_pseudo = int(len(labeled_dataset) * MAX_PSEUDO_RATIO)
    print(f"Labeled: {len(labeled_dataset)} samples, Max pseudo: {max_pseudo}")

    # Initial evaluation
    init_dice, init_iou = evaluate_holdout(model, device)
    print(f"\nGeneration 0 (Initial): DSC={init_dice:.4f}, IoU={init_iou:.4f}")

    # Compute EWC
    print("Computing EWC Fisher information...")
    ewc_loader = DataLoader(labeled_val, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
    ewc = EWC(model, ewc_loader, device)
    print("EWC computed.")

    # DICOM cases
    require_external(DICOM_DIR, "unlabeled MRA studies")
    all_cases = sorted([d for d in DICOM_DIR.iterdir() if d.is_dir()])
    total_batches = min(NUM_GENERATIONS,
                        (len(all_cases) + CASES_PER_BATCH - 1) // CASES_PER_BATCH)
    print(f"\nDICOM cases: {len(all_cases)}, Generations: {total_batches}")

    # Evolution log
    evolution_log = [{
        "generation": 0,
        "timestamp": datetime.now().isoformat(),
        "dice": init_dice,
        "iou": init_iou,
        "action": "initial",
        "cumulative_cases": 0,
    }]

    best_dice = init_dice
    best_iou = init_iou
    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    cumulative_pseudo = []

    for batch_idx in range(total_batches):
        generation = batch_idx + 1
        start_idx = batch_idx * CASES_PER_BATCH
        end_idx = min(start_idx + CASES_PER_BATCH, len(all_cases))
        batch_cases = all_cases[start_idx:end_idx]

        print(f"\n{'='*60}")
        print(f"GENERATION {generation}/{total_batches} "
              f"(cases {start_idx+1}-{end_idx})")
        print(f"{'='*60}")
        t0 = time.time()

        # Phase 1: Inference (strict threshold)
        print("  Phase 1: Inference...")
        new_pseudo, conf_stats = batch_inference(
            model, batch_cases, device, max_pseudo=max_pseudo)

        if len(new_pseudo) < 10:
            print("  Too few pseudo-labels, skipping.")
            evolution_log.append({
                "generation": generation,
                "timestamp": datetime.now().isoformat(),
                "dice": best_dice, "iou": best_iou,
                "action": "skipped",
                "cumulative_cases": end_idx,
                "confidence_stats": conf_stats,
            })
            continue

        # Phase 2: Accumulate with ratio limit
        cumulative_pseudo.extend(new_pseudo)
        # Keep top confidence, capped at max_pseudo
        cumulative_pseudo.sort(key=lambda x: x[2], reverse=True)
        cumulative_pseudo = cumulative_pseudo[:max_pseudo]
        print(f"  Pseudo-labels: {len(cumulative_pseudo)} (max: {max_pseudo})")

        # Phase 3: Re-train with weighted loss
        print("  Phase 3: Re-training (weighted loss + differential LR)...")
        model = retrain(model, labeled_dataset, cumulative_pseudo,
                        ewc, device, generation)

        # Phase 4: Evaluate
        dice, iou = evaluate_holdout(model, device)
        elapsed = time.time() - t0

        print(f"\n  Generation {generation}: DSC={dice:.4f}, IoU={iou:.4f} "
              f"(best: {best_dice:.4f}) - Time: {elapsed:.0f}s")

        # Evolutionary selection (tighter margin)
        if dice >= best_dice - 0.003:
            if dice > best_dice:
                action = "improved"
                print(f"  -> IMPROVED (+{dice - best_dice:.4f})")
                best_dice = dice
                best_iou = iou
            else:
                action = "maintained"
                print(f"  -> Maintained (within margin)")
            best_model_state = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
        else:
            action = "rollback"
            print(f"  -> ROLLBACK (dropped {best_dice - dice:.4f})")
            model.load_state_dict(best_model_state)
            model = model.to(device)

        evolution_log.append({
            "generation": generation,
            "timestamp": datetime.now().isoformat(),
            "dice": float(dice) if action != "rollback" else float(best_dice),
            "iou": float(iou) if action != "rollback" else float(best_iou),
            "pseudo_labels": len(new_pseudo),
            "cumulative_pseudo": len(cumulative_pseudo),
            "cumulative_cases": end_idx,
            "confidence_stats": conf_stats,
            "action": action,
            "elapsed_sec": elapsed,
        })

        # Save checkpoint + log
        torch.save(model.state_dict(), EVOLUTION_DIR / f"model_gen{generation:03d}.pth")
        with open(EVOLUTION_LOG_V2, "w") as f:   # written under results/
            json.dump(evolution_log, f, indent=2)

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("EVOLUTIONARY LEARNING v2 COMPLETE")
    print(f"{'='*60}")
    print(f"Initial DSC:  {init_dice:.4f}")
    print(f"Final DSC:    {best_dice:.4f}")
    print(f"Improvement:  {best_dice - init_dice:+.4f}")

    improvements = sum(1 for e in evolution_log[1:] if e["action"] == "improved")
    maintained = sum(1 for e in evolution_log[1:] if e["action"] == "maintained")
    rollbacks = sum(1 for e in evolution_log[1:] if e["action"] == "rollback")
    print(f"Improved: {improvements}, Maintained: {maintained}, Rollbacks: {rollbacks}")

    torch.save(best_model_state, EVOLUTION_DIR / "model_final_v2.pth")
    print(f"\nFinal model: {EVOLUTION_DIR / 'model_final_v2.pth'}")

    # Comparison with v1
    v1_log_path = EVOLUTION_LOG
    if v1_log_path.exists():
        with open(v1_log_path) as f:
            v1_log = json.load(f)
        print(f"\n--- Round 1 vs Round 2 ---")
        print(f"Round 1: {sum(1 for e in v1_log[1:] if e['action']=='improved')} improved, "
              f"{sum(1 for e in v1_log[1:] if e['action']=='rollback')} rollback")
        print(f"Round 2: {improvements} improved, {rollbacks} rollback")


if __name__ == "__main__":
    main()
