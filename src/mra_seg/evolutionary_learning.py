"""Evolutionary knowledge update: self-training with confidence-gated
pseudo-labels, EWC, and a selection rule with an explicit checkpoint policy.

This is the framework evaluated in the paper (six configurations, eight runs).
Each generation consumes one 100-case batch of unlabeled DICOM examinations:

  Phase 1  batch inference, per-slice confidence = mean max-softmax
  Phase 2  quality gate (threshold tau, capped pool; top-confidence or
           foreground-stratified selection)
  Phase 3  re-training on labeled + pseudo-labeled slices with EWC anchored
           to the seed parameters
  Phase 4  evaluation on the fold's holdout cases; accept or roll back

Two quantities are tracked separately on purpose: the recorded best score
(DSC_best) and the checkpoint actually retained. With delta = 0 and
replacement only on strict improvement they move together; with delta > 0
under replace-on-acceptance they can separate (Sections 2.4 and 3.3 of the
paper).

Usage:
    python evolutionary_learning.py <configuration> [fold]
      configuration = naive | strict-topconf | strict-stratified |
                      strict-large-pool | strict-weak-ewc | strict-combined
      fold = 1..5 (default 1); the seed is models/deeplabv3plus_5fold/fold<k>.pth
             and the holdout is that fold's test cases.

Batch sizes were tuned on the study hardware (bf16 autocast, channels_last,
one GPU per run) and can be overridden with EVO_TRAIN_BATCH / EVO_INFER_BATCH /
EVO_WORKERS. EVO_MAX_GEN and EVO_CASES_PER_BATCH shrink the run for smoke tests.
"""
import heapq
import itertools
import json
import os
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from torchvision.models.segmentation import deeplabv3_resnet50

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
cv2.setNumThreads(0)  # do not compete with DataLoader workers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (  # noqa: E402
    EVOLUTION_DIR, FOLD_MODEL_DIR, MRA_STUDY_DIR, RAW_JPEG_DIR, RAW_PNG_DIR,
    SEG_RESULT_DIR, ensure_dirs, require_external,
)

# ============================================================
# Configurations (Table 1 of the paper)
# ============================================================
_STRICT = dict(pseudo_weight=0.3, threshold=0.97, max_pseudo_abs=None,
               max_pseudo_ratio=2.0, lr_backbone=0.00005, lr_head=0.0005,
               ewc_lambda=1000.0, freeze_epochs=2, clip=1.0, epochs=7,
               margin=0.0, strict=True, balanced_sampler=True,
               selection="topconf")

STRATEGIES = {
    # Permissive configuration: positive tolerance, checkpoint replaced on
    # every acceptance, low threshold, equal pseudo-label loss weight.
    "naive": dict(pseudo_weight=1.0, threshold=0.80, max_pseudo_abs=20000,
                  max_pseudo_ratio=None, lr_backbone=0.001, lr_head=0.001,
                  ewc_lambda=500.0, freeze_epochs=0, clip=None, epochs=5,
                  margin=0.005, strict=False, balanced_sampler=False,
                  selection="topconf"),
    # delta = 0 family: checkpoint replaced only on strict improvement.
    "strict-topconf": dict(_STRICT),
    "strict-stratified": dict(_STRICT, selection="stratified"),
    "strict-large-pool": dict(_STRICT, max_pseudo_abs=20000, max_pseudo_ratio=None),
    "strict-weak-ewc": dict(_STRICT, ewc_lambda=100.0),
    "strict-combined": dict(_STRICT, selection="stratified", max_pseudo_abs=20000,
                            max_pseudo_ratio=None, ewc_lambda=100.0),
}

STRATEGY = sys.argv[1] if len(sys.argv) > 1 else "strict-topconf"
if STRATEGY not in STRATEGIES:
    sys.exit(f"configuration must be one of {list(STRATEGIES)}")
CFG = STRATEGIES[STRATEGY]

FOLD = int(sys.argv[2]) if len(sys.argv) > 2 else 1
TAG = f"{STRATEGY}_fold{FOLD}"
INITIAL_MODEL = FOLD_MODEL_DIR / f"fold{FOLD}.pth"
RUN_DIR = EVOLUTION_DIR / TAG
EVOLUTION_LOG = SEG_RESULT_DIR / f"evolution_log_{TAG}.json"
DICOM_DIR = MRA_STUDY_DIR
ensure_dirs(RUN_DIR, SEG_RESULT_DIR)


def _fold_test_cases(fold):
    """The fold's test cases, from the same split as the seed training."""
    from sklearn.model_selection import KFold
    ids = np.array(sorted(int(d.name) for d in RAW_JPEG_DIR.iterdir() if d.is_dir()))
    splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(ids))
    return ids[splits[fold - 1][1]].tolist()


HOLDOUT_CASES = _fold_test_cases(FOLD)
NUM_CLASSES = 2
IMG_SIZE = 512
CASES_PER_BATCH = 100
NUM_GENERATIONS = 10
SEED = 42

TRAIN_BATCH = int(os.environ.get("EVO_TRAIN_BATCH", 24))
INFER_BATCH = int(os.environ.get("EVO_INFER_BATCH", 48))
NUM_WORKERS = int(os.environ.get("EVO_WORKERS", 3))
# The Fisher estimate runs in fp32 and needs far more memory per sample.
EWC_BATCH = int(os.environ.get("EVO_EWC_BATCH", 8))

if os.environ.get("EVO_MAX_GEN"):
    NUM_GENERATIONS = int(os.environ["EVO_MAX_GEN"])
if os.environ.get("EVO_CASES_PER_BATCH"):
    CASES_PER_BATCH = int(os.environ["EVO_CASES_PER_BATCH"])

PSEUDO_WEIGHT = CFG["pseudo_weight"]
CONFIDENCE_THRESHOLD = CFG["threshold"]
EWC_LAMBDA = CFG["ewc_lambda"]
LR_BACKBONE = CFG["lr_backbone"]
LR_HEAD = CFG["lr_head"]
RE_TRAIN_EPOCHS = CFG["epochs"]
ROLLBACK_MARGIN = CFG["margin"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

AMP_DTYPE = torch.bfloat16
MEM_FMT = torch.channels_last
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

print(f"=== Evolutionary knowledge update ({STRATEGY}, fold {FOLD}) ===")
print(f"seed model : {INITIAL_MODEL.name}  (holdout {HOLDOUT_CASES})")
print(f"GPUs visible: {torch.cuda.device_count()}, "
      f"train batch {TRAIN_BATCH}, infer batch {INFER_BATCH}, workers {NUM_WORKERS}")
print(f"pseudo weight {PSEUDO_WEIGHT}, threshold {CONFIDENCE_THRESHOLD}, "
      f"EWC lambda {EWC_LAMBDA}, LR {LR_BACKBONE}/{LR_HEAD}, "
      f"epochs {RE_TRAIN_EPOCHS}, margin {ROLLBACK_MARGIN}, strict {CFG['strict']}")


def amp():
    return torch.autocast("cuda", dtype=AMP_DTYPE)


# ============================================================
# Candidate slices are held JPEG/PNG-encoded to bound memory.
# ============================================================
def encode(img_np, mask_np):
    ok1, jpg = cv2.imencode(".jpg", img_np[:, :, ::-1],
                            [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    ok2, png = cv2.imencode(".png", mask_np,
                            [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    if not (ok1 and ok2):
        raise RuntimeError("failed to encode a candidate slice")
    return jpg.tobytes(), png.tobytes()


def decode(blob):
    jpg, png = blob
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
    mask = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    return np.ascontiguousarray(img), mask


# ============================================================
# Model
# ============================================================
def create_model():
    m = deeplabv3_resnet50(weights=None, aux_loss=True)
    m.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    m.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    return m


def load_initial_model(device):
    m = create_model()
    m.load_state_dict(torch.load(INITIAL_MODEL, map_location="cpu",
                                 weights_only=True))
    return m.to(device, memory_format=MEM_FMT)


# ============================================================
# EWC (Fisher estimated once from the seed; the anchor never moves)
# ============================================================
class EWC:
    def __init__(self, model, dataloader, device):
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()
                  if p.requires_grad}
        n_samples = 0
        for batch in dataloader:
            images, masks = batch[0].to(device), batch[1].to(device)
            images = images.to(memory_format=MEM_FMT)
            model.zero_grad(set_to_none=True)
            out = model(images)["out"].float()
            nn.CrossEntropyLoss()(out, masks).backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2) * images.size(0)
            n_samples += images.size(0)
        for n in fisher:
            fisher[n] /= n_samples
        self.fisher = fisher
        self.params = {n: p.data.clone() for n, p in model.named_parameters()
                       if p.requires_grad}
        model.zero_grad(set_to_none=True)

    def penalty(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss = loss + (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return loss


# ============================================================
# Datasets
# ============================================================
NORM = A.Compose([A.Normalize(mean=MEAN.tolist(), std=STD.tolist()), ToTensorV2()])


class LabeledDataset(Dataset):
    def __init__(self, case_ids, jpeg_dir, png_dir, augment=True):
        self.samples = []
        for cid in case_ids:
            jd, pd_ = jpeg_dir / str(cid), png_dir / str(cid)
            for jp in sorted(jd.glob("*.JPG")):
                pp = pd_ / (jp.stem + ".png")
                if pp.exists():
                    self.samples.append((str(jp), str(pp)))
        self.transform = A.Compose([
            A.Rotate(limit=25, interpolation=1, border_mode=0, p=0.9),
            A.RandomScale(scale_limit=(-0.3, 0.0), p=0.9),
            A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE,
                          border_mode=0, value=0, mask_value=0),
            A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15,
                                       contrast_limit=0.15, p=0.3),
            A.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
            ToTensorV2(),
        ]) if augment else NORM

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ip, mp = self.samples[idx]
        img = np.array(Image.open(ip).convert("RGB"))
        mask = (np.array(Image.open(mp)) > 0).astype(np.uint8)
        t = self.transform(image=img, mask=mask)
        return t["image"], t["mask"].long(), 1.0


class PseudoDataset(Dataset):
    """Holds encoded pseudo-labeled slices; decodes on access."""

    def __init__(self, samples):
        self.samples = samples
        self.transform = A.Compose([
            A.Rotate(limit=15, interpolation=1, border_mode=0, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1,
                                       contrast_limit=0.1, p=0.2),
            A.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, mask = decode(self.samples[idx][0])
        t = self.transform(image=img, mask=mask)
        return t["image"], t["mask"].long(), PSEUDO_WEIGHT


class ValidationDataset(Dataset):
    def __init__(self, case_ids, jpeg_dir, png_dir):
        self.samples = []
        for cid in case_ids:
            jd, pd_ = jpeg_dir / str(cid), png_dir / str(cid)
            for jp in sorted(jd.glob("*.JPG")):
                pp = pd_ / (jp.stem + ".png")
                if pp.exists():
                    self.samples.append((str(jp), str(pp)))
        self.transform = NORM

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ip, mp = self.samples[idx]
        img = np.array(Image.open(ip).convert("RGB"))
        mask = (np.array(Image.open(mp)) > 0).astype(np.uint8)
        t = self.transform(image=img, mask=mask)
        return t["image"], t["mask"].long()


class DICOMInferenceDataset(Dataset):
    def __init__(self, case_dirs):
        self.samples = []
        for d in case_dirs:
            self.samples += sorted(d.glob("*.dcm"),
                                   key=lambda x: x.stem.split("_")[-1].zfill(5))
        self.transform = NORM

    def __getitem__(self, idx):
        ds = pydicom.dcmread(str(self.samples[idx]))
        px = ds.pixel_array.astype(np.float32)
        if px.max() > 0:
            # Per-slice min-max rescaling to 8 bit (Section 2.1 of the paper).
            px = (px - px.min()) / (px.max() - px.min()) * 255.0
        img = Image.fromarray(px.astype(np.uint8)).convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        return self.transform(image=np.array(img))["image"], idx

    def __len__(self):
        return len(self.samples)


# ============================================================
# Loss
# ============================================================
class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, pred, target, weights=None):
        ce_map = self.ce(pred, target)
        ce_loss = (ce_map * weights.view(-1, 1, 1)).mean() if weights is not None \
            else ce_map.mean()
        soft = torch.softmax(pred, dim=1)[:, 1]
        tgt = (target == 1).float()
        inter = (soft * tgt).sum(dim=(1, 2))
        card = soft.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2))
        dice_loss = 1.0 - ((2.0 * inter + 1.0) / (card + 1.0)).mean()
        return 0.5 * ce_loss + 0.5 * dice_loss


# ============================================================
# Phase 1 + 2: inference and pseudo-label selection
# ============================================================
N_STRATA = 10          # equal-width strata of the predicted foreground fraction
STRATUM_MAX = 0.60     # fractions at or above this join the top stratum


def _stratum(frac):
    return min(int(frac / STRATUM_MAX * N_STRATA), N_STRATA - 1)


def batch_inference(model, case_dirs, device, max_pseudo):
    """Run the model over one batch and select up to max_pseudo pseudo-labels.

    selection = "topconf":    keep the highest-confidence slices.
    selection = "stratified": divide the predicted foreground fraction into
        N_STRATA equal-width strata (width STRATUM_MAX / N_STRATA; slices at or
        above STRATUM_MAX join the top stratum); each stratum receives an equal
        quota of max_pseudo / N_STRATA, filled by the highest-confidence slices
        within that stratum; shortfalls are not reallocated. The threshold
        applies unchanged, before stratification.

    Candidates are kept in per-stratum min-heaps so memory stays bounded.
    """
    stratified = CFG["selection"] == "stratified"
    quota = max_pseudo // N_STRATA if stratified else max_pseudo

    ds = DICOMInferenceDataset(case_dirs)
    loader = DataLoader(ds, batch_size=INFER_BATCH, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        prefetch_factor=4)
    model.eval()

    heaps = [[] for _ in range(N_STRATA if stratified else 1)]
    tie = itertools.count()
    total = high = 0

    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device, non_blocking=True).to(memory_format=MEM_FMT)
            with amp():
                logits = model(images)["out"]
            probs = torch.softmax(logits.float(), dim=1)
            preds = probs.argmax(dim=1)
            confs = probs.max(dim=1)[0].mean(dim=(1, 2))
            fracs = (preds == 1).float().mean(dim=(1, 2))

            total += images.size(0)
            keep = (confs >= CONFIDENCE_THRESHOLD).nonzero(as_tuple=True)[0]
            high += int(keep.numel())
            if keep.numel() == 0:
                continue

            cvals = confs[keep].tolist()
            fvals = fracs[keep].tolist()
            bins = [_stratum(f) if stratified else 0 for f in fvals]

            sel = [i for i, (b, c) in enumerate(zip(bins, cvals))
                   if len(heaps[b]) < quota or c > heaps[b][0][0]]
            if not sel:
                continue
            keep = keep[torch.tensor(sel, device=keep.device)]
            cvals = [cvals[i] for i in sel]
            bins = [bins[i] for i in sel]

            imgs = images[keep].permute(0, 2, 3, 1).float().cpu().numpy()
            masks = preds[keep].to(torch.uint8).cpu().numpy()
            imgs = ((imgs * STD + MEAN) * 255.0).clip(0, 255).astype(np.uint8)

            for k, (c, b) in enumerate(zip(cvals, bins)):
                item = (c, next(tie), encode(imgs[k], masks[k]))
                if len(heaps[b]) < quota:
                    heapq.heappush(heaps[b], item)
                else:
                    heapq.heappushpop(heaps[b], item)

    pool = [(blob, c) for h in heaps for c, _, blob in h]
    pool.sort(key=lambda x: -x[1])
    selected = pool[:max_pseudo]
    per_stratum = [len(h) for h in heaps]

    del loader, ds
    torch.cuda.empty_cache()

    print(f"  Inference: {total} slices, above threshold: {high} "
          f"({high / max(total, 1) * 100:.1f}%), selected: {len(selected)}/{max_pseudo}"
          f"{'  strata=' + str(per_stratum) if stratified else ''}")
    return selected, {"total": total, "above_threshold": high,
                      "selected": len(selected), "per_stratum": per_stratum}


# ============================================================
# Phase 3: re-training
# ============================================================
def retrain(model, labeled_dataset, pseudo_samples, ewc, device, generation):
    pseudo_dataset = PseudoDataset(pseudo_samples)
    combined = ConcatDataset([labeled_dataset, pseudo_dataset])
    n_l, n_p = len(labeled_dataset), len(pseudo_dataset)

    if CFG["balanced_sampler"]:
        # Equalise the expected sampling frequency of the two sources.
        w = [1.0 / n_l] * n_l + [1.0 / max(n_p, 1)] * n_p
        sampler = WeightedRandomSampler(w, num_samples=n_l + n_p, replacement=True)
        loader = DataLoader(combined, batch_size=TRAIN_BATCH, sampler=sampler,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            drop_last=True, prefetch_factor=4)
    else:
        # naive: uniform shuffle of the concatenated set.
        loader = DataLoader(combined, batch_size=TRAIN_BATCH, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            drop_last=True, prefetch_factor=4)

    backbone, head = [], []
    for name, p in model.named_parameters():
        (backbone if "backbone" in name else head).append(p)

    criterion = DiceCELoss()
    optimizer = optim.AdamW([{"params": backbone, "lr": LR_BACKBONE},
                             {"params": head, "lr": LR_HEAD}], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=RE_TRAIN_EPOCHS, eta_min=1e-7)

    for epoch in range(RE_TRAIN_EPOCHS):
        model.train()
        frozen = epoch < CFG["freeze_epochs"]
        for p in backbone:
            p.requires_grad = not frozen

        running, seen = 0.0, 0
        for images, masks, weights in loader:
            images = images.to(device, non_blocking=True).to(memory_format=MEM_FMT)
            masks = masks.to(device, non_blocking=True)
            weights = weights.float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp():
                out = model(images)
            loss = criterion(out["out"].float(), masks, weights)
            if "aux" in out:
                loss = loss + 0.4 * criterion(out["aux"].float(), masks, weights)
            if ewc is not None:
                loss = loss + EWC_LAMBDA * ewc.penalty(model)
            loss.backward()
            if CFG["clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
            optimizer.step()
            running += loss.item() * images.size(0)
            seen += images.size(0)

        scheduler.step()
        print(f"    Gen {generation} Epoch {epoch + 1}/{RE_TRAIN_EPOCHS} "
              f"- Loss: {running / max(seen, 1):.4f} "
              f"- LR bb={optimizer.param_groups[0]['lr']:.6f} "
              f"hd={optimizer.param_groups[1]['lr']:.6f}"
              f"{' [backbone frozen]' if frozen else ''}")

    for p in model.parameters():
        p.requires_grad = True
    del loader, combined, pseudo_dataset
    return model


# ============================================================
# Phase 4: evaluation
# ============================================================
def evaluate_holdout(model, device):
    """Holdout DSC as the slice mean, the IoU, and the per-case (volume) mean.

    The selection rule uses the slice mean; both aggregations are logged
    because they are not interchangeable (Section 3.1 of the paper).
    """
    model.eval()
    slice_d, slice_i, case_d = [], [], []
    for cid in HOLDOUT_CASES:
        ds = ValidationDataset([cid], RAW_JPEG_DIR, RAW_PNG_DIR)
        loader = DataLoader(ds, batch_size=INFER_BATCH, shuffle=False,
                            num_workers=2, pin_memory=True)
        inter_sum = pn_sum = gn_sum = 0.0
        with torch.inference_mode():
            for images, masks in loader:
                images = images.to(device, non_blocking=True).to(memory_format=MEM_FMT)
                masks = masks.to(device, non_blocking=True)
                with amp():
                    logits = model(images)["out"]
                preds = logits.float().argmax(dim=1)
                p = (preds == 1).float()
                t = (masks == 1).float()
                inter = (p * t).sum(dim=(1, 2))
                pn, gn = p.sum(dim=(1, 2)), t.sum(dim=(1, 2))
                union = pn + gn - inter
                slice_d += ((2 * inter + 1e-7) / (pn + gn + 1e-7)).tolist()
                slice_i += ((inter + 1e-7) / (union + 1e-7)).tolist()
                inter_sum += float(inter.sum())
                pn_sum += float(pn.sum())
                gn_sum += float(gn.sum())
        case_d.append((2 * inter_sum + 1e-7) / (pn_sum + gn_sum + 1e-7))
        del loader, ds
    return float(np.mean(slice_d)), float(np.mean(slice_i)), float(np.mean(case_d))


# ============================================================
# Main
# ============================================================
def main():
    device = torch.device("cuda:0")
    print(f"\nLoading seed model: {INITIAL_MODEL.name}")
    model = load_initial_model(device)

    train_cases = sorted(int(d.name) for d in RAW_JPEG_DIR.iterdir()
                         if d.is_dir() and int(d.name) not in HOLDOUT_CASES)
    labeled = LabeledDataset(train_cases, RAW_JPEG_DIR, RAW_PNG_DIR, augment=True)
    labeled_val = LabeledDataset(train_cases, RAW_JPEG_DIR, RAW_PNG_DIR, augment=False)
    max_pseudo = (CFG["max_pseudo_abs"] if CFG["max_pseudo_abs"]
                  else int(len(labeled) * CFG["max_pseudo_ratio"]))
    print(f"train cases: {len(train_cases)} ({len(labeled)} slices), "
          f"max pseudo: {max_pseudo}")

    init_dice, init_iou, init_case = evaluate_holdout(model, device)
    print(f"\nGeneration 0 (seed): DSC={init_dice:.4f} (slice), "
          f"{init_case:.4f} (case), IoU={init_iou:.4f}")

    print("Computing EWC Fisher information...")
    ewc = EWC(model, DataLoader(labeled_val, batch_size=EWC_BATCH, shuffle=False,
                                num_workers=2, pin_memory=True), device)
    print("EWC computed.")

    require_external(DICOM_DIR, "unlabeled MRA studies (MRA_STUDY_DIR)")
    all_cases = sorted(d for d in DICOM_DIR.iterdir() if d.is_dir())
    n_gen = min(NUM_GENERATIONS,
                (len(all_cases) + CASES_PER_BATCH - 1) // CASES_PER_BATCH)
    print(f"DICOM cases: {len(all_cases)}, Generations: {n_gen}")

    log = [{"generation": 0, "timestamp": datetime.now().isoformat(),
            "candidate_dice": None, "candidate_iou": None,
            "seed_case_dice": init_case,
            "retained_dice": init_dice, "retained_iou": init_iou,
            "dice": init_dice, "iou": init_iou,
            "action": "initial", "cumulative_cases": 0}]
    best_dice, best_iou = init_dice, init_iou
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    pool = []

    for b in range(n_gen):
        gen = b + 1
        cases = all_cases[b * CASES_PER_BATCH:(b + 1) * CASES_PER_BATCH]
        print(f"\n{'=' * 60}\nGENERATION {gen}/{n_gen} "
              f"(cases {b * CASES_PER_BATCH + 1}-{b * CASES_PER_BATCH + len(cases)})"
              f"\n{'=' * 60}")
        t0 = time.time()

        print("  Phase 1: Inference...")
        new_pseudo, conf_stats = batch_inference(model, cases, device, max_pseudo)
        if len(new_pseudo) < 10:
            print("  Too few pseudo-labels, skipping.")
            log.append({"generation": gen, "timestamp": datetime.now().isoformat(),
                        "candidate_dice": None, "candidate_iou": None,
                        "retained_dice": best_dice, "retained_iou": best_iou,
                        "dice": best_dice, "iou": best_iou, "action": "skipped",
                        "cumulative_cases": (b + 1) * CASES_PER_BATCH,
                        "confidence_stats": conf_stats})
            continue

        # Phase 2: cumulative pool, truncated to the cap by confidence.
        pool.extend(new_pseudo)
        pool.sort(key=lambda x: -x[1])
        pool = pool[:max_pseudo]
        print(f"  Pseudo-labels: {len(pool)} (max: {max_pseudo})")

        print("  Phase 3: Re-training...")
        model = retrain(model, labeled, pool, ewc, device, gen)

        dice, iou, case_dice = evaluate_holdout(model, device)
        elapsed = time.time() - t0
        print(f"\n  Generation {gen}: DSC={dice:.4f} (slice), "
              f"{case_dice:.4f} (case), IoU={iou:.4f} "
              f"(best: {best_dice:.4f}) - Time: {elapsed:.0f}s")

        accept = dice > best_dice if CFG["strict"] else dice >= best_dice - ROLLBACK_MARGIN
        if accept:
            if dice > best_dice:
                action = "improved"
                print(f"  -> IMPROVED (+{dice - best_dice:.4f})")
                best_dice, best_iou = dice, iou
            else:
                action = "maintained"
                print("  -> Maintained (within margin)")
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            action = "rollback"
            print(f"  -> ROLLBACK (candidate {dice:.4f}, best {best_dice:.4f})")
            model.load_state_dict(best_state)
            model = model.to(device, memory_format=MEM_FMT)

        log.append({"generation": gen, "timestamp": datetime.now().isoformat(),
                    "candidate_dice": float(dice), "candidate_iou": float(iou),
                    "candidate_case_dice": float(case_dice),
                    "retained_dice": float(best_dice), "retained_iou": float(best_iou),
                    "dice": float(best_dice), "iou": float(best_iou),
                    "pseudo_labels": len(new_pseudo), "cumulative_pseudo": len(pool),
                    "cumulative_cases": (b + 1) * CASES_PER_BATCH,
                    "confidence_stats": conf_stats, "action": action,
                    "elapsed_sec": elapsed})

        torch.save(model.state_dict(), RUN_DIR / f"model_gen{gen:03d}.pth")
        EVOLUTION_LOG.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}\nRUN {TAG} COMPLETE\n{'=' * 60}")
    print(f"Seed DSC:  {init_dice:.4f}")
    print(f"Best DSC:  {best_dice:.4f}  ({best_dice - init_dice:+.4f})")
    for a in ("improved", "maintained", "rollback"):
        print(f"  {a}: {sum(1 for e in log[1:] if e['action'] == a)}")
    torch.save(best_state, RUN_DIR / "model_final.pth")
    print(f"Final model: {RUN_DIR / 'model_final.pth'}")


if __name__ == "__main__":
    main()
