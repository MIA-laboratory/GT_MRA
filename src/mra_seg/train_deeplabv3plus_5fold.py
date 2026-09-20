"""Supervised seed training: DeepLabV3+ (ResNet-50), case-level 5-fold CV.

Settings as in Section 2.2 of the paper: SGD with momentum 0.9, initial
learning rate 0.007 with polynomial decay (power 0.9), weight decay 1e-4,
batch size 24, 132 epochs, combined Dice-CE loss, online augmentation.

The test fold is evaluated every EVAL_EVERY epochs; the checkpoint with the
best test-fold DSC is saved (both the best-epoch and the final-epoch values
are logged). The saved checkpoints are the seeds of the evolutionary update.

Usage:
    python train_deeplabv3plus_5fold.py [fold ...]     # default: all 5 folds
"""
import json
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_resnet50

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (  # noqa: E402
    FOLD_MODEL_DIR, RAW_JPEG_DIR, RAW_PNG_DIR, SEG_RESULT_DIR, ensure_dirs,
)

FOLDS = [int(x) for x in sys.argv[1:]] or [1, 2, 3, 4, 5]
TAG = "_".join(str(f) for f in FOLDS)
RESULT = SEG_RESULT_DIR / f"train_5fold_f{TAG}.json"
ensure_dirs(FOLD_MODEL_DIR, SEG_RESULT_DIR)

NUM_FOLDS = 5
EPOCHS = 132
LR = 0.007
BATCH_SIZE = 24
IMG_SIZE = 512
NUM_CLASSES = 2
NUM_WORKERS = 3
EVAL_EVERY = 4
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
MEM_FMT = torch.channels_last

print(f"=== DeepLabV3+ 5-fold (SGD lr {LR}, poly 0.9, {EPOCHS} epochs) ===")
print(f"  folds {FOLDS}")


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16)


class MRADataset(Dataset):
    def __init__(self, case_ids, augment=False):
        self.samples = []
        for cid in case_ids:
            jd, pd_ = RAW_JPEG_DIR / str(cid), RAW_PNG_DIR / str(cid)
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
            A.OneOf([A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                     A.MedianBlur(blur_limit=5, p=1.0)], p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.15,
                                       contrast_limit=0.15, p=0.3),
            A.GaussNoise(std_range=(0.01, 0.03), p=0.2),
            A.Normalize(mean=MEAN, std=STD), ToTensorV2(),
        ]) if augment else A.Compose(
            [A.Normalize(mean=MEAN, std=STD), ToTensorV2()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ip, mp = self.samples[i]
        img = np.array(Image.open(ip).convert("RGB"))
        mask = (np.array(Image.open(mp)) > 0).astype(np.uint8)
        t = self.transform(image=img, mask=mask)
        return t["image"], t["mask"].long()


class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dw, self.cw = dice_weight, ce_weight

    def forward(self, pred, target):
        ce = self.ce(pred, target)
        soft = torch.softmax(pred, dim=1)[:, 1]
        tgt = (target == 1).float()
        inter = (soft * tgt).sum(dim=(1, 2))
        card = soft.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2))
        dice = 1.0 - ((2.0 * inter + 1.0) / (card + 1.0)).mean()
        return self.cw * ce + self.dw * dice


def create_model():
    m = deeplabv3_resnet50(weights="DEFAULT")
    m.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    m.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    return m.cuda().to(memory_format=MEM_FMT)


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    dices, ious = [], []
    for images, masks in loader:
        images = images.cuda(non_blocking=True).to(memory_format=MEM_FMT)
        masks = masks.cuda(non_blocking=True)
        with amp():
            logits = model(images)["out"]
        pred = logits.float().argmax(dim=1)
        p, t = (pred == 1).float(), (masks == 1).float()
        inter = (p * t).sum(dim=(1, 2))
        union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2)) - inter
        dices += ((2 * inter + 1e-7) /
                  (p.sum(dim=(1, 2)) + t.sum(dim=(1, 2)) + 1e-7)).tolist()
        ious += ((inter + 1e-7) / (union + 1e-7)).tolist()
    return float(np.mean(dices)), float(np.std(dices)), float(np.mean(ious))


def main():
    case_ids = np.array(sorted(int(d.name) for d in RAW_JPEG_DIR.iterdir()
                               if d.is_dir()))
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    splits = list(kf.split(case_ids))

    results = []
    for fold in FOLDS:
        tr_idx, te_idx = splits[fold - 1]
        train_cases = case_ids[tr_idx].tolist()
        test_cases = case_ids[te_idx].tolist()
        print(f"\n{'=' * 60}\nFOLD {fold}/{NUM_FOLDS}  test={test_cases}\n{'=' * 60}")

        tr = MRADataset(train_cases, augment=True)
        te = MRADataset(test_cases, augment=False)
        print(f"  train {len(tr)} slices / test {len(te)} slices")
        tr_loader = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               drop_last=True, persistent_workers=True,
                               prefetch_factor=4)
        te_loader = DataLoader(te, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=2, pin_memory=True)

        model = create_model()
        criterion = DiceCELoss()
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                              weight_decay=1e-4)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda e: (1 - e / EPOCHS) ** 0.9)

        best_dice, best_state, best_epoch = 0.0, None, 0
        t_start = time.time()
        for epoch in range(EPOCHS):
            model.train()
            running, seen = 0.0, 0
            for images, masks in tr_loader:
                images = images.cuda(non_blocking=True).to(memory_format=MEM_FMT)
                masks = masks.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with amp():
                    out = model(images)
                loss = criterion(out["out"].float(), masks)
                if "aux" in out:
                    loss = loss + 0.4 * criterion(out["aux"].float(), masks)
                loss.backward()
                optimizer.step()
                running += loss.item() * images.size(0)
                seen += images.size(0)
            scheduler.step()

            if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == EPOCHS:
                d, _, i = evaluate(model, te_loader)
                print(f"  Epoch {epoch + 1:3d}/{EPOCHS} - Loss {running / seen:.4f} "
                      f"- LR {scheduler.get_last_lr()[0]:.6f} - DSC {d:.4f} - IoU {i:.4f}")
                if d > best_dice:
                    best_dice, best_epoch = d, epoch + 1
                    best_state = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}
            elif (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1:3d}/{EPOCHS} - Loss {running / seen:.4f}")

        final_d, final_sd, final_i = evaluate(model, te_loader)
        elapsed = time.time() - t_start
        print(f"\n  Fold {fold}: final DSC {final_d:.4f} (IoU {final_i:.4f}) / "
              f"best DSC {best_dice:.4f} @epoch {best_epoch} / {elapsed:.0f}s")

        torch.save(best_state, FOLD_MODEL_DIR / f"fold{fold}.pth")
        results.append({"fold": fold, "train_cases": train_cases,
                        "test_cases": test_cases, "final_dice": final_d,
                        "final_dice_std": final_sd, "final_iou": final_i,
                        "best_dice": best_dice, "best_epoch": best_epoch,
                        "epochs": EPOCHS, "elapsed_sec": elapsed,
                        "timestamp": datetime.now().isoformat()})
        RESULT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if len(results) > 1:
        print(f"\n  final DSC mean {np.mean([r['final_dice'] for r in results]):.4f} "
              f"± {np.std([r['final_dice'] for r in results]):.4f}")
        print(f"  best  DSC mean {np.mean([r['best_dice'] for r in results]):.4f} "
              f"± {np.std([r['best_dice'] for r in results]):.4f}")
    print("saved:", RESULT)


if __name__ == "__main__":
    main()
