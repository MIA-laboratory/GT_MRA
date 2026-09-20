"""Direct fp32 re-measurement of saved checkpoints on the holdout cases.

Every DSC tabulated in the paper is produced this way rather than copied from
a training log: the seed and every generation checkpoint of the requested runs
are loaded and evaluated on the fold's holdout cases (slice-mean DSC, IoU, and
per-case DSC).

Usage:
    python eval_retained_models.py <fold> [configuration ...]
      default configurations: every run directory found for that fold
Output:
    results/analysis/retained_model_eval_fold<k>.json
"""
import json
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_resnet50

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (  # noqa: E402
    ANALYSIS_DIR, EVOLUTION_DIR, FOLD_MODEL_DIR, RAW_JPEG_DIR, RAW_PNG_DIR,
    ensure_dirs,
)

FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 1
CONFIGS = sys.argv[2:]

NUM_CLASSES = 2
BATCH_SIZE = 8 * max(torch.cuda.device_count(), 1)


def fold_test_cases(fold):
    from sklearn.model_selection import KFold
    ids = np.array(sorted(int(d.name) for d in RAW_JPEG_DIR.iterdir() if d.is_dir()))
    splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(ids))
    return ids[splits[fold - 1][1]].tolist()


HOLDOUT_CASES = fold_test_cases(FOLD)


class ValidationDataset(Dataset):
    def __init__(self, cases):
        self.samples = []
        for c in cases:
            jd, pd_ = RAW_JPEG_DIR / str(c), RAW_PNG_DIR / str(c)
            for jp in sorted(jd.glob("*.JPG")):
                mp = pd_ / (jp.stem + ".png")
                if mp.exists():
                    self.samples.append((str(jp), str(mp)))
        self.tf = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        jp, mp = self.samples[i]
        img = np.array(Image.open(jp).convert("RGB"))
        msk = (np.array(Image.open(mp)) > 0).astype(np.uint8)
        a = self.tf(image=img, mask=msk)
        return a["image"], a["mask"].long()


def create_model():
    m = deeplabv3_resnet50(weights=None, aux_loss=True)
    m.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    m.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    return m


def evaluate(state_path, device, loader):
    """Slice-mean DSC / IoU over all holdout slices, in fp32."""
    m = create_model()
    m.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    m = m.to(device).eval()
    dices, ious = [], []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            preds = m(images)["out"].argmax(dim=1)
            for i in range(images.size(0)):
                p = (preds[i] == 1).float()
                t = (masks[i] == 1).float()
                inter = (p * t).sum()
                union = p.sum() + t.sum() - inter
                dices.append(((2 * inter + 1e-7) / (p.sum() + t.sum() + 1e-7)).item())
                ious.append(((inter + 1e-7) / (union + 1e-7)).item())
    del m
    torch.cuda.empty_cache()
    return float(np.mean(dices)), float(np.mean(ious)), len(dices)


def per_case(state_path, device):
    out = {}
    m = create_model()
    m.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    m = m.to(device).eval()
    for c in HOLDOUT_CASES:
        ds = ValidationDataset([c])
        ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        ds_list = []
        with torch.no_grad():
            for images, masks in ld:
                images, masks = images.to(device), masks.to(device)
                preds = m(images)["out"].argmax(dim=1)
                for i in range(images.size(0)):
                    p = (preds[i] == 1).float()
                    t = (masks[i] == 1).float()
                    inter = (p * t).sum()
                    ds_list.append(((2 * inter + 1e-7)
                                    / (p.sum() + t.sum() + 1e-7)).item())
        out[c] = {"dice": float(np.mean(ds_list)), "slices": len(ds_list)}
    del m
    torch.cuda.empty_cache()
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ValidationDataset(HOLDOUT_CASES)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"fold {FOLD}  holdout: {HOLDOUT_CASES} / {len(ds)} slices  device={device}")

    configs = CONFIGS or sorted(
        d.name[: -len(f"_fold{FOLD}")] for d in EVOLUTION_DIR.glob(f"*_fold{FOLD}")
        if d.is_dir())
    targets = [(f"seed (fold {FOLD})", FOLD_MODEL_DIR / f"fold{FOLD}.pth")]
    for name in configs:
        d = EVOLUTION_DIR / f"{name}_fold{FOLD}"
        if (d / "model_final.pth").exists():
            targets.append((f"{name} final retained", d / "model_final.pth"))
        for g in sorted(d.glob("model_gen*.pth")):
            targets.append((f"{name} {g.stem}", g))

    res = {"fold": FOLD, "holdout_cases": HOLDOUT_CASES,
           "holdout_slices": len(ds), "models": {}}
    for name, path in targets:
        if not Path(path).exists():
            print(f"  -- {name}: missing ({path})")
            continue
        d, i, n = evaluate(path, device, loader)
        pc = per_case(path, device)
        res["models"][name] = {"path": str(path), "dice": d, "iou": i,
                               "slices": n, "per_case": pc}
        print(f"  {name:36} DSC={d:.4f}  IoU={i:.4f}  "
              + "  ".join(f"C{c}={v['dice']:.4f}" for c, v in pc.items()))

    ensure_dirs(ANALYSIS_DIR)
    out = ANALYSIS_DIR / f"retained_model_eval_fold{FOLD}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("saved:", out)


if __name__ == "__main__":
    main()
