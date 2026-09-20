"""Relation between the confidence score and per-slice label accuracy
(Section 3.4 of the paper).

On the fold's labeled holdout cases, where ground truth exists, the seed's
prediction on each slice is treated as the label it would generate and its
per-slice DSC against ground truth as that label's accuracy. Reported:
Pearson r and Spearman rho between confidence and per-slice DSC, and the
partial correlation controlling for the predicted foreground fraction.

Usage:
    python conf_vs_accuracy.py [fold]
Output:
    results/analysis/conf_vs_accuracy_fold<k>.json
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
    ANALYSIS_DIR, FOLD_MODEL_DIR, RAW_JPEG_DIR, RAW_PNG_DIR, ensure_dirs,
)

FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SEED_MODEL = FOLD_MODEL_DIR / f"fold{FOLD}.pth"
BATCH = 48
OUT = ANALYSIS_DIR / f"conf_vs_accuracy_fold{FOLD}.json"

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
NORM = A.Compose([A.Normalize(mean=MEAN, std=STD), ToTensorV2()])


def fold_test_cases(fold):
    from sklearn.model_selection import KFold
    ids = np.array(sorted(int(d.name) for d in RAW_JPEG_DIR.iterdir() if d.is_dir()))
    splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(ids))
    return ids[splits[fold - 1][1]].tolist()


HOLDOUT = fold_test_cases(FOLD)


class LabeledSet(Dataset):
    def __init__(self, cases):
        self.s = []
        for c in cases:
            jd, pd_ = RAW_JPEG_DIR / str(c), RAW_PNG_DIR / str(c)
            for jp in sorted(jd.glob("*.JPG")):
                pp = pd_ / (jp.stem + ".png")
                if pp.exists():
                    self.s.append((str(jp), str(pp), str(c)))

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        ip, mp, case = self.s[i]
        img = np.array(Image.open(ip).convert("RGB"))
        mask = (np.array(Image.open(mp)) > 0).astype(np.uint8)
        t = NORM(image=img, mask=mask)
        return t["image"], t["mask"].long(), case


def build():
    m = deeplabv3_resnet50(weights=None, aux_loss=True)
    m.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    m.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    m.load_state_dict(torch.load(SEED_MODEL, map_location="cpu", weights_only=True))
    return m.cuda().eval()


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return pearson(ra, rb)


def partial_r(a, b, ctrl):
    rab, rac, rbc = pearson(a, b), pearson(a, ctrl), pearson(b, ctrl)
    return float((rab - rac * rbc) / np.sqrt((1 - rac**2) * (1 - rbc**2)))


@torch.inference_mode()
def main():
    torch.backends.cudnn.benchmark = True
    model = build()
    dl = DataLoader(LabeledSet(HOLDOUT), batch_size=BATCH, num_workers=4)

    conf, dsc, frac_pred, frac_gt, cases = [], [], [], [], []
    for x, gt, case in dl:
        x = x.cuda(non_blocking=True)
        gt = gt.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)["out"]
        p = torch.softmax(logits.float(), dim=1)
        pred = p.argmax(dim=1)
        mx = p.max(dim=1)[0]
        n = pred.shape[-1] * pred.shape[-2]
        conf += mx.mean(dim=(1, 2)).tolist()
        fg, gt1 = (pred == 1), (gt == 1)
        frac_pred += (fg.sum(dim=(1, 2)).float() / n).tolist()
        frac_gt += (gt1.sum(dim=(1, 2)).float() / n).tolist()
        inter = (fg & gt1).sum(dim=(1, 2)).float()
        denom = fg.sum(dim=(1, 2)).float() + gt1.sum(dim=(1, 2)).float()
        d = torch.where(denom > 0, 2 * inter / denom, torch.ones_like(denom))
        dsc += d.tolist()
        cases += list(case)

    conf, dsc = np.array(conf), np.array(dsc)
    fp_, fg_ = np.array(frac_pred), np.array(frac_gt)

    res = {
        "seed_model": SEED_MODEL.name, "fold": FOLD,
        "holdout_cases": HOLDOUT, "n_slices": int(len(conf)),
        "slice_mean_dsc": float(dsc.mean()),
        "r_conf_dsc": pearson(conf, dsc),
        "rho_conf_dsc": spearman(conf, dsc),
        "r_conf_frac_pred": pearson(conf, fp_),
        "r_dsc_frac_gt": pearson(dsc, fg_),
        "partial_r_conf_dsc_given_frac_pred": partial_r(conf, dsc, fp_),
        "per_slice": {"conf": conf.tolist(), "dsc": dsc.tolist(),
                      "frac_pred": fp_.tolist(), "frac_gt": fg_.tolist(),
                      "case": cases},
    }
    ensure_dirs(ANALYSIS_DIR)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")

    print(f"slices={res['n_slices']}  slice-mean DSC={res['slice_mean_dsc']:.4f}")
    print(f"r(conf, DSC)          = {res['r_conf_dsc']:+.3f}")
    print(f"rho(conf, DSC)        = {res['rho_conf_dsc']:+.3f}")
    print(f"r(conf, frac_pred)    = {res['r_conf_frac_pred']:+.3f}")
    print(f"r(DSC, frac_gt)       = {res['r_dsc_frac_gt']:+.3f}")
    print(f"partial r(conf,DSC|frac_pred) = "
          f"{res['partial_r_conf_dsc_given_frac_pred']:+.3f}")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
