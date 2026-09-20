"""What the confidence score of Eq. (6) ranks (Section 3.4 of the paper).

The seed model is run over a fixed diagnostic sample of the unlabeled stream
(the first N_CASES case directories of MRA_STUDY_DIR, every slice), recording
per slice the mean-max-softmax confidence and the predicted foreground
fraction. Reported: their correlation, and the share of each foreground
stratum falling above a top-40% confidence cut within the sample. The same
protocol is also applied to the fold's labeled holdout cases.

Usage:
    python diagnose_confidence.py [fold]
Output:
    results/analysis/confidence_diagnosis_fold<k>.json
"""
import json
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import pydicom
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_resnet50

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (  # noqa: E402
    ANALYSIS_DIR, FOLD_MODEL_DIR, MRA_STUDY_DIR, RAW_JPEG_DIR, RAW_PNG_DIR,
    ensure_dirs, require_external,
)

FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SEED_MODEL = FOLD_MODEL_DIR / f"fold{FOLD}.pth"
N_CASES = 8                        # head of the first generation's batch
IMG, BATCH = 512, 48
TOP_FRAC = 0.40
STRATA = [(0.05, 0.20), (0.20, 0.40), (0.40, 1.01)]
OUT = ANALYSIS_DIR / f"confidence_diagnosis_fold{FOLD}.json"

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
                    self.s.append((str(jp), str(pp)))

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        ip, mp = self.s[i]
        img = np.array(Image.open(ip).convert("RGB"))
        mask = (np.array(Image.open(mp)) > 0).astype(np.uint8)
        t = NORM(image=img, mask=mask)
        return t["image"], t["mask"].long()


class DicomSet(Dataset):
    """Unlabeled DICOM slices, preprocessed exactly as in the framework."""

    def __init__(self, case_dirs):
        self.s = []
        for d in case_dirs:
            self.s += sorted(d.glob("*.dcm"),
                             key=lambda x: x.stem.split("_")[-1].zfill(5))

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        ds = pydicom.dcmread(str(self.s[i]))
        px = ds.pixel_array.astype(np.float32)
        if px.max() > 0:
            px = (px - px.min()) / (px.max() - px.min()) * 255.0
        img = Image.fromarray(px.astype(np.uint8)).convert("RGB").resize(
            (IMG, IMG), Image.BILINEAR)
        return NORM(image=np.array(img))["image"], 0


def build():
    m = deeplabv3_resnet50(weights=None, aux_loss=True)
    m.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    m.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    m.load_state_dict(torch.load(SEED_MODEL, map_location="cpu", weights_only=True))
    return m.cuda().eval()


@torch.inference_mode()
def run(model, loader, has_gt):
    frac_pred, frac_gt, conf_all, conf_fg, empty = [], [], [], [], 0
    for batch in loader:
        x = batch[0].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)["out"]
        p = torch.softmax(logits.float(), dim=1)
        pred = p.argmax(dim=1)
        mx = p.max(dim=1)[0]
        fg = (pred == 1)
        n = pred.shape[-1] * pred.shape[-2]
        frac_pred += (fg.sum(dim=(1, 2)).float() / n).tolist()
        conf_all += mx.mean(dim=(1, 2)).tolist()
        for k in range(x.size(0)):
            if fg[k].any():
                conf_fg.append(float(mx[k][fg[k]].mean()))
            else:
                empty += 1
        if has_gt:
            gt = batch[1].cuda()
            frac_gt += ((gt == 1).sum(dim=(1, 2)).float() / n).tolist()
    return dict(frac_pred=frac_pred, frac_gt=frac_gt, conf_all=conf_all,
                conf_fg=conf_fg, empty=empty, n=len(frac_pred))


def main():
    torch.backends.cudnn.benchmark = True
    model = build()

    require_external(MRA_STUDY_DIR, "unlabeled MRA studies (MRA_STUDY_DIR)")
    cases = sorted(d for d in Path(MRA_STUDY_DIR).iterdir() if d.is_dir())
    used = cases[:N_CASES]
    print(f"diagnostic sample: {len(used)} cases")

    dl = DataLoader(DicomSet(used), batch_size=BATCH, num_workers=4)
    jmid = run(model, dl, has_gt=False)
    ll = DataLoader(LabeledSet(HOLDOUT), batch_size=BATCH, num_workers=4)
    lab = run(model, ll, has_gt=True)

    c = np.array(jmid["conf_all"])
    f = np.array(jmid["frac_pred"])
    thr = float(np.percentile(c, 100 * (1 - TOP_FRAC)))
    strata = []
    for lo, hi in STRATA:
        m = (f >= lo) & (f < hi)
        strata.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                       "selected_pct": float(100 * (c[m] >= thr).mean())
                       if m.any() else 0.0})
    r = float(np.corrcoef(c, f)[0, 1])

    res = {"seed_model": SEED_MODEL.name, "fold": FOLD,
           "n_unlabeled_slices": jmid["n"], "top_frac": TOP_FRAC,
           "top_threshold": thr, "r_conf_vs_foreground": r,
           "strata": strata, "unlabeled": jmid, "labeled_holdout": lab}
    ensure_dirs(ANALYSIS_DIR)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")

    print(f"  unlabeled slices {jmid['n']}  r = {r:.3f}  "
          f"top-40% threshold {thr:.4f}")
    for s in strata:
        print(f"    foreground {s['lo']:.2f}-{s['hi']:.2f}: n={s['n']:4d}  "
              f"selected {s['selected_pct']:.1f}%")
    print(f"  holdout confidence all/foreground: "
          f"{np.mean(lab['conf_all']):.4f} / {np.mean(lab['conf_fg']):.4f}")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
