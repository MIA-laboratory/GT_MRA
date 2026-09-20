"""How much the DSC of identical predictions depends on the aggregation.

For the five seed checkpoints, three conventions are computed from one
inference pass per fold (Sections 3.1 and 4.3 of the paper):

  slice_mean : mean of the per-slice DSC over all slices
  case_mean  : one DSC per case from the pooled voxels, averaged over cases
  global     : all voxels of all cases pooled into a single DSC

Output: results/analysis/dsc_aggregation.json
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
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_resnet50

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (  # noqa: E402
    ANALYSIS_DIR, FOLD_MODEL_DIR, RAW_JPEG_DIR, RAW_PNG_DIR, ensure_dirs,
)

OUT = ANALYSIS_DIR / "dsc_aggregation.json"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
NORM = A.Compose([A.Normalize(mean=MEAN, std=STD), ToTensorV2()])
MEM_FMT = torch.channels_last
BATCH = 48
SEED = 42


class CaseSet(Dataset):
    def __init__(self, case_id):
        jd, pd_ = RAW_JPEG_DIR / str(case_id), RAW_PNG_DIR / str(case_id)
        self.s = [(str(jp), str(pd_ / (jp.stem + ".png")))
                  for jp in sorted(jd.glob("*.JPG"))
                  if (pd_ / (jp.stem + ".png")).exists()]

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        ip, mp = self.s[i]
        img = np.array(Image.open(ip).convert("RGB"))
        mask = (np.array(Image.open(mp)) > 0).astype(np.uint8)
        t = NORM(image=img, mask=mask)
        return t["image"], t["mask"].long()


def build(path):
    m = deeplabv3_resnet50(weights=None, aux_loss=True)
    m.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    m.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.cuda().to(memory_format=MEM_FMT).eval()


@torch.inference_mode()
def case_stats(model, case_id):
    ds = CaseSet(case_id)
    ld = DataLoader(ds, batch_size=BATCH, num_workers=2, pin_memory=True)
    inter = pred_n = gt_n = 0.0
    slice_dice = []
    for images, masks in ld:
        images = images.cuda(non_blocking=True).to(memory_format=MEM_FMT)
        masks = masks.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images)["out"]
        p = (logits.float().argmax(dim=1) == 1).float()
        t = (masks == 1).float()
        i_ = (p * t).sum(dim=(1, 2))
        pn, gn = p.sum(dim=(1, 2)), t.sum(dim=(1, 2))
        slice_dice += ((2 * i_ + 1e-7) / (pn + gn + 1e-7)).tolist()
        inter += float(i_.sum())
        pred_n += float(pn.sum())
        gt_n += float(gn.sum())
    return inter, pred_n, gt_n, slice_dice


def main():
    case_ids = np.array(sorted(int(d.name) for d in RAW_JPEG_DIR.iterdir()
                               if d.is_dir()))
    splits = list(KFold(n_splits=5, shuffle=True, random_state=SEED)
                  .split(case_ids))

    out = {}
    all_slice, all_case, gi = [], [], [0.0, 0.0, 0.0]
    for fold in range(1, 6):
        w = FOLD_MODEL_DIR / f"fold{fold}.pth"
        if not w.exists():
            print(f"  fold {fold}: no checkpoint")
            continue
        model = build(w)
        test_cases = case_ids[splits[fold - 1][1]].tolist()
        f_slice, f_case = [], []
        for c in test_cases:
            inter, pn, gn, sd = case_stats(model, c)
            f_slice += sd
            f_case.append((2 * inter + 1e-7) / (pn + gn + 1e-7))
            gi[0] += inter
            gi[1] += pn
            gi[2] += gn
        del model
        torch.cuda.empty_cache()
        out[f"fold{fold}"] = {"test_cases": test_cases,
                              "slice_mean": float(np.mean(f_slice)),
                              "case_mean": float(np.mean(f_case)),
                              "n_slices": len(f_slice)}
        all_slice += f_slice
        all_case += f_case
        print(f"  fold {fold}: slice_mean {np.mean(f_slice):.4f}  "
              f"case_mean {np.mean(f_case):.4f}  ({len(f_slice)} slices)")

    if all_slice:
        out["overall"] = {"slice_mean": float(np.mean(all_slice)),
                          "case_mean": float(np.mean(all_case)),
                          "global": float((2 * gi[0] + 1e-7) / (gi[1] + gi[2] + 1e-7))}
        print(f"\n  overall  slice_mean {out['overall']['slice_mean']:.4f}"
              f"  case_mean {out['overall']['case_mean']:.4f}"
              f"  global {out['overall']['global']:.4f}")
    ensure_dirs(ANALYSIS_DIR)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
