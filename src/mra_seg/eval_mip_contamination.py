"""Task-relevant MIP measure: contamination and loss (Sections 2.5 and 3.5).

For each holdout case the image stack is masked by the ground truth and by the
prediction, MIPs are formed in the axial, coronal, and sagittal directions,
and the two are compared pixelwise on the 0-1 intensity scale:

  contamination : mean of max(0, MIP_pred - MIP_gt)   excess brightness admitted
  loss          : mean of max(0, MIP_gt - MIP_pred)   brightness removed

bright_fp / bright_fn count false positives / negatives among pixels at or
above the 90th percentile of the intensities inside the case's ground-truth
mask (pixels bright enough to be vasculature).

Usage:
    python eval_mip_contamination.py <fold> <configuration>
      compares the fold's seed against <configuration>'s retained model
Output:
    results/analysis/mip_contamination_fold<k>_<configuration>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (  # noqa: E402
    ANALYSIS_DIR, EVOLUTION_DIR, FOLD_MODEL_DIR, RAW_JPEG_DIR, RAW_PNG_DIR,
    ensure_dirs,
)

FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 1
CONFIG = sys.argv[2] if len(sys.argv) > 2 else "strict-combined"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BATCH = 48
MEM_FMT = torch.channels_last
BRIGHT_PCTL = 90

MODELS = {
    "seed": FOLD_MODEL_DIR / f"fold{FOLD}.pth",
    CONFIG: EVOLUTION_DIR / f"{CONFIG}_fold{FOLD}" / "model_final.pth",
}
OUT_JSON = ANALYSIS_DIR / f"mip_contamination_fold{FOLD}_{CONFIG}.json"


def fold_test_cases(fold):
    from sklearn.model_selection import KFold
    ids = np.array(sorted(int(d.name) for d in RAW_JPEG_DIR.iterdir() if d.is_dir()))
    splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(ids))
    return ids[splits[fold - 1][1]].tolist()


HOLDOUT = fold_test_cases(FOLD)


def build(path):
    m = deeplabv3_resnet50(weights=None, aux_loss=True)
    m.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    m.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.cuda().to(memory_format=MEM_FMT).eval()


def load_case(case_id):
    jd, pd_ = RAW_JPEG_DIR / str(case_id), RAW_PNG_DIR / str(case_id)
    pairs = [(jp, pd_ / (jp.stem + ".png")) for jp in sorted(jd.glob("*.JPG"))
             if (pd_ / (jp.stem + ".png")).exists()]
    imgs = np.stack([np.array(Image.open(ip).convert("L"), dtype=np.float32) / 255.0
                     for ip, _ in pairs])
    gts = np.stack([(np.array(Image.open(mp)) > 0).astype(np.uint8)
                    for _, mp in pairs])
    rgb = np.stack([np.array(Image.open(ip).convert("RGB"), dtype=np.float32) / 255.0
                    for ip, _ in pairs])
    return imgs, gts, rgb


@torch.inference_mode()
def predict(model, rgb):
    out = []
    for i in range(0, len(rgb), BATCH):
        chunk = (rgb[i:i + BATCH] - MEAN) / STD
        x = torch.from_numpy(chunk.transpose(0, 3, 1, 2)).cuda().to(
            memory_format=MEM_FMT)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)["out"]
        out.append((logits.float().argmax(dim=1) == 1).cpu().numpy().astype(np.uint8))
    return np.concatenate(out)


def mips(vol):
    return {"axial": vol.max(axis=0), "coronal": vol.max(axis=1),
            "sagittal": vol.max(axis=2)}


def measure(img, gt, pred):
    m_gt = mips(img * gt)
    m_pr = mips(img * pred)
    res = {}
    for k in m_gt:
        diff = m_pr[k] - m_gt[k]
        res[f"contamination_{k}"] = float(np.clip(diff, 0, None).mean())
        res[f"loss_{k}"] = float(np.clip(-diff, 0, None).mean())
    thr = np.percentile(img[gt > 0], BRIGHT_PCTL) if (gt > 0).any() else 1.0
    bright = img >= thr
    res["bright_fp"] = int(((pred > gt) & bright).sum())
    res["bright_fn"] = int(((pred < gt) & bright).sum())
    res["bright_total"] = int((bright & (gt > 0)).sum())
    return res


def main():
    torch.backends.cudnn.benchmark = True
    results = {}
    for name, path in MODELS.items():
        if not path.exists():
            print(f"  -- {name}: missing ({path})")
            continue
        model = build(path)
        per_case = {}
        for c in HOLDOUT:
            img, gt, rgb = load_case(c)
            per_case[str(c)] = measure(img, gt, predict(model, rgb))
        del model
        torch.cuda.empty_cache()

        agg = {k: float(np.mean([v[k] for v in per_case.values()]))
               for k in next(iter(per_case.values()))}
        results[name] = {"per_case": per_case, "mean": agg}
        print(f"\n--- {name} ---")
        for d in ("axial", "coronal", "sagittal"):
            print(f"  {d:>9}: contamination {agg[f'contamination_{d}']:.5f}  "
                  f"loss {agg[f'loss_{d}']:.5f}")
        print(f"  bright FP {agg['bright_fp']:.0f} / FN {agg['bright_fn']:.0f} "
              f"(bright GT pixels {agg['bright_total']:.0f})")

    if len(results) == 2:
        a, b = list(results)
        print(f"\n=== {b} - {a} ===")
        for d in ("axial", "coronal", "sagittal"):
            dc = results[b]["mean"][f"contamination_{d}"] - results[a]["mean"][f"contamination_{d}"]
            dl = results[b]["mean"][f"loss_{d}"] - results[a]["mean"][f"loss_{d}"]
            print(f"  {d:>9}: contamination {dc:+.5f}  loss {dl:+.5f}")

    ensure_dirs(ANALYSIS_DIR)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nsaved:", OUT_JSON)


if __name__ == "__main__":
    main()
