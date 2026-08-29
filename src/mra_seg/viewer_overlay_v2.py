"""
Segmentation Overlay Viewer v2
================================
Three-panel comparison on Fold 1 test data:
  Left:   Initial model (Gen 0)
  Center: Round 1 Gen 10 (Naive self-training)
  Right:  Round 2 Final (Improved evolutionary)
Ground truth contour in green.
Mouse wheel to scroll (synchronized).
"""

import sys
import tkinter as tk
from tkinter import ttk
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image, ImageTk
from scipy import ndimage
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# All paths live in paths.py, resolved relative to the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (  # noqa: E402
    RAW_JPEG_DIR, RAW_PNG_DIR, INITIAL_MODEL, EVOLUTION_DIR, EVOLUTION_V2_DIR,
)

JPEG_DIR = RAW_JPEG_DIR
PNG_DIR = RAW_PNG_DIR

# Intermediate checkpoints may be absent. Compare whatever is present.
_MODEL_CANDIDATES = {
    "Initial (Gen 0)": INITIAL_MODEL,
    "Round 1 Gen10": EVOLUTION_DIR / "model_gen010.pth",
    "Round 2 Final": EVOLUTION_V2_DIR / "model_final_v2.pth",
}
MODELS = {name: p for name, p in _MODEL_CANDIDATES.items() if p.exists()}
for _name, _p in _MODEL_CANDIDATES.items():
    if _name not in MODELS:
        print(f"[skip] {_name}: model not found, excluded from the comparison -> {_p}")
COLORS = {
    "Initial (Gen 0)": "#4fc3f7",
    "Round 1 Gen10": "#ff8a65",
    "Round 2 Final": "#81c784",
}

# Case IDs to display. Set these for your own dataset.
TEST_CASES: list[int] = []
IMG_SIZE = 512
NUM_CLASSES = 2
PANEL_SIZE = 380


def create_model():
    model = deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    return model


def load_model(path, device):
    model = create_model()
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def load_test_data():
    samples = []
    for cid in TEST_CASES:
        jpeg_case = JPEG_DIR / str(cid)
        png_case = PNG_DIR / str(cid)
        for jp in sorted(jpeg_case.glob("*.JPG")):
            pp = png_case / (jp.stem + ".png")
            if pp.exists():
                img = np.array(Image.open(jp).convert("RGB"))
                mask = np.array(Image.open(pp))
                mask = (mask > 0).astype(np.uint8)
                samples.append((cid, jp.stem, img, mask))
    return samples


def run_inference(model, images_np, device):
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    predictions = []
    for i in range(0, len(images_np), 16):
        batch = [transform(image=img)["image"] for img in images_np[i:i+16]]
        batch = torch.stack(batch).to(device)
        with torch.no_grad():
            preds = model(batch)["out"].argmax(dim=1).cpu().numpy()
        predictions.extend([p.astype(np.uint8) for p in preds])
    return predictions


def create_overlay(image_np, pred_mask, gt_mask, alpha=0.4):
    overlay = image_np.copy()
    vessel = pred_mask == 1
    overlay[vessel, 0] = np.clip(overlay[vessel, 0].astype(np.int16) + 100, 0, 255).astype(np.uint8)
    overlay[vessel, 1] = (overlay[vessel, 1] * (1 - alpha)).astype(np.uint8)
    overlay[vessel, 2] = (overlay[vessel, 2] * (1 - alpha)).astype(np.uint8)
    if gt_mask.sum() > 0:
        dilated = ndimage.binary_dilation(gt_mask, iterations=1)
        eroded = ndimage.binary_erosion(gt_mask, iterations=1)
        contour = (dilated.astype(np.uint8) - eroded.astype(np.uint8)) > 0
        overlay[contour, 0] = 0
        overlay[contour, 1] = 255
        overlay[contour, 2] = 0
    return overlay


def compute_dice(pred, gt):
    pf = (pred == 1).astype(np.float32)
    gf = (gt == 1).astype(np.float32)
    inter = (pf * gf).sum()
    return (2 * inter + 1e-7) / (pf.sum() + gf.sum() + 1e-7)


class TripleViewer:
    def __init__(self, root, panels_data, labels, dice_data, model_names):
        self.root = root
        self.root.title("MRA Segmentation: Initial vs Round1 vs Round2")
        self.root.configure(bg="#1e1e1e")

        self.panels_data = panels_data  # {name: [overlay_list]}
        self.labels = labels
        self.dice_data = dice_data  # {name: [dice_list]}
        self.model_names = model_names
        self.current_idx = 0
        self.total = len(labels)

        # Header
        header = tk.Frame(root, bg="#1e1e1e")
        header.pack(fill=tk.X, padx=10, pady=5)
        for name in model_names:
            tk.Label(header, text=name, font=("Consolas", 12, "bold"),
                     fg=COLORS[name], bg="#1e1e1e").pack(side=tk.LEFT, expand=True)

        # Canvases
        canvas_frame = tk.Frame(root, bg="#1e1e1e")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10)

        self.canvases = []
        for i, name in enumerate(model_names):
            c = tk.Canvas(canvas_frame, width=PANEL_SIZE, height=PANEL_SIZE,
                          bg="#000000", highlightthickness=0)
            px = (0, 3) if i == 0 else ((3, 0) if i == len(model_names)-1 else (3, 3))
            c.pack(side=tk.LEFT, padx=px)
            self.canvases.append(c)

        self.tk_imgs = [None] * len(model_names)

        # Info
        info_frame = tk.Frame(root, bg="#1e1e1e")
        info_frame.pack(fill=tk.X, padx=10, pady=3)

        self.info_label = tk.Label(info_frame, text="", font=("Consolas", 11),
                                   fg="#e0e0e0", bg="#1e1e1e")
        self.info_label.pack()

        self.dice_label = tk.Label(info_frame, text="", font=("Consolas", 10),
                                   fg="#aaaaaa", bg="#1e1e1e")
        self.dice_label.pack()

        # Slider
        slider_frame = tk.Frame(root, bg="#1e1e1e")
        slider_frame.pack(fill=tk.X, padx=10, pady=3)
        self.slider = ttk.Scale(slider_frame, from_=0, to=self.total - 1,
                                orient=tk.HORIZONTAL, command=self._on_slider)
        self.slider.pack(fill=tk.X)

        # Legend
        legend_frame = tk.Frame(root, bg="#1e1e1e")
        legend_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        tk.Label(legend_frame, text="Red: Prediction    Green contour: Ground truth    Scroll: navigate",
                 font=("Consolas", 9), fg="#666666", bg="#1e1e1e").pack()

        # Bindings
        root.bind("<MouseWheel>", self._on_mousewheel)
        root.bind("<Up>", lambda e: self._navigate(-1))
        root.bind("<Down>", lambda e: self._navigate(1))
        root.bind("<Left>", lambda e: self._navigate(-1))
        root.bind("<Right>", lambda e: self._navigate(1))
        root.bind("<Home>", lambda e: self._goto(0))
        root.bind("<End>", lambda e: self._goto(self.total - 1))

        self._update_display()

    def _on_mousewheel(self, event):
        self._navigate(-1 if event.delta > 0 else 1)

    def _navigate(self, delta):
        new = max(0, min(self.total - 1, self.current_idx + delta))
        if new != self.current_idx:
            self.current_idx = new
            self.slider.set(new)
            self._update_display()

    def _goto(self, idx):
        self.current_idx = idx
        self.slider.set(idx)
        self._update_display()

    def _on_slider(self, val):
        idx = int(float(val))
        if idx != self.current_idx:
            self.current_idx = idx
            self._update_display()

    def _update_display(self):
        idx = self.current_idx
        for i, name in enumerate(self.model_names):
            img = Image.fromarray(self.panels_data[name][idx]).resize(
                (PANEL_SIZE, PANEL_SIZE), Image.BILINEAR)
            self.tk_imgs[i] = ImageTk.PhotoImage(img)
            self.canvases[i].delete("all")
            self.canvases[i].create_image(0, 0, anchor=tk.NW, image=self.tk_imgs[i])

        self.info_label.config(text=f"Slice {idx+1}/{self.total}  |  {self.labels[idx]}")

        dice_parts = []
        for name in self.model_names:
            d = self.dice_data[name][idx]
            short = name.split("(")[0].strip() if "(" in name else name
            dice_parts.append(f"{short}: {d:.4f}")
        # Difference between the last loaded model and the initial model
        if len(self.model_names) >= 2:
            d_init = self.dice_data[self.model_names[0]][idx]
            d_last = self.dice_data[self.model_names[-1]][idx]
            dice_parts.append(f"Diff(Last-Init): {d_last - d_init:+.4f}")
        self.dice_label.config(text="  |  ".join(dice_parts))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load models
    model_names = list(MODELS.keys())
    models = {}
    for name, path in MODELS.items():
        print(f"Loading {name}: {path.name}")
        models[name] = load_model(path, device)

    # Load data
    print("Loading test data (Fold 1)...")
    samples = load_test_data()
    print(f"  {len(samples)} slices from cases {TEST_CASES}")
    images_np = [s[2] for s in samples]
    gt_masks = [s[3] for s in samples]

    # Inference
    all_preds = {}
    for name, model in models.items():
        print(f"Running inference: {name}...")
        all_preds[name] = run_inference(model, images_np, device)

    # Free GPU
    del models
    torch.cuda.empty_cache()

    # Generate overlays
    print("Generating overlays...")
    panels_data = {name: [] for name in model_names}
    dice_data = {name: [] for name in model_names}
    labels = []

    for i, (cid, sname, img, gt) in enumerate(samples):
        labels.append(f"Case {cid} / {sname}")
        for name in model_names:
            panels_data[name].append(create_overlay(img, all_preds[name][i], gt))
            dice_data[name].append(compute_dice(all_preds[name][i], gt))

    # Summary
    for name in model_names:
        m = np.mean(dice_data[name])
        print(f"  {name}: Mean DSC = {m:.4f}")

    print(f"\nLaunching viewer ({len(samples)} slices, {len(model_names)} panels)...")

    root = tk.Tk()
    w = PANEL_SIZE * len(model_names) + 30
    h = PANEL_SIZE + 160
    root.geometry(f"{w}x{h}")
    root.resizable(True, True)
    TripleViewer(root, panels_data, labels, dice_data, model_names)
    root.mainloop()


if __name__ == "__main__":
    main()
