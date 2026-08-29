"""
Segmentation Overlay Viewer
============================
Side-by-side comparison of two model predictions on Fold 1 test data.
Left: Initial model (Gen 0 / fold5)
Right: Evolutionary model (Gen 10)
Ground truth contour shown in green on both panels.
Mouse wheel to scroll through slices (synchronized).
"""

import sys
import tkinter as tk
from tkinter import ttk
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image, ImageTk, ImageDraw
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# ============================================================
# Paths
# ============================================================
# All paths live in paths.py, resolved relative to the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import RAW_JPEG_DIR, RAW_PNG_DIR, INITIAL_MODEL, EVOLUTION_DIR  # noqa: E402

JPEG_DIR = RAW_JPEG_DIR
PNG_DIR = RAW_PNG_DIR
MODEL_INITIAL = INITIAL_MODEL
MODEL_GEN10 = EVOLUTION_DIR / "model_gen010.pth"   # intermediate checkpoint

# Fold 1 test cases
# Case IDs to display. Set these for your own dataset.
TEST_CASES: list[int] = []
IMG_SIZE = 512
NUM_CLASSES = 2
DISPLAY_SIZE = 480  # display size per panel


# ============================================================
# Model
# ============================================================
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


# ============================================================
# Data loading
# ============================================================
def load_test_data():
    """Load all test slices: returns list of (case_id, slice_name, image_np, gt_mask_np)"""
    samples = []
    for cid in TEST_CASES:
        jpeg_case = JPEG_DIR / str(cid)
        png_case = PNG_DIR / str(cid)
        jpegs = sorted(jpeg_case.glob("*.JPG"))
        for jp in jpegs:
            pp = png_case / (jp.stem + ".png")
            if pp.exists():
                img = np.array(Image.open(jp).convert("RGB"))
                mask = np.array(Image.open(pp))
                mask = (mask > 0).astype(np.uint8)
                samples.append((cid, jp.stem, img, mask))
    return samples


# ============================================================
# Inference
# ============================================================
def run_inference(model, images_np, device):
    """Run batch inference, return list of predicted masks."""
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    predictions = []
    batch_size = 16
    for i in range(0, len(images_np), batch_size):
        batch_imgs = images_np[i:i + batch_size]
        tensors = []
        for img in batch_imgs:
            t = transform(image=img)["image"]
            tensors.append(t)
        batch = torch.stack(tensors).to(device)

        with torch.no_grad():
            out = model(batch)["out"]
            preds = out.argmax(dim=1).cpu().numpy()

        for p in preds:
            predictions.append(p.astype(np.uint8))

    return predictions


# ============================================================
# Overlay rendering
# ============================================================
def create_overlay(image_np, pred_mask, gt_mask, alpha=0.4):
    """
    Create overlay image:
    - Red semi-transparent: predicted vessel region
    - Green contour: ground truth boundary
    """
    overlay = image_np.copy()

    # Red overlay for prediction
    vessel_region = pred_mask == 1
    overlay[vessel_region, 0] = np.clip(
        overlay[vessel_region, 0].astype(np.int16) + 100, 0, 255).astype(np.uint8)
    overlay[vessel_region, 1] = (overlay[vessel_region, 1] * (1 - alpha)).astype(np.uint8)
    overlay[vessel_region, 2] = (overlay[vessel_region, 2] * (1 - alpha)).astype(np.uint8)

    # Green contour for ground truth
    from scipy import ndimage
    if gt_mask.sum() > 0:
        dilated = ndimage.binary_dilation(gt_mask, iterations=1)
        eroded = ndimage.binary_erosion(gt_mask, iterations=1)
        contour = dilated.astype(np.uint8) - eroded.astype(np.uint8)
        contour = contour > 0
        overlay[contour, 0] = 0
        overlay[contour, 1] = 255
        overlay[contour, 2] = 0

    return overlay


# ============================================================
# Metrics
# ============================================================
def compute_dice(pred, gt):
    pred_f = (pred == 1).astype(np.float32)
    gt_f = (gt == 1).astype(np.float32)
    inter = (pred_f * gt_f).sum()
    return (2 * inter + 1e-7) / (pred_f.sum() + gt_f.sum() + 1e-7)


# ============================================================
# GUI Viewer
# ============================================================
class OverlayViewer:
    def __init__(self, root, overlays_left, overlays_right, labels, dice_left, dice_right):
        self.root = root
        self.root.title("MRA Segmentation Overlay Viewer")
        self.root.configure(bg="#1e1e1e")

        self.overlays_left = overlays_left
        self.overlays_right = overlays_right
        self.labels = labels
        self.dice_left = dice_left
        self.dice_right = dice_right
        self.current_idx = 0
        self.total = len(overlays_left)

        # --- Header ---
        header = tk.Frame(root, bg="#1e1e1e")
        header.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(header, text="Initial Model (Gen 0)",
                 font=("Consolas", 14, "bold"), fg="#4fc3f7", bg="#1e1e1e").pack(side=tk.LEFT, expand=True)
        tk.Label(header, text="Evolutionary Model (Gen 10)",
                 font=("Consolas", 14, "bold"), fg="#ff8a65", bg="#1e1e1e").pack(side=tk.RIGHT, expand=True)

        # --- Image panels ---
        img_frame = tk.Frame(root, bg="#1e1e1e")
        img_frame.pack(fill=tk.BOTH, expand=True, padx=10)

        self.canvas_left = tk.Canvas(img_frame, width=DISPLAY_SIZE, height=DISPLAY_SIZE,
                                     bg="#000000", highlightthickness=0)
        self.canvas_left.pack(side=tk.LEFT, padx=(0, 5))

        self.canvas_right = tk.Canvas(img_frame, width=DISPLAY_SIZE, height=DISPLAY_SIZE,
                                      bg="#000000", highlightthickness=0)
        self.canvas_right.pack(side=tk.RIGHT, padx=(5, 0))

        # --- Info bar ---
        info_frame = tk.Frame(root, bg="#1e1e1e")
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        self.info_label = tk.Label(info_frame, text="", font=("Consolas", 12),
                                   fg="#e0e0e0", bg="#1e1e1e")
        self.info_label.pack()

        self.dice_label = tk.Label(info_frame, text="", font=("Consolas", 11),
                                   fg="#aaaaaa", bg="#1e1e1e")
        self.dice_label.pack()

        # --- Slider ---
        slider_frame = tk.Frame(root, bg="#1e1e1e")
        slider_frame.pack(fill=tk.X, padx=10, pady=5)

        self.slider = ttk.Scale(slider_frame, from_=0, to=self.total - 1,
                                orient=tk.HORIZONTAL, command=self._on_slider)
        self.slider.pack(fill=tk.X)

        # --- Legend ---
        legend_frame = tk.Frame(root, bg="#1e1e1e")
        legend_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Label(legend_frame, text="Red: Model prediction    Green contour: Ground truth",
                 font=("Consolas", 10), fg="#888888", bg="#1e1e1e").pack()

        # --- Bindings ---
        root.bind("<MouseWheel>", self._on_mousewheel)
        root.bind("<Up>", lambda e: self._navigate(-1))
        root.bind("<Down>", lambda e: self._navigate(1))
        root.bind("<Left>", lambda e: self._navigate(-1))
        root.bind("<Right>", lambda e: self._navigate(1))
        root.bind("<Home>", lambda e: self._goto(0))
        root.bind("<End>", lambda e: self._goto(self.total - 1))

        self._update_display()

    def _on_mousewheel(self, event):
        delta = -1 if event.delta > 0 else 1
        self._navigate(delta)

    def _navigate(self, delta):
        new_idx = max(0, min(self.total - 1, self.current_idx + delta))
        if new_idx != self.current_idx:
            self.current_idx = new_idx
            self.slider.set(new_idx)
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

        # Left panel
        img_l = Image.fromarray(self.overlays_left[idx]).resize(
            (DISPLAY_SIZE, DISPLAY_SIZE), Image.BILINEAR)
        self.tk_img_left = ImageTk.PhotoImage(img_l)
        self.canvas_left.delete("all")
        self.canvas_left.create_image(0, 0, anchor=tk.NW, image=self.tk_img_left)

        # Right panel
        img_r = Image.fromarray(self.overlays_right[idx]).resize(
            (DISPLAY_SIZE, DISPLAY_SIZE), Image.BILINEAR)
        self.tk_img_right = ImageTk.PhotoImage(img_r)
        self.canvas_right.delete("all")
        self.canvas_right.create_image(0, 0, anchor=tk.NW, image=self.tk_img_right)

        # Info
        label = self.labels[idx]
        self.info_label.config(
            text=f"Slice {idx + 1}/{self.total}  |  {label}")
        self.dice_label.config(
            text=f"DSC  Initial: {self.dice_left[idx]:.4f}    Gen10: {self.dice_right[idx]:.4f}    "
                 f"Diff: {self.dice_right[idx] - self.dice_left[idx]:+.4f}")


# ============================================================
# Main
# ============================================================
def main():
    print("Loading models...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Point at the v2 viewer when this checkpoint is unavailable.
    if not MODEL_GEN10.exists():
        raise SystemExit(
            f"[missing model] {MODEL_GEN10} was not found.\n"
            "  Intermediate checkpoints of the naive strategy are not distributed.\n"
            "  Use viewer_overlay_v2.py to compare the initial and final models."
        )

    model_initial = load_model(MODEL_INITIAL, device)
    model_gen10 = load_model(MODEL_GEN10, device)
    print(f"  Initial model: {MODEL_INITIAL.name}")
    print(f"  Gen10 model:   {MODEL_GEN10.name}")

    print("Loading test data (Fold 1 test cases)...")
    samples = load_test_data()
    print(f"  {len(samples)} slices from cases {TEST_CASES}")

    images_np = [s[2] for s in samples]
    gt_masks = [s[3] for s in samples]

    print("Running inference with initial model...")
    preds_initial = run_inference(model_initial, images_np, device)

    print("Running inference with Gen 10 model...")
    preds_gen10 = run_inference(model_gen10, images_np, device)

    print("Generating overlays...")
    import scipy  # verify scipy available

    overlays_left = []
    overlays_right = []
    labels = []
    dice_left = []
    dice_right = []

    for i, (cid, sname, img, gt) in enumerate(samples):
        ol = create_overlay(img, preds_initial[i], gt)
        orr = create_overlay(img, preds_gen10[i], gt)
        overlays_left.append(ol)
        overlays_right.append(orr)
        labels.append(f"Case {cid} / {sname}")
        dice_left.append(compute_dice(preds_initial[i], gt))
        dice_right.append(compute_dice(preds_gen10[i], gt))

    # Free GPU memory
    del model_initial, model_gen10
    torch.cuda.empty_cache()

    # Summary
    mean_dl = np.mean(dice_left)
    mean_dr = np.mean(dice_right)
    print(f"\nMean DSC  Initial: {mean_dl:.4f}  Gen10: {mean_dr:.4f}  Diff: {mean_dr - mean_dl:+.4f}")
    print(f"\nLaunching viewer ({len(samples)} slices)...")

    # Launch GUI
    root = tk.Tk()
    root.geometry(f"{DISPLAY_SIZE * 2 + 30}x{DISPLAY_SIZE + 150}")
    root.resizable(True, True)
    viewer = OverlayViewer(root, overlays_left, overlays_right, labels, dice_left, dice_right)
    root.mainloop()


if __name__ == "__main__":
    main()
