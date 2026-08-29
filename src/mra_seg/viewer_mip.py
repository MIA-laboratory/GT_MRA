"""
MIP (Maximum Intensity Projection) Viewer v3
==============================================
Features:
  - Axial / Coronal / Sagittal MIP views
  - Correct aspect ratio using DICOM spacing (PixelSpacing + SliceSpacing)
  - WW/WL (Window Width / Window Level) sliders
  - Side-by-side comparison of the original volume, the ground truth and each model
"""

import sys
import math
import tkinter as tk
from tkinter import ttk
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image, ImageTk
import pydicom
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# All paths live in paths.py, resolved relative to the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (  # noqa: E402
    RAW_JPEG_DIR, RAW_PNG_DIR, MRA_DICOM_DIR,
    INITIAL_MODEL, EVOLUTION_DIR, EVOLUTION_V2_DIR,
)

JPEG_DIR = RAW_JPEG_DIR
PNG_DIR = RAW_PNG_DIR
DICOM_DIR = MRA_DICOM_DIR

# Intermediate checkpoints may be absent. Load whatever is present and
# drop the panel for anything that is not.
_MODEL_CANDIDATES = {
    "Initial": INITIAL_MODEL,
    "Round1 Gen10": EVOLUTION_DIR / "model_gen010.pth",
    "Round2 Final": EVOLUTION_V2_DIR / "model_final_v2.pth",
}
MODELS = {name: p for name, p in _MODEL_CANDIDATES.items() if p.exists()}
for _name, _p in _MODEL_CANDIDATES.items():
    if _name not in MODELS:
        print(f"[skip] {_name}: model not found, panel omitted -> {_p}")

# Case IDs to display. Set these for your own dataset.
TEST_CASES: list[int] = []
IMG_SIZE = 512
NUM_CLASSES = 2
PANEL_SIZE = 300


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


def get_spacing_from_dicom(case_id):
    """
    Read PixelSpacing and actual slice spacing from DICOM headers.
    Returns (pixel_spacing_xy, slice_spacing_z) in mm.
    """
    dcm_case = DICOM_DIR / str(case_id)
    dcm_files = sorted(dcm_case.glob("*.dcm"))

    ds1 = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
    pixel_spacing = float(ds1.PixelSpacing[0])

    # Compute actual slice spacing from ImagePositionPatient
    if len(dcm_files) >= 2:
        ds2 = pydicom.dcmread(str(dcm_files[1]), stop_before_pixels=True)
        pos1 = [float(x) for x in ds1.ImagePositionPatient]
        pos2 = [float(x) for x in ds2.ImagePositionPatient]
        slice_spacing = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
    else:
        slice_spacing = float(getattr(ds1, 'SpacingBetweenSlices',
                                       getattr(ds1, 'SliceThickness', 1.0)))

    return pixel_spacing, slice_spacing


def load_case_data(case_id):
    jpeg_case = JPEG_DIR / str(case_id)
    png_case = PNG_DIR / str(case_id)
    images = []
    masks = []
    for jp in sorted(jpeg_case.glob("*.JPG")):
        pp = png_case / (jp.stem + ".png")
        if pp.exists():
            images.append(np.array(Image.open(jp).convert("RGB")))
            m = np.array(Image.open(pp))
            masks.append((m > 0).astype(np.uint8))
    return images, masks


def run_inference_batch(model, images_np, device):
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


def build_volume(images_rgb, masks=None):
    slices = []
    for i, img in enumerate(images_rgb):
        gray = np.mean(img, axis=2).astype(np.float32)
        if masks is not None:
            gray = gray * masks[i]
        slices.append(gray)
    return np.array(slices, dtype=np.float32)  # (Z, H, W)


def compute_mip(volume, axis=0):
    return np.max(volume, axis=axis)


def apply_wwwl(mip_float, ww, wl):
    lower = wl - ww / 2.0
    upper = wl + ww / 2.0
    img = (mip_float - lower) / max(upper - lower, 1e-7) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def mip_to_display(mip_u8, axis, pixel_spacing, slice_spacing, panel_size):
    """
    Resize MIP image with correct aspect ratio for display.

    Volume shape: (Z, H, W)
    - Axial MIP (axis=0): projects along Z -> result is (H, W), both at pixel_spacing
    - Coronal MIP (axis=1): projects along H -> result is (Z, W), Z=slice, W=pixel
    - Sagittal MIP (axis=2): projects along W -> result is (Z, H), Z=slice, H=pixel

    Physical dimensions determine the aspect ratio.
    """
    h_pix, w_pix = mip_u8.shape

    if axis == 0:
        # Axial: H and W both at pixel_spacing -> square aspect
        phys_h = h_pix * pixel_spacing
        phys_w = w_pix * pixel_spacing
    elif axis == 1:
        # Coronal: result shape (Z, W) -> h=Z at slice_spacing, w=W at pixel_spacing
        phys_h = h_pix * slice_spacing
        phys_w = w_pix * pixel_spacing
    else:
        # Sagittal: result shape (Z, H) -> h=Z at slice_spacing, w=H at pixel_spacing
        phys_h = h_pix * slice_spacing
        phys_w = w_pix * pixel_spacing

    aspect = phys_w / max(phys_h, 1e-7)

    if aspect >= 1.0:
        # Wider than tall
        disp_w = panel_size
        disp_h = int(panel_size / aspect)
    else:
        # Taller than wide
        disp_h = panel_size
        disp_w = int(panel_size * aspect)

    disp_h = max(disp_h, 1)
    disp_w = max(disp_w, 1)

    rgb = np.stack([mip_u8, mip_u8, mip_u8], axis=2)
    img = Image.fromarray(rgb).resize((disp_w, disp_h), Image.BILINEAR)

    # Center on black background
    bg = Image.new("RGB", (panel_size, panel_size), (0, 0, 0))
    offset_x = (panel_size - disp_w) // 2
    offset_y = (panel_size - disp_h) // 2
    bg.paste(img, (offset_x, offset_y))
    return bg


def compute_dice_case(preds, gts):
    dices = []
    for p, g in zip(preds, gts):
        pf = (p == 1).astype(np.float32)
        gf = (g == 1).astype(np.float32)
        inter = (pf * gf).sum()
        d = (2 * inter + 1e-7) / (pf.sum() + gf.sum() + 1e-7)
        dices.append(d)
    return np.mean(dices)


# ============================================================
# GUI
# ============================================================
class MIPViewer:
    def __init__(self, root, case_data):
        self.root = root
        self.root.title("MRA MIP Viewer (WW/WL + Axial/Coronal/Sagittal)")
        self.root.configure(bg="#1e1e1e")

        self.case_data = case_data
        self.current_case = 0
        self.n_cases = len(case_data)
        self.current_axis = 0
        self.axis_names = ["Axial", "Coronal", "Sagittal"]

        # Lay out only the panels whose model was actually loaded
        _all_cols = ["Original", "GT Masked", "Initial", "Round1 Gen10", "Round2 Final"]
        _all_colors = ["#bbbbbb", "#4caf50", "#4fc3f7", "#ff8a65", "#81c784"]
        _present = case_data[0]["volumes"]
        self.col_names = [n for n in _all_cols if n in _present]
        self.col_colors = [c for n, c in zip(_all_cols, _all_colors) if n in _present]

        self._compute_all_mips()
        self._find_global_range()

        self.ww = self.global_max - self.global_min
        self.wl = (self.global_max + self.global_min) / 2.0

        # ---- Layout ----

        # View buttons
        view_frame = tk.Frame(root, bg="#1e1e1e")
        view_frame.pack(fill=tk.X, padx=10, pady=(8, 2))

        tk.Label(view_frame, text="View: ", font=("Consolas", 11),
                 fg="#e0e0e0", bg="#1e1e1e").pack(side=tk.LEFT)

        self.view_buttons = []
        for i, name in enumerate(self.axis_names):
            btn = tk.Button(view_frame, text=name, font=("Consolas", 11, "bold"),
                            command=lambda ax=i: self._set_axis(ax),
                            bg="#444444", fg="#ffffff", relief=tk.FLAT, padx=15, pady=2)
            btn.pack(side=tk.LEFT, padx=3)
            self.view_buttons.append(btn)
        self._highlight_axis_button()

        # Spacing info label
        self.spacing_label = tk.Label(view_frame, text="", font=("Consolas", 9),
                                      fg="#666666", bg="#1e1e1e")
        self.spacing_label.pack(side=tk.RIGHT, padx=10)

        # Column headers
        header = tk.Frame(root, bg="#1e1e1e")
        header.pack(fill=tk.X, padx=10, pady=(5, 2))
        for i, name in enumerate(self.col_names):
            tk.Label(header, text=name, font=("Consolas", 10, "bold"),
                     fg=self.col_colors[i], bg="#1e1e1e").pack(side=tk.LEFT, expand=True)

        # Canvases
        canvas_frame = tk.Frame(root, bg="#1e1e1e")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10)

        self.canvases = []
        self.tk_imgs = [None] * len(self.col_names)
        for i in range(len(self.col_names)):
            c = tk.Canvas(canvas_frame, width=PANEL_SIZE, height=PANEL_SIZE,
                          bg="#000000", highlightthickness=0)
            c.pack(side=tk.LEFT, padx=2)
            self.canvases.append(c)

        # WW/WL sliders
        wl_frame = tk.Frame(root, bg="#1e1e1e")
        wl_frame.pack(fill=tk.X, padx=10, pady=(5, 0))

        tk.Label(wl_frame, text="WL (Level):", font=("Consolas", 10),
                 fg="#aaaaaa", bg="#1e1e1e").pack(side=tk.LEFT)
        self.wl_var = tk.DoubleVar(value=self.wl)
        self.wl_slider = ttk.Scale(wl_frame, from_=0, to=self.global_max,
                                    orient=tk.HORIZONTAL, variable=self.wl_var,
                                    command=self._on_wwwl_change)
        self.wl_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.wl_label = tk.Label(wl_frame, text=f"{self.wl:.0f}", font=("Consolas", 10),
                                 fg="#e0e0e0", bg="#1e1e1e", width=6)
        self.wl_label.pack(side=tk.LEFT)

        ww_frame = tk.Frame(root, bg="#1e1e1e")
        ww_frame.pack(fill=tk.X, padx=10, pady=(2, 0))

        tk.Label(ww_frame, text="WW (Width):", font=("Consolas", 10),
                 fg="#aaaaaa", bg="#1e1e1e").pack(side=tk.LEFT)
        self.ww_var = tk.DoubleVar(value=self.ww)
        self.ww_slider = ttk.Scale(ww_frame, from_=1, to=self.global_max,
                                    orient=tk.HORIZONTAL, variable=self.ww_var,
                                    command=self._on_wwwl_change)
        self.ww_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.ww_label = tk.Label(ww_frame, text=f"{self.ww:.0f}", font=("Consolas", 10),
                                 fg="#e0e0e0", bg="#1e1e1e", width=6)
        self.ww_label.pack(side=tk.LEFT)

        # Reset button
        reset_frame = tk.Frame(root, bg="#1e1e1e")
        reset_frame.pack(fill=tk.X, padx=10, pady=(2, 0))
        tk.Button(reset_frame, text="Reset WW/WL", font=("Consolas", 9),
                  command=self._reset_wwwl,
                  bg="#333333", fg="#ffffff", relief=tk.FLAT, padx=10).pack(side=tk.RIGHT)

        # Info bar
        info_frame = tk.Frame(root, bg="#1e1e1e")
        info_frame.pack(fill=tk.X, padx=10, pady=3)

        self.info_label = tk.Label(info_frame, text="", font=("Consolas", 12, "bold"),
                                   fg="#e0e0e0", bg="#1e1e1e")
        self.info_label.pack()

        self.dice_label = tk.Label(info_frame, text="", font=("Consolas", 10),
                                   fg="#aaaaaa", bg="#1e1e1e")
        self.dice_label.pack()

        # Navigation
        nav_frame = tk.Frame(root, bg="#1e1e1e")
        nav_frame.pack(fill=tk.X, padx=10, pady=(3, 8))

        tk.Button(nav_frame, text="<< Prev Case", font=("Consolas", 11),
                  command=lambda: self._navigate_case(-1),
                  bg="#333333", fg="#ffffff", relief=tk.FLAT, padx=20).pack(side=tk.LEFT, padx=5)

        self.case_label = tk.Label(nav_frame, text="", font=("Consolas", 11),
                                   fg="#e0e0e0", bg="#1e1e1e")
        self.case_label.pack(side=tk.LEFT, expand=True)

        tk.Button(nav_frame, text="Next Case >>", font=("Consolas", 11),
                  command=lambda: self._navigate_case(1),
                  bg="#333333", fg="#ffffff", relief=tk.FLAT, padx=20).pack(side=tk.RIGHT, padx=5)

        # Bindings
        root.bind("<Left>", lambda e: self._navigate_case(-1))
        root.bind("<Right>", lambda e: self._navigate_case(1))
        root.bind("<MouseWheel>", lambda e: self._navigate_case(-1 if e.delta > 0 else 1))
        root.bind("a", lambda e: self._set_axis(0))
        root.bind("c", lambda e: self._set_axis(1))
        root.bind("s", lambda e: self._set_axis(2))

        self._update_display()

    def _compute_all_mips(self):
        self.mips = []
        for cd in self.case_data:
            mips = {}
            for name in self.col_names:
                vol = cd["volumes"][name]
                mips[name] = compute_mip(vol, axis=self.current_axis)
            self.mips.append(mips)

    def _find_global_range(self):
        all_vals = []
        for case_mips in self.mips:
            for name, mip in case_mips.items():
                if mip.max() > 0:
                    all_vals.append(mip.max())
        self.global_max = max(all_vals) if all_vals else 255.0
        self.global_min = 0.0

    def _highlight_axis_button(self):
        for i, btn in enumerate(self.view_buttons):
            btn.configure(bg="#0078d4" if i == self.current_axis else "#444444")

    def _set_axis(self, axis):
        if axis != self.current_axis:
            self.current_axis = axis
            self._highlight_axis_button()
            self._compute_all_mips()
            self._find_global_range()
            self.wl_slider.configure(to=self.global_max)
            self.ww_slider.configure(to=self.global_max)
            self._reset_wwwl()

    def _reset_wwwl(self):
        self.ww = self.global_max - self.global_min
        self.wl = (self.global_max + self.global_min) / 2.0
        self.ww_var.set(self.ww)
        self.wl_var.set(self.wl)
        self.ww_label.config(text=f"{self.ww:.0f}")
        self.wl_label.config(text=f"{self.wl:.0f}")
        self._update_display()

    def _on_wwwl_change(self, *args):
        self.ww = max(self.ww_var.get(), 1)
        self.wl = self.wl_var.get()
        self.ww_label.config(text=f"{self.ww:.0f}")
        self.wl_label.config(text=f"{self.wl:.0f}")
        self._update_display()

    def _navigate_case(self, delta):
        new = max(0, min(self.n_cases - 1, self.current_case + delta))
        if new != self.current_case:
            self.current_case = new
            self._update_display()

    def _update_display(self):
        idx = self.current_case
        cd = self.case_data[idx]
        case_mips = self.mips[idx]
        ps = cd["pixel_spacing"]
        ss = cd["slice_spacing"]

        for i, name in enumerate(self.col_names):
            mip_float = case_mips[name]
            mip_u8 = apply_wwwl(mip_float, self.ww, self.wl)
            pil_img = mip_to_display(mip_u8, self.current_axis, ps, ss, PANEL_SIZE)
            self.tk_imgs[i] = ImageTk.PhotoImage(pil_img)
            self.canvases[i].delete("all")
            self.canvases[i].create_image(0, 0, anchor=tk.NW, image=self.tk_imgs[i])

        view_str = self.axis_names[self.current_axis]
        self.info_label.config(
            text=f"Case {cd['case_id']}  ({cd['n_slices']} slices)  |  View: {view_str}")

        self.spacing_label.config(
            text=f"Pixel: {ps:.3f}mm  Slice: {ss:.3f}mm")

        dsc_parts = []
        for mname in ["Initial", "Round1 Gen10", "Round2 Final"]:
            d = cd["dsc"][mname]
            dsc_parts.append(f"{mname}: {d:.4f}")
        self.dice_label.config(text="DSC  " + "  |  ".join(dsc_parts))

        self.case_label.config(text=f"Case {idx + 1} / {self.n_cases}")


# ============================================================
# Main
# ============================================================
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    models = {}
    for name, path in MODELS.items():
        print(f"Loading {name}: {path.name}")
        models[name] = load_model(path, device)

    case_data = []
    for cid in TEST_CASES:
        print(f"\nProcessing Case {cid}...")
        images, gt_masks = load_case_data(cid)
        ps, ss = get_spacing_from_dicom(cid)
        n_slices = len(images)
        print(f"  {n_slices} slices, pixel={ps:.4f}mm, slice={ss:.4f}mm")

        volumes = {
            "Original": build_volume(images),
            "GT Masked": build_volume(images, gt_masks),
        }
        dsc = {}

        for mname, model in models.items():
            preds = run_inference_batch(model, images, device)
            volumes[mname] = build_volume(images, preds)
            dsc[mname] = compute_dice_case(preds, gt_masks)
            print(f"  {mname}: DSC={dsc[mname]:.4f}")

        case_data.append({
            "case_id": cid,
            "n_slices": n_slices,
            "pixel_spacing": ps,
            "slice_spacing": ss,
            "volumes": volumes,
            "dsc": dsc,
        })

    del models
    torch.cuda.empty_cache()

    print(f"\nLaunching MIP viewer ({len(case_data)} cases)...")
    print("Keys: A=Axial, C=Coronal, S=Sagittal, Left/Right=Cases")

    root = tk.Tk()
    w = PANEL_SIZE * (len(models) + 2) + 30   # original + ground truth + models
    h = PANEL_SIZE + 340
    root.geometry(f"{w}x{h}")
    root.resizable(True, True)
    MIPViewer(root, case_data)
    root.mainloop()


if __name__ == "__main__":
    main()
