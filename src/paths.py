"""Project paths, resolved relative to the repository root.

Nothing is hard-coded: every path is derived from the location of this file,
so the tree can be moved to another drive or machine without editing code.

Usage from src/mra_seg/*.py:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/
    from paths import RAW_JPEG_DIR, FOLD_MODEL_DIR
"""

from pathlib import Path

# paths.py is expected at <PROJECT_ROOT>/src/paths.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

PYTHON_EXE = PROJECT_ROOT / "python" / "python.exe"

# ------------------------------------------------------------------
# Labeled dataset
# ------------------------------------------------------------------
MRA_SEG_DIR = DATA_DIR / "mra_seg"
RAW_JPEG_DIR = MRA_SEG_DIR / "rawJPEG"       # input slices, per-case folders
RAW_PNG_DIR = MRA_SEG_DIR / "rawPNG"         # ground-truth masks, same filenames
MRA_DICOM_DIR = MRA_SEG_DIR / "DICOMdata"    # source DICOM (spacing / MIP)

# Unlabeled studies consumed by the evolutionary update. Not distributed with
# this repository; point MRA_STUDY_DIR at your own store.
MRA_STUDY_DIR = DATA_DIR / "mra_studies"

# ------------------------------------------------------------------
# Models and results
# ------------------------------------------------------------------
# Seed checkpoints written by train_deeplabv3plus_5fold.py (fold1.pth .. fold5.pth)
FOLD_MODEL_DIR = MODELS_DIR / "deeplabv3plus_5fold"

# Evolutionary runs: one sub-directory per run, named <configuration>_fold<k>
EVOLUTION_DIR = MODELS_DIR / "evolution"

SEG_RESULT_DIR = RESULTS_DIR / "mra_seg"
SEG_FIGURE_DIR = SEG_RESULT_DIR / "figures"
ANALYSIS_DIR = RESULTS_DIR / "analysis"


def require_external(path, what):
    """Stop with a clear message when data that is not distributed is missing."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"[missing data] {what} was not found at: {p}\n"
            f"  This dataset is not distributed with the repository.\n"
            f"  Edit MRA_STUDY_DIR in paths.py to point at your own store."
        )
    return p


def ensure_dirs(*dirs):
    """Create output directories."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"PROJECT_ROOT = {PROJECT_ROOT}\n")
    for name, p in [
        ("RAW_JPEG_DIR", RAW_JPEG_DIR), ("RAW_PNG_DIR", RAW_PNG_DIR),
        ("MRA_DICOM_DIR", MRA_DICOM_DIR), ("MRA_STUDY_DIR", MRA_STUDY_DIR),
        ("FOLD_MODEL_DIR", FOLD_MODEL_DIR), ("EVOLUTION_DIR", EVOLUTION_DIR),
        ("SEG_RESULT_DIR", SEG_RESULT_DIR), ("ANALYSIS_DIR", ANALYSIS_DIR),
        ("PYTHON_EXE", PYTHON_EXE),
    ]:
        print(f"  [{'OK ' if p.exists() else '-- '}] {name:20s} {p}")
