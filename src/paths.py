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
FOLD_MODEL_DIR = MODELS_DIR / "deeplabv3plus_5fold"
FOLD_MODEL_V1_DIR = MODELS_DIR / "deeplabv3plus_5fold_v1"
INITIAL_MODEL = FOLD_MODEL_DIR / "deeplabv3plus_fold5.pth"

EVOLUTION_DIR = MODELS_DIR / "evolution"        # naive strategy
EVOLUTION_V2_DIR = MODELS_DIR / "evolution_v2"  # improved strategy

SEG_RESULT_DIR = RESULTS_DIR / "mra_seg"
SEG_FIGURE_DIR = SEG_RESULT_DIR / "figures"
EVOLUTION_LOG = SEG_RESULT_DIR / "evolution_log.json"
EVOLUTION_LOG_V2 = SEG_RESULT_DIR / "evolution_log_v2.json"


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
        ("FOLD_MODEL_DIR", FOLD_MODEL_DIR), ("INITIAL_MODEL", INITIAL_MODEL),
        ("EVOLUTION_DIR", EVOLUTION_DIR), ("EVOLUTION_V2_DIR", EVOLUTION_V2_DIR),
        ("SEG_RESULT_DIR", SEG_RESULT_DIR), ("PYTHON_EXE", PYTHON_EXE),
    ]:
        print(f"  [{'OK ' if p.exists() else '-- '}] {name:20s} {p}")
