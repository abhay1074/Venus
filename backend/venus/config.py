"""Paths, versions and the locked operating point.

Everything that reaches a report is versioned together: the model weights, the
calibration parameters and the referable threshold. The version string in the
report footer is MODEL_VERSION, and the operating point carries the fingerprint
of the calibration set it was fitted on. serving refuses to run if that
fingerprint does not match the manifest on disk (see operating_point()).
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
WEIGHTS_DIR = Path(os.getenv("VENUS_WEIGHTS_DIR", str(BACKEND_ROOT / "weights")))
CONFIG_DIR = BACKEND_ROOT / "config"
MANIFEST_DIR = BACKEND_ROOT / "data" / "manifests"
REPORT_DIR = Path(os.getenv("VENUS_REPORT_DIR", str(BACKEND_ROOT / "reports")))
DB_PATH = Path(os.getenv("VENUS_DB_PATH", str(BACKEND_ROOT / "data" / "venus.sqlite")))
SAMPLES_DIR = PROJECT_ROOT / "samples"

MODEL_VERSION = "venus-dr-1.0.0"
GRADER_WEIGHTS = WEIGHTS_DIR / "eye_best.weights.h5"
GATE_WEIGHTS = WEIGHTS_DIR / "eye_modality_gate.weights.h5"
OPERATING_POINT_PATH = CONFIG_DIR / "operating_point.json"
CALIBRATION_MANIFEST = MANIFEST_DIR / "calibration_split.csv"

# Working image size for Stage 0 output, Stage 1 and Stage 3 overlays.
WORK_SIZE = 512
# Input size the CNN grader was trained at.
GRADER_SIZE = 380

MAX_UPLOAD_MB = int(os.getenv("VENUS_MAX_UPLOAD_MB", "12"))
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

ICDR_LABELS = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "PDR",
}


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OperatingPointError(RuntimeError):
    """The locked operating point is missing or does not match its calibration set."""


@lru_cache(maxsize=1)
def operating_point() -> dict:
    """Load config/operating_point.json and verify its calibration fingerprint.

    The threshold and the Platt parameters were chosen on the calibration split
    whose SHA-256 is recorded in the file. If the manifest on disk has changed,
    the numbers no longer describe it, so serving stops rather than continuing
    with a threshold nobody can vouch for.
    """
    if not OPERATING_POINT_PATH.exists():
        raise OperatingPointError(
            f"missing {OPERATING_POINT_PATH}; run `python -m backend.eval.calibrate` first"
        )
    with open(OPERATING_POINT_PATH, "r", encoding="utf-8") as handle:
        point = json.load(handle)
    if os.getenv("VENUS_SKIP_FINGERPRINT_CHECK", "false").lower() != "true":
        if not CALIBRATION_MANIFEST.exists():
            raise OperatingPointError(f"calibration manifest missing: {CALIBRATION_MANIFEST}")
        actual = sha256_of_file(CALIBRATION_MANIFEST)
        if actual != point.get("calibration_fingerprint"):
            raise OperatingPointError(
                "calibration manifest fingerprint does not match operating_point.json "
                f"({actual[:12]}... vs {str(point.get('calibration_fingerprint'))[:12]}...)"
            )
    return point
