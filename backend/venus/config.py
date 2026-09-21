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

# Checkpoints. The grader served is v2 (EfficientNet-B3 at 512, trained here)
# when its weights are present, else the v1 EfficientNet-B4/380 checkpoint.
# The quality CNN and the lesion U-Net are optional: without them Stage 0 uses
# handcrafted features only and Stage 1 the classical detectors, and each
# result says which it was.
GRADER_V2_WEIGHTS = WEIGHTS_DIR / "grader_v2.weights.h5"
GRADER_V1_WEIGHTS = WEIGHTS_DIR / "eye_best.weights.h5"
GATE_WEIGHTS = WEIGHTS_DIR / "eye_modality_gate.weights.h5"
QUALITY_WEIGHTS = WEIGHTS_DIR / "quality_cnn.weights.h5"
UNET_WEIGHTS = WEIGHTS_DIR / "lesion_unet.weights.h5"
LESION_THRESHOLDS_PATH = CONFIG_DIR / "lesion_thresholds.json"
# Optional second lesion network at a larger frame, used only for the lesion
# classes its thresholds file lists under "serves" (microaneurysms: 1-3 px
# at 512). Absent files simply mean the 512 px network reads every class.
UNET_HIRES_WEIGHTS = WEIGHTS_DIR / "lesion_unet_1024.weights.h5"
LESION_THRESHOLDS_HIRES_PATH = Path(os.getenv("VENUS_UNET_HIRES_SPEC", str(CONFIG_DIR / "lesion_thresholds_1024.json")))  # point at a missing file to disable
REVIEW_POLICY_PATH = CONFIG_DIR / "review_policy.json"
# VENUS_GRADER_TAG names the grader an environment without checkpoints (CI,
# a docs build) is standing in for, so the committed operating point can be
# read; serving never sets it, the weights on disk decide.
GRADER_TAG = os.getenv("VENUS_GRADER_TAG") or ("grader_v2" if GRADER_V2_WEIGHTS.exists() else "legacy_v1")
GRADER_WEIGHTS = GRADER_V2_WEIGHTS if GRADER_TAG == "grader_v2" else GRADER_V1_WEIGHTS
MODEL_VERSION = "venus-dr-2.0.0" if GRADER_TAG == "grader_v2" else "venus-dr-1.0.0"
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


@lru_cache(maxsize=1)
def review_policy() -> dict:
    """Human-review parameters chosen on the validation sample by
    backend.eval.review_policy: abstain band (logit half-width) and the
    attention-agreement floor. Defaults are the architecture's if absent."""
    defaults = {"abstain_band_logit": None, "abstain_band": 0.05, "attention_floor": 0.15, "attention_min_lift": 1.5, "chosen_on": "architecture defaults"}
    if not REVIEW_POLICY_PATH.exists():
        return defaults
    with open(REVIEW_POLICY_PATH, "r", encoding="utf-8") as handle:
        policy = json.load(handle)
    return {**defaults, **policy, "attention_min_lift": None}


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
        manifest = MANIFEST_DIR / point.get("calibration_manifest", CALIBRATION_MANIFEST.name)
        if not manifest.exists():
            raise OperatingPointError(f"calibration manifest missing: {manifest}")
        actual = sha256_of_file(manifest)
        if point.get("grader_tag", "legacy_v1") != GRADER_TAG:
            raise OperatingPointError(
                f"operating_point.json was locked for grader '{point.get('grader_tag')}' but the served grader is "
                f"'{GRADER_TAG}'; a threshold from another model is meaningless here. Re-run backend.eval.calibrate."
            )
        if actual != point.get("calibration_fingerprint"):
            raise OperatingPointError(
                "calibration manifest fingerprint does not match operating_point.json "
                f"({actual[:12]}... vs {str(point.get('calibration_fingerprint'))[:12]}...)"
            )
    return point
