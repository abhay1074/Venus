"""Stage 2 — DR severity grading.

Two independent graders run and are fused; their disagreement is itself an
output.

CNN grader   EfficientNet-B4 at 380x380 with an ordinal head (four sigmoid
             thresholds P(grade >= k), k = 1..4). Grade = number of thresholds
             passed. P(grade >= 2) is the referable signal; it is passed through
             the Platt scaling fitted on the calibration split so the served
             number is a probability, and compared with the locked threshold.
             P(grade >= 4) is the PDR / neovascularization evidence.

Rule grader  A deterministic function of the Stage 1 lesion counts applying
             the International Clinical DR scale as written (4-2-1 rule for
             severe; MA-only is mild; NV probability > 0.5 is PDR). A clinician
             can check it against the ICDR table by hand.

Fusion       The CNN grade is primary. Disagreement of two or more levels, a
             CNN referable call with no lesion at all, or a calibrated
             probability inside the abstain band routes the case to human
             review with both pieces of evidence attached.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from backend.venus.config import GRADER_SIZE, GRADER_WEIGHTS, ICDR_LABELS, operating_point
from backend.venus.imaging import clahe

_model = None
_lock = threading.Lock()
_load_error: str | None = None

# Referable DR is ICDR level 2 or worse.
REFERABLE_LEVEL = 2
# 4-2-1 rule: more than 20 hemorrhages in each of four quadrants.
SEVERE_HE_PER_QUADRANT = 20
# Hemorrhage area suggesting preretinal / vitreous bleed, as a fraction of FOV.
PDR_HE_AREA_FRACTION = 0.04
# Component thresholds: the classical detectors produce isolated false
# components on healthy retinas, so a lesion type counts as present only above
# these floors (count, or total area as a fraction of the FOV).
EVIDENCE_FLOOR = {"MA": (2, 0.0), "HE": (1, 0.00025), "EX": (3, 0.00020), "SE": (1, 0.00040)}


# ----------------------------------------------------------------- model --

def build_grader():
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(GRADER_SIZE, GRADER_SIZE, 3), name="fundus_image")
    backbone = keras.applications.EfficientNetB4(include_top=False, weights=None, input_tensor=inputs)
    features = layers.GlobalAveragePooling2D(name="fundus_gap")(backbone.output)

    dr = layers.Dense(256, activation="relu", name="dr_dense")(features)
    dr = layers.Dropout(0.5, name="dr_dropout")(dr)
    dr_out = layers.Dense(4, activation="sigmoid", dtype="float32", name="dr_ordinal_thresholds")(dr)

    # The checkpoint also carries a multi-label disease head. It is built so the
    # weights load, and its output is never read: nothing it computes has been
    # validated and the report must not carry numbers nobody can vouch for.
    disease = layers.Dense(256, activation="relu", name="disease_dense")(features)
    disease = layers.Dropout(0.5, name="disease_dropout")(disease)
    disease_out = layers.Dense(8, activation="sigmoid", dtype="float32", name="disease_multilabel")(disease)

    return keras.Model(inputs, {"dr_ordinal_thresholds": dr_out, "disease_multilabel": disease_out},
                       name="venus_dr_grader")


def load_grader():
    global _model, _load_error
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        if not GRADER_WEIGHTS.exists():
            _load_error = f"missing grader weights: {GRADER_WEIGHTS.name}"
            return None
        try:
            model = build_grader()
            model.load_weights(GRADER_WEIGHTS)
            _model = model
        except Exception as exc:  # pragma: no cover
            _load_error = f"{type(exc).__name__}: {exc}"
            return None
    return _model


def grader_status() -> dict:
    return {"loaded": _model is not None, "error": _load_error,
            "backbone": "EfficientNet-B4, ordinal head, 380x380", "weights": GRADER_WEIGHTS.name}


# ---------------------------------------------------------- preprocessing --

def grader_input(image_bgr: np.ndarray) -> np.ndarray:
    """The exact preprocessing the grader was trained and validated with.

    Fundus bounding-box crop -> Ben Graham illumination correction (sigma 10)
    -> CLAHE on green -> 380x380 RGB, float32 in 0-255 (EfficientNet rescales
    internally; normalising here would hand the backbone a near-zero constant).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fundus = image_bgr
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        fundus = image_bgr[y:y + h, x:x + w]
    enhanced = cv2.addWeighted(fundus, 4, cv2.GaussianBlur(fundus, (0, 0), sigmaX=10), -4, 128)
    enhanced[:, :, 1] = clahe(enhanced[:, :, 1], clip=2.0, tile=8)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (GRADER_SIZE, GRADER_SIZE)).astype(np.float32)


def _tta_views(x: np.ndarray) -> np.ndarray:
    """Four rotations and one flip. Rotation invariance is free for fundus."""
    views = [x, np.rot90(x, 1), np.rot90(x, 2), np.rot90(x, 3), np.fliplr(x)]
    return np.stack([np.ascontiguousarray(v) for v in views])


# ------------------------------------------------------------ CNN grader --

def logit(p: float) -> float:
    p = min(max(p, 1e-7), 1 - 1e-7)
    return float(np.log(p / (1 - p)))


def calibrate(raw: float, point: dict) -> float:
    a, b = point["calibration"]["a"], point["calibration"]["b"]
    return float(1.0 / (1.0 + np.exp(-(a * logit(raw) + b))))


def cnn_grade(image_bgr: np.ndarray, tta: bool = False) -> dict:
    model = load_grader()
    if model is None:
        raise RuntimeError(f"CNN grader unavailable: {_load_error}")
    x = grader_input(image_bgr)
    batch = _tta_views(x) if tta else x[None, ...]
    out = model.predict(batch, verbose=0)
    ordinal = np.asarray(out["dr_ordinal_thresholds"]).mean(axis=0)
    grade = int((ordinal >= 0.5).sum())
    point = operating_point()
    raw_referable = float(ordinal[1])
    p_referable = calibrate(raw_referable, point)
    # Five-grade probability bar from the ordinal thresholds:
    # P(g=k) = P(g>=k) - P(g>=k+1), with the monotone cumulative enforced.
    cum = np.concatenate([[1.0], np.minimum.accumulate(ordinal), [0.0]])
    per_grade = np.clip(cum[:-1] - cum[1:], 0, 1)
    per_grade = per_grade / max(per_grade.sum(), 1e-6)
    return {
        "grade": grade,
        "grade_label": ICDR_LABELS[grade],
        "ordinal_thresholds": [round(float(v), 4) for v in ordinal],
        "grade_probabilities": [round(float(v), 4) for v in per_grade],
        "referable_raw": round(raw_referable, 6),
        "referable_probability": round(p_referable, 4),
        "nv_probability": round(float(ordinal[3]), 4),
        "tta": tta,
        "input": x,
    }


# ------------------------------------------------------------ rule grader --

def rule_grade(stage1: dict, nv_probability: float) -> dict:
    """ICDR scale applied literally to Stage 1 counts. Returns the grade and a trace."""
    lesions = stage1["lesions"]
    raw_counts = {k: lesions[k]["count"] for k in ("MA", "HE", "EX", "SE")}
    present = {}
    for key, (min_count, min_area) in EVIDENCE_FLOOR.items():
        present[key] = lesions[key]["count"] if (
            lesions[key]["count"] >= min_count and lesions[key]["area_fraction"] >= min_area) else 0
    ma, he, ex, se = (present[k] for k in ("MA", "HE", "EX", "SE"))
    he_area = lesions["HE"]["area_fraction"]
    quadrants = stage1["hemorrhages_per_quadrant"]
    trace = []

    if nv_probability > 0.5:
        grade = 4
        trace.append(f"P(NV) = {nv_probability:.2f} > 0.5 from the grader (classifier, not localised) → PDR")
    elif he_area > PDR_HE_AREA_FRACTION:
        grade = 4
        trace.append(f"hemorrhage area {he_area * 100:.1f}% of FOV suggests vitreous/preretinal bleed → PDR")
    elif all(q > SEVERE_HE_PER_QUADRANT for q in quadrants):
        grade = 3
        trace.append(f"4-2-1 rule: >{SEVERE_HE_PER_QUADRANT} hemorrhages in each of 4 quadrants ({quadrants}) → severe NPDR")
    elif he > 0 or ex > 0 or se > 0:
        grade = 2
        parts = []
        if he: parts.append(f"{he} hemorrhage{'s' if he != 1 else ''} (quadrants {quadrants})")
        if ex: parts.append(f"{ex} hard exudate{'s' if ex != 1 else ''}")
        if se: parts.append(f"{se} soft exudate{'s' if se != 1 else ''}")
        if ma: parts.append(f"{ma} microaneurysm{'s' if ma != 1 else ''}")
        trace.append("more than microaneurysms only, less than severe: " + "; ".join(parts) + " → moderate NPDR")
    elif ma > 0:
        grade = 1
        trace.append(f"microaneurysms only ({ma}) → mild NPDR")
    else:
        grade = 0
        trace.append("no MA, HE or EX above component thresholds → no DR")

    below = [f"{k} {raw_counts[k]}" for k in raw_counts if raw_counts[k] and not present[k]]
    if below:
        trace.append("below component threshold, not counted: " + ", ".join(below))
    trace.append("venous beading and IRMA are not detected by this build and are not part of the severe criterion")
    return {
        "grade": grade,
        "grade_label": ICDR_LABELS[grade],
        "referable": grade >= REFERABLE_LEVEL,
        "counts": {"MA": ma, "HE": he, "EX": ex, "SE": se},
        "raw_counts": raw_counts,
        "evidence_floor": {k: {"min_count": v[0], "min_area_fraction": v[1]} for k, v in EVIDENCE_FLOOR.items()},
        "hemorrhages_per_quadrant": quadrants,
        "nv_probability": round(nv_probability, 4),
        "trace": trace,
        "criteria_text": trace[0],
    }


# ----------------------------------------------------------------- fusion --

def fuse(cnn: dict, rule: dict) -> dict:
    point = operating_point()
    threshold = point["thresholds"]["referable"]
    band = point["thresholds"]["abstain_band"]
    p = cnn["referable_probability"]
    referable = p >= threshold
    abstain = abs(p - threshold) <= band
    disagreement = abs(cnn["grade"] - rule["grade"])
    total_lesions = sum(rule["counts"].values())

    reasons = []
    if disagreement >= 2:
        reasons.append(f"CNN grade {cnn['grade']} and rule grade {rule['grade']} differ by {disagreement} levels")
    if referable and total_lesions == 0:
        reasons.append("CNN calls referable but no lesion was found")
    if abstain:
        reasons.append(f"P(referable) {p:.2f} is within ±{band:.2f} of the threshold {threshold:.2f}")

    return {
        "grade": cnn["grade"],
        "grade_label": cnn["grade_label"],
        "referable": bool(referable),
        "p_referable": p,
        "threshold": threshold,
        "abstain": bool(abstain),
        "abstain_band": band,
        "grades_agree_within_one": disagreement <= 1,
        "disagreement_levels": disagreement,
        "flag_for_review": bool(reasons),
        "flag_reasons": reasons,
        "agreement_text": f"CNN grade {cnn['grade']}, rule grade {rule['grade']}",
        "confidence_text": f"Referable: {p:.2f} (calibrated, threshold {threshold:.2f}). "
                           f"Agreement: CNN grade {cnn['grade']}, rule grade {rule['grade']}",
    }


# -------------------------------------------------------------------- run --

def run(image_bgr: np.ndarray, stage1: dict, tta: bool = False) -> dict:
    started = time.perf_counter()
    cnn = cnn_grade(image_bgr, tta=tta)
    cnn_ms = int((time.perf_counter() - started) * 1000)
    rule = rule_grade(stage1, cnn["nv_probability"])
    fusion = fuse(cnn, rule)
    return {
        "cnn": {k: v for k, v in cnn.items() if k != "input"},
        "rule": rule,
        "fusion": fusion,
        "grader_input": cnn["input"],
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "cnn_ms": cnn_ms,
    }
