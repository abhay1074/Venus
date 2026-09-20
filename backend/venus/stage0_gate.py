"""Stage 0 — input gate and image quality.

Decides three things before any diagnostic model runs: is this a fundus
photograph, is it gradable, and does it need enhancement. It is the direct
answer to the problem statement's line about variable image quality from
portable cameras.

    image -> modality check -> FOV mask + crop -> quality features -> enhancement
                 |                                       |
             reject: not retinal                 retake: reason shown to operator

The modality classifier is an EfficientNet-B0 gate that was pre-registered and
scored once on a held-out set (fundus false-rejection 0.67%, dermoscopy
acceptance 0.17%). Its threshold is locked; see GATE_THRESHOLD.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from backend.venus.config import GATE_WEIGHTS, QUALITY_WEIGHTS, WORK_SIZE
from backend.venus.imaging import clahe

# Locked 2026-09-02 on a selection split under a 0.5% false-rejection target,
# then the held-out set was scored exactly once. Not tuned since.
GATE_THRESHOLD = 0.987005
GATE_CLASSES = ["fundus", "dermoscopy", "face", "sclera", "other"]
GATE_SIZE = 224

# Hard limits on the handcrafted quality features. Fixed from the distribution
# of public "good" fundus images (percentile-based), not on any test image.
QUALITY_LIMITS = {
    "sharpness_reject": 6.0,        # variance of Laplacian on green inside FOV
    "sharpness_usable": 14.0,
    "illumination_reject": 0.28,    # periphery/centre luminance ratio
    "illumination_usable": 0.45,
    "saturated_reject": 0.12,       # fraction of FOV pixels saturated
    "saturated_usable": 0.05,
    "dark_reject": 0.55,            # fraction of FOV pixels near-black
    "dark_usable": 0.35,
    "coverage_reject": 0.30,        # FOV area / image area
    "circularity_reject": 0.55,
}

_gate_model = None
_gate_lock = threading.Lock()
_gate_error: str | None = None
_quality_model = None
_quality_error: str | None = None
QUALITY_SIZE = 256
QUALITY_CLASSES = ["good", "usable", "reject"]


# ---------------------------------------------------------------- modality --

def _build_gate():
    from tensorflow import keras

    backbone = keras.applications.EfficientNetB0(
        include_top=False, weights=None, input_shape=(GATE_SIZE, GATE_SIZE, 3)
    )
    x = keras.layers.GlobalAveragePooling2D(name="gap")(backbone.output)
    x = keras.layers.Dropout(0.3, name="drop")(x)
    out = keras.layers.Dense(len(GATE_CLASSES), activation="softmax", dtype="float32",
                             name="modality")(x)
    return keras.Model(backbone.input, out, name="venus_modality_gate")


def load_gate():
    """Load the modality gate once. Returns None (and records why) on failure."""
    global _gate_model, _gate_error
    if _gate_model is not None:
        return _gate_model
    with _gate_lock:
        if _gate_model is not None:
            return _gate_model
        if not GATE_WEIGHTS.exists():
            _gate_error = f"missing modality gate weights: {GATE_WEIGHTS.name}"
            return None
        try:
            model = _build_gate()
            model.load_weights(GATE_WEIGHTS)
            _gate_model = model
        except Exception as exc:  # pragma: no cover - depends on TF state
            _gate_error = f"{type(exc).__name__}: {exc}"
            return None
    return _gate_model


def gate_status() -> dict:
    return {"loaded": _gate_model is not None, "error": _gate_error, "threshold": GATE_THRESHOLD}


def modality_check(image_bgr: np.ndarray) -> dict:
    """P(fundus) on a plain 224 resize of the raw image.

    Deliberately not the enhanced image: enhancement is built to make a fundus
    look like a fundus and would erase the acquisition cues the gate reads.
    A gate that cannot load must not pass everything through, so a load failure
    is a rejection with the reason attached.
    """
    model = load_gate()
    if model is None:
        return {
            "ran": False,
            "accepted": False,
            "fundus_probability": None,
            "class_probabilities": None,
            "threshold": GATE_THRESHOLD,
            "reason": ("The modality gate could not be loaded, so the image cannot be "
                       f"verified as a fundus photograph. ({_gate_error})"),
        }
    resized = cv2.cvtColor(cv2.resize(image_bgr, (GATE_SIZE, GATE_SIZE)), cv2.COLOR_BGR2RGB)
    probs = model.predict(resized[None, ...].astype(np.float32), verbose=0)[0]
    p_fundus = float(probs[0])
    accepted = p_fundus >= GATE_THRESHOLD
    return {
        "ran": True,
        "accepted": accepted,
        "fundus_probability": round(p_fundus, 6),
        "class_probabilities": {c: round(float(p), 6) for c, p in zip(GATE_CLASSES, probs)},
        "threshold": GATE_THRESHOLD,
        "reason": None if accepted else (
            "This does not look like a fundus (retinal) photograph. Nothing was analysed."
        ),
    }


def load_quality():
    """The EyeQ-trained quality CNN, if present. Optional: without it the
    label comes from the handcrafted features alone, and the result says so."""
    global _quality_model, _quality_error
    if _quality_model is not None:
        return _quality_model
    with _gate_lock:
        if _quality_model is not None:
            return _quality_model
        if not QUALITY_WEIGHTS.exists():
            _quality_error = "quality CNN weights not present; handcrafted features only"
            return None
        try:
            from backend.venus import nets
            model = nets.quality_cnn()
            model.load_weights(QUALITY_WEIGHTS)
            _quality_model = model
        except Exception as exc:  # pragma: no cover
            _quality_error = f"{type(exc).__name__}: {exc}"
            return None
    return _quality_model


def quality_status() -> dict:
    return {"loaded": _quality_model is not None, "error": _quality_error}


def quality_cnn_probabilities(image512: np.ndarray) -> dict | None:
    model = load_quality()
    if model is None:
        return None
    resized = cv2.cvtColor(cv2.resize(image512, (QUALITY_SIZE, QUALITY_SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    probs = model.predict(resized[None, ...].astype(np.float32), verbose=0)[0]
    return {c: round(float(p), 4) for c, p in zip(QUALITY_CLASSES, probs)}


# --------------------------------------------------------------------- FOV --

def fov_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Circular field-of-view mask: Otsu on the red channel, largest component."""
    red = image_bgr[:, :, 2]
    blurred = cv2.GaussianBlur(red, (0, 0), 3)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # A very dark surround can pull Otsu too high; fall back to a fixed floor.
    if mask.mean() < 25:
        _, mask = cv2.threshold(blurred, 12, 255, cv2.THRESH_BINARY)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    # Convex hull of the largest component: fills holes (dark macula, lesions)
    # and smooths the jagged edge a dark, vignetted periphery leaves behind.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        cv2.drawContours(filled, [hull], -1, 255, thickness=-1)
        return filled
    return mask


def normalise_fov(image_bgr: np.ndarray, size: int = WORK_SIZE):
    """Crop to the FOV bounding box, pad to square, resize to `size`.

    Returns (image, mask, geometry). geometry carries what the quality features
    need: coverage and circularity of the mask before resizing.
    """
    mask = fov_mask(image_bgr)
    h, w = mask.shape
    coords = cv2.findNonZero(mask)
    if coords is None:
        raise ValueError("No field of view could be found in the image.")
    x, y, bw, bh = cv2.boundingRect(coords)
    area = float(cv2.countNonZero(mask))
    perimeter = 0.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        perimeter = float(cv2.arcLength(max(contours, key=cv2.contourArea), True))
    circularity = float(4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    coverage = area / float(h * w)

    crop = image_bgr[y:y + bh, x:x + bw]
    crop_mask = mask[y:y + bh, x:x + bw]
    side = max(bw, bh)
    canvas = np.zeros((side, side, 3), np.uint8)
    canvas_mask = np.zeros((side, side), np.uint8)
    oy, ox = (side - bh) // 2, (side - bw) // 2
    canvas[oy:oy + bh, ox:ox + bw] = crop
    canvas_mask[oy:oy + bh, ox:ox + bw] = crop_mask
    image = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    out_mask = cv2.resize(canvas_mask, (size, size), interpolation=cv2.INTER_NEAREST)
    geometry = {
        "bbox": [int(x), int(y), int(bw), int(bh)],
        "pad": [int(oy), int(ox), int(side)],
        "coverage": round(coverage, 4),
        "circularity": round(min(circularity, 1.0), 4),
        "source_size": [int(w), int(h)],
        "size": int(size),
    }
    return image, out_mask, geometry


def warp_like(array: np.ndarray, geometry: dict, nearest: bool = True) -> np.ndarray:
    """Apply the crop/pad/resize that normalise_fov applied, to another array
    of the same source size (a lesion mask, for instance)."""
    x, y, bw, bh = geometry["bbox"]
    oy, ox, side = geometry["pad"]
    size = geometry["size"]
    crop = array[y:y + bh, x:x + bw]
    shape = (side, side) + array.shape[2:]
    canvas = np.zeros(shape, array.dtype)
    canvas[oy:oy + bh, ox:ox + bw] = crop
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(canvas, (size, size), interpolation=interp)


# ----------------------------------------------------------------- quality --

def quality_features(image512: np.ndarray, mask512: np.ndarray, geometry: dict) -> dict:
    inside = mask512 > 0
    green = image512[:, :, 1]
    gray = cv2.cvtColor(image512, cv2.COLOR_BGR2GRAY)

    lap = cv2.Laplacian(green, cv2.CV_64F)
    sharpness = float(lap[inside].var()) if inside.any() else 0.0

    # Illumination uniformity on a 5x5 grid: darkest periphery cell that lies
    # inside the FOV against the mean of the centre 3x3 cells.
    h, w = gray.shape
    cell_h, cell_w = h // 5, w // 5
    cells = []
    for r in range(5):
        for c in range(5):
            block = gray[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
            block_mask = inside[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
            if block_mask.mean() > 0.6:
                cells.append(((r, c), float(block[block_mask].mean())))
    centre = [v for (r, c), v in cells if 1 <= r <= 3 and 1 <= c <= 3]
    periphery = [v for (r, c), v in cells if not (1 <= r <= 3 and 1 <= c <= 3)]
    if centre and periphery:
        illumination = float(min(periphery) / max(np.mean(centre), 1e-6))
    else:
        illumination = 1.0
    illumination = float(np.clip(illumination, 0.0, 1.5))

    saturated = float(np.mean(gray[inside] >= 245)) if inside.any() else 1.0
    dark = float(np.mean(gray[inside] <= 12)) if inside.any() else 1.0

    return {
        "sharpness": round(sharpness, 2),
        "illumination_uniformity": round(illumination, 3),
        "saturated_fraction": round(saturated, 4),
        "dark_fraction": round(dark, 4),
        "fov_coverage": geometry["coverage"],
        "fov_circularity": geometry["circularity"],
    }


def quality_label(features: dict):
    """(label, score in [0,1], operator-facing retake reason or None).

    Label is "good", "usable" or "reject". The learned quality classifier is not
    part of this build; the label comes from the handcrafted features against
    fixed limits, and the report says so (quality.model).
    """
    L = QUALITY_LIMITS
    reasons_reject, reasons_usable = [], []
    if features["sharpness"] < L["sharpness_reject"]:
        reasons_reject.append("Image too blurry — hold still and refocus")
    elif features["sharpness"] < L["sharpness_usable"]:
        reasons_usable.append("slightly soft focus")
    if features["illumination_uniformity"] < L["illumination_reject"]:
        reasons_reject.append("Uneven illumination — centre the pupil and check the flash")
    elif features["illumination_uniformity"] < L["illumination_usable"]:
        reasons_usable.append("vignetting")
    if features["saturated_fraction"] > L["saturated_reject"]:
        reasons_reject.append("Overexposed — reduce flash intensity")
    elif features["saturated_fraction"] > L["saturated_usable"]:
        reasons_usable.append("bright reflections")
    if features["dark_fraction"] > L["dark_reject"]:
        reasons_reject.append("Underexposed — increase illumination or dilate")
    elif features["dark_fraction"] > L["dark_usable"]:
        reasons_usable.append("dark exposure")
    if features["fov_coverage"] < L["coverage_reject"]:
        reasons_reject.append("Retina fills too little of the frame — move closer or zoom")
    if features["fov_circularity"] < L["circularity_reject"]:
        reasons_reject.append("Partial capture — the field of view is cut off; re-centre")

    # Score: mean of per-feature soft scores, so the operator sees a number
    # move as they fix the capture, not just a label flipping.
    s = [
        np.clip(features["sharpness"] / 40.0, 0, 1),
        np.clip((features["illumination_uniformity"] - 0.2) / 0.6, 0, 1),
        np.clip(1 - features["saturated_fraction"] / 0.15, 0, 1),
        np.clip(1 - features["dark_fraction"] / 0.6, 0, 1),
        np.clip((features["fov_circularity"] - 0.4) / 0.5, 0, 1),
    ]
    score = float(np.mean(s))
    if reasons_reject:
        return "reject", round(score, 3), reasons_reject[0], reasons_usable
    if reasons_usable:
        return "usable", round(score, 3), None, reasons_usable
    return "good", round(score, 3), None, reasons_usable


# ------------------------------------------------------------- enhancement --

def enhance(image512: np.ndarray, mask512: np.ndarray) -> np.ndarray:
    """Adaptive enhancement for images labelled usable.

    1. Illumination correction: divide by a large-kernel Gaussian estimate of
       the illumination (sigma = width / 30) and rescale to the FOV's target
       brightness. Unlike the 4x high-pass the grader's own preprocessing
       uses, this keeps colour and local contrast as they are, so the lesion
       detectors see a normally lit retina rather than an amplified one.
    2. Bilateral filter with a small spatial sigma so microaneurysms survive.
       (CLAHE on green happens inside the Stage 1 detectors.)
    """
    inside = mask512 > 0
    sigma = image512.shape[1] / 30.0
    gray = cv2.cvtColor(image512, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Normalised convolution: blur only what is inside the FOV, so the black
    # surround does not drag the illumination estimate (and hence the gain)
    # to extremes along the rim.
    weight = inside.astype(np.float32)
    illumination = cv2.GaussianBlur(gray * weight, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(weight, (0, 0), sigma), 1e-3)
    illumination[~inside] = 1.0
    target = max(float(np.percentile(gray[inside], 60)), 1.0) if inside.any() else 1.0
    # A dark capture is lifted to a normal exposure; an already-bright one is
    # only flattened, never pushed above a mid-grey target.
    target = float(np.clip(max(target, 90.0), 90.0, 140.0))
    gain = np.clip(target / np.maximum(illumination, 8.0), 0.6, 4.0)
    corrected = np.clip(image512.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)
    # Green-channel CLAHE is applied inside each Stage 1 detector on its own
    # terms; applying it here as well would double it and tint the periphery.
    denoised = cv2.bilateralFilter(corrected, d=5, sigmaColor=25, sigmaSpace=3)
    denoised[~inside] = 0
    return denoised


# --------------------------------------------------------------------- run --

def run(image_bgr: np.ndarray) -> dict:
    """Stage 0 entry. Returns the stage struct plus working arrays.

    Keys:
      accepted        bool — False means stop: modality rejected or quality reject
      modality        gate decision
      quality         label, score, features, reason, enhanced flag
      fov             geometry
      image           512x512 BGR working image (enhanced when usable)
      mask            512x512 FOV mask
      original        512x512 BGR before enhancement (overlays draw on this)
    """
    started = time.perf_counter()
    modality = modality_check(image_bgr)
    if not modality["accepted"]:
        return {
            "accepted": False,
            "stop_reason": modality["reason"],
            "modality": modality,
            "quality": None,
            "fov": None,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }

    image, mask, geometry = normalise_fov(image_bgr)
    features = quality_features(image, mask, geometry)
    label, score, reason, notes = quality_label(features)
    cnn = quality_cnn_probabilities(image)
    if cnn is not None:
        # The learned label decides; the handcrafted hard limits can only
        # override it to reject (with the operator-facing reason).
        learned = max(cnn, key=cnn.get)
        if label != "reject":
            label = learned
            reason = None
            if learned == "reject":
                reason = "Image quality too low to grade (learned quality classifier) — retake"
            score = round(float(cnn["good"] + 0.5 * cnn["usable"]), 3)
    enhanced = label == "usable"
    working = enhance(image, mask) if enhanced else image.copy()
    working[mask == 0] = 0

    quality = {
        "label": label,
        "score": score,
        "features": features,
        "enhanced": enhanced,
        "notes": notes,
        "retake_reason": reason,
        "cnn_probabilities": cnn,
        "model": ("EyeQ-trained quality CNN, overridden to reject by handcrafted hard limits" if cnn is not None
                  else "handcrafted features against fixed limits (quality CNN weights not present)"),
    }
    return {
        "accepted": label != "reject",
        "stop_reason": reason,
        "modality": modality,
        "quality": quality,
        "fov": geometry,
        "image": working,
        "mask": mask,
        "original": image,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }
