"""Stage 3 — explainability and report.

Four kinds of evidence for every graded image, assembled into one page a
clinician can check in under a minute:

    attention map        Grad-CAM on the grader's last conv block, for the
                         referable head and for the predicted grade, clipped
                         to the FOV and overlaid at 40% opacity
    lesion overlay       Stage 1 component outlines on the ORIGINAL image
                         (MA red, HE dark red, EX yellow, SE white), with the
                         optic disc and fovea marked
    clinical criteria    the rule-grader trace: which ICDR criterion fired
    confidence           calibrated P(referable), five-grade bar, abstain flag
    attention agreement  fraction of Grad-CAM mass inside detected lesion
                         pixels, 0..1; low values are flagged

Attention agreement is what makes the heatmap a decision input rather than a
decoration: Grad-CAM shows where the network looked, the lesion masks say
whether that was disease, and the overlap is a number.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from backend.venus.config import ICDR_LABELS, review_policy
from backend.venus.imaging import disk, to_png_base64
from backend.venus.stage2_grade import load_grader

CAM_LAYER = "top_activation"
OVERLAY_ALPHA = 0.40
# Attention agreement: share of Grad-CAM mass inside a neighbourhood of the
# detected lesions. The neighbourhood radius matches one cell of the 12x12
# CAM grid at the 512 working size, because the CAM cannot localise finer
# than that. A case is flagged when the share is low AND no better than
# chance (the neighbourhood's share of the FOV), so a broad heatmap over a
# lesion-covered retina and a heatmap far from a single lesion are told apart.
ATTENTION_RADIUS = 20
ATTENTION_FLOOR = 0.15
ATTENTION_MIN_LIFT = 1.5
_cam_model = None

LESION_COLOURS_BGR = {
    "MA": (0, 0, 255),        # red
    "HE": (0, 0, 140),        # dark red
    "EX": (0, 220, 255),      # yellow
    "SE": (255, 255, 255),    # white
}


# --------------------------------------------------------------- Grad-CAM --

def _cam_function():
    """Build (once) a compiled function returning one CAM per ordinal head.

    Compiling the tape with tf.function takes Grad-CAM from ~2.7 s to ~80 ms
    on a laptop CPU, which is what keeps Stage 3 inside its budget. All four
    heads are computed in the same pass; picking two of them afterwards is
    free.
    """
    import tensorflow as tf
    from tensorflow import keras

    global _cam_model
    if _cam_model is not None:
        return _cam_model
    model = load_grader()
    head = model.output["dr_ordinal_thresholds"] if isinstance(model.output, dict) else model.output
    cam_model = keras.Model(model.inputs, [model.get_layer(CAM_LAYER).output, head])

    @tf.function(reduce_retracing=True)
    @tf.autograph.experimental.do_not_convert
    def cams(inp):
        with tf.GradientTape(persistent=True) as tape:
            conv, out = cam_model(inp, training=False)
            scores = [out[:, k] for k in range(4)]
        maps = []
        for score in scores:
            grads = tape.gradient(score, conv)
            weights = tf.reduce_mean(grads, axis=(1, 2))              # (1, C)
            cam = tf.nn.relu(tf.reduce_sum(conv * weights[:, None, None, :], axis=-1))[0]
            maps.append(cam / (tf.reduce_max(cam) + 1e-8))
        return tf.stack(maps)

    _cam_model = cams
    return cams


def gradcam(grader_input: np.ndarray, targets: list[int]) -> list[np.ndarray]:
    """One heatmap per target ordinal-threshold index (0..3).

    Returns heatmaps in [0, 1] at the conv-block resolution (12x12 for B4 at 380).
    """
    import tensorflow as tf

    fn = _cam_function()
    stack = fn(tf.constant(grader_input[None, ...])).numpy()
    return [stack[t] for t in targets]


def heatmap_to_frame(cam: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Resize a CAM to the 512 working frame and clip it to the FOV."""
    size = mask.shape[0]
    resized = cv2.resize(cam.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    resized = np.clip(resized, 0, 1)
    resized[mask == 0] = 0
    return resized


def overlay_heatmap(image: np.ndarray, heat: np.ndarray, alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    colour = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(image, 1 - alpha, colour, alpha, 0)
    blended[heat <= 0.02] = image[heat <= 0.02]
    return blended


# ------------------------------------------------------------- overlays --

def lesion_overlay(original: np.ndarray, stage1: dict) -> np.ndarray:
    out = original.copy()
    masks = stage1["masks"]
    for key, colour in LESION_COLOURS_BGR.items():
        contours, _ = cv2.findContours(masks[key], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, colour, 2)
    od = stage1["optic_disc"]
    cv2.circle(out, tuple(od["centre"]), int(od["radius"]), (255, 200, 0), 2)
    fx, fy = stage1["fovea"]["centre"]
    cv2.drawMarker(out, (fx, fy), (255, 200, 0), cv2.MARKER_CROSS, 18, 2)
    return out


def vessel_overlay(original: np.ndarray, stage1: dict) -> np.ndarray:
    out = original.copy()
    out[stage1["masks"]["vessels"] > 0] = (0, 200, 0)
    return cv2.addWeighted(original, 0.4, out, 0.6, 0)


# ---------------------------------------------------- attention agreement --

def attention_agreement(heat: np.ndarray, stage1: dict, mask: np.ndarray, counted: int,
                        referable: bool) -> dict:
    """Fraction of Grad-CAM mass that falls inside (dilated) lesion pixels.

    `counted` is the number of lesions the rule grader accepted above its
    component thresholds. The review flag applies only when lesions were
    counted AND the CNN called the image referable: Grad-CAM for a negative
    call has no disease to point at, so a low score there means nothing.
    """
    lesion_union = np.zeros_like(mask)
    for key in LESION_COLOURS_BGR:
        lesion_union = cv2.bitwise_or(lesion_union, stage1["masks"][key])
    lesion_pixels = int(cv2.countNonZero(lesion_union))
    if lesion_pixels == 0:
        return {"score": None, "lesion_pixels": 0, "flag": False,
                "note": "no lesions detected, so agreement is undefined"}
    wide = cv2.dilate(lesion_union, disk(ATTENTION_RADIUS))
    total = float(heat[mask > 0].sum())
    inside = float(heat[wide > 0].sum())
    score = inside / total if total > 1e-6 else 0.0
    # Chance level: the share of FOV the dilated lesions occupy. Reported so a
    # reader can tell 0.30 on a lesion-covered retina from 0.30 on one lesion.
    chance = float(cv2.countNonZero(cv2.bitwise_and(wide, mask)) / max(cv2.countNonZero(mask), 1))
    lift = score / chance if chance > 0 else 0.0
    policy = review_policy()
    floor = float(policy.get("attention_floor", ATTENTION_FLOOR))
    min_lift = policy.get("attention_min_lift", ATTENTION_MIN_LIFT)
    # Validation-chosen policy: a plain floor (the lift condition is dropped,
    # the floor already selects calls whose error rate is >= 60%).
    low = score < floor and (min_lift is None or lift < min_lift)
    applies = counted > 0 and referable
    return {
        "score": round(score, 3),
        "chance_level": round(chance, 3),
        "lift": round(lift, 2),
        "floor": floor,
        "lesion_pixels": lesion_pixels,
        "flag": bool(low and applies),
        "note": ("attention not on lesions" if (low and applies)
                 else "only sub-threshold detections; not used for review" if counted == 0
                 else "not referable by CNN; heatmap not used for review" if not referable
                 else "attention overlaps lesion evidence"),
    }


# ------------------------------------------------------------------ run --

def run(stage0: dict, stage1: dict, stage2: dict) -> dict:
    started = time.perf_counter()
    mask = stage0["mask"]
    original = stage0["original"]
    grade = stage2["cnn"]["grade"]
    # Target indices in the ordinal head: 1 is P(grade >= 2) (referable);
    # the grade target is the threshold the prediction most recently crossed.
    grade_target = max(grade - 1, 0)
    targets = [1] if grade_target == 1 else [1, grade_target]
    cams = gradcam(stage2["grader_input"], targets)
    cam_ms = int((time.perf_counter() - started) * 1000)

    # v2 sees the working frame itself, so its CAM resizes straight onto it.
    # v1 saw a bounding-box crop of the raw image; the working frame is a padded
    # square of the FOV, and both are axis-aligned crops of the same circle, so
    # mapping the CAM onto the FOV bounding box in the working frame aligns them.
    def to_working(cam):
        if stage2.get("input_frame", "raw_bbox") == "working":
            return heatmap_to_frame(cam, mask)
        coords = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(coords)
        frame = np.zeros(mask.shape, np.float32)
        frame[y:y + h, x:x + w] = cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        frame = np.clip(frame, 0, 1)
        frame[mask == 0] = 0
        return frame

    heat_referable = to_working(cams[0])
    heat_grade = to_working(cams[1]) if len(cams) > 1 else heat_referable

    counted = sum(stage2["rule"]["counts"].values())
    agreement = attention_agreement(heat_referable, stage1, mask, counted, stage2["fusion"]["referable"])
    overlays = {
        "gradcam_referable": to_png_base64(overlay_heatmap(original, heat_referable)),
        "gradcam_grade": to_png_base64(overlay_heatmap(original, heat_grade)),
        "lesions": to_png_base64(lesion_overlay(original, stage1)),
        "vessels": to_png_base64(vessel_overlay(original, stage1)),
        "original": to_png_base64(original),
        "enhanced": to_png_base64(stage0["image"]),
    }

    rule = stage2["rule"]
    fusion = stage2["fusion"]
    criteria_text = (f"Grade {rule['grade']} ({ICDR_LABELS[rule['grade']]}) by ICDR criteria: "
                     f"{rule['criteria_text']}")
    return {
        "gradcam_layer": CAM_LAYER,
        "gradcam_targets": {"referable": "P(grade >= 2)", "grade": f"P(grade >= {grade_target + 1})"},
        "attention_agreement": agreement,
        "criteria_text": criteria_text,
        "confidence_text": fusion["confidence_text"],
        "overlays": overlays,
        "heat_referable": heat_referable,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "gradcam_ms": cam_ms,
    }
