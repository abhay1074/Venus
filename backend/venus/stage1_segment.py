"""Stage 1 — retinal structure segmentation.

Turns the Stage 0 working image into lesion maps and landmarks. Two consumers:
the rule grader in Stage 2 (counts per lesion type, hemorrhages per ETDRS
quadrant, P(NV)) and the overlays in Stage 3.

Landmarks always come from the classical, no-GPU path the architecture names:
brightest-blob optic disc, geometric fovea prior, Frangi vesselness for vessels.

Lesions have two paths, and every result carries `method` saying which ran:

- "unet" when `backend/weights/lesion_unet.weights.h5` and
  `config/lesion_thresholds.json` are present. A U-Net (4 levels, 32 base
  filters, 512x512, four sigmoid channels MA/HE/EX/SE) trained on the DDR
  lesion-segmentation set (383 train / 149 valid / 225 test), applied to the
  un-enhanced frame after LAB colour normalisation to the DDR statistics, with
  per-class thresholds chosen on the DDR valid split (max pixel F1) and the
  test split scored once — see docs/VALIDATION.md. The same rim/disc
  post-processing and component-area floors as the classical path apply.
  `config.LESION_THRESHOLDS_HIRES_PATH` can add a second network at a larger
  frame for the classes it names; no such file ships (see
  config/experiments/lesion_unet_1024.json for the measurement behind that).
- "classical" otherwise: morphological black-hat (red lesions) and top-hat
  (bright lesions) with robust MAD thresholds, so the pipeline still produces
  lesion evidence on a machine with no checkpoints.

Neovascularization is never segmented; Stage 2 uses the grader's
P(grade >= 4) as PDR evidence and says "classifier, not localised".

All coordinates are in the 512x512 working frame.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from skimage.filters import frangi

from backend.venus.imaging import clahe, disk

import json
import threading

from backend.venus.config import (CONFIG_DIR, LESION_THRESHOLDS_HIRES_PATH, LESION_THRESHOLDS_PATH, UNET_HIRES_WEIGHTS,
                                  UNET_WEIGHTS)

COLOUR_REFERENCE_PATH = CONFIG_DIR / "unet_colour_reference.json"
_colour_reference: dict | None = None

_unet = None
_unet_thresholds: dict | None = None
_unet_frame_size = 512
_hires = None                      # second network, larger frame, some classes only
_hires_thresholds: dict | None = None
_hires_frame_size = 1024
_hires_serves: list[str] = []
_hires_note: str | None = None
_unet_lock = threading.Lock()
_unet_error: str | None = None

LESION_TYPES = ["MA", "HE", "EX", "SE"]
LESION_NAMES = {
    "MA": "microaneurysms",
    "HE": "hemorrhages",
    "EX": "hard exudates",
    "SE": "soft exudates",
}
# Minimum component area (px at 512) per lesion type; below this it is noise.
MIN_AREA = {"MA": 4, "HE": 15, "EX": 5, "SE": 60}
# At 1024 px, ground-truth microaneurysms on DDR valid have a median area of
# 17 px and a 10th percentile of ~6 px (2,555 components); 6 px keeps the
# small ones the larger frame exists for while dropping single-pixel speckle.
MIN_AREA_HIRES = {"MA": 6}
# Red-lesion size split: components at or above this area are hemorrhages.
MA_MAX_AREA = 40


# ------------------------------------------------------------ optic disc --

# Retina, including the palest optic disc, is chromatic; printed labels, text
# and specular glare are not. Measured chroma, (max-min)/max over a 20 px patch:
# discs 0.30-0.63 on the shipped samples, a white image label 0.00.
ACHROMATIC_CHROMA = 0.12


def optic_disc(image: np.ndarray, mask: np.ndarray, original: np.ndarray | None = None) -> dict:
    """Brightest smooth chromatic blob after flat-fielding, away from the FOV rim.

    Three corrections to a brightest-blob detector, each measured
    (config/experiments/landmark_check.json, clinician-marked fovea centres on
    1,008 MESSIDOR images, design / held-out halves):

    - Located on `original`, the un-enhanced frame. Stage 0's enhancement
      divides out illumination at sigma = width/30 (~17 px), smaller than a
      disc (~70 px across), so on enhanced frames - 96 % of real images - the
      disc was no longer the brightest region.
    - Flat-fielded at sigma = FOV/4 instead: that removes vignetting and uneven
      exposure, which is much larger than a disc, while the disc stays a
      local peak.
    - Achromatic pixels excluded: image labels, text and glare are grey or
      white, and a white "A" in a corner is brighter than any disc.

    Held out: fovea within one disc diameter of the clinician's mark on 99.8 %
    of images, against 83.7 % for the previous detector. Known hard case:
    neovascularization at the disc hides its brightness, and a bright fibrous
    patch elsewhere can win.
    """
    src = image if original is None else original
    h, w = mask.shape
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32)
    channels = src.astype(np.float32)
    top, bottom = channels.max(axis=2), channels.min(axis=2)
    achromatic = ((top - bottom) / np.maximum(top, 1.0) < ACHROMATIC_CHROMA) & (top > 60) & (mask > 0)
    achromatic = cv2.dilate(achromatic.astype(np.uint8) * 255, disk(6)) > 0
    # The disc is bright in all channels; the red channel saturates on it so
    # the mean of green and red works better than either alone.
    lum = 0.5 * channels[:, :, 1] + 0.5 * channels[:, :, 2]
    lum[achromatic] = 0.0                 # no bleed of a label's brightness into its neighbours
    # Flat-field by normalised convolution inside the FOV (the black surround
    # must not drag the illumination estimate down along the rim).
    inside = (mask > 0).astype(np.float32)
    fov_d = 2 * np.sqrt(max(inside.sum(), 1.0) / np.pi)
    illumination = (cv2.GaussianBlur(lum * inside, (0, 0), fov_d / 4)
                    / np.maximum(cv2.GaussianBlur(inside, (0, 0), fov_d / 4), 1e-3))
    lum = lum / np.maximum(illumination, 8.0)
    smooth = cv2.GaussianBlur(lum, (0, 0), 12)
    inner = cv2.erode(mask, disk(10))
    smooth[inner == 0] = -1
    smooth[achromatic] = -1
    gray[achromatic] = 0.0
    cy, cx = np.unravel_index(int(np.argmax(smooth)), smooth.shape)
    # Radius: a disc spans roughly a sixth to a seventh of the FOV diameter.
    fov_diameter = 2 * np.sqrt(cv2.countNonZero(mask) / np.pi)
    radius = float(fov_diameter / 13.0)
    # Refine with a bright-region fit around the candidate.
    local = np.zeros_like(mask)
    cv2.circle(local, (int(cx), int(cy)), int(radius * 2.2), 255, -1)
    region = gray.copy()
    region[local == 0] = 0
    thresh = np.percentile(gray[(local > 0) & (mask > 0)], 90) if (local > 0).any() else 255
    bright = ((region >= thresh) & (local > 0)).astype(np.uint8) * 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, disk(4))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > np.pi * (radius * 0.4) ** 2:
            (fx, fy), fr = cv2.minEnclosingCircle(largest)
            if 0.6 * radius < fr < 1.8 * radius:
                cx, cy, radius = fx, fy, fr
    od_mask = np.zeros_like(mask)
    cv2.circle(od_mask, (int(cx), int(cy)), int(radius), 255, -1)
    return {
        "centre": [int(cx), int(cy)],
        "radius": round(float(radius), 1),
        "mask": od_mask,
        "method": "brightest-blob prior with local bright-region refinement",
    }


# ----------------------------------------------------------------- fovea --

def fovea(image: np.ndarray, mask: np.ndarray, od: dict) -> dict:
    """Darkest smooth region 2-3 disc diameters from the disc, near horizontal.

    Which side is temporal is unknown for a single image, so both sides are
    searched and the darker candidate wins; the confidence records the
    contrast between them.
    """
    cx, cy = od["centre"]
    dd = 2.0 * od["radius"]
    green = image[:, :, 1].astype(np.float32)
    smooth = cv2.GaussianBlur(green, (0, 0), 10)
    # The rim is dark too: erode by a full disc diameter so the annulus never
    # touches it, and penalise distance from the FOV centre because the macula
    # sits near the centre of a macula-centred or disc-centred 45-degree field.
    inner = cv2.erode(mask, disk(max(int(dd), 20)))
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(xx - cx, yy - cy)
    band = (dist > 1.8 * dd) & (dist < 3.2 * dd) & (np.abs(yy - cy) < 0.9 * dd) & (inner > 0)
    moments = cv2.moments(mask, binaryImage=True)
    mcx, mcy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    centre_penalty = 0.12 * np.hypot(xx - mcx, yy - mcy)
    smooth = smooth + centre_penalty.astype(np.float32)
    if not band.any():
        # Geometric prior: 2.5 DD towards the image centre.
        direction = 1 if cx < w / 2 else -1
        return {"centre": [int(cx + direction * 2.5 * dd), int(cy)], "confidence": 0.0,
                "method": "geometric prior"}
    candidates = np.where(band, smooth, np.inf)
    fy, fx = np.unravel_index(int(np.argmin(candidates)), candidates.shape)
    contrast = float(np.median(smooth[band]) - smooth[fy, fx])
    confidence = float(np.clip(contrast / 12.0, 0, 1))
    return {"centre": [int(fx), int(fy)], "confidence": round(confidence, 3),
            "method": "darkest blob in a 2-3 DD horizontal annulus"}


# --------------------------------------------------------------- vessels --

def vessels(image: np.ndarray, mask: np.ndarray) -> dict:
    """Frangi vesselness on the inverted green channel, thresholded."""
    green = clahe(image[:, :, 1], clip=2.5, tile=8).astype(np.float32) / 255.0
    inner = cv2.erode(mask, disk(6))
    response = frangi(green, sigmas=(1.0, 1.5, 2.0, 3.0, 4.5), black_ridges=True, gamma=15)
    response = np.nan_to_num(response)
    response[inner == 0] = 0
    inside = response[inner > 0]
    if inside.size == 0:
        return {"mask": np.zeros_like(mask), "fraction": 0.0, "method": "frangi"}
    thresh = np.percentile(inside, 88)
    vessel = ((response >= max(thresh, 1e-6)) & (inner > 0)).astype(np.uint8) * 255
    vessel = cv2.morphologyEx(vessel, cv2.MORPH_OPEN, disk(1))
    # Drop specks: real vessel segments are elongated and connected.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(vessel, connectivity=8)
    keep = np.zeros_like(vessel)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= 30:
            keep[labels == i] = 255
    fraction = float(cv2.countNonZero(keep) / max(cv2.countNonZero(mask), 1))
    return {"mask": keep, "fraction": round(fraction, 4), "method": "Frangi vesselness (sigmas 1-4.5), 88th-percentile threshold"}


# --------------------------------------------------------------- lesions --

def _robust_threshold(values: np.ndarray, k: float, floor: float) -> float:
    """median + k * MAD: the optic disc and vessel reflexes inflate a mean/std
    threshold until faint lesions fall under it; the median and MAD do not care."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826
    return max(median + k * max(mad, 1.0), floor)


def _elongation(region: np.ndarray) -> float:
    """Ratio of principal axes from second-order moments (1 = round)."""
    m = cv2.moments(region.astype(np.uint8), binaryImage=True)
    if m["m00"] == 0:
        return 1.0
    mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
    common = np.sqrt(max((mu20 - mu02) ** 2 + 4 * mu11 ** 2, 0))
    l1, l2 = (mu20 + mu02 + common) / 2, (mu20 + mu02 - common) / 2
    return float(np.sqrt(l1 / max(l2, 1e-6)))


def _components(binary: np.ndarray, min_area: int):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    comps = []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            comps.append({
                "label": i,
                "area": area,
                "centroid": [int(centroids[i][0]), int(centroids[i][1])],
                "bbox": [int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                         int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])],
            })
    return labels, comps


def red_lesions(image: np.ndarray, mask: np.ndarray, vessel_mask: np.ndarray, od_mask: np.ndarray):
    """Microaneurysms and hemorrhages: dark blobs that are not vessels.

    Black-hat on the CLAHE green channel with a disc large enough to hold a
    hemorrhage responds to any locally dark structure. Vessels are the main
    false positive, so the (dilated) vessel map is subtracted, and what is
    left is split by size: small round components are MA, larger ones HE.
    """
    green = clahe(image[:, :, 1], clip=2.5, tile=8)
    green = cv2.medianBlur(green, 3)
    inner = cv2.erode(mask, disk(10))
    blackhat = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, disk(9))
    blackhat[inner == 0] = 0
    values = blackhat[inner > 0]
    if values.size == 0:
        return np.zeros_like(mask), [], np.zeros_like(mask), []
    # Adaptive threshold: well above the background texture.
    thresh = _robust_threshold(values, 8.0, 14.0)
    candidate = ((blackhat >= thresh) & (inner > 0)).astype(np.uint8) * 255
    vessel_wide = cv2.dilate(vessel_mask, disk(4))
    candidate[vessel_wide > 0] = 0
    candidate[cv2.dilate(od_mask, disk(8)) > 0] = 0
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, disk(1))

    vessel_near = cv2.dilate(vessel_mask, disk(5)) > 0
    labels, comps = _components(candidate, MIN_AREA["MA"])
    ma_mask = np.zeros_like(mask)
    he_mask = np.zeros_like(mask)
    ma, he = [], []
    for comp in comps:
        region = labels == comp["label"]
        elongation = _elongation(region)
        if comp["area"] < MA_MAX_AREA and elongation <= 2.2:
            ma_mask[region] = 255
            ma.append(comp)
        elif comp["area"] >= MIN_AREA["HE"]:
            # Long thin residues, or blobs that mostly sit on the vessel map,
            # are vessel fragments the Frangi threshold missed.
            overlap = float(vessel_near[region].mean())
            if elongation > 3.5 or overlap > 0.35:
                continue
            he_mask[region] = 255
            he.append(comp)
    return ma_mask, ma, he_mask, he


def bright_lesions(image: np.ndarray, mask: np.ndarray, od_mask: np.ndarray,
                   vessel_mask: np.ndarray | None = None):
    """Hard exudates (sharp, saturated yellow-white) and soft exudates (fuzzy).

    Top-hat on the green channel finds locally bright structures; the optic
    disc is the dominant false positive so a dilated disc mask is excluded.
    Sharpness of the component boundary separates hard from soft exudates.
    """
    green = clahe(image[:, :, 1], clip=2.0, tile=8)
    inner = cv2.erode(mask, disk(22))
    tophat = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, disk(9))
    tophat[inner == 0] = 0
    od_wide = cv2.dilate(od_mask, disk(14))
    values = tophat[(inner > 0) & (od_wide == 0)]
    if values.size == 0:
        return np.zeros_like(mask), [], np.zeros_like(mask), []
    thresh = _robust_threshold(values, 4.5, 18.0)
    candidate = ((tophat >= thresh) & (inner > 0)).astype(np.uint8) * 255
    candidate[od_wide > 0] = 0
    # Arterial light reflexes are bright ridges that run along vessels; they
    # are the dominant false positive on a healthy young retina.
    if vessel_mask is not None:
        candidate[cv2.dilate(vessel_mask, disk(3)) > 0] = 0
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, disk(1))

    # Exudates are yellow lipid; vessel reflexes and nerve-fibre sheen are
    # white. LAB b* against a local background separates the two.
    lab_b = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 2].astype(np.float32)
    lab_b_bg = cv2.GaussianBlur(lab_b, (0, 0), 15)
    yellowness_map = lab_b - lab_b_bg

    # Edge strength per component decides hard vs soft.
    edges = cv2.Sobel(green.astype(np.float32), cv2.CV_32F, 1, 0) ** 2 + \
        cv2.Sobel(green.astype(np.float32), cv2.CV_32F, 0, 1) ** 2
    edges = np.sqrt(edges)
    labels, comps = _components(candidate, MIN_AREA["EX"])
    ex_mask = np.zeros_like(mask)
    se_mask = np.zeros_like(mask)
    ex, se = [], []
    for comp in comps:
        region = labels == comp["label"]
        if _elongation(region) > 3.0:
            continue  # ridge-shaped: a reflex or vessel fragment, not an exudate
        yellowness = float(yellowness_map[region].mean())
        if yellowness < 2.5:
            continue  # white, not yellow: reflex or sheen
        comp["yellowness"] = round(yellowness, 2)
        boundary = cv2.dilate(region.astype(np.uint8), disk(1)) - region.astype(np.uint8)
        edge_strength = float(edges[boundary > 0].mean()) if boundary.any() else 0.0
        if comp["area"] >= MIN_AREA["SE"] and edge_strength < 22.0:
            se_mask[region] = 255
            se.append(comp)
        else:
            ex_mask[region] = 255
            ex.append(comp)
    return ex_mask, ex, se_mask, se


# ----------------------------------------------------------------- U-Net --

def load_unet():
    """Load the lesion U-Net once, if its weights and thresholds are present."""
    global _unet, _unet_thresholds, _unet_error, _unet_frame_size
    if _unet is not None:
        return _unet
    with _unet_lock:
        if _unet is not None:
            return _unet
        if not (UNET_WEIGHTS.exists() and LESION_THRESHOLDS_PATH.exists()):
            _unet_error = "lesion U-Net weights or thresholds not present; classical detectors in use"
            return None
        try:
            from backend.venus import nets
            with open(LESION_THRESHOLDS_PATH, "r", encoding="utf-8") as handle:
                thresholds = json.load(handle)
            # The frame size the network was trained on travels with its thresholds.
            _unet_frame_size = int(thresholds.get("frame_size", 512))
            model = nets.lesion_unet(size=_unet_frame_size)
            model.load_weights(UNET_WEIGHTS)
            _unet_thresholds = thresholds["thresholds"]
            _load_hires(nets)
            _unet = model
        except Exception as exc:  # pragma: no cover
            _unet_error = f"{type(exc).__name__}: {exc}"
            return None
    return _unet


def _load_hires(nets) -> None:
    """The optional larger-frame network; a failure here leaves the 512 px
    network reading every class and says so in the status."""
    global _hires, _hires_thresholds, _hires_frame_size, _hires_serves, _hires_note
    if not (UNET_HIRES_WEIGHTS.exists() and LESION_THRESHOLDS_HIRES_PATH.exists()):
        _hires_note = "no larger-frame lesion network; the 512 px network reads every class"
        return
    try:
        with open(LESION_THRESHOLDS_HIRES_PATH, "r", encoding="utf-8") as handle:
            spec = json.load(handle)
        serves = [k for k in spec.get("serves", []) if k in LESION_TYPES]
        if not serves:
            _hires_note = "larger-frame network present but serves no class"
            return
        model = nets.lesion_unet(size=int(spec.get("frame_size", 1024)))
        model.load_weights(UNET_HIRES_WEIGHTS)
        _hires_frame_size = int(spec.get("frame_size", 1024))
        _hires_thresholds = {k: float(spec["thresholds"][k]) for k in serves}
        _hires_serves = serves
        _hires = model
        _hires_note = f"{', '.join(serves)} read at {_hires_frame_size} px ({spec.get('tag', 'hires')})"
    except Exception as exc:  # pragma: no cover
        _hires_note = f"larger-frame network not loaded: {type(exc).__name__}: {exc}"


def unet_status() -> dict:
    return {"loaded": _unet is not None, "error": _unet_error, "thresholds": _unet_thresholds,
            "frame_size": _unet_frame_size if _unet is not None else None,
            "hires": {"loaded": _hires is not None, "frame_size": _hires_frame_size if _hires is not None else None,
                      "serves": list(_hires_serves), "thresholds": _hires_thresholds, "note": _hires_note}}


def lesion_method() -> str:
    if _unet is None:
        return "classical"
    if _hires is None:
        return "unet"
    return f"unet ({'/'.join(_hires_serves)} at {_hires_frame_size} px)"


def colour_normalise(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Match the FOV's per-channel LAB mean and std to the DDR training
    statistics (config/unet_colour_reference.json).

    The U-Net learned DDR's cameras; a very red or very pale retina from
    another device produced spurious hemorrhages (13 on a healthy CC0 image)
    that this removes without touching real lesions. It is applied to the
    un-enhanced frame, because the network was trained on raw frames."""
    global _colour_reference
    if _colour_reference is None:
        if not COLOUR_REFERENCE_PATH.exists():
            return image
        with open(COLOUR_REFERENCE_PATH, "r", encoding="utf-8") as handle:
            _colour_reference = json.load(handle)
    ref_mean, ref_std = _colour_reference["lab_mean"], _colour_reference["lab_std"]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    inside = mask > 0
    for c in range(3):
        channel = lab[:, :, c]
        mu, sd = float(channel[inside].mean()), float(channel[inside].std())
        lab[:, :, c] = (channel - mu) / max(sd, 1e-3) * ref_std[c] + ref_mean[c]
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    out[~inside] = 0
    return out


def unet_lesions(image: np.ndarray, mask: np.ndarray, od_mask: np.ndarray, raw: np.ndarray | None = None):
    """Per-lesion probability maps from the U-Net on the colour-normalised
    frame, thresholded at the values chosen on the DDR valid split, with the
    same rim/disc post-processing the classical path applies.

    A network trained on larger frames (microaneurysms are 1-3 px at 512)
    gets the same FOV normalisation of the raw upload at its own size and its
    probabilities are averaged back down to the 512 working frame, so every
    overlay and count stays in working coordinates. When the optional
    larger-frame network is loaded, the classes it serves take their
    probability channel and threshold from it.
    Returns {key: (mask, components)}."""
    model = load_unet()
    size = image.shape[0]
    thresholds = dict(_unet_thresholds)
    if _unet_frame_size != size and raw is not None:
        from backend.venus.stage0_gate import normalise_fov
        hi_image, hi_mask, _ = normalise_fov(raw, _unet_frame_size)
        rgb = cv2.cvtColor(colour_normalise(hi_image, hi_mask), cv2.COLOR_BGR2RGB).astype(np.float32)
        probs = model.predict(rgb[None, ...], verbose=0)[0]
        probs = cv2.resize(probs, (size, size), interpolation=cv2.INTER_AREA)
    else:
        if _unet_frame_size != size:
            image = cv2.resize(image, (_unet_frame_size, _unet_frame_size), interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(mask, (_unet_frame_size, _unet_frame_size), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(colour_normalise(image, mask), cv2.COLOR_BGR2RGB).astype(np.float32)
        probs = model.predict(rgb[None, ...], verbose=0)[0]
        if probs.shape[0] != size:
            probs = cv2.resize(probs, (size, size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    inner = cv2.erode(mask, disk(6))
    od_wide = cv2.dilate(od_mask, disk(6))

    def extract(prob, key, threshold, inner_mask, od_mask_wide, scale):
        binary = ((prob >= threshold) & (inner_mask > 0)).astype(np.uint8) * 255
        if key in ("EX", "SE"):
            binary[od_mask_wide > 0] = 0     # the disc rim is the classic exudate false positive
        labels, comps = _components(binary, MIN_AREA_HIRES.get(key, MIN_AREA[key] * scale * scale) if scale > 1 else MIN_AREA[key])
        keep = np.zeros(binary.shape, np.uint8)
        for comp in comps:
            keep[labels == comp["label"]] = 255
        return keep, comps

    out = {}
    for c, key in enumerate(LESION_TYPES):
        out[key] = extract(probs[:, :, c], key, float(thresholds[key]), inner, od_wide, 1)

    if _hires is not None and raw is not None:
        # Served classes are thresholded and counted at the network's own
        # resolution (a 1-2 px microaneurysm would not survive averaging to
        # 512), then mask and centroids come back to the working frame.
        from backend.venus.stage0_gate import normalise_fov
        hi_image, hi_mask, _ = normalise_fov(raw, _hires_frame_size)
        hi_rgb = cv2.cvtColor(colour_normalise(hi_image, hi_mask), cv2.COLOR_BGR2RGB).astype(np.float32)
        hi_probs = _hires.predict(hi_rgb[None, ...], verbose=0)[0]
        scale = _hires_frame_size // size
        hi_inner = cv2.erode(hi_mask, disk(6 * scale))
        hi_od = cv2.dilate(cv2.resize(od_mask, (_hires_frame_size, _hires_frame_size), interpolation=cv2.INTER_NEAREST), disk(6 * scale))
        for key in _hires_serves:
            c = LESION_TYPES.index(key)
            keep, comps = extract(hi_probs[:, :, c], key, _hires_thresholds[key], hi_inner, hi_od, scale)
            keep = (cv2.resize(keep, (size, size), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8) * 255
            for comp in comps:
                comp["centroid"] = [comp["centroid"][0] // scale, comp["centroid"][1] // scale]
                comp["bbox"] = [v // scale for v in comp["bbox"]]
                comp["area"] = max(comp["area"] // (scale * scale), 1)
            out[key] = (keep, comps)
    return out


# ------------------------------------------------------------- quadrants --

def quadrant_counts(components: list, fovea_centre: list, size: int) -> list:
    """Hemorrhages per ETDRS quadrant, split by the diagonals through the fovea.

    Quadrant order: superior, temporal-or-nasal (right), inferior, left.
    """
    fx, fy = fovea_centre
    counts = [0, 0, 0, 0]
    for comp in components:
        x, y = comp["centroid"]
        dx, dy = x - fx, y - fy
        if abs(dx) >= abs(dy):
            counts[1 if dx >= 0 else 3] += 1
        else:
            counts[2 if dy >= 0 else 0] += 1
    return counts


# ------------------------------------------------------------------- run --

def run(image: np.ndarray, mask: np.ndarray, original: np.ndarray | None = None, raw: np.ndarray | None = None) -> dict:
    """`image` is the Stage 0 working frame (enhanced when usable) for the
    landmark and classical detectors; `original` is the un-enhanced frame the
    U-Net sees (it was trained on raw frames). Defaults to `image`."""
    started = time.perf_counter()
    original = image if original is None else original
    fov_area = max(cv2.countNonZero(mask), 1)
    od = optic_disc(image, mask, original=original)
    fov = fovea(image, mask, od)
    ves = vessels(image, mask)
    if load_unet() is not None:
        method = lesion_method() if raw is not None else "unet"
        found = unet_lesions(original, mask, od["mask"], raw=raw)
        ma_mask, ma = found["MA"]; he_mask, he = found["HE"]; ex_mask, ex = found["EX"]; se_mask, se = found["SE"]
    else:
        method = "classical"
        ma_mask, ma, he_mask, he = red_lesions(image, mask, ves["mask"], od["mask"])
        ex_mask, ex, se_mask, se = bright_lesions(image, mask, od["mask"], ves["mask"])

    def summary(comps, lesion_mask):
        return {
            "count": len(comps),
            "area_fraction": round(float(cv2.countNonZero(lesion_mask) / fov_area), 5),
            "centroids": [c["centroid"] for c in comps[:200]],
        }

    lesions = {
        "MA": summary(ma, ma_mask),
        "HE": summary(he, he_mask),
        "EX": summary(ex, ex_mask),
        "SE": summary(se, se_mask),
    }
    quadrants = quadrant_counts(he, fov["centre"], mask.shape[0])
    return {
        "method": method,
        "optic_disc": {"centre": od["centre"], "radius": od["radius"], "method": od["method"]},
        "fovea": fov,
        "vessels": {"fraction": ves["fraction"], "method": ves["method"]},
        "lesions": lesions,
        "hemorrhages_per_quadrant": quadrants,
        "neovascularization": {
            "segmented": False,
            "note": "not segmented; Stage 2 reports P(NV) from the grader as PDR evidence",
        },
        "masks": {
            "optic_disc": od["mask"], "vessels": ves["mask"],
            "MA": ma_mask, "HE": he_mask, "EX": ex_mask, "SE": se_mask,
        },
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }
