"""Stage 1 — retinal structure segmentation.

Turns the Stage 0 working image into lesion maps and landmarks. Two consumers:
the rule grader in Stage 2 (counts per lesion type, hemorrhages per ETDRS
quadrant, P(NV)) and the overlays in Stage 3.

This build uses the classical, no-GPU path the architecture names as the
fallback for every structure: brightest-blob optic disc, geometric fovea prior,
Frangi vesselness for vessels, and morphological top-hat / black-hat lesion
detection. No pixel-level lesion network is included — the only public
pixel-labelled DR set (IDRiD) has 81 images and a U-Net trained on it is not
something this build can validate — so every output carries method="classical"
and the report states it. Neovascularization is never segmented; Stage 2 uses
the grader's P(grade >= 4) as PDR evidence and says "classifier, not localised".

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

from backend.venus.config import LESION_THRESHOLDS_PATH, UNET_WEIGHTS

_unet = None
_unet_thresholds: dict | None = None
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
# Red-lesion size split: components at or above this area are hemorrhages.
MA_MAX_AREA = 40


# ------------------------------------------------------------ optic disc --

def optic_disc(image: np.ndarray, mask: np.ndarray) -> dict:
    """Brightest smooth blob in the luminance, away from the FOV rim."""
    h, w = mask.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # The disc is bright in all channels; the red channel saturates on it so
    # the mean of green and red works better than either alone.
    lum = 0.5 * image[:, :, 1].astype(np.float32) + 0.5 * image[:, :, 2].astype(np.float32)
    smooth = cv2.GaussianBlur(lum, (0, 0), 12)
    inner = cv2.erode(mask, disk(10))
    smooth[inner == 0] = -1
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
    global _unet, _unet_thresholds, _unet_error
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
            model = nets.lesion_unet()
            model.load_weights(UNET_WEIGHTS)
            with open(LESION_THRESHOLDS_PATH, "r", encoding="utf-8") as handle:
                _unet_thresholds = json.load(handle)["thresholds"]
            _unet = model
        except Exception as exc:  # pragma: no cover
            _unet_error = f"{type(exc).__name__}: {exc}"
            return None
    return _unet


def unet_status() -> dict:
    return {"loaded": _unet is not None, "error": _unet_error, "thresholds": _unet_thresholds}


def unet_lesions(image: np.ndarray, mask: np.ndarray, od_mask: np.ndarray):
    """Per-lesion probability maps from the U-Net, thresholded at the values
    chosen on the DDR valid split, with the same rim/disc post-processing the
    classical path applies. Returns {key: (mask, components)}."""
    model = load_unet()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    probs = model.predict(rgb[None, ...], verbose=0)[0]
    inner = cv2.erode(mask, disk(6))
    od_wide = cv2.dilate(od_mask, disk(6))
    out = {}
    for c, key in enumerate(LESION_TYPES):
        binary = ((probs[:, :, c] >= float(_unet_thresholds[key])) & (inner > 0)).astype(np.uint8) * 255
        if key in ("EX", "SE"):
            binary[od_wide > 0] = 0          # the disc rim is the classic exudate false positive
        labels, comps = _components(binary, MIN_AREA[key])
        keep = np.zeros_like(mask)
        for comp in comps:
            keep[labels == comp["label"]] = 255
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

def run(image: np.ndarray, mask: np.ndarray) -> dict:
    started = time.perf_counter()
    fov_area = max(cv2.countNonZero(mask), 1)
    od = optic_disc(image, mask)
    fov = fovea(image, mask, od)
    ves = vessels(image, mask)
    if load_unet() is not None:
        method = "unet"
        found = unet_lesions(image, mask, od["mask"])
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
