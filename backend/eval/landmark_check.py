"""Optic-disc detectors, refereed by clinician-marked fovea centres.

    wsl bash scripts/wsl-gpu.sh backend.eval.landmark_check --labels <data.xls>

Ground truth: the University of Huelva annotations for the MESSIDOR images
(Gegundez-Arias et al., "A Method for Locating Fovea Center Position in Digital
Retinal Images"), https://www.uhu.es/retinopathy/ -> Provided_Information.zip:
for 1,136 images, a clinician's fovea centre and the standard optic-disc
diameter, in original pixel coordinates. The images are the MESSIDOR ones
inside our Messidor-2 folder.

Why fovea labels referee a disc detector: Stage 1 places the fovea 2-3 disc
diameters from the disc it found, so a disc found on an image label, an
exudate or a vignette puts the fovea in the wrong place, and the ETDRS
hemorrhage quadrants are drawn around that fovea. Fovea error is therefore
the end-to-end cost of a wrong disc. A second, disc-only check: the true
disc-fovea distance is about 2.5 disc diameters, so a disc between 1.5 and
3.5 DD from the true fovea is anatomically plausible.

Scope: Messidor-2 is a frozen external test set for the *grader*; what was
scored on it (AUC, sensitivity, specificity, calibration) involves no landmark,
so nothing recorded is re-scored here. To keep the landmark choice honest the
annotated images are split in two by a hash of the file name: the one free
design parameter (the vessel weight) is chosen on the design half, and the
numbers reported are from the held-out half.

Detectors (all designs fixed before this was run):
  old        brightest blob on the enhanced working frame (the pre-fix code)
  original   brightest blob on the un-enhanced frame, achromatic pixels excluded
  flatfield  as `original`, after dividing out illumination at sigma = FOV/4
             (removes vignetting, keeps a ~70 px disc as a local peak)
  combined   flatfield brightness + w x vessel density: the disc is where it is
             bright AND the major vessels converge (neovascularization at the
             disc hides the brightness; vessels still converge there)

Writes backend/config/experiments/landmark_check.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from backend.venus import stage0_gate, stage1_segment
from backend.venus.config import CONFIG_DIR
from backend.venus.imaging import disk

OUT = CONFIG_DIR / "experiments" / "landmark_check.json"
CHROMA = 0.12
WEIGHTS = (0.5, 1.0, 2.0)


def achromatic_mask(src, mask):
    ch = src.astype(np.float32)
    top, bottom = ch.max(axis=2), ch.min(axis=2)
    a = ((top - bottom) / np.maximum(top, 1.0) < CHROMA) & (top > 60) & (mask > 0)
    return cv2.dilate(a.astype(np.uint8) * 255, disk(6)) > 0


def brightness(src, mask, flatfield: bool):
    ach = achromatic_mask(src, mask)
    ch = src.astype(np.float32)
    lum = 0.5 * ch[:, :, 1] + 0.5 * ch[:, :, 2]
    lum[ach] = 0.0
    if flatfield:
        inside = (mask > 0).astype(np.float32)
        d = 2 * np.sqrt(inside.sum() / np.pi)
        illum = cv2.GaussianBlur(lum * inside, (0, 0), d / 4) / np.maximum(cv2.GaussianBlur(inside, (0, 0), d / 4), 1e-3)
        lum = lum / np.maximum(illum, 8.0)
    smooth = cv2.GaussianBlur(lum, (0, 0), 12)
    smooth[cv2.erode(mask, disk(10)) == 0] = -np.inf
    smooth[ach] = -np.inf
    return smooth


def normalise(score, mask):
    finite = np.isfinite(score) & (mask > 0)
    if not finite.any():
        return np.zeros_like(score)
    lo, hi = np.percentile(score[finite], 1), score[finite].max()
    out = (score - lo) / max(hi - lo, 1e-6)
    out[~finite] = -np.inf
    return out


def pick(score):
    cy, cx = np.unravel_index(int(np.argmax(score)), score.shape)
    return int(cx), int(cy)


def old_pick(image, mask):
    lum = 0.5 * image[:, :, 1].astype(np.float32) + 0.5 * image[:, :, 2].astype(np.float32)
    smooth = cv2.GaussianBlur(lum, (0, 0), 12)
    smooth[cv2.erode(mask, disk(10)) == 0] = -1
    return pick(smooth)


def fovea_for(image, mask, centre, radius):
    od_mask = np.zeros_like(mask)
    cv2.circle(od_mask, centre, int(radius), 255, -1)
    return stage1_segment.fovea(image, mask, {"centre": list(centre), "radius": radius, "mask": od_mask})["centre"]


def to_working(x, y, geometry):
    bx, by, _, _ = geometry["bbox"]
    oy, ox, side = geometry["pad"]
    s = geometry["size"] / side
    return (x - bx + ox) * s, (y - by + oy) * s, s


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="Huelva data.xls (MESSIDOR fovea centres)")
    parser.add_argument("--images", default=None, help="folder with the MESSIDOR images (default: VENUS_DATA_ROOT/eye/messidor2/IMAGES)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    labels = pd.read_excel(args.labels)
    labels.columns = ["file", "od_diameter", "fx", "fy", "method_fx", "method_fy"]
    from backend.data.sources import EYE
    folder = Path(args.images) if args.images else EYE / "messidor2" / "IMAGES"
    on_disk = {p.stem: p for p in folder.iterdir()}
    labels["stem"] = labels["file"].str.rsplit(".", n=1).str[0]
    labels = labels[labels["stem"].isin(on_disk)].copy()
    labels["half"] = ["design" if int(hashlib.md5(s.encode()).hexdigest(), 16) % 2 == 0 else "held_out" for s in labels["stem"]]
    if args.limit:
        labels = labels.head(args.limit)
    print(f"{len(labels)} annotated images on disk ({(labels.half == 'held_out').sum()} held out)", flush=True)
    stage0_gate.load_gate(); stage0_gate.load_quality()

    rows, started = [], time.perf_counter()
    for i, lab in enumerate(labels.itertuples(), 1):
        raw = cv2.imread(str(on_disk[lab.stem]))
        if raw is None:
            continue
        s0 = stage0_gate.run(raw)
        if not s0["accepted"]:
            continue
        # Geometry of the normalisation Stage 0 applied, to map the labels in.
        _, _, geometry = stage0_gate.normalise_fov(raw)
        tfx, tfy, s = to_working(lab.fx, lab.fy, geometry)
        dd = lab.od_diameter * s
        work, orig, mask = s0["image"], s0["original"], s0["mask"]
        radius = float(2 * np.sqrt(cv2.countNonZero(mask) / np.pi) / 13.0)
        ves = stage1_segment.vessels(work, mask)["mask"].astype(np.float32) / 255.0
        density = cv2.GaussianBlur(ves, (0, 0), radius)
        density[mask == 0] = 0
        bright_ff = normalise(brightness(orig, mask, True), mask)
        vess_n = normalise(np.where(mask > 0, density, -np.inf), mask)
        picks = {"old": old_pick(work, mask),
                 "original": pick(brightness(orig, mask, False)),
                 "flatfield": pick(bright_ff)}
        for w in WEIGHTS:
            picks[f"combined_w{w}"] = pick(bright_ff + w * vess_n)
        rec = {"stem": lab.stem, "half": lab.half, "dd_px": round(float(dd), 1), "enhanced": bool(s0["quality"]["enhanced"])}
        for name, c in picks.items():
            fov = fovea_for(work, mask, c, radius)
            rec[f"{name}_fovea_err_dd"] = round(float(np.hypot(fov[0] - tfx, fov[1] - tfy) / dd), 3)
            rec[f"{name}_disc_to_true_fovea_dd"] = round(float(np.hypot(c[0] - tfx, c[1] - tfy) / dd), 3)
        rows.append(rec)
        if i % 100 == 0:
            print(f"  {i}/{len(labels)}  {(time.perf_counter() - started) / i:.2f} s/img", flush=True)

    r = pd.DataFrame(rows)
    names = ["old", "original", "flatfield"] + [f"combined_w{w}" for w in WEIGHTS]

    def summarise(frame):
        out = {}
        for n in names:
            err = frame[f"{n}_fovea_err_dd"]; dist = frame[f"{n}_disc_to_true_fovea_dd"]
            out[n] = {"fovea_within_0.5dd": round(float((err <= 0.5).mean()), 4),
                      "fovea_within_1dd": round(float((err <= 1.0).mean()), 4),
                      "fovea_median_err_dd": round(float(err.median()), 3),
                      "disc_plausible_1.5_to_3.5dd": round(float(dist.between(1.5, 3.5).mean()), 4)}
        return out

    design, held = r[r.half == "design"], r[r.half == "held_out"]
    d = summarise(design)
    best_w = max(WEIGHTS, key=lambda w: (d[f"combined_w{w}"]["fovea_within_1dd"], d[f"combined_w{w}"]["disc_plausible_1.5_to_3.5dd"]))
    summary = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "question": "Which optic-disc detector places the disc - and so the fovea - where a clinician marked the fovea?",
        "ground_truth": "University of Huelva MESSIDOR annotations: clinician-marked fovea centre and standard optic-disc diameter "
                        "(Gegundez-Arias et al.), https://www.uhu.es/retinopathy/",
        "n_images": int(len(r)), "n_design": int(len(design)), "n_held_out": int(len(held)),
        "split": "md5(file stem) mod 2; the vessel weight w is chosen on the design half only",
        "chosen_vessel_weight": best_w,
        "design_half": d, "held_out_half": summarise(held),
        "held_out_by_enhancement": {"enhanced": summarise(held[held.enhanced]) if held.enhanced.any() else None,
                                    "not_enhanced": summarise(held[~held.enhanced]) if (~held.enhanced).any() else None},
        "note": "Messidor-2 is a frozen external test set for the grader; no grading metric is recomputed here and the "
                "landmark stage enters none of the recorded external-test numbers.",
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"chosen vessel weight on the design half: {best_w}")
    print("held-out half:")
    for n, m in summary["held_out_half"].items():
        print(f"  {n:16s} fovea<=0.5DD {m['fovea_within_0.5dd']:.3f}  <=1DD {m['fovea_within_1dd']:.3f}  "
              f"median {m['fovea_median_err_dd']:.2f} DD  disc plausible {m['disc_plausible_1.5_to_3.5dd']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
