"""Fit the calibration, lock the operating point, score the external test once.

Implements section 6.4 of the architecture with the data this build has: the
grader's raw referable scores on the frozen EyePACS external manifest
(1,500 images, 300 per ICDR grade, never trained on, a different acquisition
source from the training sets).

    1. Split the manifest by patient into a calibration half and a test half.
       The split is written to data/manifests/calibration_split.csv and its
       SHA-256 is the fingerprint that config/operating_point.json carries.
    2. Fit Platt scaling (a, b) on the calibration half so the served number
       is a probability, not a ranking score. Report ECE before and after.
    3. Choose the referable threshold on the calibration half at the point
       that gives 90% sensitivity. Lock it. Also record the 85% point because
       a district officer may prefer it; both are chosen here, at the same time.
    4. Score the test half exactly once with those numbers. Bootstrap 2,000
       resamples for 95% CIs. Write everything to config/operating_point.json.

The test half is scored once per model version: config/external_test.lock
records the version, and a second run with the same version refuses unless
--force is passed (and then says so in the output file).

Run:  python -m backend.eval.calibrate
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from backend.venus.config import (CALIBRATION_MANIFEST, CONFIG_DIR, MANIFEST_DIR, MODEL_VERSION,
                                  OPERATING_POINT_PATH, sha256_of_file)

FROZEN = MANIFEST_DIR / "eyepacs_frozen_predictions.csv"
LOCK = CONFIG_DIR / "external_test.lock"
SEED = 42
INDIAN_PREVALENCE = 0.18
ABSTAIN_BAND = 0.05
EPS = 1e-7


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def expected_calibration_error(prob: np.ndarray, y: np.ndarray, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    ece, diagram = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (prob >= lo) & (prob < hi) if hi < 1 else (prob >= lo) & (prob <= hi)
        if sel.sum() == 0:
            diagram.append({"bin": [round(lo, 2), round(hi, 2)], "n": 0})
            continue
        conf, acc = float(prob[sel].mean()), float(y[sel].mean())
        ece += sel.mean() * abs(conf - acc)
        diagram.append({"bin": [round(lo, 2), round(hi, 2)], "n": int(sel.sum()),
                        "confidence": round(conf, 4), "accuracy": round(acc, 4)})
    return float(ece), diagram


def threshold_for_sensitivity(prob: np.ndarray, y: np.ndarray, target: float) -> float:
    """Largest threshold at which sensitivity is still >= target."""
    positives = np.sort(prob[y == 1])
    # Sensitivity >= target means at least ceil(target * n) positives at or above t.
    k = int(np.ceil(target * len(positives)))
    return float(positives[len(positives) - k])


def confusion(prob, y, t):
    pred = prob >= t
    tp = int((pred & (y == 1)).sum()); fn = int((~pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum()); tn = int((~pred & (y == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    ppv = (sens * INDIAN_PREVALENCE) / max(sens * INDIAN_PREVALENCE + (1 - spec) * (1 - INDIAN_PREVALENCE), EPS)
    npv = (spec * (1 - INDIAN_PREVALENCE)) / max(spec * (1 - INDIAN_PREVALENCE) + (1 - sens) * INDIAN_PREVALENCE, EPS)
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "sensitivity": sens, "specificity": spec,
            "ppv_at_indian_prevalence": ppv, "npv_at_indian_prevalence": npv,
            "referred_fraction": float(pred.mean())}


def bootstrap(prob, y, t, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    stats = {"auc": [], "sensitivity": [], "specificity": [], "ppv_at_indian_prevalence": []}
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        c = confusion(prob[s], y[s], t)
        stats["auc"].append(roc_auc_score(y[s], prob[s]))
        for k in ("sensitivity", "specificity", "ppv_at_indian_prevalence"):
            stats[k].append(c[k])
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
            for k, v in stats.items()}


def roc_points(prob, y):
    """Sensitivity/specificity pairs the district simulation sweeps along."""
    points = []
    for target in (0.80, 0.85, 0.90, 0.95, 0.975):
        t = threshold_for_sensitivity(prob, y, target)
        c = confusion(prob, y, t)
        points.append({"sensitivity_target": target, "threshold": round(t, 6),
                       "sensitivity": round(c["sensitivity"], 4), "specificity": round(c["specificity"], 4)})
    return points


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-score the test half for a version already scored")
    args = parser.parse_args(argv)

    rows = list(csv.DictReader(open(FROZEN, encoding="utf-8")))
    images = np.array([r["image"] for r in rows])
    patients = np.array([r["image"].split("_")[0] for r in rows])
    grade = np.array([int(r["true_grade"]) for r in rows])
    raw = np.array([float(r["referable_score"]) for r in rows])
    y = (grade >= 2).astype(int)

    # Patient-disjoint split, seeded.
    rng = np.random.default_rng(SEED)
    unique = np.unique(patients)
    rng.shuffle(unique)
    cal_patients = set(unique[: len(unique) // 2])
    split = np.array(["calibration" if p in cal_patients else "test" for p in patients])

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_MANIFEST, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "patient", "true_grade", "referable_raw", "split"])
        for i in np.argsort(images):
            writer.writerow([images[i], patients[i], grade[i], f"{raw[i]:.9g}", split[i]])
    fingerprint = sha256_of_file(CALIBRATION_MANIFEST)

    cal, test = split == "calibration", split == "test"
    assert not (set(patients[cal]) & set(patients[test])), "split is not patient-disjoint"

    # Platt scaling on the calibration half.
    x_cal = logit(raw[cal]).reshape(-1, 1)
    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(x_cal, y[cal])
    a, b = float(platt.coef_[0][0]), float(platt.intercept_[0])
    calibrated = sigmoid(a * logit(raw) + b)

    ece_raw_cal, _ = expected_calibration_error(raw[cal], y[cal])
    ece_cal, diagram_cal = expected_calibration_error(calibrated[cal], y[cal])

    t90 = threshold_for_sensitivity(calibrated[cal], y[cal], 0.90)
    t85 = threshold_for_sensitivity(calibrated[cal], y[cal], 0.85)
    cal_at_90 = confusion(calibrated[cal], y[cal], t90)
    cal_at_85 = confusion(calibrated[cal], y[cal], t85)

    # Lock check: the test half is scored once per model version.
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    previously = json.load(open(LOCK)) if LOCK.exists() else {}
    if previously.get("model_version") == MODEL_VERSION and not args.force:
        print(f"refusing: test half already scored for {MODEL_VERSION} on {previously.get('scored_at')}. "
              "Pass --force to re-score (the output file will say so).", file=sys.stderr)
        return 2

    test_prob, test_y = calibrated[test], y[test]
    test_auc = float(roc_auc_score(test_y, test_prob))
    test_at_90 = confusion(test_prob, test_y, t90)
    test_at_85 = confusion(test_prob, test_y, t85)
    ci = bootstrap(test_prob, test_y, t90)
    ece_test, diagram_test = expected_calibration_error(test_prob, test_y)
    # Grade-2 sensitivity alone: the hardest referable class.
    g2 = test & (grade == 2)
    sens_g2 = float((calibrated[g2] >= t90).mean()) if g2.any() else None
    per_grade = {int(g): round(float((calibrated[test & (grade == g)] >= t90).mean()), 4)
                 for g in range(5) if (test & (grade == g)).any()}

    point = {
        "model_version": MODEL_VERSION,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "calibration_fingerprint": fingerprint,
        "calibration_manifest": CALIBRATION_MANIFEST.name,
        "frozen_source": {
            "file": FROZEN.name,
            "description": "EyePACS external set, 1,500 images stratified 300 per grade, never trained on, cross-source",
            "n_calibration": int(cal.sum()),
            "n_test": int(test.sum()),
            "n_patients_calibration": len(cal_patients),
            "split": "by patient id, seed 42",
        },
        "calibration": {
            "method": "Platt scaling on logit(raw referable score)",
            "a": a, "b": b,
            "ece_raw": round(ece_raw_cal, 4),
            "ece_calibrated": round(ece_cal, 4),
            "reliability_diagram": diagram_cal,
        },
        "thresholds": {
            "referable": round(t90, 6),
            "referable_85pc_alternative": round(t85, 6),
            "chosen_on": "calibration half, at 90% sensitivity, then locked",
            "abstain_band": ABSTAIN_BAND,
            "calibration_half_at_90": {k: round(v, 4) if isinstance(v, float) else v for k, v in cal_at_90.items()},
            "calibration_half_at_85": {k: round(v, 4) if isinstance(v, float) else v for k, v in cal_at_85.items()},
        },
        "external_test": {
            "scored_once": not (previously.get("model_version") == MODEL_VERSION),
            "rescored_with_force": bool(previously.get("model_version") == MODEL_VERSION and args.force),
            "auc": round(test_auc, 4),
            "at_locked_threshold": {k: round(v, 4) if isinstance(v, float) else v for k, v in test_at_90.items()},
            "at_85pc_threshold": {k: round(v, 4) if isinstance(v, float) else v for k, v in test_at_85.items()},
            "ci95_bootstrap_2000": ci,
            "ece": round(ece_test, 4),
            "reliability_diagram": diagram_test,
            "sensitivity_grade2_only": round(sens_g2, 4) if sens_g2 is not None else None,
            "referred_fraction_by_true_grade": per_grade,
            "prevalence_for_ppv": INDIAN_PREVALENCE,
            "roc_points_for_simulation": roc_points(test_prob, test_y),
        },
        "targets": {
            "sensitivity": 0.90, "specificity": 0.85, "auc": 0.95, "ece": 0.05,
            "sensitivity_met": test_at_90["sensitivity"] >= 0.90,
            "specificity_met": test_at_90["specificity"] >= 0.85,
            "auc_met": test_auc >= 0.95,
            "ece_met": ece_test <= 0.05,
        },
        "notes": [
            "The raw grader score ranks well but is not a probability (Brier 0.268 on the full frozen set); Platt scaling is what makes P(referable) reportable.",
            "The frozen set is 60% referable by construction; PPV/NPV are recomputed at 18% Indian prevalence.",
            "EyePACS labels are single-grader ICDR grades; an adjudicated set (Messidor-2) is the planned replacement.",
            "A missed target is reported as the achieved number with its CI, not a re-tuned threshold.",
        ],
    }
    with open(OPERATING_POINT_PATH, "w", encoding="utf-8") as handle:
        json.dump(point, handle, indent=2)
    with open(LOCK, "w", encoding="utf-8") as handle:
        json.dump({"model_version": MODEL_VERSION, "scored_at": point["written_at"],
                   "calibration_fingerprint": fingerprint}, handle, indent=2)

    print(f"calibration n={cal.sum()}  test n={test.sum()}  fingerprint {fingerprint[:16]}...")
    print(f"Platt a={a:.4f} b={b:.4f}   ECE raw {ece_raw_cal:.3f} -> calibrated {ece_cal:.3f}")
    print(f"threshold@90 sens = {t90:.4f}  (cal spec {cal_at_90['specificity']:.3f})")
    print(f"TEST  AUC {test_auc:.3f} {ci['auc']}  sens {test_at_90['sensitivity']:.3f} {ci['sensitivity']}  "
          f"spec {test_at_90['specificity']:.3f} {ci['specificity']}  PPV@18% {test_at_90['ppv_at_indian_prevalence']:.3f}")
    print(f"wrote {OPERATING_POINT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
