"""Fit the calibration, lock the operating point, score the external test once.

Implements section 6.4 / 11 of the architecture:

    1. Read the grader's predictions on the CALIBRATION manifest (held out by
       patient before training) and fit Platt scaling (a, b) so the served
       number is a probability. Report ECE before and after.
    2. Choose the referable threshold on the calibration predictions at 90 %
       sensitivity. Lock it. Record the 85 % alternative at the same time.
    3. Score the frozen EXTERNAL test exactly once with those numbers:
       sensitivity, specificity, PPV/NPV at 18 % Indian prevalence, AUC, ECE,
       per-grade referral rates, 2,000-resample bootstrap CIs, and the ROC
       points the district simulation sweeps along.
    4. Score the secondary held-out set (EyePACS patients frozen from the
       earlier grader, within-source) the same way, labelled as such.
    5. Write config/operating_point.json with the calibration manifest's
       SHA-256; serving refuses to start if the manifest on disk differs.

The external test is scored once per model version (config/external_test.lock);
a second run refuses without --force, and a forced run says so in the output.

    python -m backend.eval.calibrate --tag grader_v2 --model-version venus-dr-2.0.0
    python -m backend.eval.calibrate --legacy          # the v1 grader's frozen EyePACS CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from backend.venus.config import CONFIG_DIR, MANIFEST_DIR, MODEL_VERSION, OPERATING_POINT_PATH, sha256_of_file

CACHE_DIR = Path(os.getenv("VENUS_CACHE_DIR", str(Path.home() / "venus-cache")))
LOCK = CONFIG_DIR / "external_test.lock"
SEED = 42
INDIAN_PREVALENCE = 0.18
ABSTAIN_BAND = 0.05
EPS = 1e-7


# ------------------------------------------------------------ helpers --

def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def expected_calibration_error(prob, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    ece, diagram = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (prob >= lo) & (prob < hi) if hi < 1 else (prob >= lo) & (prob <= hi)
        if sel.sum() == 0:
            diagram.append({"bin": [round(lo, 2), round(hi, 2)], "n": 0})
            continue
        conf, acc = float(prob[sel].mean()), float(y[sel].mean())
        ece += sel.mean() * abs(conf - acc)
        diagram.append({"bin": [round(lo, 2), round(hi, 2)], "n": int(sel.sum()), "confidence": round(conf, 4), "accuracy": round(acc, 4)})
    return float(ece), diagram


def threshold_for_sensitivity(prob, y, target):
    positives = np.sort(prob[y == 1])
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
            "ppv_at_indian_prevalence": ppv, "npv_at_indian_prevalence": npv, "referred_fraction": float(pred.mean())}


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
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in stats.items()}


def roc_points(prob, y):
    points = []
    for target in (0.80, 0.85, 0.90, 0.95, 0.975):
        t = threshold_for_sensitivity(prob, y, target)
        c = confusion(prob, y, t)
        points.append({"sensitivity_target": target, "threshold": round(t, 6),
                       "sensitivity": round(c["sensitivity"], 4), "specificity": round(c["specificity"], 4)})
    return points


def _round(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def score_set(name, prob, grade, t90, t85, description):
    y = (grade >= 2).astype(int)
    at90, at85 = confusion(prob, y, t90), confusion(prob, y, t85)
    ece, diagram = expected_calibration_error(prob, y)
    pred_grade = None
    per_grade = {int(g): round(float((prob[grade == g] >= t90).mean()), 4) for g in range(5) if (grade == g).any()}
    return {
        "name": name, "description": description, "n": int(len(y)), "n_referable": int(y.sum()),
        "auc": round(float(roc_auc_score(y, prob)), 4),
        "at_locked_threshold": _round(at90), "at_85pc_threshold": _round(at85),
        "ci95_bootstrap_2000": bootstrap(prob, y, t90), "ece": round(ece, 4), "reliability_diagram": diagram,
        "sensitivity_grade2_only": round(float((prob[grade == 2] >= t90).mean()), 4) if (grade == 2).any() else None,
        "referred_fraction_by_true_grade": per_grade,
        "prevalence_for_ppv": INDIAN_PREVALENCE,
        "roc_points_for_simulation": roc_points(prob, y),
    }


def grade_metrics(df: pd.DataFrame) -> dict | None:
    """Five-grade decode quality, reported for contrast (not a claim)."""
    if not {"p_ge1", "p_ge3", "p_ge4"} <= set(df.columns):
        return None
    from sklearn.metrics import cohen_kappa_score
    probs = df[["p_ge1", "p_ge2", "p_ge3", "p_ge4"]].values
    pred = (probs >= 0.5).sum(axis=1)
    return {"quadratic_weighted_kappa": round(float(cohen_kappa_score(df["grade"], pred, weights="quadratic")), 4),
            "exact_grade_accuracy": round(float((pred == df["grade"]).mean()), 4),
            "within_one_grade": round(float((np.abs(pred - df["grade"]) <= 1).mean()), 4)}


# --------------------------------------------------------------- data --

def load_v2(tag: str, site: str | None = None):
    """`site` names a site manifest (backend.data.site_manifest) whose
    predictions replace the EyePACS calibration set; the external test stays."""
    pred_dir = CACHE_DIR / "predictions"
    cal = pd.read_csv(pred_dir / f"{tag}_{site or 'calibration'}.csv")
    ext = pd.read_csv(pred_dir / f"{tag}_external_test_ddr.csv")
    held_path = pred_dir / f"{tag}_heldout_eyepacs_frozen.csv"
    held = pd.read_csv(held_path) if held_path.exists() else None
    for frame in (cal, ext, held):
        if frame is not None:
            frame["referable_raw"] = frame["p_ge2"]
    return cal, ext, held, f"{site or 'calibration'}.csv", "external_test_ddr.csv"


def load_legacy():
    """The v1 grader's raw referable scores on the frozen EyePACS manifest,
    split by patient into calibration and test halves (written to
    calibration_split.csv, which is the fingerprinted manifest)."""
    rows = list(csv.DictReader(open(MANIFEST_DIR / "eyepacs_frozen_predictions.csv", encoding="utf-8")))
    df = pd.DataFrame({"image_id": [r["image"] for r in rows],
                       "patient": [r["image"].split("_")[0] for r in rows],
                       "grade": [int(r["true_grade"]) for r in rows],
                       "referable_raw": [float(r["referable_score"]) for r in rows]})
    rng = np.random.default_rng(SEED)
    patients = np.unique(df["patient"].values); rng.shuffle(patients)
    cal_patients = set(patients[: len(patients) // 2])
    df["split"] = np.where(df["patient"].isin(cal_patients), "calibration", "test")
    out = MANIFEST_DIR / "calibration_split.csv"
    df.sort_values("image_id")[["image_id", "patient", "grade", "referable_raw", "split"]].to_csv(out, index=False)
    return df[df["split"] == "calibration"].copy(), df[df["split"] == "test"].copy(), None, "calibration_split.csv", None


# --------------------------------------------------------------- main --

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="grader_v2")
    parser.add_argument("--legacy", action="store_true", help="use the v1 grader's frozen EyePACS CSV")
    parser.add_argument("--model-version", default=MODEL_VERSION)
    parser.add_argument("--force", action="store_true", help="re-score the external test for a version already scored")
    parser.add_argument("--site", default=None, help="site manifest stem (site_<name>): calibrate on the site's labelled images instead of EyePACS")
    parser.add_argument("--out", default=None, help="output path; default config/operating_point.json, or config/operating_point_<site>.json with --site")
    args = parser.parse_args(argv)
    if args.site:
        # A site operating point is a distinct model version: same grader, its
        # own calibration set and threshold. Activating it (copying it over
        # operating_point.json) is a deliberate step, see scripts/site-calibrate.sh.
        args.model_version = f"{args.model_version}+{args.site}"
    out_path = Path(args.out) if args.out else (CONFIG_DIR / f"operating_point_{args.site}.json" if args.site else OPERATING_POINT_PATH)

    if args.legacy:
        cal, ext, held, cal_manifest, ext_manifest = load_legacy()
        source = {"description": "v1 grader (EfficientNet-B4/380) raw scores on the frozen EyePACS manifest, 1,500 images stratified 300 per grade; split by patient",
                  "external": "EyePACS test half (within the same frozen set; cross-source relative to the v1 training data)"}
    else:
        cal, ext, held, cal_manifest, ext_manifest = load_v2(args.tag, args.site)
        source = {"description": (f"{args.tag} predictions on the site calibration manifest {args.site} (the site's own labelled images, read through Stage 0)"
                                  if args.site else f"{args.tag} predictions on the patient-disjoint calibration manifest (EyePACS, held out before training)"),
                  "external": "DDR test split: a different acquisition source from every training image, ungradables removed, never trained on"}
    fingerprint = sha256_of_file(MANIFEST_DIR / cal_manifest)
    ext_fingerprint = sha256_of_file(MANIFEST_DIR / ext_manifest) if ext_manifest else None

    y_cal = (cal["grade"].values >= 2).astype(int)
    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit(cal["referable_raw"]).reshape(-1, 1), y_cal)
    a, b = float(platt.coef_[0][0]), float(platt.intercept_[0])
    cal_prob = sigmoid(a * logit(cal["referable_raw"]) + b)
    ece_raw, _ = expected_calibration_error(cal["referable_raw"].values, y_cal)
    ece_cal, diagram_cal = expected_calibration_error(cal_prob, y_cal)
    t90 = threshold_for_sensitivity(cal_prob, y_cal, 0.90)
    t85 = threshold_for_sensitivity(cal_prob, y_cal, 0.85)

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lock = json.load(open(LOCK)) if LOCK.exists() else {}
    # Site operating points keep their own lock entries so the primary
    # version's single scoring is never forgotten.
    previously = lock.get("sites", {}).get(args.site, {}) if args.site else lock
    already = previously.get("model_version") == args.model_version
    if already and not args.force:
        print(f"refusing: external test already scored for {args.model_version} on {previously.get('scored_at')}. "
              "Pass --force to re-score (the output file will say so).", file=sys.stderr)
        return 2

    ext_prob = sigmoid(a * logit(ext["referable_raw"]) + b)
    external = score_set("external_test", ext_prob, ext["grade"].values.astype(int), t90, t85, source["external"])
    external["scored_once"] = not already
    external["rescored_with_force"] = bool(already and args.force)
    external["grade_metrics_for_contrast"] = grade_metrics(ext)
    secondary = None
    if held is not None and len(held):
        held_prob = sigmoid(a * logit(held["referable_raw"]) + b)
        secondary = score_set("heldout_eyepacs_frozen", held_prob, held["grade"].values.astype(int), t90, t85,
                              "EyePACS patients frozen from the earlier grader: held-out patients, same source as part of training (within-source)")
        secondary["grade_metrics_for_contrast"] = grade_metrics(held)

    at = external["at_locked_threshold"]
    point = {
        "model_version": args.model_version,
        "grader_tag": "legacy_v1" if args.legacy else args.tag,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "calibration_fingerprint": fingerprint,
        "calibration_manifest": cal_manifest,
        "external_test_manifest": ext_manifest,
        "external_test_fingerprint": ext_fingerprint,
        "source": {**source, "n_calibration": int(len(cal)), "n_external": int(len(ext)),
                   "n_patients_calibration": int(cal["patient"].nunique())},
        "calibration": {"method": "Platt scaling on logit(raw P(grade >= 2))", "a": a, "b": b,
                        "ece_raw": round(ece_raw, 4), "ece_calibrated": round(ece_cal, 4), "reliability_diagram": diagram_cal},
        "thresholds": {"referable": round(t90, 6), "referable_85pc_alternative": round(t85, 6),
                       "chosen_on": "calibration manifest, at 90% sensitivity, then locked", "abstain_band": ABSTAIN_BAND,
                       "calibration_at_90": _round(confusion(cal_prob, y_cal, t90)),
                       "calibration_at_85": _round(confusion(cal_prob, y_cal, t85))},
        "external_test": external,
        "secondary_heldout": secondary,
        "targets": {"sensitivity": 0.90, "specificity": 0.85, "auc": 0.95, "ece": 0.05,
                    "sensitivity_met": at["sensitivity"] >= 0.90, "specificity_met": at["specificity"] >= 0.85,
                    "auc_met": external["auc"] >= 0.95, "ece_met": external["ece"] <= 0.05},
        "notes": [
            "The raw grader output ranks; Platt scaling on the held-out calibration set is what makes P(referable) reportable as a probability.",
            "PPV/NPV are recomputed at 18% Indian prevalence because the test sets' prevalence is a sampling artefact.",
            "A missed target is reported as the achieved number with its CI, not a re-tuned threshold.",
            "Five-grade metrics are reported for contrast; the referable decision (grade >= 2) is the claim.",
        ],
    }
    if args.site:
        point["site"] = {"name": args.site, "note": "calibration (Platt a, b) and the 90 % sensitivity threshold chosen on this site's labelled images; "
                                                    "the external test is the same DDR split, scored once under this site version"}
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(point, handle, indent=2)
    entry = {"model_version": args.model_version, "scored_at": point["written_at"], "calibration_fingerprint": fingerprint,
             "external_test_fingerprint": ext_fingerprint}
    if args.site:
        lock.setdefault("sites", {})[args.site] = entry
    else:
        lock = {**lock, **entry}
    with open(LOCK, "w", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2)

    print(f"calibration n={len(cal)}  external n={len(ext)}  fingerprint {fingerprint[:16]}...")
    print(f"Platt a={a:.4f} b={b:.4f}   ECE raw {ece_raw:.3f} -> calibrated {ece_cal:.3f}")
    print(f"threshold@90 = {t90:.4f}  (calibration spec {point['thresholds']['calibration_at_90']['specificity']:.3f})")
    ci = external["ci95_bootstrap_2000"]
    print(f"EXTERNAL  AUC {external['auc']:.3f} {ci['auc']}  sens {at['sensitivity']:.3f} {ci['sensitivity']}  "
          f"spec {at['specificity']:.3f} {ci['specificity']}  PPV@18% {at['ppv_at_indian_prevalence']:.3f}  ECE {external['ece']:.3f}")
    if secondary:
        s = secondary["at_locked_threshold"]
        print(f"HELD-OUT  AUC {secondary['auc']:.3f}  sens {s['sensitivity']:.3f}  spec {s['specificity']:.3f}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
