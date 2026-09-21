"""What a site-specific calibration set buys: measured on Messidor-2.

    wsl bash scripts/wsl-gpu.sh backend.eval.site_calibration --weights ~/venus-cache/models/grader_v2.weights.h5

The locked operating point over-refers on Messidor-2 (specificity 0.645 at
sensitivity 0.980 while the set's own ROC reaches 0.90/0.90): discrimination
transfers between acquisition sources, calibration does not. docs and README
state that a site calibration set is a prerequisite for deployment. This
script puts a number on that statement without touching the locked point:

  1. the grader's raw P(grade >= 2) for every Messidor-2 image (predictions are
     cached under <cache>/predictions/grader_v2_external_test_messidor2.csv)
  2. patients are split at random into a "site calibration" half and an
     "evaluation" half (patient-disjoint), 20 repeats
  3. on the calibration half: Platt (a, b) re-fitted and the threshold re-chosen
     at 90 % sensitivity, exactly as backend.eval.calibrate does on EyePACS
  4. on the evaluation half: sensitivity / specificity / ECE at (i) the locked
     operating point and (ii) the site calibration, and the same for site
     samples of 100, 200, 400 and 800 images so a programme knows how many
     labelled images it needs

Writes config/site_calibration_messidor2.json. Nothing here changes what is
served; it is an evaluation of a deployment step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from backend.eval.calibrate import confusion, expected_calibration_error, logit, sigmoid, threshold_for_sensitivity
from backend.venus.config import CONFIG_DIR, MANIFEST_DIR, OPERATING_POINT_PATH

CACHE_DIR = Path(os.getenv("VENUS_CACHE_DIR", str(Path.home() / "venus-cache")))


def predictions(args) -> pd.DataFrame:
    path = CACHE_DIR / "predictions" / f"grader_v2_{args.manifest}.csv"
    if path.exists():
        return pd.read_csv(path)
    import tensorflow as tf
    from tensorflow import keras
    from backend.training.train_grader import build_model, dataset
    df = pd.read_csv(MANIFEST_DIR / f"{args.manifest}.csv")
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    keras.mixed_precision.set_global_policy("mixed_float16")
    model = build_model(args.backbone)
    model.load_weights(os.path.expanduser(args.weights))
    probs = model.predict(dataset(df, args.batch, train=False), verbose=0)
    for k in range(4):
        df[f"p_ge{k + 1}"] = probs[:, k]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def fit_site(raw, y, target):
    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit(raw).reshape(-1, 1), y)
    a, b = float(platt.coef_[0][0]), float(platt.intercept_[0])
    prob = sigmoid(a * logit(raw) + b)
    return a, b, threshold_for_sensitivity(prob, y, target)


def evaluate(raw, y, a, b, t):
    prob = sigmoid(a * logit(raw) + b)
    c = confusion(prob, y, t)
    ece, _ = expected_calibration_error(prob, y)
    return {"sensitivity": c["sensitivity"], "specificity": c["specificity"], "ece": ece, "referred_fraction": c["referred_fraction"],
            "auc": float(roc_auc_score(y, prob))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="external_test_messidor2")
    parser.add_argument("--weights", default=str(CACHE_DIR / "models" / "grader_v2.weights.h5"))
    parser.add_argument("--backbone", default="B3")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--sizes", type=int, nargs="*", default=[100, 200, 400, 800])
    parser.add_argument("--target", type=float, default=0.90)
    args = parser.parse_args(argv)

    point = json.load(open(OPERATING_POINT_PATH, encoding="utf-8"))
    a0, b0, t0 = point["calibration"]["a"], point["calibration"]["b"], point["thresholds"]["referable"]
    df = predictions(args)
    raw = df["p_ge2"].values.astype(float); y = (df["grade"].values >= 2).astype(int); patients = df["patient"].values
    unique = np.unique(patients)
    rng = np.random.default_rng(42)

    per_size = {str(n): [] for n in args.sizes}
    locked, full = [], []
    for _ in range(args.repeats):
        perm = rng.permutation(unique)
        cal_patients = set(perm[: len(perm) // 2])
        cal = np.array([p in cal_patients for p in patients]); ev = ~cal
        locked.append(evaluate(raw[ev], y[ev], a0, b0, t0))
        a, b, t = fit_site(raw[cal], y[cal], args.target)
        full.append(evaluate(raw[ev], y[ev], a, b, t))
        cal_idx = np.flatnonzero(cal)
        for n in args.sizes:
            if n > len(cal_idx):
                continue
            sub = rng.choice(cal_idx, size=n, replace=False)
            if y[sub].sum() < 5:
                continue
            a, b, t = fit_site(raw[sub], y[sub], args.target)
            per_size[str(n)].append(evaluate(raw[ev], y[ev], a, b, t))

    def summarise(rows):
        if not rows:
            return None
        keys = rows[0].keys()
        return {k: {"mean": round(float(np.mean([r[k] for r in rows])), 4), "p5": round(float(np.percentile([r[k] for r in rows], 5)), 4),
                    "p95": round(float(np.percentile([r[k] for r in rows], 95)), 4)} for k in keys}

    out = {
        "written_at": datetime.now(timezone.utc).isoformat(), "manifest": args.manifest, "n_images": int(len(df)), "n_patients": int(len(unique)),
        "n_referable": int(y.sum()), "model_version": point["model_version"], "grader": point["grader_tag"],
        "protocol": f"{args.repeats} random patient-disjoint halves; site Platt (a, b) and the {int(args.target * 100)} % sensitivity threshold fitted on the "
                    "calibration half (or a random subset of it of the stated size), evaluated on the other half; the locked operating point evaluated on the same halves",
        "locked_operating_point": summarise(locked),
        "site_calibration_full_half": {"n_mean": int(len(df) // 2), **summarise(full)},
        "site_calibration_by_sample_size": {n: ({"n": int(n), "repeats": len(rows), **summarise(rows)} if rows else None) for n, rows in per_size.items()},
        "note": "Evaluation of a deployment step; the served operating point is unchanged. Site calibration re-fits Platt and re-chooses the threshold at "
                "90 % sensitivity on the site's own labelled sample, which is what backend.eval.calibrate does on the EyePACS calibration set.",
    }
    with open(CONFIG_DIR / f"site_calibration_{args.manifest.replace('external_test_', '')}.json", "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    L, F = out["locked_operating_point"], out["site_calibration_full_half"]
    print(f"locked point on held-out halves: sens {L['sensitivity']['mean']:.3f} spec {L['specificity']['mean']:.3f} ECE {L['ece']['mean']:.3f}")
    print(f"site calibration (n≈{F['n_mean']}):    sens {F['sensitivity']['mean']:.3f} spec {F['specificity']['mean']:.3f} ECE {F['ece']['mean']:.3f}")
    for n, s in out["site_calibration_by_sample_size"].items():
        if s:
            print(f"  site sample {n:>4}: sens {s['sensitivity']['mean']:.3f} [{s['sensitivity']['p5']:.3f}, {s['sensitivity']['p95']:.3f}]  spec {s['specificity']['mean']:.3f} [{s['specificity']['p5']:.3f}, {s['specificity']['p95']:.3f}]  ECE {s['ece']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
