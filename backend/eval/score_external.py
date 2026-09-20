"""Score an additional frozen external test set ONCE with the locked operating point.

    wsl bash scripts/wsl-gpu.sh backend.eval.score_external --manifest external_test_messidor2 \
        --weights ~/venus-cache/models/grader_v2.weights.h5

Runs the grader over the manifest, applies the calibration (a, b) and the
threshold already locked in config/operating_point.json — nothing is chosen
here — and records the same metrics as the primary external test (AUC,
sensitivity, specificity, PPV/NPV at 18 % prevalence, ECE, per-grade referral,
2,000-resample bootstrap CIs, five-grade contrast) under
operating_point.json["additional_external_tests"][<manifest>]. One scoring per
model version per manifest is allowed (config/external_test.lock); a second
refuses without --force and is then marked as re-scored.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from backend.eval.calibrate import LOCK, grade_metrics, logit, score_set, sigmoid
from backend.training.train_grader import build_model, dataset
from backend.venus.config import MANIFEST_DIR, OPERATING_POINT_PATH, sha256_of_file


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--backbone", default="B3")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--description", default="Messidor-2 (ADCIS), Krause et al. 2018 adjudicated ICDR grades; never trained on; a third acquisition source")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    point = json.load(open(OPERATING_POINT_PATH, encoding="utf-8"))
    lock = json.load(open(LOCK, encoding="utf-8")) if LOCK.exists() else {}
    scored = lock.setdefault("additional", {})
    already = scored.get(args.manifest) == point["model_version"]
    if already and not args.force:
        print(f"refusing: {args.manifest} already scored for {point['model_version']}. Pass --force to re-score (recorded).", file=sys.stderr)
        return 2

    path = MANIFEST_DIR / f"{args.manifest}.csv"
    df = pd.read_csv(path)
    fingerprint = sha256_of_file(path)
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    keras.mixed_precision.set_global_policy("mixed_float16")
    model = build_model(args.backbone)
    model.load_weights(os.path.expanduser(args.weights))
    probs = model.predict(dataset(df, args.batch, train=False), verbose=0)
    df["p_ge1"], df["p_ge2"], df["p_ge3"], df["p_ge4"] = (probs[:, k] for k in range(4))

    a, b = point["calibration"]["a"], point["calibration"]["b"]
    t90, t85 = point["thresholds"]["referable"], point["thresholds"]["referable_85pc_alternative"]
    prob = sigmoid(a * logit(df["p_ge2"].values) + b)
    result = score_set(args.manifest, prob, df["grade"].values.astype(int), t90, t85, args.description)
    result.update({"manifest_fingerprint": fingerprint, "n_patients": int(df["patient"].nunique()),
                   "scored_once": not already, "rescored_with_force": bool(already and args.force),
                   "scored_at": datetime.now(timezone.utc).isoformat(), "grade_metrics_for_contrast": grade_metrics(df),
                   "threshold_used": t90, "note": "threshold and calibration taken from the locked operating point; nothing chosen on this set"})
    if "adjudicated_dme" in df.columns:
        result["referred_fraction_by_adjudicated_dme"] = {str(int(k)): round(float((prob[df["adjudicated_dme"].values == k] >= t90).mean()), 4)
                                                          for k in sorted(df["adjudicated_dme"].dropna().unique())}
    point.setdefault("additional_external_tests", {})[args.manifest] = result
    with open(OPERATING_POINT_PATH, "w", encoding="utf-8") as handle:
        json.dump(point, handle, indent=2)
    scored[args.manifest] = point["model_version"]
    with open(LOCK, "w", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2)

    at, ci = result["at_locked_threshold"], result["ci95_bootstrap_2000"]
    print(f"{args.manifest}: n={len(df)} ({result['n_referable']} referable)  AUC {result['auc']:.3f} {ci['auc']}  "
          f"sens {at['sensitivity']:.3f} {ci['sensitivity']}  spec {at['specificity']:.3f} {ci['specificity']}  "
          f"PPV@18% {at['ppv_at_indian_prevalence']:.3f}  ECE {result['ece']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
