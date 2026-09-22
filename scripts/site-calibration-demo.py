"""Live demo: what a site calibration set buys, in under two minutes.

    python scripts/site-calibration-demo.py [--size 400] [--repeats 20]

The story this tells, on stage, with real numbers computed in front of the
audience: the locked threshold was chosen on EyePACS; on Messidor-2 — a third
acquisition source — it over-refers badly. Re-fitting the calibration on a few
hundred labelled images from that site restores the intended operating point.

What is precomputed and what is live. The grader's raw score for each of the
1,744 Messidor-2 images is read from docs/examples/messidor2_raw_scores.csv:
running the network over them takes minutes on a laptop CPU and is the one-time
cost a real site pays once. Everything that follows — the patient-disjoint
split, the Platt fit, the threshold at 90 % sensitivity, and both evaluations —
is computed now, in about a second, and that is the step being demonstrated.

Nothing here touches the locked operating point or any frozen test set: the
site fit is evaluated on the half of the patients it was not fitted on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "docs" / "examples" / "messidor2_raw_scores.csv"
POINT = ROOT / "backend" / "config" / "operating_point.json"
EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def threshold_at_sensitivity(prob, y, target):
    positives = np.sort(prob[y == 1])
    return float(positives[len(positives) - int(np.ceil(target * len(positives)))])


def evaluate(prob, y, t):
    pred = prob >= t
    tp = int((pred & (y == 1)).sum()); fn = int((~pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum()); tn = int((~pred & (y == 0)).sum())
    edges = np.linspace(0, 1, 11)
    ece = sum(sel.mean() * abs(prob[sel].mean() - y[sel].mean())
              for lo, hi in zip(edges[:-1], edges[1:])
              if (sel := ((prob >= lo) & (prob < hi if hi < 1 else prob <= hi))).sum())
    return {"sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "ece": float(ece), "referred": float(pred.mean()), "auc": float(roc_auc_score(y, prob))}


def bar(value: float, width: int = 28) -> str:
    """ASCII on purpose: the Windows console this is demonstrated on is cp1252
    and block-drawing characters raise UnicodeEncodeError mid-demo."""
    filled = int(round(value * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=400, help="labelled images the site provides")
    parser.add_argument("--repeats", type=int, default=20, help="random patient-disjoint splits to average over")
    parser.add_argument("--target", type=float, default=0.90)
    args = parser.parse_args(argv)

    if not SCORES.exists():
        print(f"missing {SCORES}", file=sys.stderr)
        return 2
    point = json.load(open(POINT, encoding="utf-8"))
    a0, b0 = point["calibration"]["a"], point["calibration"]["b"]
    t0 = point["thresholds"]["referable"]
    df = pd.read_csv(SCORES)
    raw = df["p_referable_raw"].values.astype(float)
    y = (df["grade"].values >= 2).astype(int)
    patients = df["patient"].values
    unique = np.unique(patients)

    started = time.perf_counter()
    print(f"\n  Site: Messidor-2 ({len(df):,} images, {len(unique)} patients, {int(y.sum())} referable)")
    print(f"  Served operating point: {point['model_version']}, threshold {t0:.4f},")
    print(f"  chosen on {point['source']['n_calibration']:,} EyePACS images and locked.\n")

    rng = np.random.default_rng(42)
    before, after, sized = [], [], []
    for _ in range(args.repeats):
        perm = rng.permutation(unique)
        cal_patients = set(perm[: len(perm) // 2])
        cal = np.array([p in cal_patients for p in patients])
        ev = ~cal
        before.append(evaluate(sigmoid(a0 * logit(raw[ev]) + b0), y[ev], t0))

        def fit_on(idx):
            platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit(raw[idx]).reshape(-1, 1), y[idx])
            a, b = float(platt.coef_[0][0]), float(platt.intercept_[0])
            t = threshold_at_sensitivity(sigmoid(a * logit(raw[idx]) + b), y[idx], args.target)
            return evaluate(sigmoid(a * logit(raw[ev]) + b), y[ev], t)

        after.append(fit_on(np.flatnonzero(cal)))
        subset = rng.choice(np.flatnonzero(cal), size=min(args.size, int(cal.sum())), replace=False)
        sized.append(fit_on(subset))

    def mean(rows, key):
        return float(np.mean([r[key] for r in rows]))

    rows = [
        ("BEFORE  locked EyePACS threshold", before, "over-refers: four in ten healthy eyes sent to a doctor"),
        (f"AFTER   re-fitted on {args.size} site images", sized, "the intended operating point, restored"),
        ("        re-fitted on the full half", after, f"({len(unique) // 2} patients, for reference"),
    ]
    print(f"  {'':44s}{'sensitivity':>13}{'specificity':>13}{'ECE':>8}")
    for label, data, note in rows:
        s, sp, e = mean(data, "sensitivity"), mean(data, "specificity"), mean(data, "ece")
        print(f"  {label:44s}{s:>13.3f}{sp:>13.3f}{e:>8.3f}")
        print(f"  {'':44s}{bar(s):>13}  {bar(sp)}")
    print()
    print(f"  Discrimination never moved: AUC {mean(before, 'auc'):.3f} before, {mean(sized, 'auc'):.3f} after -")
    print("  the model ranks these eyes just as well either way. What transfers between")
    print("  sources is the ranking; what does not is the operating point.")
    print(f"\n  Unnecessary referrals avoided, per 1,000 screened: "
          f"{int(1000 * (mean(before, 'referred') - mean(sized, 'referred')))}")
    print(f"  {args.repeats} random patient-disjoint splits, {time.perf_counter() - started:.1f} s.")
    print("  The site fit is evaluated only on patients it was not fitted on; the locked")
    print("  operating point and every frozen test set are untouched.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
