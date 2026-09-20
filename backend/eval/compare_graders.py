"""Compare grader checkpoints, and their average, on the calibration / val
predictions only — the decision aid for "is a second seed worth an ensemble".

    wsl bash scripts/wsl-gpu.sh backend.eval.compare_graders --tags grader_v2 grader_v2_s7
    python -m backend.eval.compare_graders --tags grader_v2 grader_v2_s7 --manifest val

Reads <cache>/predictions/<tag>_<manifest>[_tta].csv (backend.eval.score_grader
output), reports referable AUC with a 2,000-resample bootstrap CI per tag and
for the mean of P(grade >= k) across tags, plus the paired bootstrap difference
ensemble - best single. It never opens an external test manifest: the manifest
argument is restricted to calibration and val. Promotion of an ensemble is a
separate, deliberate step (a new grader tag, calibrate, external test once).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CACHE_DIR = Path(os.getenv("VENUS_CACHE_DIR", str(Path.home() / "venus-cache")))
ALLOWED = ("calibration", "val")


def bootstrap_auc(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if y[s].min() == y[s].max():
            continue
        vals.append(roc_auc_score(y[s], p[s]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_difference(y, p_a, p_b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    diffs = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if y[s].min() == y[s].max():
            continue
        diffs.append(roc_auc_score(y[s], p_a[s]) - roc_auc_score(y[s], p_b[s]))
    diffs = np.array(diffs)
    return {"mean": round(float(diffs.mean()), 4), "ci95": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
            "p_le_0": round(float((diffs <= 0).mean()), 4)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--manifest", default="calibration", choices=ALLOWED)
    parser.add_argument("--tta", action="store_true", help="use the *_tta.csv predictions")
    args = parser.parse_args(argv)

    suffix = "_tta" if args.tta else ""
    frames = {}
    for tag in args.tags:
        path = CACHE_DIR / "predictions" / f"{tag}_{args.manifest}{suffix}.csv"
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 2
        frames[tag] = pd.read_csv(path).set_index("image_id").sort_index()
    ids = frames[args.tags[0]].index
    for tag, df in frames.items():
        if not df.index.equals(ids):
            print(f"{tag}: image set differs from {args.tags[0]}", file=sys.stderr)
            return 2
    y = (frames[args.tags[0]]["grade"].values >= 2).astype(int)

    report = {"manifest": args.manifest, "n": int(len(y)), "n_referable": int(y.sum()), "tta": args.tta, "single": {}}
    for tag, df in frames.items():
        p = df["p_ge2"].values
        report["single"][tag] = {"auc": round(float(roc_auc_score(y, p)), 4), "ci95": [round(v, 4) for v in bootstrap_auc(y, p)]}
    if len(frames) > 1:
        mean_p = np.mean([df["p_ge2"].values for df in frames.values()], axis=0)
        best_tag = max(report["single"], key=lambda t: report["single"][t]["auc"])
        report["ensemble_mean"] = {"tags": args.tags, "auc": round(float(roc_auc_score(y, mean_p)), 4),
                                   "ci95": [round(v, 4) for v in bootstrap_auc(y, mean_p)],
                                   "vs_best_single": best_tag,
                                   "paired_auc_difference": paired_difference(y, mean_p, frames[best_tag]["p_ge2"].values)}
        # five-grade agreement as a secondary view
        for tag, df in frames.items():
            g = (df[["p_ge1", "p_ge2", "p_ge3", "p_ge4"]].values >= 0.5).sum(axis=1)
            report["single"][tag]["grade_accuracy"] = round(float((g == df["grade"].values).mean()), 4)
        ens = np.mean([df[["p_ge1", "p_ge2", "p_ge3", "p_ge4"]].values for df in frames.values()], axis=0)
        report["ensemble_mean"]["grade_accuracy"] = round(float(((ens >= 0.5).sum(axis=1) == frames[args.tags[0]]["grade"].values).mean()), 4)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
