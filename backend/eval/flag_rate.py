"""Measure the human-review flag rate and the attention-agreement distribution.

    wsl bash scripts/wsl-gpu.sh backend.eval.flag_rate --n 600

Runs Stage 0 -> 1 -> 2 -> 3 (the served path, current checkpoints, no report)
over a grade-stratified sample of the VALIDATION manifest and records, per
image: the CNN grade and calibrated P(referable), the rule grade and counts,
every fusion flag, the attention-agreement score, and the true grade. From
those:

    * human-review flag rate (and the share of each reason) -- the workload
      the district simulation absorbs (architecture §6.3 expects 10-15 %)
    * attention agreement for correct vs incorrect referable decisions
      (architecture §3.3: "if it separates them, it is a third review signal")
    * the rule grader alone: sensitivity / specificity for referable DR and
      exact-grade agreement, so its role as a consistency check is quantified
    * how often a flag lands on an image the CNN got wrong (flag usefulness)

Writes backend/config/validation_flags.json. Validation images only: the
external test stays untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import pandas as pd

from backend.venus import stage0_gate, stage1_segment, stage2_grade, stage3_explain
from backend.venus.config import CONFIG_DIR, GRADER_TAG, MANIFEST_DIR, MODEL_VERSION, operating_point


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--manifest", default="val")
    args = parser.parse_args(argv)

    df = pd.read_csv(MANIFEST_DIR / f"{args.manifest}.csv")
    per_grade = max(args.n // 5, 1)
    sample = pd.concat([g.sample(min(per_grade, len(g)), random_state=42) for _, g in df.groupby("grade")])
    point = operating_point()
    stage0_gate.load_gate(); stage0_gate.load_quality(); stage2_grade.load_grader(); stage1_segment.load_unet()
    rows, started = [], time.perf_counter()
    for i, row in enumerate(sample.itertuples(), 1):
        image = cv2.imread(row.cache_path)
        if image is None:
            continue
        s0 = stage0_gate.run(image)
        rec = {"image_id": row.image_id, "grade": int(row.grade), "accepted": s0["accepted"],
               "quality": s0["quality"]["label"] if s0["quality"] else "reject"}
        if s0["accepted"]:
            s1 = stage1_segment.run(s0["image"], s0["mask"], original=s0["original"])
            s2 = stage2_grade.run(image, s1, stage0_image=s0["original"])
            s3 = stage3_explain.run(s0, s1, s2)
            f = s2["fusion"]
            if s3["attention_agreement"]["flag"]:
                f["flag_reasons"].append("attention"); f["flag_for_review"] = True
            rec.update({"cnn_grade": s2["cnn"]["grade"], "p_referable": f["p_referable"], "referable": f["referable"],
                        "rule_grade": s2["rule"]["grade"], "rule_referable": s2["rule"]["referable"],
                        "counts": s2["rule"]["counts"], "flag": f["flag_for_review"], "reasons": f["flag_reasons"],
                        "attention": s3["attention_agreement"]["score"], "lesion_method": s1["method"]})
        rows.append(rec)
        if i % 50 == 0:
            print(f"  {i}/{len(sample)}  {(time.perf_counter() - started) / i:.2f} s/img", flush=True)

    g = [r for r in rows if r["accepted"]]
    y = np.array([r["grade"] >= 2 for r in g]); pred = np.array([r["referable"] for r in g])
    correct = y == pred
    flags = np.array([r["flag"] for r in g])
    reasons = {}
    for r in g:
        for reason in r["reasons"]:
            key = ("disagreement" if "differ" in reason else "no_lesion" if "no lesion" in reason
                   else "abstain" if "within" in reason else "attention")
            reasons[key] = reasons.get(key, 0) + 1
    att = np.array([r["attention"] if r["attention"] is not None else np.nan for r in g])
    have = ~np.isnan(att) & np.array([r["referable"] for r in g])

    def describe(v):
        v = v[~np.isnan(v)]
        return {"n": int(len(v)), "mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4),
                "p25": round(float(np.percentile(v, 25)), 4), "p75": round(float(np.percentile(v, 75)), 4)} if len(v) else {"n": 0}

    rule_y = np.array([r["rule_referable"] for r in g])
    rule_grade = np.array([r["rule_grade"] for r in g]); true_grade = np.array([r["grade"] for r in g])
    cnn_grade = np.array([r["cnn_grade"] for r in g])
    out = {
        "model_version": MODEL_VERSION, "grader": GRADER_TAG, "written_at": datetime.now(timezone.utc).isoformat(),
        "manifest": args.manifest, "n_sampled": len(rows), "n_gradable": len(g),
        "retake_rate": round(1 - len(g) / max(len(rows), 1), 4),
        "quality_labels": {k: int(v) for k, v in pd.Series([r["quality"] for r in rows]).value_counts().items()},
        "lesion_method": g[0]["lesion_method"] if g else None,
        "flag_rate": round(float(flags.mean()), 4),
        "flag_reasons": reasons,
        "flag_precision_for_cnn_errors": round(float((~correct[flags]).mean()), 4) if flags.any() else None,
        "cnn_error_rate_flagged_vs_unflagged": {"flagged": round(float((~correct[flags]).mean()), 4) if flags.any() else None,
                                                "unflagged": round(float((~correct[~flags]).mean()), 4) if (~flags).any() else None},
        "referable_accuracy_cnn": round(float(correct.mean()), 4),
        "attention_agreement": {
            "note": "referable CNN calls only, where the heatmap is meaningful",
            "correct": describe(att[have & correct]), "incorrect": describe(att[have & ~correct]),
        },
        "rule_grader_alone": {
            "referable_sensitivity": round(float(rule_y[y].mean()), 4) if y.any() else None,
            "referable_specificity": round(float((~rule_y[~y]).mean()), 4) if (~y).any() else None,
            "exact_grade_agreement_with_truth": round(float((rule_grade == true_grade).mean()), 4),
            "within_one_of_truth": round(float((np.abs(rule_grade - true_grade) <= 1).mean()), 4),
            "agreement_with_cnn_within_one": round(float((np.abs(rule_grade - cnn_grade) <= 1).mean()), 4),
        },
        "seconds_per_image": round((time.perf_counter() - started) / max(len(rows), 1), 3),
        "rows": rows,
    }
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_DIR / "validation_flags.json", "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else o)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
