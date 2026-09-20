"""Freeze an additional external test set (Messidor-2) with a fingerprint.

    wsl bash scripts/wsl-gpu.sh backend.data.freeze_external --dataset messidor2

1. Runs Stage 0 once over the images (same cache job as the training data,
   written under <cache>/stage0_512/<dataset>/), with quality features and the
   256-bit perceptual hash.
2. Checks every image against the hashes of train / val / calibration at the
   de-duplication threshold; any near-duplicate is reported and excluded, so
   the set is guaranteed disjoint from everything the grader saw.
3. Writes backend/data/manifests/external_test_<dataset>.csv (ungradable
   images excluded, kept in a side manifest) and records its SHA-256 in
   manifests.json. Scoring happens once, in backend.eval.score_external.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from backend.data import sources
from backend.data.build_manifests import DEDUP_HAMMING
from backend.data.cache_stage0 import CACHE_DIR, _process_grading
from backend.data.rehash import phash256
from backend.venus.config import MANIFEST_DIR, sha256_of_file


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="messidor2", choices=["messidor2"])
    parser.add_argument("--workers", type=int, default=max(os.cpu_count() - 2, 1))
    args = parser.parse_args(argv)

    df = sources.messidor2()
    rows = df.to_dict("records")
    with Pool(args.workers) as pool:
        results = pool.map(_process_grading, rows, chunksize=8)
    ok = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    print(f"cached {len(ok)} images, {len(errors)} errors")
    out = pd.DataFrame(ok).merge(df[["image_id", "adjudicated_dme", "adjudicated_gradable"]], on="image_id")
    with Pool(args.workers) as pool:
        out["phash256"] = pool.map(phash256, out["cache_path"].astype(str).tolist(), chunksize=32)

    # Disjointness from everything the grader saw (train / val / calibration).
    seen = pd.concat([pd.read_csv(MANIFEST_DIR / f"{n}.csv", dtype={"phash256": str}) for n in ("train", "val", "calibration")])
    seen_hashes = [int(h, 16) for h in seen["phash256"]]
    near = []
    for h in out["phash256"]:
        hi = int(h, 16)
        d = min(hamming(hi, s) for s in seen_hashes)
        near.append(d)
    out["nearest_train_distance"] = near
    dup = out["nearest_train_distance"] <= DEDUP_HAMMING
    print(f"near-duplicates of training/val/calibration images: {int(dup.sum())} (excluded); "
          f"nearest-distance percentiles 1/5/50: {np.percentile(near, [1, 5, 50])}")
    out = out[~dup]

    gradable = out[out["grade"] <= 4]
    ungradable = out[out["grade"] == 5]
    columns = ["image_id", "dataset", "grade", "patient", "source_split", "cache_path", "quality_label", "quality_score",
               "phash256", "adjudicated_dme", "adjudicated_gradable"]
    name = f"external_test_{args.dataset}"
    path = MANIFEST_DIR / f"{name}.csv"
    gradable.sort_values("image_id")[columns].to_csv(path, index=False)
    ungradable.sort_values("image_id")[columns].to_csv(MANIFEST_DIR / f"ungradable_{args.dataset}.csv", index=False)
    summary_path = MANIFEST_DIR / "manifests.json"
    summary = json.load(open(summary_path, encoding="utf-8")) if summary_path.exists() else {"manifests": {}}
    grades = gradable["grade"].value_counts().sort_index().to_dict()
    summary["manifests"][name] = {
        "n": int(len(gradable)), "sha256": sha256_of_file(path), "datasets": {args.dataset: int(len(gradable))},
        "grades": {int(k): int(v) for k, v in grades.items()},
        "referable_fraction": round(float((gradable["grade"] >= 2).mean()), 4),
        "patients": int(gradable["patient"].nunique()),
        "near_duplicates_of_training_excluded": int(dup.sum()),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "labels": "Krause et al. 2018 adjudicated ICDR grades (messidor_data.csv)",
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"{name}: n={len(gradable)} grades {grades} referable {summary['manifests'][name]['referable_fraction']} "
          f"fingerprint {summary['manifests'][name]['sha256'][:16]}...  (ungradable {len(ungradable)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
