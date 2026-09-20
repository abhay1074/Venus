"""Build the split manifests from the Stage 0 index, with fingerprints.

Split discipline (architecture §10):

    * De-duplication by perceptual hash across ALL datasets before splitting:
      256-bit DCT hash (backend.data.rehash), Hamming distance <= DEDUP_HAMMING.
      The threshold sits in the empty gap of the nearest-neighbour distance
      distribution (exact/near duplicates at 0-4, natural neighbours from ~12),
      recorded in <cache>/stage0_nn_distances.json. Within a duplicate group
      the earliest image_id is kept; any duplicate of a test image is dropped
      from training, never the other way round.
    * EyePACS is split by patient (the file name encodes it); APTOS and DDR
      have no patient id, so each image is its own group.
    * External test, frozen: the DDR test split (a different acquisition source
      from everything in training), ungradable images removed. Written with a
      SHA-256 fingerprint; scored once per model version.
    * EyePACS frozen set: the 1,500 images of the previous external manifest are
      excluded from training by PATIENT, so they remain a within-source,
      held-out-patient test for continuity with the earlier grader.
    * Calibration: 2,000 EyePACS images, patient-disjoint from everything else,
      held out BEFORE training. The operating point is chosen here.
    * Validation: ~8% of the remaining pool by group (checkpoint selection only).
    * Train: everything else. DDR's ungradable class (5) is kept in a separate
      manifest for the quality classifier and never enters grading training.

Outputs (backend/data/manifests/):
    train.csv  val.csv  calibration.csv  external_test_ddr.csv
    heldout_eyepacs_frozen.csv  ungradable.csv  manifests.json (counts + SHA-256)

    python -m backend.data.build_manifests
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backend.venus.config import MANIFEST_DIR, sha256_of_file

CACHE_DIR = Path(os.getenv("VENUS_CACHE_DIR", str(Path.home() / "venus-cache")))
SEED = 42
DEDUP_HAMMING = 10
CALIBRATION_N = 2000
VAL_FRACTION = 0.08
PREVIOUS_FROZEN = MANIFEST_DIR / "eyepacs_frozen_predictions.csv"


def hamming_groups(hashes: np.ndarray, max_distance: int) -> np.ndarray:
    """Group index per row: rows whose hashes are within max_distance share a group.

    Hashes as Python ints of `bits` bits; exact-bucket pre-filter on 16-bit
    slices (a pair within distance < bits/16 must agree exactly on at least one
    slice, by pigeonhole) keeps this near O(n).
    """
    bits = 256
    assert max_distance < bits // 16
    n = len(hashes)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for shift in range(0, bits, 16):
        buckets: dict[int, list[int]] = {}
        for i, h in enumerate(hashes):
            buckets.setdefault((h >> shift) & 0xFFFF, []).append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            for a_idx in range(len(members)):
                a = members[a_idx]
                for b in members[a_idx + 1:]:
                    d = bin(int(hashes[a]) ^ int(hashes[b])).count("1")
                    if d <= max_distance:
                        union(a, b)
    return np.array([find(i) for i in range(n)])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=str(CACHE_DIR / "stage0_index.csv"))
    args = parser.parse_args(argv)

    df = pd.read_csv(args.index, dtype={"phash": str, "phash256": str})
    if "phash256" not in df.columns:
        raise SystemExit("run `python -m backend.data.rehash` first (256-bit hashes)")
    df = df.sort_values("image_id").reset_index(drop=True)
    print(f"index rows {len(df)}  per dataset {df['dataset'].value_counts().to_dict()}")

    # ---------------------------------------------------------- dedup --
    hashes = [int(h, 16) for h in df["phash256"]]
    df["dup_group"] = hamming_groups(hashes, DEDUP_HAMMING)
    group_sizes = df.groupby("dup_group")["image_id"].transform("size")
    dup_rows = int((group_sizes > 1).sum())
    print(f"duplicate groups: {int((df.groupby('dup_group').size() > 1).sum())} groups covering {dup_rows} images")

    # Roles decided before dedup so that a test image always wins its group.
    df["is_ddr_test"] = (df["dataset"] == "ddr") & (df["source_split"] == "test")
    prev = pd.read_csv(PREVIOUS_FROZEN) if PREVIOUS_FROZEN.exists() else pd.DataFrame(columns=["image"])
    prev_patients = set("eyepacs/" + prev["image"].astype(str).str.split("_").str[0])
    df["is_prev_frozen"] = df["patient"].isin(prev_patients)
    # Within a group keep: DDR-test first, then previous-frozen, then the first id.
    df["_rank"] = np.where(df["is_ddr_test"], 0, np.where(df["is_prev_frozen"], 1, 2))
    keep = df.sort_values(["dup_group", "_rank", "image_id"]).drop_duplicates("dup_group", keep="first")
    dropped = df.loc[~df.index.isin(keep.index)]
    print(f"dropped {len(dropped)} near-duplicates ({dropped['dataset'].value_counts().to_dict()})")
    cross = dropped.merge(keep[["dup_group", "dataset"]].rename(columns={"dataset": "kept_dataset"}), on="dup_group")
    cross = cross[cross["dataset"] != cross["kept_dataset"]]
    print(f"  of which cross-dataset: {len(cross)}")
    df = keep.drop(columns=["_rank"]).reset_index(drop=True)

    # ------------------------------------------------------------ roles --
    ungradable = df[df["grade"] == 5]
    df = df[df["grade"] <= 4]
    external = df[df["is_ddr_test"]]
    heldout_prev = df[df["is_prev_frozen"] & ~df["is_ddr_test"]]
    pool = df[~df["is_ddr_test"] & ~df["is_prev_frozen"]]

    rng = np.random.default_rng(SEED)
    eyepacs_patients = np.array(sorted(pool.loc[pool["dataset"] == "eyepacs", "patient"].unique()), dtype=object)
    rng.shuffle(eyepacs_patients)
    cal_patients, cal_count = [], 0
    counts = pool[pool["dataset"] == "eyepacs"].groupby("patient").size()
    for p in eyepacs_patients:
        if cal_count >= CALIBRATION_N:
            break
        cal_patients.append(p)
        cal_count += int(counts[p])
    cal_patients = set(cal_patients)
    calibration = pool[pool["patient"].isin(cal_patients)]
    rest = pool[~pool["patient"].isin(cal_patients)]

    groups = np.array(sorted(rest["patient"].unique()), dtype=object)
    rng.shuffle(groups)
    n_val_groups = int(round(len(groups) * VAL_FRACTION))
    val_groups = set(groups[:n_val_groups])
    val = rest[rest["patient"].isin(val_groups)]
    train = rest[~rest["patient"].isin(val_groups)]

    # Leakage assertions: no patient in two roles, no image in two roles.
    roles = {"train": train, "val": val, "calibration": calibration, "external_test_ddr": external,
             "heldout_eyepacs_frozen": heldout_prev}
    names = list(roles)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (set(roles[a]["patient"]) & set(roles[b]["patient"])), f"patient overlap {a}/{b}"
            assert not (set(roles[a]["dup_group"]) & set(roles[b]["dup_group"])), f"duplicate overlap {a}/{b}"

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["image_id", "dataset", "grade", "patient", "source_split", "cache_path", "quality_label", "quality_score", "phash256"]
    summary = {"written_at": datetime.now(timezone.utc).isoformat(), "seed": SEED, "dedup_hamming": DEDUP_HAMMING,
               "duplicates_dropped": int(len(dropped)), "cross_dataset_duplicates": int(len(cross)), "manifests": {}}
    for name, frame in {**roles, "ungradable": ungradable}.items():
        path = MANIFEST_DIR / f"{name}.csv"
        frame = frame.sort_values("image_id")
        frame[columns].to_csv(path, index=False)
        grades = frame["grade"].value_counts().sort_index().to_dict() if len(frame) else {}
        summary["manifests"][name] = {"n": int(len(frame)), "sha256": sha256_of_file(path),
                                      "datasets": frame["dataset"].value_counts().to_dict(),
                                      "grades": {int(k): int(v) for k, v in grades.items()},
                                      "referable_fraction": round(float(frame["grade"].between(2, 4).mean()), 4) if len(frame) else None}
        print(f"{name:24s} n={len(frame):6d}  {summary['manifests'][name]['datasets']}  grades {summary['manifests'][name]['grades']}")
    with open(MANIFEST_DIR / "manifests.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"external test fingerprint {summary['manifests']['external_test_ddr']['sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
