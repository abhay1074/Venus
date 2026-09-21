"""Turn a site's labelled fundus images into a fingerprinted calibration manifest.

    python -m backend.data.site_manifest --name <site> --images <folder> --labels labels.csv

labels.csv has one row per image: `image` (file name in the folder, or a path
relative to it), `grade` (ICDR 0-4 from the site's reader) and optionally
`patient` (so the manifest is patient-aware; defaults to the file stem).
Ungradable images are left out of labels.csv, or given grade -1 and skipped.

Every image goes through the same Stage 0 normalisation as the training data
(cache_stage0._process_grading -> 512 px FOV frame, quality features, hash),
and the manifest backend/data/manifests/site_<name>.csv is written with the
same columns as calibration.csv. Its SHA-256 is printed: that is the value the
site operating point will carry and the server will verify on every start.

Next steps (scripts/site-calibrate.sh does all three):
    backend.eval.score_grader --weights ... --manifests site_<name>
    backend.eval.calibrate --tag grader_v2 --site site_<name>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from backend.data.cache_stage0 import CACHE_DIR, _process_grading, run
from backend.venus.config import MANIFEST_DIR, sha256_of_file


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="site identifier, e.g. district-north-phc")
    parser.add_argument("--images", required=True, help="folder with the site's fundus images")
    parser.add_argument("--labels", required=True, help="CSV with image,grade[,patient]")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    folder = Path(args.images)
    labels = pd.read_csv(args.labels)
    if not {"image", "grade"} <= set(labels.columns):
        print("labels.csv needs columns image,grade[,patient]", file=sys.stderr)
        return 2
    labels = labels[labels["grade"].astype(int) >= 0].copy()
    labels["path"] = [str(folder / str(name)) for name in labels["image"]]
    missing = [p for p in labels["path"] if not Path(p).exists()]
    if missing:
        print(f"{len(missing)} labelled images not found, e.g. {missing[0]}", file=sys.stderr)
        return 2
    name = args.name.strip().lower().replace(" ", "-")
    dataset = f"site_{name}"
    rows = [{"image_id": f"{dataset}/{Path(p).stem}", "dataset": dataset, "path": p, "grade": int(g),
             "patient": f"{dataset}/{r.get('patient', Path(p).stem)}" if isinstance(r.get("patient"), str) else f"{dataset}/{Path(p).stem}",
             "source_split": "site"}
            for (p, g, r) in zip(labels["path"], labels["grade"], labels.to_dict("records"))]
    index_path = CACHE_DIR / f"{dataset}_index.csv"
    run(rows, _process_grading, index_path, args.workers)
    index = pd.read_csv(index_path)
    bad = index[index["error"].notna()] if "error" in index.columns else index.iloc[0:0]
    good = index[index["error"].isna()] if "error" in index.columns else index
    cols = ["image_id", "dataset", "grade", "patient", "source_split", "cache_path", "quality_label", "quality_score", "phash"]
    manifest = good[cols].sort_values("image_id")
    out = MANIFEST_DIR / f"{dataset}.csv"
    manifest.to_csv(out, index=False)
    fingerprint = sha256_of_file(out)
    counts = manifest["grade"].value_counts().sort_index().to_dict()
    print(f"site manifest {out.name}: {len(manifest)} images ({len(bad)} unreadable), {manifest['patient'].nunique()} patients, grades {counts}")
    print(f"quality labels {manifest['quality_label'].value_counts().to_dict()}")
    print(f"SHA-256 {fingerprint}")
    if (manifest["grade"] >= 2).sum() < 30:
        print("warning: fewer than 30 referable images; the 90 % sensitivity threshold will be coarse (backend/eval/site_calibration.py: ~400 images, "
              "~100 referable, gave a stable operating point on Messidor-2)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
