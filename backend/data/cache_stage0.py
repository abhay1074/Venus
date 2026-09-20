"""Run Stage 0 once over every raw image and cache the result.

For each grading image: FOV-normalised 512x512 JPEG (quality 95), the quality
features, and a perceptual hash (for de-duplication across datasets). For each
DDR lesion image: the same, plus the four masks warped by the identical
transform, saved as one PNG with one bit per lesion type.

    python -m backend.data.cache_stage0 [--workers 14] [--limit N] [--only grading|lesions]

Output under VENUS_CACHE_DIR (default ~/venus-cache on WSL):

    stage0_512/<dataset>/<stem>.jpg
    lesions_512/<stem>.jpg, <stem>_mask.png
    stage0_index.csv        one row per image with quality features + phash
    lesion_index.csv

Reading is from the raw archive (possibly /mnt/c); writing is to the Linux
filesystem. Idempotent: existing outputs are skipped unless --force.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from backend.data import sources
from backend.venus.stage0_gate import normalise_fov, quality_features, quality_label, warp_like

CACHE_DIR = Path(os.getenv("VENUS_CACHE_DIR", str(Path.home() / "venus-cache")))
SIZE = 512
LESION_BITS = {"MA": 1, "HE": 2, "EX": 4, "SE": 8}


def phash(image512: np.ndarray, hash_size: int = 8) -> str:
    """Perceptual hash (DCT low frequencies), 64 bits as hex."""
    gray = cv2.cvtColor(image512, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:hash_size, :hash_size]
    med = np.median(dct[1:, 1:])
    bits = (dct > med).flatten()
    return "".join("1" if b else "0" for b in bits)


def _process_grading(row: dict) -> dict | None:
    out = CACHE_DIR / "stage0_512" / row["dataset"] / (Path(row["path"]).stem + ".jpg")
    try:
        if out.exists() and not row.get("force"):
            image = cv2.imread(str(out))
            if image is None:
                raise ValueError("unreadable cache")
            mask = None
            geometry = None
        else:
            raw = cv2.imread(row["path"], cv2.IMREAD_COLOR)
            if raw is None:
                return {"image_id": row["image_id"], "error": "unreadable"}
            image, mask, geometry = normalise_fov(raw, SIZE)
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if mask is None:
            # Recompute the mask/features from the cached image (cheap).
            image, mask, geometry = normalise_fov(image, SIZE)
        features = quality_features(image, mask, geometry)
        label, score, reason, _ = quality_label(features)
        return {"image_id": row["image_id"], "dataset": row["dataset"], "grade": row["grade"],
                "patient": row["patient"], "source_split": row["source_split"], "cache_path": str(out),
                "phash": phash(image), "quality_label": label, "quality_score": score, **features}
    except Exception as exc:  # noqa: BLE001 - one bad file must not kill the job
        return {"image_id": row["image_id"], "error": f"{type(exc).__name__}: {exc}"}


def _process_lesion(row: dict) -> dict | None:
    stem = Path(row["path"]).stem
    size = int(row.get("size", SIZE))
    folder = CACHE_DIR / f"lesions_{size}"
    out_img = folder / f"{stem}.jpg"
    out_mask = folder / f"{stem}_mask.png"
    try:
        if out_img.exists() and out_mask.exists() and not row.get("force"):
            return {"image_id": row["image_id"], "source_split": row["source_split"],
                    "cache_path": str(out_img), "mask_path": str(out_mask)}
        raw = cv2.imread(row["path"], cv2.IMREAD_COLOR)
        if raw is None:
            return {"image_id": row["image_id"], "error": "unreadable"}
        image, _, geometry = normalise_fov(raw, size)
        combined = np.zeros(raw.shape[:2], np.uint8)
        counts = {}
        for key, bit in LESION_BITS.items():
            m = cv2.imread(row[f"mask_{key}"], cv2.IMREAD_GRAYSCALE)
            if m is None:
                return {"image_id": row["image_id"], "error": f"mask {key} unreadable"}
            if m.shape != combined.shape:
                m = cv2.resize(m, (combined.shape[1], combined.shape[0]), interpolation=cv2.INTER_NEAREST)
            present = m > 0
            combined[present] |= bit
            counts[f"px_{key}"] = int(present.sum())
        warped = warp_like(combined, geometry, nearest=True)
        out_img.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(str(out_mask), warped)
        return {"image_id": row["image_id"], "source_split": row["source_split"],
                "cache_path": str(out_img), "mask_path": str(out_mask), **counts}
    except Exception as exc:  # noqa: BLE001
        return {"image_id": row["image_id"], "error": f"{type(exc).__name__}: {exc}"}


def run(rows: list[dict], fn, index_path: Path, workers: int) -> None:
    started = time.perf_counter()
    results, errors = [], []
    with Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(fn, rows, chunksize=8), 1):
            if result is None:
                continue
            (errors if "error" in result else results).append(result)
            if i % 500 == 0 or i == len(rows):
                rate = i / (time.perf_counter() - started)
                print(f"  {i}/{len(rows)}  {rate:.1f} img/s  errors {len(errors)}", flush=True)
    if results:
        keys = sorted({k for r in results for k in r}, key=lambda k: (k != "image_id", k))
        with open(index_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
    if errors:
        with open(index_path.with_suffix(".errors.csv"), "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_id", "error"])
            writer.writeheader()
            writer.writerows(errors)
    print(f"wrote {index_path}  ({len(results)} rows, {len(errors)} errors, {(time.perf_counter() - started) / 60:.1f} min)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(os.cpu_count() - 2, 1))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", choices=["grading", "lesions"], default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--size", type=int, default=SIZE, help="lesion cache frame size (the grading cache is always 512)")
    args = parser.parse_args(argv)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"cache dir {CACHE_DIR}  data root {sources.DATA_ROOT}  workers {args.workers}")

    if args.only in (None, "grading"):
        df = sources.all_grading()
        if args.limit:
            df = df.groupby("dataset", group_keys=False).head(args.limit)
        rows = df.to_dict("records")
        for r in rows:
            r["force"] = args.force
        print(f"grading images: {len(rows)}")
        run(rows, _process_grading, CACHE_DIR / "stage0_index.csv", args.workers)

    if args.only in (None, "lesions"):
        df = sources.ddr_lesions()
        if args.limit:
            df = df.groupby("source_split", group_keys=False).head(args.limit)
        rows = df.to_dict("records")
        for r in rows:
            r["force"] = args.force
            r["size"] = args.size
        print(f"lesion images: {len(rows)} at {args.size} px")
        suffix = "" if args.size == SIZE else f"_{args.size}"
        run(rows, _process_lesion, CACHE_DIR / f"lesion_index{suffix}.csv", args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
