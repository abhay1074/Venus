"""Compute a 256-bit perceptual hash for every cached image and report the
nearest-neighbour distance distribution, so the de-duplication threshold is
chosen from the data rather than guessed.

    python -m backend.data.rehash

Adds a `phash256` column to stage0_index.csv (in place) and writes
stage0_nn_distances.json with the histogram of each image's nearest-neighbour
Hamming distance, overall and across datasets.
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CACHE_DIR = Path(os.getenv("VENUS_CACHE_DIR", str(Path.home() / "venus-cache")))
HASH_SIZE = 16   # 16x16 DCT block -> 256 bits


def phash256(path: str) -> str:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return ""
    small = cv2.resize(image, (HASH_SIZE * 4, HASH_SIZE * 4), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:HASH_SIZE, :HASH_SIZE]
    med = np.median(dct[1:, 1:])
    bits = np.packbits((dct > med).flatten())
    return bits.tobytes().hex()


def nearest_distances(hashes: np.ndarray, block: int = 2048) -> np.ndarray:
    """Hamming distance to the nearest other image, brute force in blocks."""
    n = len(hashes)
    bits = np.unpackbits(hashes, axis=1).astype(np.uint8)          # (n, 256)
    out = np.full(n, 999, np.int32)
    for start in range(0, n, block):
        chunk = bits[start:start + block].astype(np.int32)
        # distance = popcount(a xor b) = |a| + |b| - 2 a.b
        dots = chunk @ bits.T.astype(np.int32)
        dist = chunk.sum(1)[:, None] + bits.sum(1)[None, :] - 2 * dots
        idx = np.arange(start, start + len(chunk))
        dist[np.arange(len(chunk)), idx] = 999
        out[start:start + len(chunk)] = dist.min(axis=1)
    return out


def main() -> int:
    index_path = CACHE_DIR / "stage0_index.csv"
    df = pd.read_csv(index_path, dtype={"phash": str})
    started = time.perf_counter()
    with Pool(max(os.cpu_count() - 2, 1)) as pool:
        df["phash256"] = pool.map(phash256, df["cache_path"].astype(str).tolist(), chunksize=64)
    print(f"hashed {len(df)} images in {(time.perf_counter() - started) / 60:.1f} min")
    df.to_csv(index_path, index=False)

    hashes = np.array([np.frombuffer(bytes.fromhex(h), np.uint8) for h in df["phash256"]])
    nn = nearest_distances(hashes)
    df["nn_distance"] = nn
    hist = {int(d): int(c) for d, c in zip(*np.unique(nn, return_counts=True))}
    same_patient = df.assign(nn=nn).groupby("dataset")["nn"].describe().to_dict()
    report = {"hash_bits": HASH_SIZE * HASH_SIZE, "nearest_neighbour_histogram": hist,
              "percentiles": {p: float(np.percentile(nn, p)) for p in (0.1, 1, 5, 10, 25, 50)},
              "per_dataset": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in same_patient.items()}}
    with open(CACHE_DIR / "stage0_nn_distances.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("nearest-neighbour distance percentiles:", report["percentiles"])
    print("histogram (d: count) for d <= 40:", {d: c for d, c in hist.items() if d <= 40})
    return 0


if __name__ == "__main__":
    sys.exit(main())
