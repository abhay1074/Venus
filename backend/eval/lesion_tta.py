"""Can microaneurysm segmentation be improved cheaply? Measured on DDR VALID.

    wsl bash scripts/wsl-gpu.sh backend.eval.lesion_tta

The served lesion U-Net's weakest class by a distance is MA: pixel AUPR 0.079 on
the DDR test split. This script measures candidate improvements on the DDR
**valid** split only — the test split is not touched, so nothing here can be
tuned against the number that gets reported:

  1. plain          the served 512 px network, as deployed (baseline)
  2. tta            the same network, 4 rotations + horizontal flip averaged
  3. hires          the 1024 px network from config/experiments/lesion_unet_1024.json
  4. hires_tta      that network with the same averaging
  5. ensemble       mean of plain and hires probabilities, at 512

Writes backend/config/experiments/ma_improvement.json. Whether any of these is
worth shipping is a separate judgement recorded in docs/VALIDATION.md; a gain in
pixel AUPR on valid is not by itself a reason to change the served path, because
the downstream decision is the referable one and MA-only findings are ICDR 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backend.venus.config import CONFIG_DIR

OUT = CONFIG_DIR / "experiments" / "ma_improvement.json"
LESIONS = ["MA", "HE", "EX", "SE"]


def tta_predict(model, batch: np.ndarray, tta: bool) -> np.ndarray:
    """Dihedral averaging in probability space; each view is un-transformed
    before averaging so the maps stay in register."""
    if not tta:
        return np.asarray(model.predict_on_batch(batch), np.float32)
    views = []
    for k in range(4):
        rotated = np.rot90(batch, k, axes=(1, 2))
        out = np.asarray(model.predict_on_batch(np.ascontiguousarray(rotated)), np.float32)
        views.append(np.rot90(out, -k, axes=(1, 2)))
    flipped = np.asarray(model.predict_on_batch(np.ascontiguousarray(batch[:, :, ::-1])), np.float32)
    views.append(flipped[:, :, ::-1])
    return np.mean(views, axis=0)


def score(probs_by_index, masks, size) -> dict:
    """Per-class pixel AUPR over the whole split, from the same 8,192-bin
    histogram estimator train_lesion_unet.evaluate uses (verified equal to
    sklearn's average_precision_score to 4 dp)."""
    from backend.training.train_lesion_unet import BITS, BINS
    pos = np.zeros((4, BINS), np.int64)
    neg = np.zeros((4, BINS), np.int64)
    for i, probs in probs_by_index:
        idx = np.clip((probs * BINS).astype(np.int32), 0, BINS - 1)
        for c, key in enumerate(LESIONS):
            y = (masks[i] & BITS[key]) > 0
            pos[c] += np.bincount(idx[..., c][y], minlength=BINS)
            neg[c] += np.bincount(idx[..., c][~y], minlength=BINS)
    out = {}
    for c, key in enumerate(LESIONS):
        n_pos = int(pos[c].sum())
        if n_pos == 0:
            out[key] = None
            continue
        tp = np.cumsum(pos[c][::-1])[::-1].astype(np.float64)
        fp = np.cumsum(neg[c][::-1])[::-1].astype(np.float64)
        recall = tp / n_pos
        precision = tp / np.maximum(tp + fp, 1)
        out[key] = round(float(np.sum((recall - np.append(recall[1:], 0.0)) * precision)), 4)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args(argv)

    import tensorflow as tf
    from tensorflow import keras
    from backend.training.train_lesion_unet import CACHE_DIR, build_unet, load_split
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)

    models_dir = CACHE_DIR / "models"
    runs, results = {}, {}

    def evaluate_model(weights: Path, size: int, tta: bool, index_suffix: str):
        model = build_unet(size=size)
        model.load_weights(weights)
        index = pd.read_csv(CACHE_DIR / f"lesion_index{index_suffix}.csv")
        images, masks = load_split(index, "valid", size)
        maps = []
        for i in range(0, len(images), args.batch):
            batch = images[i:i + args.batch].astype(np.float32)
            probs = tta_predict(model, batch, tta)
            for j in range(len(batch)):
                maps.append((i + j, probs[j]))
        keras.backend.clear_session()
        return maps, masks, images.shape[1]

    plain_w = models_dir / "lesion_unet.weights.h5"
    hires_w = models_dir / "lesion_unet_1024.weights.h5"

    print("plain 512 ...", flush=True)
    maps, masks, size = evaluate_model(plain_w, 512, False, "")
    results["plain_512"] = score(maps, masks, size)
    plain_maps = {i: p for i, p in maps}

    print("tta 512 ...", flush=True)
    maps, masks, size = evaluate_model(plain_w, 512, True, "")
    results["tta_512"] = score(maps, masks, size)

    if hires_w.exists():
        import cv2
        print("plain 1024 ...", flush=True)
        maps1024, masks1024, _ = evaluate_model(hires_w, 1024, False, "_1024")
        results["plain_1024"] = score(maps1024, masks1024, 1024)

        print("tta 1024 ...", flush=True)
        maps_t, _, _ = evaluate_model(hires_w, 1024, True, "_1024")
        results["tta_1024"] = score(maps_t, masks1024, 1024)

        # Ensemble at the served 512 frame: the 1024 maps are averaged down,
        # which is what the serving path would have to do anyway.
        print("ensemble 512+1024 at 512 ...", flush=True)
        ens = []
        for i, p1024 in maps1024:
            down = cv2.resize(p1024, (512, 512), interpolation=cv2.INTER_AREA)
            ens.append((i, (plain_maps[i] + down) / 2.0))
        results["ensemble_at_512"] = score(ens, masks, 512)

    baseline = results["plain_512"]["MA"]
    summary = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "question": "Can microaneurysm pixel AUPR be improved cheaply, measured on the DDR valid split only?",
        "split": "DDR valid (the DDR test split was not touched)",
        "metric": "pixel AUPR over every pixel of the split, per lesion class",
        "results": results,
        "ma_delta_vs_served": {k: round(v["MA"] - baseline, 4) for k, v in results.items() if v.get("MA") is not None},
        "baseline_note": ("All five variants were measured in float32 in one run, which is what the CPU serving path "
                          "uses; the training-time evaluation recorded a slightly different figure for the same "
                          "weights because it ran in mixed float16 on the GPU. The comparison between rows is "
                          "internally consistent, which is what the decision rests on."),
        "note": "Any change adopted from this table would still carry the test number recorded for the model version "
                "that was scored once; the DDR test split is not re-scored to advertise an improvement.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    for name, r in results.items():
        print(f"{name:20s} " + "  ".join(f"{k} {r[k]}" for k in LESIONS))
    print(f"\nMA vs served: {summary['ma_delta_vs_served']}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
