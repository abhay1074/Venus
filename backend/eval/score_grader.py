"""Run a trained grader over the calibration / test manifests and save predictions.

    wsl bash scripts/wsl-gpu.sh backend.eval.score_grader --weights ~/venus-cache/models/grader_v2.weights.h5

Writes <cache>/predictions/<tag>_<manifest>.csv with one row per image:
image_id, patient, dataset, grade, p_ge1..p_ge4. Scoring is mechanical here;
the decision about thresholds happens in backend.eval.calibrate, which reads
the calibration predictions only, and touches the external test once.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from backend.training.train_grader import SIZE, build_model, dataset
from backend.venus.config import MANIFEST_DIR

CACHE_DIR = Path(os.getenv("VENUS_CACHE_DIR", str(Path.home() / "venus-cache")))
MANIFESTS = ["calibration", "external_test_ddr", "heldout_eyepacs_frozen", "val"]


def tta_predict(model: keras.Model, ds: tf.data.Dataset, tta: bool) -> np.ndarray:
    if not tta:
        return model.predict(ds, verbose=0)
    outs = []
    for k in range(4):
        outs.append(model.predict(ds.map(lambda x, y: (tf.image.rot90(x, k), y)), verbose=0))
    outs.append(model.predict(ds.map(lambda x, y: (tf.image.flip_left_right(x), y)), verbose=0))
    return np.mean(outs, axis=0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--backbone", default="B3", choices=["B0", "B3"])
    parser.add_argument("--tag", default="grader_v2")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--manifests", nargs="*", default=MANIFESTS)
    args = parser.parse_args(argv)

    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    keras.mixed_precision.set_global_policy("mixed_float16")
    model = build_model(args.backbone)
    model.load_weights(os.path.expanduser(args.weights))
    out_dir = CACHE_DIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.manifests:
        path = MANIFEST_DIR / f"{name}.csv"
        if not path.exists():
            print(f"skip {name}: no manifest")
            continue
        df = pd.read_csv(path)
        probs = tta_predict(model, dataset(df, args.batch, train=False), args.tta)
        out = df[["image_id", "patient", "dataset", "grade"]].copy()
        for k in range(4):
            out[f"p_ge{k + 1}"] = np.round(probs[:, k].astype(np.float64), 6)
        out_path = out_dir / f"{args.tag}_{name}{'_tta' if args.tta else ''}.csv"
        out.to_csv(out_path, index=False)
        # The external test is scored by backend.eval.calibrate exactly once,
        # after the threshold is locked; nothing about it is printed here.
        if name.startswith("external"):
            print(f"{name:26s} n={len(df):5d}  (not summarised here)  -> {out_path.name}", flush=True)
            continue
        from sklearn.metrics import roc_auc_score
        referable = (df["grade"] >= 2).astype(int)
        auc = roc_auc_score(referable, probs[:, 1]) if referable.nunique() > 1 else float("nan")
        print(f"{name:26s} n={len(df):5d}  referable AUC {auc:.4f}  -> {out_path.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
