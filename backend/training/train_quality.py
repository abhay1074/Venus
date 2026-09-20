"""Train the image-quality classifier (good / usable / reject) on EyeQ labels.

    wsl bash scripts/wsl-gpu.sh backend.training.train_quality --epochs 12

EyeQ (Fu et al. 2019) labels 12,543 EyePACS training images we have on disk
(its test labels refer to EyePACS test images we do not). They are split here
by patient into train / val / test (70 / 10 / 20). EfficientNet-B0 at 256x256
on the Stage 0 cache, softmax over three classes, class-weighted
cross-entropy. Reported: 3-class accuracy, and "ungradable detection" AUC
(reject vs the rest, the architecture's > 0.95 target) on the held-out
patients, plus recall on DDR's ungradable class as an out-of-source check
(recall only: DDR's gradable images carry no quality label).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score
from tensorflow import keras

from backend.venus import nets
from backend.venus.config import MANIFEST_DIR

CACHE_DIR = Path(os.getenv("VENUS_CACHE_DIR", str(Path.home() / "venus-cache")))
MODEL_DIR = CACHE_DIR / "models"
EXTERNAL = Path(__file__).resolve().parents[1] / "data" / "external"
SIZE = 256
CLASSES = ["good", "usable", "reject"]


def load_eyeq() -> pd.DataFrame:
    labels = pd.read_csv(EXTERNAL / "Label_EyeQ_train.csv")
    labels["stem"] = labels["image"].str.replace(".jpeg", "", regex=False)
    labels["image_id"] = "eyepacs/" + labels["stem"]
    labels["patient"] = labels["stem"].str.split("_").str[0]
    index = pd.read_csv(CACHE_DIR / "stage0_index.csv")[["image_id", "cache_path"]]
    df = labels.merge(index, on="image_id", how="inner")
    rng = np.random.default_rng(42)
    patients = np.unique(df["patient"].values)
    rng.shuffle(patients)
    n = len(patients)
    split = {p: "train" for p in patients[: int(0.7 * n)]}
    split.update({p: "val" for p in patients[int(0.7 * n): int(0.8 * n)]})
    split.update({p: "test" for p in patients[int(0.8 * n):]})
    df["split"] = df["patient"].map(split)
    return df[["image_id", "cache_path", "quality", "patient", "split"]]


def _decode(path, label):
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    image = tf.image.resize(image, (SIZE, SIZE))
    return image, label


def _augment(image, label):
    image = tf.image.rot90(image, tf.random.uniform([], 0, 4, dtype=tf.int32))
    image = tf.image.random_flip_left_right(image)
    return image, label


def dataset(df, batch, train):
    ds = tf.data.Dataset.from_tensor_slices((df["cache_path"].astype(str).values, df["quality"].values.astype(np.int32)))
    if train:
        ds = ds.shuffle(len(df), seed=42)
    ds = ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
    if train:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch, drop_remainder=train).prefetch(tf.data.AUTOTUNE)


def build_model() -> keras.Model:
    return nets.quality_cnn(weights="imagenet")


def evaluate(model, df, batch):
    probs = model.predict(dataset(df, batch, False), verbose=0)
    y = df["quality"].values
    pred = probs.argmax(axis=1)
    out = {"accuracy": round(float((pred == y).mean()), 4),
           "confusion": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
           "ungradable_detection_auc": round(float(roc_auc_score((y == 2).astype(int), probs[:, 2])), 4) if (y == 2).any() and (y != 2).any() else None,
           "good_vs_rest_auc": round(float(roc_auc_score((y == 0).astype(int), probs[:, 0])), 4) if (y == 0).any() and (y != 0).any() else None}
    return out, probs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--tag", default="quality_cnn")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    keras.mixed_precision.set_global_policy("mixed_float16")
    keras.utils.set_random_seed(42)

    df = load_eyeq()
    if args.limit:
        df = df.groupby("split", group_keys=False).head(args.limit)
    train_df, val_df, test_df = (df[df["split"] == s] for s in ("train", "val", "test"))
    print(f"train {len(train_df)} val {len(val_df)} test {len(test_df)}  class counts {train_df['quality'].value_counts().sort_index().to_dict()}", flush=True)
    counts = train_df["quality"].value_counts().reindex([0, 1, 2], fill_value=0).values.astype(float)
    class_weight = {i: float(counts.sum() / (3 * max(counts[i], 1.0))) for i in range(3)}

    model = build_model()
    steps = len(train_df) // args.batch
    model.compile(optimizer=keras.optimizers.AdamW(keras.optimizers.schedules.CosineDecay(args.lr, steps * args.epochs), weight_decay=1e-5),
                  loss="sparse_categorical_crossentropy", jit_compile=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_path = MODEL_DIR / f"{args.tag}.weights.h5"
    train_ds = dataset(train_df, args.batch, True)
    best, history = -1.0, []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        loss = float(model.fit(train_ds, epochs=1, verbose=0, class_weight=class_weight).history["loss"][0])
        val, _ = evaluate(model, val_df, args.batch)
        score = val["ungradable_detection_auc"] or 0.0
        improved = score > best
        if improved:
            best = score
            model.save_weights(best_path)
        history.append({"epoch": epoch + 1, "loss": round(loss, 4), **{k: v for k, v in val.items() if k != "confusion"}})
        print(f"epoch {epoch + 1:2d}  loss {loss:.4f}  val acc {val['accuracy']:.3f}  ungradable AUC {score:.4f}  {(time.perf_counter() - t0) / 60:.1f} min{'  *' if improved else ''}", flush=True)

    model.load_weights(best_path)
    test, _ = evaluate(model, test_df, args.batch)
    ddr = pd.read_csv(MANIFEST_DIR / "ungradable.csv") if (MANIFEST_DIR / "ungradable.csv").exists() else None
    ddr_recall = None
    if ddr is not None and len(ddr):
        ddr = ddr.assign(quality=2)
        probs = model.predict(dataset(ddr, args.batch, False), verbose=0)
        ddr_recall = {"n": int(len(ddr)), "reject_recall_argmax": round(float((probs.argmax(axis=1) == 2).mean()), 4),
                      "not_good_recall": round(float((probs.argmax(axis=1) != 0).mean()), 4)}
    summary = {"tag": args.tag, "architecture": "EfficientNet-B0 at 256, 3-class softmax", "classes": CLASSES,
               "labels": "EyeQ train labels on EyePACS images, split by patient 70/10/20", "epochs": args.epochs,
               "train_n": int(len(train_df)), "val_n": int(len(val_df)), "test_n": int(len(test_df)),
               "class_weight": class_weight, "history": history, "test_heldout_patients": test,
               "ddr_ungradable_external_check": ddr_recall, "weights": str(best_path),
               "finished_at": datetime.now(timezone.utc).isoformat(), "total_minutes": round((time.perf_counter() - started) / 60, 1)}
    with open(MODEL_DIR / f"{args.tag}.summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"TEST acc {test['accuracy']}  ungradable AUC {test['ungradable_detection_auc']}  DDR ungradable {ddr_recall}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
