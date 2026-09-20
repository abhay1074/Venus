"""Train the DR grader v2: EfficientNet-B3 at 512 with an ordinal head.

    wsl bash scripts/wsl-gpu.sh backend.training.train_grader --epochs 15

Head: four sigmoid outputs P(grade >= k), k = 1..4 (CORAL-style ordinal
regression). Grade = number of thresholds passed; P(grade >= 2) is the
referable signal the whole system is optimised for, so that threshold's loss
carries double weight. Neovascularization evidence is P(grade >= 4).

Data: the Stage 0 cache (512x512 FOV-normalised JPEGs) via the manifests in
backend/data/manifests. Train on train.csv, select the checkpoint on val.csv by
referable AUC. The calibration and external test manifests are never read here.

Augmentation: the eight dihedral symmetries (rotation invariance is free for
fundus), random FOV zoom 90-100%, brightness/contrast/saturation jitter.
No cutout: it corrupts the lesion evidence the label depends on.

Precision: mixed_float16 for the backbone with a float32 head (softmax /
sigmoid saturation in float16 was a documented defect of the earlier work),
XLA-compiled train step. On the RTX 5060 this is ~130 img/s, ~6 min per epoch.
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
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from tensorflow import keras

from backend.venus import nets
from backend.venus.config import MANIFEST_DIR

CACHE_DIR = Path(os.getenv("VENUS_CACHE_DIR", str(Path.home() / "venus-cache")))
MODEL_DIR = CACHE_DIR / "models"
SIZE = 512
THRESHOLD_WEIGHTS = np.array([1.0, 2.0, 1.0, 1.0], np.float32)


# --------------------------------------------------------------- data --

def cumulative_targets(grades: np.ndarray) -> np.ndarray:
    """grade g -> [g>=1, g>=2, g>=3, g>=4]."""
    return np.stack([(grades >= k).astype(np.float32) for k in (1, 2, 3, 4)], axis=1)


def _decode(path, target):
    raw = tf.io.read_file(path)
    image = tf.io.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, (SIZE, SIZE)) if tf.shape(image)[0] != SIZE else tf.cast(image, tf.float32)
    return image, target


def _augment(image, target):
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    image = tf.image.random_flip_left_right(image)
    # Random zoom 90-100% of the FOV, keeping the centre.
    zoom = tf.random.uniform([], 0.9, 1.0)
    crop = tf.cast(tf.round(SIZE * zoom), tf.int32)
    image = tf.image.resize_with_crop_or_pad(image, crop, crop)
    image = tf.image.resize(image, (SIZE, SIZE))
    image = tf.image.random_brightness(image, 0.2 * 255)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, target


def dataset(manifest: pd.DataFrame, batch: int, train: bool) -> tf.data.Dataset:
    paths = manifest["cache_path"].astype(str).values
    targets = cumulative_targets(manifest["grade"].values.astype(np.int32))
    ds = tf.data.Dataset.from_tensor_slices((paths, targets))
    if train:
        ds = ds.shuffle(len(paths), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
    if train:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch, drop_remainder=train).prefetch(tf.data.AUTOTUNE)


# -------------------------------------------------------------- model --

def build_model(backbone_name: str = "B3", dropout: float = 0.3) -> keras.Model:
    return nets.grader_v2(backbone_name, dropout, weights="imagenet")


def weighted_bce(pos_weight: np.ndarray):
    """BCE per threshold, weighted per threshold and with a positive-class
    weight so the rarer 'grade >= k' targets are not drowned by grade 0."""
    pw = tf.constant(pos_weight, tf.float32)
    tw = tf.constant(THRESHOLD_WEIGHTS, tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-6, 1 - 1e-6)
        per = -(pw * y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        return tf.reduce_mean(per * tw)
    return loss


# ------------------------------------------------------------ metrics --

def evaluate(model: keras.Model, manifest: pd.DataFrame, batch: int) -> dict:
    ds = dataset(manifest, batch, train=False)
    probs = model.predict(ds, verbose=0)
    grades = manifest["grade"].values.astype(int)
    pred_grade = (probs >= 0.5).sum(axis=1)
    referable = (grades >= 2).astype(int)
    out = {
        "referable_auc": float(roc_auc_score(referable, probs[:, 1])) if len(set(referable)) > 1 else None,
        "qwk": float(cohen_kappa_score(grades, pred_grade, weights="quadratic")),
        "exact": float((pred_grade == grades).mean()),
        "within_one": float((np.abs(pred_grade - grades) <= 1).mean()),
        "any_dr_auc": float(roc_auc_score((grades >= 1).astype(int), probs[:, 0])) if len(set(grades >= 1)) > 1 else None,
        "pdr_auc": float(roc_auc_score((grades >= 4).astype(int), probs[:, 3])) if (grades >= 4).any() and (grades < 4).any() else None,
    }
    return out, probs


# --------------------------------------------------------------- main --

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", default="B3", choices=["B0", "B3"])
    parser.add_argument("--limit", type=int, default=None, help="images per split, for a smoke run")
    parser.add_argument("--tag", default="grader_v2")
    parser.add_argument("--no-xla", action="store_true")
    args = parser.parse_args(argv)

    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    keras.mixed_precision.set_global_policy("mixed_float16")
    keras.utils.set_random_seed(42)

    train_df = pd.read_csv(MANIFEST_DIR / "train.csv")
    val_df = pd.read_csv(MANIFEST_DIR / "val.csv")
    if args.limit:
        train_df = train_df.sample(min(args.limit, len(train_df)), random_state=42)
        val_df = val_df.sample(min(args.limit // 4, len(val_df)), random_state=42)
    print(f"train {len(train_df)}  val {len(val_df)}  grades {train_df['grade'].value_counts().sort_index().to_dict()}", flush=True)

    # Positive-class weight per threshold from the training prevalence
    # (capped so grade >= 4 does not get a weight of 50).
    targets = cumulative_targets(train_df["grade"].values)
    prevalence = targets.mean(axis=0)
    pos_weight = np.clip((1 - prevalence) / np.maximum(prevalence, 1e-3), 1.0, 6.0).astype(np.float32)
    print("threshold prevalence", np.round(prevalence, 3), "pos_weight", np.round(pos_weight, 2), flush=True)

    model = build_model(args.backbone)
    steps_per_epoch = len(train_df) // args.batch
    schedule = keras.optimizers.schedules.CosineDecay(args.lr, decay_steps=steps_per_epoch * args.epochs, warmup_target=None)
    optimizer = keras.optimizers.AdamW(learning_rate=schedule, weight_decay=1e-5)
    model.compile(optimizer=optimizer, loss=weighted_bce(pos_weight), jit_compile=not args.no_xla)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_path = MODEL_DIR / f"{args.tag}.weights.h5"
    history, best_auc = [], -1.0
    train_ds = dataset(train_df, args.batch, train=True)
    started = time.perf_counter()
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        fit = model.fit(train_ds, epochs=1, verbose=0)
        loss = float(fit.history["loss"][0])
        metrics, _ = evaluate(model, val_df, args.batch)
        elapsed = time.perf_counter() - t0
        record = {"epoch": epoch + 1, "loss": round(loss, 4), "minutes": round(elapsed / 60, 1), **{k: (round(v, 4) if v is not None else None) for k, v in metrics.items()}}
        history.append(record)
        improved = metrics["referable_auc"] is not None and metrics["referable_auc"] > best_auc
        if improved:
            best_auc = metrics["referable_auc"]
            model.save_weights(best_path)
        print(f"epoch {epoch + 1:2d}/{args.epochs}  loss {loss:.4f}  val AUC {metrics['referable_auc']:.4f}  QWK {metrics['qwk']:.3f}  "
              f"exact {metrics['exact']:.3f}  {elapsed / 60:.1f} min{'  *' if improved else ''}", flush=True)
        with open(MODEL_DIR / f"{args.tag}.history.json", "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

    summary = {
        "tag": args.tag, "backbone": f"EfficientNet-{args.backbone}", "input": SIZE, "head": "ordinal, 4 cumulative sigmoids",
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "schedule": "AdamW, cosine decay", "precision": "mixed_float16, float32 head",
        "xla": not args.no_xla, "augmentation": "dihedral, zoom 0.9-1.0, brightness/contrast/saturation ±20%",
        "loss": "weighted BCE on cumulative targets, threshold weights [1,2,1,1], pos_weight " + str([round(float(w), 2) for w in pos_weight]),
        "train_n": int(len(train_df)), "val_n": int(len(val_df)), "best_val_referable_auc": round(best_auc, 4),
        "history": history, "weights": str(best_path), "finished_at": datetime.now(timezone.utc).isoformat(),
        "total_minutes": round((time.perf_counter() - started) / 60, 1),
    }
    with open(MODEL_DIR / f"{args.tag}.summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"best val referable AUC {best_auc:.4f}  weights {best_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
