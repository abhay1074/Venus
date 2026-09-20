"""Train the lesion U-Net on the DDR lesion-segmentation set.

    wsl bash scripts/wsl-gpu.sh backend.training.train_lesion_unet --epochs 60

Input: 512x512 Stage 0 images from the cache; targets: four binary masks
(MA, HE, EX, SE) warped by the same transform. One U-Net (4 levels, 32 base
filters) with four sigmoid output channels, trained with Dice + BCE per
channel, dihedral + photometric augmentation. The DDR splits are used as
shipped: train for fitting, valid for checkpoint selection and per-class
threshold choice (max F1), test for the one reported number (pixel AUPR per
class, and Dice at the chosen threshold).

The thresholds go to config/lesion_thresholds.json; Stage 1 uses the network
when its weights are present and falls back to the classical detectors
otherwise, and either way reports which method produced the evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score
from tensorflow import keras

from backend.venus import nets
from backend.venus.config import CONFIG_DIR

CACHE_DIR = Path(os.getenv("VENUS_CACHE_DIR", str(Path.home() / "venus-cache")))
MODEL_DIR = CACHE_DIR / "models"
SIZE = 512
LESIONS = ["MA", "HE", "EX", "SE"]
BITS = {"MA": 1, "HE": 2, "EX": 4, "SE": 8}


# --------------------------------------------------------------- data --

def load_split(index: pd.DataFrame, split: str):
    rows = index[index["source_split"] == split]
    images = np.zeros((len(rows), SIZE, SIZE, 3), np.uint8)
    masks = np.zeros((len(rows), SIZE, SIZE, 4), np.uint8)
    for i, row in enumerate(rows.itertuples()):
        img = cv2.imread(row.cache_path)
        images[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        m = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)
        for c, key in enumerate(LESIONS):
            masks[i, :, :, c] = (m & BITS[key]) > 0
    return images, masks


def _augment(image, mask):
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image, mask = tf.image.rot90(image, k), tf.image.rot90(mask, k)
    if tf.random.uniform([]) < 0.5:
        image, mask = tf.image.flip_left_right(image), tf.image.flip_left_right(mask)
    image = tf.image.random_brightness(image, 0.15 * 255)
    image = tf.image.random_contrast(image, 0.85, 1.15)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, mask


def dataset(images, masks, batch, train):
    ds = tf.data.Dataset.from_tensor_slices((images, masks))
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)), num_parallel_calls=tf.data.AUTOTUNE)
    if train:
        ds = ds.shuffle(len(images), seed=42).map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch, drop_remainder=train).prefetch(tf.data.AUTOTUNE)


# -------------------------------------------------------------- model --

def build_unet(base: int = 32, levels: int = 4) -> keras.Model:
    return nets.lesion_unet(base, levels)


def dice_bce_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1 - 1e-6)
    bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    # Lesion pixels are < 1% of the image; weight positives so BCE does not
    # learn "all background" first.
    bce = tf.reduce_mean(bce * (1 + 9 * y_true))
    inter = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    denom = tf.reduce_sum(y_true + y_pred, axis=(1, 2))
    dice = 1 - (2 * inter + 1) / (denom + 1)
    return bce + tf.reduce_mean(dice)


# ------------------------------------------------------------ metrics --

def evaluate(model, images, masks, batch, thresholds=None):
    probs = model.predict(dataset(images, masks, batch, False), verbose=0)
    out = {}
    for c, key in enumerate(LESIONS):
        y = masks[:, :, :, c].reshape(-1).astype(np.uint8)
        p = probs[:, :, :, c].reshape(-1)
        if y.sum() == 0:
            out[key] = {"aupr": None}
            continue
        # Subsample negatives for the PR computation (300M pixels otherwise).
        rng = np.random.default_rng(0)
        pos = np.flatnonzero(y)
        neg = rng.choice(np.flatnonzero(y == 0), size=min(2_000_000, int((y == 0).sum())), replace=False)
        sel = np.concatenate([pos, neg])
        aupr = float(average_precision_score(y[sel], p[sel]))
        entry = {"aupr": round(aupr, 4)}
        if thresholds is None:
            best_f1, best_t = 0.0, 0.5
            for t in np.linspace(0.1, 0.9, 17):
                pred = p[sel] >= t
                tp = int((pred & (y[sel] == 1)).sum()); fp = int((pred & (y[sel] == 0)).sum()); fn = int((~pred & (y[sel] == 1)).sum())
                f1 = 2 * tp / max(2 * tp + fp + fn, 1)
                if f1 > best_f1:
                    best_f1, best_t = f1, float(t)
            entry["threshold"], entry["f1_at_threshold"] = round(best_t, 3), round(best_f1, 4)
        else:
            t = thresholds[key]
            pred = p >= t
            tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
            entry["threshold"] = t
            entry["dice_at_threshold"] = round(2 * tp / max(2 * tp + fp + fn, 1), 4)
            entry["precision"] = round(tp / max(tp + fp, 1), 4)
            entry["recall"] = round(tp / max(tp + fn, 1), 4)
        out[key] = entry
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tag", default="lesion_unet")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    keras.mixed_precision.set_global_policy("mixed_float16")
    keras.utils.set_random_seed(42)

    index = pd.read_csv(CACHE_DIR / "lesion_index.csv")
    if args.limit:
        index = index.groupby("source_split", group_keys=False).head(args.limit)
    train_x, train_y = load_split(index, "train")
    val_x, val_y = load_split(index, "valid")
    test_x, test_y = load_split(index, "test")
    print(f"train {len(train_x)}  valid {len(val_x)}  test {len(test_x)}  positive px per class (train): "
          f"{dict(zip(LESIONS, train_y.reshape(-1, 4).sum(axis=0).tolist()))}", flush=True)

    model = build_unet()
    steps = len(train_x) // args.batch
    schedule = keras.optimizers.schedules.CosineDecay(args.lr, decay_steps=steps * args.epochs)
    model.compile(optimizer=keras.optimizers.AdamW(schedule, weight_decay=1e-5), loss=dice_bce_loss, jit_compile=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_path = MODEL_DIR / f"{args.tag}.weights.h5"
    train_ds = dataset(train_x, train_y, args.batch, True)
    history, best = [], -1.0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        loss = float(model.fit(train_ds, epochs=1, verbose=0).history["loss"][0])
        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            val = evaluate(model, val_x, val_y, args.batch)
            mean_aupr = float(np.mean([v["aupr"] for v in val.values() if v["aupr"] is not None]))
            improved = mean_aupr > best
            if improved:
                best = mean_aupr
                model.save_weights(best_path)
            history.append({"epoch": epoch + 1, "loss": round(loss, 4), "val": val, "mean_aupr": round(mean_aupr, 4)})
            print(f"epoch {epoch + 1:3d}  loss {loss:.4f}  val AUPR " + "  ".join(f"{k} {v['aupr']}" for k, v in val.items())
                  + f"  mean {mean_aupr:.4f}  {(time.perf_counter() - t0) / 60:.1f} min{'  *' if improved else ''}", flush=True)
        else:
            print(f"epoch {epoch + 1:3d}  loss {loss:.4f}  {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    # Thresholds on valid (max F1), then the test split scored once.
    model.load_weights(best_path)
    val = evaluate(model, val_x, val_y, args.batch)
    thresholds = {k: v.get("threshold", 0.5) for k, v in val.items()}
    test = evaluate(model, test_x, test_y, args.batch, thresholds)
    summary = {
        "tag": args.tag, "architecture": "U-Net, 4 levels, 32 base filters, 512x512, 4 sigmoid channels",
        "loss": "BCE (positive weight 10) + Dice, per channel", "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "train_n": int(len(train_x)), "valid_n": int(len(val_x)), "test_n": int(len(test_x)),
        "thresholds_chosen_on_valid_max_f1": thresholds, "valid": val, "test_scored_once": test,
        "history": history, "weights": str(best_path), "finished_at": datetime.now(timezone.utc).isoformat(),
        "total_minutes": round((time.perf_counter() - started) / 60, 1),
    }
    with open(MODEL_DIR / f"{args.tag}.summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_DIR / "lesion_thresholds.json", "w", encoding="utf-8") as handle:
        json.dump({"tag": args.tag, "thresholds": thresholds, "chosen_on": "DDR valid split, max pixel F1",
                   "test_aupr": {k: v["aupr"] for k, v in test.items()}, "test_dice": {k: v.get("dice_at_threshold") for k, v in test.items()},
                   "written_at": summary["finished_at"]}, handle, indent=2)
    print("TEST (once): " + "  ".join(f"{k} AUPR {v['aupr']} Dice {v.get('dice_at_threshold')}" for k, v in test.items()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
