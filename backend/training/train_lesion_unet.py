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
from tensorflow import keras

from backend.venus import nets
from backend.venus.config import CONFIG_DIR

CACHE_DIR = Path(os.getenv("VENUS_CACHE_DIR", str(Path.home() / "venus-cache")))
MODEL_DIR = CACHE_DIR / "models"
SIZE = 512
LESIONS = ["MA", "HE", "EX", "SE"]
BITS = {"MA": 1, "HE": 2, "EX": 4, "SE": 8}


# --------------------------------------------------------------- data --

def load_split(index: pd.DataFrame, split: str, size: int = SIZE):
    """Images (N, S, S, 3) uint8 and masks (N, S, S) uint8 with one bit per
    lesion class (the cache's own encoding); the four channels are unpacked
    on the fly so a 1024 px split fits in memory."""
    rows = index[index["source_split"] == split]
    images = np.zeros((len(rows), size, size, 3), np.uint8)
    masks = np.zeros((len(rows), size, size), np.uint8)
    for i, row in enumerate(rows.itertuples()):
        img = cv2.imread(row.cache_path)
        images[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masks[i] = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE) & 15
    return images, masks


def positive_pixels(masks: np.ndarray) -> dict:
    return {key: int(((masks & BITS[key]) > 0).sum()) for key in LESIONS}


def unpack_mask(packed):
    """(…, S, S) uint8 bit mask -> (…, S, S, 4) float32."""
    bits = tf.constant([BITS[k] for k in LESIONS], tf.uint8)
    return tf.cast(tf.bitwise.bitwise_and(packed[..., None], bits) > 0, tf.float32)


def _augment(image, mask):
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image, mask = tf.image.rot90(image, k), tf.image.rot90(mask, k)
    if tf.random.uniform([]) < 0.5:
        image, mask = tf.image.flip_left_right(image), tf.image.flip_left_right(mask)
    image = tf.image.random_brightness(image, 0.15 * 255)
    image = tf.image.random_contrast(image, 0.85, 1.15)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, mask


def _patch_fn(patch: int, lesion_bias: float = 0.5):
    """Random crop of `patch` px; half the crops are centred near a lesion
    pixel (lesions cover < 1 % of a frame, uniform crops would mostly be
    background)."""
    def fn(image, mask):
        size = tf.shape(image)[0]
        coords = tf.cast(tf.where(tf.reduce_max(mask, axis=-1) > 0), tf.int32)
        n = tf.shape(coords)[0]

        def centred():
            c = coords[tf.random.uniform([], 0, n, dtype=tf.int32)]
            jitter = tf.random.uniform([2], -patch // 4, patch // 4, dtype=tf.int32)
            y0 = tf.clip_by_value(c[0] - patch // 2 + jitter[0], 0, size - patch)
            x0 = tf.clip_by_value(c[1] - patch // 2 + jitter[1], 0, size - patch)
            return y0, x0

        def uniform():
            return (tf.random.uniform([], 0, size - patch + 1, dtype=tf.int32),
                    tf.random.uniform([], 0, size - patch + 1, dtype=tf.int32))

        y0, x0 = tf.cond(tf.logical_and(n > 0, tf.random.uniform([]) < lesion_bias), centred, uniform)
        return image[y0:y0 + patch, x0:x0 + patch], mask[y0:y0 + patch, x0:x0 + patch]
    return fn


def dataset(images, masks, batch, train, patch: int | None = None, repeats: int = 1):
    """Index-shuffled pipeline over the in-memory arrays: the frames are
    fetched by index (no second copy as a graph constant, no shuffle buffer
    of full frames), unpacked, cropped and augmented on the fly."""
    size = images.shape[1]

    def fetch(i):
        x, y = tf.numpy_function(lambda j: (images[j], masks[j]), [i], [tf.uint8, tf.uint8])
        x.set_shape((size, size, 3)); y.set_shape((size, size))
        return tf.cast(x, tf.float32), unpack_mask(y)

    ds = tf.data.Dataset.range(len(images))
    if train:
        ds = ds.shuffle(len(images), seed=42, reshuffle_each_iteration=True)
        if repeats > 1:
            ds = ds.repeat(repeats)
    ds = ds.map(fetch, num_parallel_calls=tf.data.AUTOTUNE)
    if train:
        if patch:
            ds = ds.map(_patch_fn(patch), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch, drop_remainder=train).prefetch(tf.data.AUTOTUNE)


# -------------------------------------------------------------- model --

def build_unet(base: int = 32, levels: int = 4, size: int | None = None) -> keras.Model:
    return nets.lesion_unet(base, levels, size)


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

BINS = 8192


def evaluate(model, images, masks, batch, thresholds=None, subsample=None):
    """Per-class pixel AUPR over every pixel of the split, computed from
    8,192-bin histograms of the probabilities (positives and negatives
    separately) so memory does not grow with the frame size: precision and
    recall are exact at each bin edge, AUPR is the step-wise sum over those
    edges (the same estimator sklearn's average_precision_score uses, on a
    1/8192 probability grid). Thresholds by max F1 when none are given, else
    Dice / precision / recall at the given thresholds. `subsample` is
    accepted for the in-training monitor and ignored: the full split is cheap
    once nothing is materialised."""
    pos = np.zeros((4, BINS), np.int64)
    neg = np.zeros((4, BINS), np.int64)
    for i in range(0, len(images), batch):
        x = images[i:i + batch].astype(np.float32)
        probs = np.asarray(model.predict_on_batch(x), np.float32)
        idx = np.clip((probs * BINS).astype(np.int32), 0, BINS - 1)
        for c, key in enumerate(LESIONS):
            y = (masks[i:i + batch] & BITS[key]) > 0
            pos[c] += np.bincount(idx[..., c][y], minlength=BINS)
            neg[c] += np.bincount(idx[..., c][~y], minlength=BINS)
    out = {}
    edges = np.arange(BINS) / BINS
    for c, key in enumerate(LESIONS):
        n_pos, n_neg = int(pos[c].sum()), int(neg[c].sum())
        if n_pos == 0:
            out[key] = {"aupr": None}
            continue
        # cumulative from the top: tp(t) = positives with p >= edge t
        tp = np.cumsum(pos[c][::-1])[::-1].astype(np.float64)
        fp = np.cumsum(neg[c][::-1])[::-1].astype(np.float64)
        recall = tp / n_pos
        precision = tp / np.maximum(tp + fp, 1)
        # AP = sum over thresholds of (R_k - R_{k+1}) * P_k, descending thresholds
        recall_next = np.append(recall[1:], 0.0)
        aupr = float(np.sum((recall - recall_next) * precision))
        entry = {"aupr": round(aupr, 4), "pixels": n_pos + n_neg, "positives": n_pos, "negatives_subsampled": False}
        f1 = 2 * tp / np.maximum(2 * tp + fp + (n_pos - tp), 1)
        if thresholds is None:
            grid = np.linspace(0.05, 0.98, 32)
            k = np.clip((grid * BINS).astype(int), 0, BINS - 1)
            best = int(np.argmax(f1[k]))
            entry["threshold"], entry["f1_at_threshold"] = round(float(grid[best]), 3), round(float(f1[k][best]), 4)
        else:
            t = thresholds[key]
            k = int(np.clip(t * BINS, 0, BINS - 1))
            entry["threshold"] = t
            entry["dice_at_threshold"] = round(float(f1[k]), 4)
            entry["precision"] = round(float(precision[k]), 4)
            entry["recall"] = round(float(recall[k]), 4)
        out[key] = entry
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tag", default="lesion_unet")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--size", type=int, default=SIZE, help="frame size of the lesion cache to train on (512 or 1024)")
    parser.add_argument("--patch", type=int, default=None, help="train on random lesion-biased crops of this size")
    parser.add_argument("--repeats", type=int, default=4, help="crops per image per epoch when --patch is set")
    args = parser.parse_args(argv)

    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    keras.mixed_precision.set_global_policy("mixed_float16")
    keras.utils.set_random_seed(42)

    suffix = "" if args.size == SIZE else f"_{args.size}"
    index = pd.read_csv(CACHE_DIR / f"lesion_index{suffix}.csv")
    if args.limit:
        index = index.groupby("source_split", group_keys=False).head(args.limit)
    train_x, train_y = load_split(index, "train", args.size)
    val_x, val_y = load_split(index, "valid", args.size)
    test_x, test_y = load_split(index, "test", args.size)
    print(f"train {len(train_x)}  valid {len(val_x)}  test {len(test_x)}  positive px per class (train): "
          f"{positive_pixels(train_y)}", flush=True)

    # Fully convolutional: train on crops, evaluate and serve on full frames
    # through a second graph that shares the weights.
    repeats = args.repeats if args.patch else 1
    model = build_unet(size=args.patch or args.size)
    full = build_unet(size=args.size) if args.patch else model

    def full_model():
        if args.patch:
            full.set_weights(model.get_weights())
        return full

    eval_batch = max(args.batch // 4, 1) if args.patch else args.batch
    steps = (len(train_x) * repeats) // args.batch
    schedule = keras.optimizers.schedules.CosineDecay(args.lr, decay_steps=steps * args.epochs)
    model.compile(optimizer=keras.optimizers.AdamW(schedule, weight_decay=1e-5), loss=dice_bce_loss, jit_compile=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_path = MODEL_DIR / f"{args.tag}.weights.h5"
    train_ds = dataset(train_x, train_y, args.batch, True, patch=args.patch, repeats=repeats)
    history, best = [], -1.0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        loss = float(model.fit(train_ds, epochs=1, verbose=0).history["loss"][0])
        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            val = evaluate(full_model(), val_x, val_y, eval_batch, subsample=2_000_000)
            scored = [v["aupr"] for v in val.values() if v["aupr"] is not None]
            mean_aupr = float(np.mean(scored)) if scored else 0.0
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
    served = full_model()
    val = evaluate(served, val_x, val_y, eval_batch)
    thresholds = {k: v.get("threshold", 0.5) for k, v in val.items()}
    test = evaluate(served, test_x, test_y, eval_batch, thresholds)
    summary = {
        "tag": args.tag, "frame_size": args.size, "patch": args.patch, "repeats": repeats,
        "architecture": f"U-Net, 4 levels, 32 base filters, {args.size}x{args.size} frames"
                        + (f", trained on {args.patch} px lesion-biased crops ({repeats} per image per epoch)" if args.patch else "")
                        + ", 4 sigmoid channels",
        "loss": "BCE (positive weight 10) + Dice, per channel", "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "train_n": int(len(train_x)), "valid_n": int(len(val_x)), "test_n": int(len(test_x)),
        "thresholds_chosen_on_valid_max_f1": thresholds, "valid": val, "test_scored_once": test,
        "history": history, "weights": str(best_path), "finished_at": datetime.now(timezone.utc).isoformat(),
        "total_minutes": round((time.perf_counter() - started) / 60, 1),
    }
    with open(MODEL_DIR / f"{args.tag}.summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    # Only the default tag writes the served thresholds; an experiment's go
    # next to its weights and are promoted by hand (scripts/finalise-models.sh).
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    thresholds_path = CONFIG_DIR / "lesion_thresholds.json" if args.tag == "lesion_unet" else MODEL_DIR / f"{args.tag}.thresholds.json"
    with open(thresholds_path, "w", encoding="utf-8") as handle:
        json.dump({"tag": args.tag, "thresholds": thresholds, "frame_size": args.size, "chosen_on": "DDR valid split, max pixel F1",
                   "test_aupr": {k: v["aupr"] for k, v in test.items()}, "test_dice": {k: v.get("dice_at_threshold") for k, v in test.items()},
                   "written_at": summary["finished_at"]}, handle, indent=2)
    print("TEST (once): " + "  ".join(f"{k} AUPR {v['aupr']} Dice {v.get('dice_at_threshold')}" for k, v in test.items()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
