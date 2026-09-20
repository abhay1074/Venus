"""Where each public dataset lives and how its labels are read.

Every reader returns a DataFrame with the same columns:

    image_id   unique across datasets: "<dataset>/<file stem>"
    dataset    aptos | eyepacs | ddr
    path       absolute path to the raw image
    grade      ICDR 0-4, or 5 for DDR's "ungradable" class
    patient    grouping key for patient-level splits (EyePACS encodes it in
               the file name; APTOS and DDR have none, so the image is its own group)
    source_split  the split the dataset shipped with, kept for reference
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(os.getenv("VENUS_DATA_ROOT", r"C:\Users\anilm\Downloads\MediScan-main\MediScan-main\backend\data\raw"))
EYE = DATA_ROOT / "eye"


def aptos() -> pd.DataFrame:
    root = EYE / "aptos"
    frames = []
    for csv, folder, split in (("train_1.csv", "train_images/train_images", "train"),
                               ("valid.csv", "val_images/val_images", "valid"),
                               ("test.csv", "test_images/test_images", "test")):
        df = pd.read_csv(root / csv)
        df = df.rename(columns={"id_code": "stem", "diagnosis": "grade"})
        df["path"] = [str(root / folder / f"{s}.png") for s in df["stem"]]
        df["source_split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["image_id"] = "aptos/" + df["stem"]
    df["dataset"] = "aptos"
    df["patient"] = df["image_id"]
    return df[["image_id", "dataset", "path", "grade", "patient", "source_split"]]


def eyepacs() -> pd.DataFrame:
    root = EYE / "eyepacs"
    df = pd.read_csv(root / "trainLabels.csv").rename(columns={"image": "stem", "level": "grade"})
    df["path"] = [str(root / "resized_train" / "resized_train" / f"{s}.jpeg") for s in df["stem"]]
    df["image_id"] = "eyepacs/" + df["stem"]
    df["dataset"] = "eyepacs"
    df["patient"] = "eyepacs/" + df["stem"].str.split("_").str[0]
    df["source_split"] = "train"
    return df[["image_id", "dataset", "path", "grade", "patient", "source_split"]]


def ddr() -> pd.DataFrame:
    root = EYE / "ddr" / "extracted" / "DDR-dataset" / "DR_grading"
    frames = []
    for split in ("train", "valid", "test"):
        rows = []
        for line in (root / f"{split}.txt").read_text().splitlines():
            if not line.strip():
                continue
            name, grade = line.split()
            rows.append({"stem": Path(name).stem, "path": str(root / split / name), "grade": int(grade), "source_split": split})
        frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True)
    df["image_id"] = "ddr/" + df["stem"]
    df["dataset"] = "ddr"
    df["patient"] = df["image_id"]
    return df[["image_id", "dataset", "path", "grade", "patient", "source_split"]]


def ddr_lesions() -> pd.DataFrame:
    """DDR lesion-segmentation set: image + four lesion masks (EX, HE, MA, SE)."""
    root = EYE / "ddr" / "extracted" / "DDR-dataset" / "lesion_segmentation"
    rows = []
    for split in ("train", "valid", "test"):
        image_dir = root / split / "image"
        if not image_dir.exists():
            continue
        label_dir = root / split / "label"
        if not label_dir.exists():
            label_dir = root / split / "segmentation label"   # DDR's valid split names it differently
        for path in sorted(image_dir.iterdir()):
            masks = {k: label_dir / k / f"{path.stem}.tif" for k in ("EX", "HE", "MA", "SE")}
            if not all(m.exists() for m in masks.values()):
                continue
            rows.append({"image_id": f"ddr_seg/{path.stem}", "path": str(path), "source_split": split,
                         **{f"mask_{k}": str(v) for k, v in masks.items()}})
    return pd.DataFrame(rows)


def all_grading() -> pd.DataFrame:
    return pd.concat([aptos(), eyepacs(), ddr()], ignore_index=True)
