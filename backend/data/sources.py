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


def messidor2() -> pd.DataFrame:
    """Messidor-2 (ADCIS, 1,748 images) with the Krause et al. 2018 adjudicated
    ICDR grades (messidor_data.csv). Images labelled ungradable (grade NaN) are
    kept with grade 5. The ADCIS left/right pairing (messidor-2.csv) gives the
    patient key. Never used for training: this is the adjudicated external test."""
    root = EYE / "messidor2"
    grades = pd.read_csv(root / "messidor_data.csv")
    on_disk = {p.stem: p for p in (root / "IMAGES").iterdir()}
    grades["stem"] = grades["image_id"].str.rsplit(".", n=1).str[0]
    grades = grades[grades["stem"].isin(on_disk)].copy()
    pairs = pd.read_csv(root / "messidor-2.csv", sep=";")
    pairs.columns = [c.strip() for c in pairs.columns]
    patient_of = {}
    for i, row in enumerate(pairs.itertuples()):
        for side in ("left", "right"):
            patient_of[str(getattr(row, side)).strip().rsplit(".", 1)[0]] = f"messidor2/pair{i:04d}"
    df = pd.DataFrame({
        "image_id": "messidor2/" + grades["stem"],
        "dataset": "messidor2",
        "path": [str(on_disk[s]) for s in grades["stem"]],
        "grade": grades["adjudicated_dr_grade"].fillna(5).astype(int).values,
        "patient": [patient_of.get(s, f"messidor2/{s}") for s in grades["stem"]],
        "source_split": "test",
        "adjudicated_dme": grades["adjudicated_dme"].values,
        "adjudicated_gradable": grades["adjudicated_gradable"].values,
    })
    return df.reset_index(drop=True)


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
