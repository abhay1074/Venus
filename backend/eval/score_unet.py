"""Re-score a trained lesion U-Net with full-pixel AUPR: thresholds on the DDR
valid split (max F1), the DDR test split scored once. Rewrites the summary's
valid/test sections and config/lesion_thresholds.json.

    wsl bash scripts/wsl-gpu.sh backend.eval.score_unet --weights ~/venus-cache/models/lesion_unet.weights.h5
"""
import argparse, json, os, sys
from datetime import datetime, timezone
import pandas as pd, tensorflow as tf
from tensorflow import keras
from backend.training import train_lesion_unet as U
from backend.venus.config import CONFIG_DIR


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--weights", required=True); ap.add_argument("--tag", default="lesion_unet")
    ap.add_argument("--batch", type=int, default=4); a = ap.parse_args(argv)
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    keras.mixed_precision.set_global_policy("mixed_float16")
    index = pd.read_csv(U.CACHE_DIR / "lesion_index.csv")
    val_x, val_y = U.load_split(index, "valid"); test_x, test_y = U.load_split(index, "test")
    model = U.build_unet(); model.load_weights(os.path.expanduser(a.weights))
    val = U.evaluate(model, val_x, val_y, a.batch)
    thresholds = {k: v.get("threshold", 0.5) for k, v in val.items()}
    test = U.evaluate(model, test_x, test_y, a.batch, thresholds)
    summary_path = U.MODEL_DIR / f"{a.tag}.summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {"tag": a.tag}
    summary.update({"thresholds_chosen_on_valid_max_f1": thresholds, "valid": val, "test_scored_once": test,
                    "aupr_note": "full-pixel AUPR (all negatives); in-training monitoring used subsampled negatives",
                    "rescored_at": datetime.now(timezone.utc).isoformat()})
    json.dump(summary, open(summary_path, "w"), indent=2)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({"tag": a.tag, "thresholds": thresholds, "chosen_on": "DDR valid split, max pixel F1 (all pixels)",
               "test_aupr": {k: v["aupr"] for k, v in test.items()}, "test_dice": {k: v.get("dice_at_threshold") for k, v in test.items()},
               "written_at": summary["rescored_at"]}, open(CONFIG_DIR / "lesion_thresholds.json", "w"), indent=2)
    print("VALID:", {k: (v["aupr"], v.get("threshold")) for k, v in val.items()})
    print("TEST (once, full pixels):", {k: (v["aupr"], v.get("dice_at_threshold"), v.get("precision"), v.get("recall")) for k, v in test.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
