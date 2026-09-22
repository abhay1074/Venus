"""Export the served networks for MATLAB (importNetworkFromTensorFlow / ONNX).

    wsl bash scripts/wsl-gpu.sh backend.eval.export_models [--onnx]

Writes models/export/<name>/ (TensorFlow SavedModel via Keras 3 `export`) and,
with --onnx and tf2onnx installed, models/export/<name>.onnx, for every
checkpoint present in backend/weights:

    grader_v2      EfficientNet-B3 at 512 -> 4 sigmoids P(grade >= k)
    modality_gate  EfficientNet-B0 at 224 -> 5-way softmax
    quality_cnn    EfficientNet-B0 at 256 -> 3-way softmax
    lesion_unet    U-Net at 512 -> 4 sigmoid channels
    lesion_unet_1024  the same U-Net at 1024, served for microaneurysms only

The exported graphs take raw 0-255 float inputs exactly as the Python
serving path feeds them, so MATLAB's `predict` on the imported dlnetwork
needs no extra normalisation. A manifest with SHA-256 of each export is
written next to them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from backend.venus import nets
from backend.venus.config import (GATE_WEIGHTS, GRADER_V2_WEIGHTS, MODEL_VERSION, PROJECT_ROOT, QUALITY_WEIGHTS,
                                  UNET_HIRES_WEIGHTS, UNET_WEIGHTS)

EXPORT_DIR = PROJECT_ROOT / "models" / "export"
SPECS = [
    ("grader_v2", GRADER_V2_WEIGHTS, lambda: nets.grader_v2("B3"), (512, 512, 3)),
    ("modality_gate", GATE_WEIGHTS, nets.modality_gate, (224, 224, 3)),
    ("quality_cnn", QUALITY_WEIGHTS, nets.quality_cnn, (256, 256, 3)),
    ("lesion_unet", UNET_WEIGHTS, nets.lesion_unet, (512, 512, 3)),
    ("lesion_unet_1024", UNET_HIRES_WEIGHTS, lambda: nets.lesion_unet(size=1024), (1024, 1024, 3)),
]


def sha256_dir(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", action="store_true")
    args = parser.parse_args(argv)
    import tensorflow as tf
    from tensorflow import keras

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"model_version": MODEL_VERSION, "written_at": datetime.now(timezone.utc).isoformat(), "models": {}}
    for name, weights, builder, shape in SPECS:
        if not weights.exists():
            print(f"skip {name}: {weights.name} not present")
            continue
        model = builder()
        model.load_weights(weights)
        out = EXPORT_DIR / name
        if out.exists():
            shutil.rmtree(out)
        model.export(str(out), format="tf_saved_model")
        entry = {"input": list(shape), "output": model.output_shape[1:] if hasattr(model, "output_shape") else None,
                 "saved_model": out.name, "sha256": sha256_dir(out), "weights": weights.name}
        if args.onnx:
            try:
                import tf2onnx  # noqa: F401
                onnx_path = EXPORT_DIR / f"{name}.onnx"
                os.system(f"{sys.executable} -m tf2onnx.convert --saved-model {out} --output {onnx_path} --opset 17 > /dev/null 2>&1")
                if onnx_path.exists():
                    entry["onnx"] = onnx_path.name
                    entry["onnx_sha256"] = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
            except ImportError:
                entry["onnx"] = "tf2onnx not installed"
        if name == "grader_v2":
            # Grad-CAM in MATLAB: the importer folds the network into one
            # opaque layer, so the feature maps (top_activation) are exported
            # as their own model and the dense head's weights as JSON; with a
            # GAP + sigmoid head, Grad-CAM is exactly ReLU(sum_c w_kc A_c).
            feats = keras.Model(model.inputs, model.get_layer("top_activation").output, name="grader_v2_features")
            fout = EXPORT_DIR / "grader_v2_features"
            if fout.exists():
                shutil.rmtree(fout)
            feats.export(str(fout), format="tf_saved_model")
            kernel, bias = model.get_layer("dr_ordinal_thresholds").get_weights()
            head = {"cam_layer": "top_activation", "feature_shape": list(feats.output_shape[1:]),
                    "dense_kernel": kernel.tolist(), "dense_bias": bias.tolist(),
                    "note": "Grad-CAM for head k = ReLU(sum_c dense_kernel[c,k] * A_c), normalised to max 1 (the sigmoid slope is a positive scalar)"}
            with open(EXPORT_DIR / "grader_v2_head.json", "w", encoding="utf-8") as handle:
                json.dump(head, handle)
            entry["features_saved_model"] = fout.name
            entry["head_weights"] = "grader_v2_head.json"
            print(f"exported grader_v2_features -> {fout} (+ head weights)")
        manifest["models"][name] = entry
        print(f"exported {name} -> {out}")
    with open(EXPORT_DIR / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"wrote {EXPORT_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
