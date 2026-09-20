"""Network definitions shared by training and serving.

Everything that loads a checkpoint builds the graph from here, so a training
run and the served model can never drift apart.

    grader_v2(backbone)   EfficientNet-B3 (or B0) at 512, ordinal head, 4 sigmoids
    grader_v1()           EfficientNet-B4 at 380, ordinal + (unused) disease head
    modality_gate()       EfficientNet-B0 at 224, 5-way softmax
    quality_cnn()         EfficientNet-B0 at 256, 3-way softmax (good/usable/reject)
    lesion_unet()         U-Net, 4 levels, 32 base filters, 512, 4 sigmoid channels
"""

from __future__ import annotations

GRADER_V2_SIZE = 512
GRADER_V1_SIZE = 380
GATE_SIZE = 224
QUALITY_SIZE = 256
UNET_SIZE = 512
LESIONS = ["MA", "HE", "EX", "SE"]


def grader_v2(backbone: str = "B3", dropout: float = 0.3, weights=None):
    from tensorflow import keras

    inputs = keras.Input(shape=(GRADER_V2_SIZE, GRADER_V2_SIZE, 3), name="fundus_512")
    cls = {"B0": keras.applications.EfficientNetB0, "B3": keras.applications.EfficientNetB3}[backbone]
    net = cls(include_top=False, weights=weights, input_tensor=inputs)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(net.output)
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    out = keras.layers.Dense(4, activation="sigmoid", dtype="float32", name="dr_ordinal_thresholds")(x)
    return keras.Model(inputs, out, name=f"venus_grader_v2_{backbone}")


def grader_v1():
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(GRADER_V1_SIZE, GRADER_V1_SIZE, 3), name="fundus_image")
    backbone = keras.applications.EfficientNetB4(include_top=False, weights=None, input_tensor=inputs)
    features = layers.GlobalAveragePooling2D(name="fundus_gap")(backbone.output)
    dr = layers.Dense(256, activation="relu", name="dr_dense")(features)
    dr = layers.Dropout(0.5, name="dr_dropout")(dr)
    dr_out = layers.Dense(4, activation="sigmoid", dtype="float32", name="dr_ordinal_thresholds")(dr)
    # The checkpoint carries an unvalidated disease head; built so the weights load, never read.
    disease = layers.Dense(256, activation="relu", name="disease_dense")(features)
    disease = layers.Dropout(0.5, name="disease_dropout")(disease)
    disease_out = layers.Dense(8, activation="sigmoid", dtype="float32", name="disease_multilabel")(disease)
    return keras.Model(inputs, {"dr_ordinal_thresholds": dr_out, "disease_multilabel": disease_out}, name="venus_grader_v1")


def modality_gate(n_classes: int = 5):
    from tensorflow import keras

    backbone = keras.applications.EfficientNetB0(include_top=False, weights=None, input_shape=(GATE_SIZE, GATE_SIZE, 3))
    x = keras.layers.GlobalAveragePooling2D(name="gap")(backbone.output)
    x = keras.layers.Dropout(0.3, name="drop")(x)
    out = keras.layers.Dense(n_classes, activation="softmax", dtype="float32", name="modality")(x)
    return keras.Model(backbone.input, out, name="venus_modality_gate")


def quality_cnn(weights=None):
    from tensorflow import keras

    inputs = keras.Input(shape=(QUALITY_SIZE, QUALITY_SIZE, 3), name="fundus_256")
    backbone = keras.applications.EfficientNetB0(include_top=False, weights=weights, input_tensor=inputs)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(backbone.output)
    x = keras.layers.Dropout(0.3, name="dropout")(x)
    out = keras.layers.Dense(3, activation="softmax", dtype="float32", name="quality")(x)
    return keras.Model(inputs, out, name="venus_quality_cnn")


def _conv_block(x, filters, name):
    from tensorflow import keras

    x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_c1")(x)
    x = keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = keras.layers.Activation("relu", name=f"{name}_a1")(x)
    x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_c2")(x)
    x = keras.layers.BatchNormalization(name=f"{name}_bn2")(x)
    return keras.layers.Activation("relu", name=f"{name}_a2")(x)


def lesion_unet(base: int = 32, levels: int = 4, size: int | None = None):
    from tensorflow import keras

    size = size or UNET_SIZE
    inputs = keras.Input(shape=(size, size, 3), name=f"fundus_{size}")
    x = keras.layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)
    skips = []
    for level in range(levels):
        x = _conv_block(x, base * 2 ** level, f"enc{level}")
        skips.append(x)
        x = keras.layers.MaxPooling2D(2, name=f"pool{level}")(x)
    x = _conv_block(x, base * 2 ** levels, "bottleneck")
    for level in reversed(range(levels)):
        x = keras.layers.Conv2DTranspose(base * 2 ** level, 2, strides=2, name=f"up{level}")(x)
        x = keras.layers.Concatenate(name=f"cat{level}")([x, skips[level]])
        x = _conv_block(x, base * 2 ** level, f"dec{level}")
    out = keras.layers.Conv2D(4, 1, activation="sigmoid", dtype="float32", name="lesions")(x)
    return keras.Model(inputs, out, name="venus_lesion_unet")
