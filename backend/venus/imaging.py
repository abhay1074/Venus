"""Small image helpers shared by every stage."""

from __future__ import annotations

import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def decode_image(payload: bytes) -> np.ndarray:
    """Bytes -> BGR uint8. Raises ValueError for anything OpenCV cannot read."""
    array = np.frombuffer(payload, np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a readable image.")
    return image


def to_png_base64(image_bgr: np.ndarray) -> str:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def to_jpeg_bytes(image_bgr: np.ndarray, quality: int = 90) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("could not encode image")
    return encoded.tobytes()


def clahe(channel: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(channel)


def disk(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
