"""Image files in and out, as 8-bit BGR arrays (OpenCV/PlantCV order)."""
import os

import cv2
import numpy as np

from .cli import LeafError


def load_image(path):
    """Read any supported image as a 3-channel uint8 BGR array."""
    if os.path.isdir(path):
        raise LeafError(f"{path}: is a directory, expected an image")
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise LeafError(f"{path}: not a readable image")
    return img


def save_image(path, img):
    """Write img to path; the extension picks the format."""
    ext = os.path.splitext(path)[1]
    try:
        ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    except cv2.error:
        ok = False
    if not ok:
        raise LeafError(f"{path}: cannot encode image as '{ext}'")
    buf.tofile(path)


def to_rgb(img):
    """BGR or grayscale array to RGB for matplotlib."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
