"""Six geometric augmentations, each (BGR image, RNG) -> new image.

Uncovered areas take the background colour so the leaf mask ignores them.
"""
import zlib

import cv2
import numpy as np


def _background(img):
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    return tuple(int(v) for v in np.median(border, axis=0))


def _warp_affine(img, matrix):
    h, w = img.shape[:2]
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=_background(img))


def flip(img, rng):
    return cv2.flip(img, int(rng.choice([1, 0, -1])))


def rotate(img, rng):
    h, w = img.shape[:2]
    angle = rng.uniform(15, 45) * rng.choice([-1, 1])
    return _warp_affine(img, cv2.getRotationMatrix2D((w / 2, h / 2),
                                                     angle, 1.0))


def skew(img, rng):
    """Projective tilt: one edge of the image shrinks toward its centre."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src.copy()
    d = rng.uniform(0.12, 0.25)
    edge = int(rng.integers(4))  # top, right, bottom, left
    along = 0 if edge in (0, 2) else 1  # coordinate running along the edge
    centre = (w if along == 0 else h) / 2
    for corner in (edge, (edge + 1) % 4):
        dst[corner, along] += (centre - dst[corner, along]) * 2 * d
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=_background(img))


def shear(img, rng):
    h, w = img.shape[:2]
    s = rng.uniform(0.15, 0.3) * rng.choice([-1, 1])
    if rng.random() < 0.5:
        matrix = np.float32([[1, s, -s * h / 2], [0, 1, 0]])
    else:
        matrix = np.float32([[1, 0, 0], [s, 1, -s * w / 2]])
    return _warp_affine(img, matrix)


def crop(img, rng):
    h, w = img.shape[:2]
    scale = rng.uniform(0.7, 0.85)
    ch, cw = max(1, int(h * scale)), max(1, int(w * scale))
    y, x = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
    return cv2.resize(img[y:y + ch, x:x + cw], (w, h),
                      interpolation=cv2.INTER_LINEAR)


def distortion(img, rng):
    """Smooth sinusoidal wave applied to the pixel grid."""
    h, w = img.shape[:2]
    amp = rng.uniform(0.03, 0.05) * min(h, w)
    period = rng.uniform(0.5, 0.9) * min(h, w)
    phase = rng.uniform(0, 2 * np.pi, size=2)
    ys, xs = np.indices((h, w), dtype=np.float32)
    map_x = xs + amp * np.sin(2 * np.pi * ys / period + phase[0])
    map_y = ys + amp * np.sin(2 * np.pi * xs / period + phase[1])
    return cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=_background(img))


AUGMENTATIONS = {
    "Flip": flip,
    "Rotate": rotate,
    "Skew": skew,
    "Shear": shear,
    "Crop": crop,
    "Distortion": distortion,
}


def augment(img, name, key="", seed=0):
    """Apply one augmentation, reproducible for a given (key, name, seed)."""
    rng = np.random.default_rng([seed, zlib.crc32(f"{key}:{name}".encode())])
    return AUGMENTATIONS[name](img, rng)
