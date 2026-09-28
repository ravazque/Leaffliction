"""Leaf segmentation with PlantCV on the LAB b - a channel (leaf high).

Otsu gives sure leaf; weaker pixels (lesions, glare) join if connected.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from plantcv import plantcv as pcv

pcv.params.debug = None
pcv.params.verbose = False
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
MIN_CONTRAST = 16  # a flatter channel holds no distinct object


@dataclass
class Segmentation:
    blur: np.ndarray  # blurred Otsu threshold
    mask: np.ndarray  # final 0/255 leaf mask


def leaf_channel(img):
    a = pcv.rgb2gray_lab(img, "a").astype(np.int16)
    b = pcv.rgb2gray_lab(img, "b").astype(np.int16)
    return np.clip(b - a + 128, 0, 255).astype(np.uint8)


def _largest_blob(binary):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return np.zeros_like(binary)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def _hysteresis(channel):
    """Otsu pixels, plus those halfway to background that touch them."""
    level, _ = cv2.threshold(channel, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sure = pcv.threshold.binary(channel, int(level), "light")
    if sure.all() or not sure.any():
        return sure
    low = (np.median(channel[sure == 0]) + level) / 2
    weak = pcv.threshold.binary(channel, int(low), "light")
    n, labels = cv2.connectedComponents(weak, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[labels[sure > 0]] = True
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def segment(img):
    """Threshold, smooth, keep the largest blob and fill its holes."""
    channel = leaf_channel(img)
    if int(channel.max()) - int(channel.min()) < MIN_CONTRAST:
        empty = np.zeros(channel.shape, np.uint8)
        return Segmentation(empty, empty)
    blur = pcv.gaussian_blur(_hysteresis(channel), ksize=(5, 5))
    mask = np.where(blur > 127, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(_largest_blob(mask), cv2.MORPH_CLOSE, KERNEL)
    mask = np.where(pcv.fill_holes(mask) > 0, 255, 0).astype(np.uint8)
    return Segmentation(blur, mask)


def leaf_mask(img):
    return segment(img).mask


def remove_background(img, mask):
    return pcv.apply_mask(img, mask, "white")
