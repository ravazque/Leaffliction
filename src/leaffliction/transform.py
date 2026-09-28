"""Feature-extraction views of a leaf, built on its PlantCV segmentation."""
import os

import cv2
import numpy as np
from plantcv import plantcv as pcv

from .cli import LeafError
from .figures import SERIES, bgr, histogram_figure
from .histogram import color_histograms
from .imageio import load_image, save_image
from .mask import remove_background, segment

BLUE, ORANGE, MAGENTA = bgr(SERIES[0]), bgr(SERIES[1]), bgr(SERIES[4])
GREEN = (0, 200, 0)  # BGR


def gaussian_blur(img, seg):
    return seg.blur


def mask(img, seg):
    return remove_background(img, seg.mask)


def roi_objects(img, seg):
    """Objects kept by a rectangular region of interest, tinted green."""
    h, w = img.shape[:2]
    m = max(2, min(h, w) // 50)
    roi = pcv.roi.rectangle(img, x=m, y=m, h=h - 2 * m, w=w - 2 * m)
    kept = pcv.roi.filter(seg.mask, roi, roi_type="partial") > 0
    out = img.copy()
    out[kept] = (0.35 * img[kept] + 0.65 * np.array(GREEN)).astype(np.uint8)
    cv2.rectangle(out, (m // 2, m // 2), (w - 1 - m // 2, h - 1 - m // 2),
                  BLUE, thickness=m)
    return out


def analyze_object(img, seg):
    """PlantCV shape analysis: outline, convex hull, centre, extents."""
    out = pcv.analyze.size(img, seg.mask, n_labels=1)
    pcv.outputs.clear()
    return out


def pseudolandmarks(img, seg):
    """Landmarks spaced along the x axis: top, bottom and centre rows."""
    out = img.copy()
    if not seg.mask.any():
        return out
    rows = pcv.homology.x_axis_pseudolandmarks(img, seg.mask)
    pcv.outputs.clear()
    radius = max(2, min(img.shape[:2]) // 64)
    for points, color in zip(rows, (BLUE, MAGENTA, ORANGE)):
        for x, y in np.asarray(points, dtype=float).reshape(-1, 2):
            cv2.circle(out, (int(x), int(y)), radius, color, cv2.FILLED)
    return out


TRANSFORMATIONS = {
    "Blur": ("Gaussian blur", gaussian_blur),
    "Mask": ("Mask", mask),
    "ROI": ("ROI objects", roi_objects),
    "Analyze": ("Analyze object", analyze_object),
    "Pseudolandmarks": ("Pseudolandmarks", pseudolandmarks),
}


def transform(img, names=None):
    """Return (segmentation, [(name, title, image)]) for the chosen views."""
    seg = segment(img)
    views = [(name, title, fn(img, seg))
             for name, (title, fn) in TRANSFORMATIONS.items()
             if names is None or name in names]
    return seg, views


def save_transformations(job):
    """(image, output dir, names): write '<stem>_<Name>' files.

    Returns None on success, else a message explaining the skip.
    """
    path, out_dir, names = job
    try:
        img = load_image(path)
        seg, views = transform(img, names)
        if not seg.mask.any():
            return f"{path}: no leaf found, skipped"
        stem, ext = os.path.splitext(os.path.basename(path))
        os.makedirs(out_dir, exist_ok=True)
        for name, _, view in views:
            save_image(os.path.join(out_dir, f"{stem}_{name}{ext}"), view)
        if "Histogram" in names:
            fig = histogram_figure(color_histograms(img, seg.mask),
                                   f"{stem}{ext} colour histogram")
            fig.savefig(os.path.join(out_dir, f"{stem}_Histogram.png"),
                        dpi=80)
    except (LeafError, OSError) as err:
        return str(err)
    return None
