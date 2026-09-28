"""Colour histograms of the leaf pixels in the RGB, HSV and LAB spaces."""
import cv2

# space -> (conversion from BGR, ((channel name, channel index), ...))
SPACES = {
    "RGB": (None, (("blue", 0), ("red", 2), ("green", 1))),
    "HSV": (cv2.COLOR_BGR2HSV_FULL,
            (("hue", 0), ("saturation", 1), ("value", 2))),
    "LAB": (cv2.COLOR_BGR2LAB,
            (("lightness", 0), ("green-magenta", 1), ("blue-yellow", 2))),
}


def color_histograms(img, mask):
    """{space: {channel: 256 values}}, each value a % of the leaf pixels."""
    out = {}
    for space, (code, channels) in SPACES.items():
        conv = img if code is None else cv2.cvtColor(img, code)
        out[space] = {}
        for name, idx in channels:
            hist = cv2.calcHist([conv], [idx], mask, [256], [0, 256]).ravel()
            out[space][name] = hist * 100 / max(hist.sum(), 1)
    return out
