"""Shared fixtures: synthetic leaves and a helper to run the programs."""
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
IMAGES = os.path.join(ROOT, "images")
sys.path.insert(0, SRC)

BACKGROUND = (150, 128, 140)  # purple-grey, BGR
GREEN, BROWN = (40, 150, 60), (30, 70, 120)


def leaf_image(size=128, spots=0, seed=0):
    """Green elliptic leaf, optional brown spots; returns (image, mask)."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), BACKGROUND, np.float64)
    img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    mask = np.zeros((size, size), np.uint8)
    centre = tuple(int(size / 2 + d) for d in rng.integers(-5, 6, 2))
    axes = (int(size * 0.36), int(size * 0.22))
    angle = float(rng.uniform(0, 180))
    cv2.ellipse(img, centre, axes, angle, 0, 360, GREEN, -1)
    cv2.ellipse(mask, centre, axes, angle, 0, 360, 255, -1)
    ys, xs = np.nonzero(mask)
    for i in rng.choice(len(xs), size=spots, replace=False):
        cv2.circle(img, (int(xs[i]), int(ys[i])), max(2, size // 24),
                   BROWN, -1)
    return img, mask


def write(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    assert ok
    buf.tofile(path)
    return path


@pytest.fixture
def leaf_file(tmp_path):
    """One synthetic leaf saved with spaces and parentheses in its name."""
    return write(str(tmp_path / "Plant_healthy" / "leaf (1).JPG"),
                 leaf_image()[0])


@pytest.fixture
def dataset(tmp_path):
    """Two classes: 'Plant_healthy' (clean) and 'Plant_spot' (spotted)."""
    root = tmp_path / "data"
    for label, spots in (("Plant_healthy", 0), ("Plant_spot", 12)):
        for i in range(10):
            img, _ = leaf_image(size=96, spots=spots, seed=i + 100 * spots)
            write(str(root / label / f"image ({i}).JPG"), img)
    return str(root)


@pytest.fixture
def cli(tmp_path):
    """Run a program headless in tmp_path; returns the CompletedProcess."""
    env = dict(os.environ, MPLBACKEND="Agg")

    def run(program, *args):
        result = subprocess.run(
            [sys.executable, os.path.join(SRC, program), *map(str, args)],
            cwd=tmp_path, env=env, capture_output=True, text=True,
            timeout=600)
        assert "Traceback" not in result.stderr, result.stderr
        return result

    return run
