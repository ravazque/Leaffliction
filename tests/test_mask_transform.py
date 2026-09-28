import os

import numpy as np
import pytest
from conftest import IMAGES, leaf_image, write

from leaffliction.data import leaf_only, to_array
from leaffliction.dataset import iter_images
from leaffliction.histogram import color_histograms
from leaffliction.imageio import load_image
from leaffliction.mask import leaf_mask, segment
from leaffliction.transform import (TRANSFORMATIONS, save_transformations,
                                    transform)

ALL = list(TRANSFORMATIONS) + ["Histogram"]


def iou(a, b):
    a, b = a > 0, b > 0
    return (a & b).sum() / max((a | b).sum(), 1)


@pytest.mark.parametrize("spots", [0, 15])
def test_mask_matches_synthetic_leaf(spots):
    img, truth = leaf_image(size=128, spots=spots, seed=spots)
    mask = leaf_mask(img)
    assert set(np.unique(mask)) <= {0, 255}
    assert iou(mask, truth) > 0.93


@pytest.mark.skipif(not os.path.isdir(IMAGES), reason="dataset not present")
def test_mask_covers_a_plausible_share_of_real_leaves():
    paths = [p for i, p in enumerate(iter_images(IMAGES)) if i % 250 == 0]
    ratios = [leaf_mask(load_image(p)).mean() / 255 for p in paths]
    assert min(ratios) > 0.1 and max(ratios) < 0.9


@pytest.mark.parametrize("value", [0, 128, 255])
def test_uniform_images_never_crash(value):
    img = np.full((64, 64, 3), value, np.uint8)
    seg, views = transform(img)
    assert seg.mask.shape == (64, 64)
    assert [name for name, _, _ in views] == list(TRANSFORMATIONS)
    for _, _, view in views:
        assert view.shape[:2] == (64, 64)
    assert leaf_only(img).shape == img.shape


def test_views_have_image_size():
    img = leaf_image(size=100, spots=4)[0]
    seg, views = transform(img, ["Mask", "Pseudolandmarks"])
    assert [n for n, _, _ in views] == ["Mask", "Pseudolandmarks"]
    masked = views[0][2]
    assert (masked[seg.mask == 0] == 255).all()


def test_histograms_are_percentages_of_leaf_pixels():
    img, _ = leaf_image(spots=6)
    hists = color_histograms(img, leaf_mask(img))
    assert list(hists) == ["RGB", "HSV", "LAB"]
    for channels in hists.values():
        assert len(channels) == 3
        for values in channels.values():
            assert values.shape == (256,)
            assert values.sum() == pytest.approx(100, abs=1e-3)
    empty = color_histograms(img, np.zeros(img.shape[:2], np.uint8))
    assert all(v.sum() == 0 for c in empty.values() for v in c.values())


def test_to_array_is_channels_first_rgb():
    img = np.zeros((10, 20, 3), np.uint8)
    img[..., 0] = 7  # blue
    out = to_array(img, 8)
    assert out.shape == (3, 8, 8) and out.dtype == np.uint8
    assert out[2].min() == 7 and out[0].max() == 0


def test_save_transformations_writes_named_files(tmp_path, leaf_file):
    out = tmp_path / "out"
    assert save_transformations((leaf_file, str(out), ALL)) is None
    assert sorted(os.listdir(out)) == sorted(
        [f"leaf (1)_{n}.JPG" for n in TRANSFORMATIONS]
        + ["leaf (1)_Histogram.png"])


def test_save_transformations_skips_blank_and_broken(tmp_path):
    blank = write(str(tmp_path / "blank.png"),
                  np.zeros((32, 32, 3), np.uint8))
    broken = tmp_path / "broken.JPG"
    broken.write_text("x")
    out = str(tmp_path / "out")
    assert "no leaf" in save_transformations((blank, out, ALL))
    assert "not a readable" in save_transformations((str(broken), out, ALL))


def test_segment_blur_is_the_smoothed_threshold():
    img = leaf_image(spots=3)[0]
    seg = segment(img)
    assert seg.blur.shape == img.shape[:2]
    assert len(np.unique(seg.blur)) > 2  # soft edges add grey levels
