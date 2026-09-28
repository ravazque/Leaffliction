import os

import numpy as np
import pytest
from conftest import leaf_image, write

from leaffliction.augment import AUGMENTATIONS, augment
from leaffliction.balance import augmented_name, balance_plan, write_image
from leaffliction.imageio import load_image

NAMES = list(AUGMENTATIONS)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("shape", [(96, 96), (40, 70), (5, 7)])
def test_augmentation_keeps_shape_and_changes_pixels(name, shape):
    img = leaf_image(size=96, spots=5)[0][:shape[0], :shape[1]].copy()
    out = augment(img, name, key="k", seed=1)
    assert out.shape == img.shape and out.dtype == np.uint8
    if min(shape) > 10:
        assert not np.array_equal(out, img)


@pytest.mark.parametrize("name", NAMES)
def test_augmentation_is_reproducible(name):
    img = leaf_image(spots=5)[0]
    first = augment(img, name, key="a", seed=5)
    assert np.array_equal(first, augment(img, name, key="a", seed=5))


def test_augmented_name_keeps_extension():
    assert augmented_name("d/image (1).JPG", "Flip") == "image (1)_Flip.JPG"


def test_plan_reaches_target_with_distinct_augmentations():
    files = {"big": [f"b{i}.JPG" for i in range(12)],
             "small": ["s0.JPG", "s1.JPG", "s2.JPG"]}
    plan = balance_plan(files, seed=0)
    assert plan["big"] == []
    assert len(plan["small"]) == 9
    assert len(set(plan["small"])) == 9
    per_source = {src: [n for s, n in plan["small"] if s == src]
                  for src in files["small"]}
    assert all(len(v) == 3 for v in per_source.values())


def test_plan_is_capped_and_warns(capsys):
    plan = balance_plan({"a": ["a.JPG"], "b": [f"{i}.JPG" for i in range(20)]})
    assert len(plan["a"]) == len(NAMES)
    assert "can only grow" in capsys.readouterr().err


def test_plan_skips_existing_names_and_augmented_sources():
    files = {"a": ["x.JPG", "x_Flip.JPG"], "b": [f"{i}.JPG" for i in range(4)]}
    jobs = balance_plan(files)["a"]
    assert ("x.JPG", "Flip") not in jobs
    assert all(src == "x.JPG" for src, _ in jobs)


def test_plan_explicit_target_below_size_adds_nothing():
    assert balance_plan({"a": ["1.JPG", "2.JPG"]}, target=1) == {"a": []}


def test_write_image_copies_augments_and_reports_errors(tmp_path):
    src = write(str(tmp_path / "leaf.JPG"), leaf_image()[0])
    copy, aug = str(tmp_path / "copy.JPG"), str(tmp_path / "aug.JPG")
    assert write_image((src, None, copy, "", 0)) is None
    assert write_image((src, "Rotate", aug, "k", 0)) is None
    assert load_image(aug).shape == load_image(src).shape
    bad = tmp_path / "bad.JPG"
    bad.write_text("not an image")
    assert "not a readable image" in write_image(
        (str(bad), "Flip", str(tmp_path / "o.JPG"), "", 0))
    assert not os.path.exists(tmp_path / "o.JPG")
