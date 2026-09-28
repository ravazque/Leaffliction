import os
import shutil
import zlib

import pytest
from conftest import leaf_image, write

from leaffliction.cli import LeafError
from leaffliction.dataset import (find_classes, is_augmented, is_image,
                                  leaf_groups, source_key, split_augmented,
                                  split_dataset)


def make(root, *names):
    """Write a distinct small leaf per name."""
    for name in names:
        img = leaf_image(size=32, seed=zlib.crc32(name.encode()))[0]
        write(os.path.join(root, name), img)


def test_is_image_ignores_case_hidden_and_other_files():
    assert all(is_image(n) for n in ("a.JPG", "b.jpeg", "c.PNG", "d.tif"))
    assert not any(is_image(n) for n in ("notes.txt", ".hidden.jpg", "jpg"))


def test_flat_layout_uses_name_prefix_as_plant(tmp_path):
    make(tmp_path, "Apple_rust/a.JPG", "Grape_spot/b.jpg", "Grape_spot/c.png")
    (tmp_path / "Grape_spot" / "notes.txt").write_text("not an image")
    (tmp_path / "empty").mkdir()
    classes = find_classes(str(tmp_path))
    assert [(c.label, c.plant, len(c.files)) for c in classes] == [
        ("Apple_rust", "Apple", 1), ("Grape_spot", "Grape", 2)]


def test_nested_layout_uses_parent_as_plant(tmp_path):
    make(tmp_path, "Apple/apple_healthy/a.JPG", "Apple/apple_scab/b.JPG")
    assert {c.plant for c in find_classes(str(tmp_path))} == {"Apple"}
    assert {c.plant for c in find_classes(str(tmp_path / "Apple"))} == {
        "apple"}


def test_root_holding_images_is_a_single_class(tmp_path):
    make(tmp_path, "Apple_rust/a.JPG", "Apple_rust/b.JPG")
    [cls] = find_classes(str(tmp_path / "Apple_rust"))
    assert (cls.label, cls.plant, len(cls.files)) == ("Apple_rust",
                                                      "Apple", 2)


def test_duplicate_folder_names_get_relative_labels(tmp_path):
    make(tmp_path, "A/healthy/a.JPG", "B/healthy/b.JPG")
    labels = [c.label for c in find_classes(str(tmp_path))]
    assert labels == ["A/healthy", "B/healthy"]


@pytest.mark.parametrize("setup", ["missing", "file", "empty", "no_images"])
def test_find_classes_rejects_bad_roots(tmp_path, setup):
    target = tmp_path / "target"
    if setup == "file":
        target.write_text("x")
    elif setup == "empty":
        target.mkdir()
    elif setup == "no_images":
        target.mkdir()
        (target / "readme.md").write_text("x")
    with pytest.raises(LeafError):
        find_classes(str(target))


@pytest.mark.parametrize("name, expected", [
    ("image (1).JPG", ("image (1)", [])),
    ("image (1)_Flip.JPG", ("image (1)", ["Flip"])),
    ("image (1)_Flip_Rotate.JPG", ("image (1)", ["Flip", "Rotate"])),
    ("leaf_2.jpg", ("leaf_2", [])),
    ("my_Crop_leaf.png", ("my_Crop_leaf", [])),
])
def test_split_augmented(name, expected):
    assert split_augmented(name) == expected


def test_source_key_groups_copies_of_one_leaf():
    assert source_key("c/x_Skew.JPG") == source_key("c/x.JPG")
    assert source_key("c/x.JPG") != source_key("d/x.JPG")


def test_split_is_per_leaf_stratified_and_reproducible(tmp_path):
    for label in ("A_x", "B_y"):
        for i in range(10):
            make(tmp_path, f"{label}/img{i}.JPG", f"{label}/img{i}_Flip.JPG")
    classes = find_classes(str(tmp_path))
    train, val = split_dataset(classes, 0.2, seed=3)
    train_keys = {source_key(p) for p, _ in train}
    assert not train_keys & {source_key(p) for p, _ in val}
    assert not any(is_augmented(p) for p, _ in val)
    assert {label for _, label in val} == {"A_x", "B_y"}
    assert len(val) == 4 and len(train) == 32
    assert split_dataset(classes, 0.2, seed=3) == (train, val)


def test_split_keeps_one_leaf_for_training_even_with_huge_ratio(tmp_path):
    make(tmp_path, "A_x/1.JPG", "A_x/2.JPG", "B_y/1.JPG", "B_y/2.JPG")
    train, val = split_dataset(find_classes(str(tmp_path)), 0.99, seed=0)
    assert len(train) == 2 and len(val) == 2


def test_split_needs_two_original_leaves_per_class(tmp_path):
    make(tmp_path, "A_x/1.JPG", "A_x/2.JPG", "B_y/1.JPG", "B_y/1_Flip.JPG")
    with pytest.raises(LeafError, match="B_y"):
        split_dataset(find_classes(str(tmp_path)), 0.2, seed=0)


def test_identical_photos_count_as_one_leaf(tmp_path):
    make(tmp_path, "A_x/a.JPG", "A_x/b.JPG", "A_x/b_Flip.JPG")
    shutil.copy(tmp_path / "A_x" / "a.JPG", tmp_path / "A_x" / "twin.JPG")
    groups = leaf_groups(find_classes(str(tmp_path))[0].files)
    assert sorted(len(files) for files in groups.values()) == [2, 2]
