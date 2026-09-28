"""Dataset discovery: classes, plant types, augmented copies and splits."""
import hashlib
import os
import random
from collections import defaultdict
from dataclasses import dataclass

from .augment import AUGMENTATIONS
from .cli import LeafError, require_dir

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class ImageClass:
    label: str
    path: str
    plant: str
    files: list


def is_image(name):
    return (not name.startswith(".")
            and os.path.splitext(name)[1].lower() in IMAGE_EXTS)


def iter_images(root):
    """Yield every image path under root in a stable order."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if is_image(name):
                yield os.path.join(dirpath, name)


def plant_of(directory, root):
    """Parent folder when nested (Apple/apple_scab), else the name prefix."""
    directory, root = os.path.abspath(directory), os.path.abspath(root)
    parent = os.path.dirname(directory)
    if directory != root and parent != root:
        return os.path.basename(parent)
    return os.path.basename(directory).split("_")[0]


def find_classes(root):
    """Each directory that directly holds images is one class."""
    require_dir(root)
    by_dir = defaultdict(list)
    for path in iter_images(root):
        by_dir[os.path.dirname(path)].append(path)
    if not by_dir:
        raise LeafError(f"{root}: no images found")
    names = [os.path.basename(os.path.abspath(d)) for d in by_dir]
    classes = []
    for directory, files in by_dir.items():
        label = os.path.basename(os.path.abspath(directory))
        if names.count(label) > 1:
            label = os.path.relpath(directory, root).replace(os.sep, "/")
        classes.append(ImageClass(label, directory,
                                  plant_of(directory, root), files))
    return sorted(classes, key=lambda c: c.label)


def split_augmented(path):
    """Split 'img_Flip_Rotate.JPG' into ('img', ['Flip', 'Rotate'])."""
    stem = os.path.splitext(os.path.basename(path))[0]
    suffixes = []
    while "_" in stem and stem.rsplit("_", 1)[1] in AUGMENTATIONS:
        stem, name = stem.rsplit("_", 1)
        suffixes.insert(0, name)
    return stem, suffixes


def is_augmented(path):
    return bool(split_augmented(path)[1])


def source_key(path):
    """Identifies the original leaf an image (or its augmentation) shows."""
    return os.path.join(os.path.dirname(path), split_augmented(path)[0])


def _digest(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def leaf_groups(files):
    """{leaf: files}; augmented copies and byte-identical photos merge."""
    alias, first = {}, {}
    for path in files:
        if not is_augmented(path):
            key = source_key(path)
            alias[key] = first.setdefault(_digest(path), key)
    groups = defaultdict(list)
    for path in files:
        key = source_key(path)
        groups[alias.get(key, key)].append(path)
    return groups


def split_dataset(classes, val_ratio, seed):
    """Stratified split per leaf: a leaf and its copies stay on one side.

    Validation keeps original images only, so nothing leaks from training.
    """
    train, val = [], []
    for cls in classes:
        groups = leaf_groups(cls.files)
        keys = sorted(groups)
        originals = [k for k in keys
                     if not all(is_augmented(p) for p in groups[k])]
        if len(originals) < 2:
            raise LeafError(f"class '{cls.label}' needs at least 2 "
                            "original images to be split")
        random.Random(f"{seed}:{cls.label}").shuffle(originals)
        n_val = round(len(originals) * val_ratio)
        val_keys = set(originals[:min(len(originals) - 1, max(1, n_val))])
        for key in keys:
            if key in val_keys:
                val += [(p, cls.label) for p in groups[key]
                        if not is_augmented(p)]
            else:
                train += [(p, cls.label) for p in groups[key]]
    return train, val
