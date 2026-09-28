"""Model inputs: the same leaf-only preprocessing for training and predict.

Jobs return an array, or an error string so one bad file never stops a batch.
"""
import cv2
import numpy as np

from .augment import augment
from .cli import LeafError
from .imageio import load_image, save_image
from .mask import leaf_mask, remove_background


def leaf_only(img):
    """What the network sees: the segmented leaf on a white background."""
    return remove_background(img, leaf_mask(img))


def to_array(img, size):
    """BGR image to a size x size RGB uint8 array, channels first."""
    small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(small[:, :, ::-1].transpose(2, 0, 1))


def to_float(batch):
    """uint8 tensor batch to floats in [-1, 1]."""
    return batch.float().div_(127.5).sub_(1.0)


def prepare_file(job):
    """(path, size) -> model input array."""
    path, size = job
    try:
        return to_array(leaf_only(load_image(path)), size)
    except LeafError as err:
        return str(err)


def build_train_image(job):
    """(source, augmentation or None, output, size, key, seed): save the
    leaf-only training image and return its model input array."""
    src, name, out, size, key, seed = job
    try:
        img = load_image(src)
        if name:
            img = augment(img, name, key=key, seed=seed)
        leaf = leaf_only(img)
        save_image(out, leaf)
        return to_array(leaf, size)
    except LeafError as err:
        return str(err)
