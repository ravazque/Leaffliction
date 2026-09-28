"""Class balancing: which augmentations bring every class to a target."""
import os
import random
import shutil

from .augment import AUGMENTATIONS, augment
from .cli import LeafError, warn
from .dataset import is_augmented
from .imageio import load_image, save_image

NAMES = list(AUGMENTATIONS)


def augmented_name(path, name):
    stem, ext = os.path.splitext(os.path.basename(path))
    return f"{stem}_{name}{ext}"


def balance_plan(files, target=None, seed=0):
    """{label: [(original, augmentation)]} filling every class to target.

    Each original gets each augmentation once at most; names never repeat.
    """
    target = target or max(len(paths) for paths in files.values())
    plan = {}
    for label, paths in files.items():
        taken = {os.path.basename(p) for p in paths}
        order = sorted(p for p in paths if not is_augmented(p))
        random.Random(f"{seed}:{label}").shuffle(order)
        jobs = [(path, NAMES[(i + r) % len(NAMES)])
                for r in range(len(NAMES)) for i, path in enumerate(order)]
        jobs = [(p, n) for p, n in jobs if augmented_name(p, n) not in taken]
        need = target - len(paths)
        if need > len(jobs):
            warn(f"class '{label}' can only grow to "
                 f"{len(paths) + len(jobs)} images (target {target})")
        plan[label] = jobs[:max(0, need)]
    return plan


def write_image(job):
    """(source, augmentation or None, output, key, seed): copy or augment.

    Returns None on success, else an error message.
    """
    src, name, out, key, seed = job
    try:
        if name is None:
            shutil.copy2(src, out)
        else:
            save_image(out, augment(load_image(src), name, key, seed))
    except (LeafError, OSError) as err:
        return str(err)
    return None
