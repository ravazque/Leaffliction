#!/usr/bin/env python3
"""Augment one image (and show it) or balance a whole dataset."""
import argparse
import os

from leaffliction.augment import AUGMENTATIONS, augment
from leaffliction.balance import augmented_name, balance_plan, write_image
from leaffliction.cli import LeafError, is_within, run, show, warn
from leaffliction.dataset import find_classes
from leaffliction.imageio import load_image, save_image
from leaffliction.parallel import parallel_map

DEFAULT_DST = "augmented_directory"


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Augmentation.py",
        description="Image: save and display its 6 augmentations "
                    f"({', '.join(AUGMENTATIONS)}) next to it. "
                    "Directory: copy it to --dst and add augmented images "
                    "until every class has as many images as the largest.")
    parser.add_argument("path", help="an image or a dataset directory")
    parser.add_argument("--dst", help="output directory (image: next to "
                        f"the original, directory: {DEFAULT_DST})")
    parser.add_argument("--target", type=int,
                        help="images per class when balancing "
                             "(default: size of the largest class)")
    parser.add_argument("--seed", type=int, default=7,
                        help="random seed (default: 7)")
    args = parser.parse_args()
    if args.target is not None and args.target < 1:
        parser.error("--target must be positive")
    return args


def augment_image(args):
    import matplotlib.pyplot as plt

    from leaffliction.figures import image_grid

    img = load_image(args.path)
    out_dir = args.dst or os.path.dirname(args.path) or "."
    os.makedirs(out_dir, exist_ok=True)
    key = os.path.basename(args.path)
    views = [("Original", img)]
    for name in AUGMENTATIONS:
        out = augment(img, name, key=key, seed=args.seed)
        path = os.path.join(out_dir, augmented_name(args.path, name))
        save_image(path, out)
        print(f"Saved {path}")
        views.append((name, out))
    fig = image_grid(views, key, cols=len(views), size=2.4,
                     figure=plt.figure)
    show([(fig, "augmentation")])


def balance_directory(args):
    src, dst = args.path, args.dst or DEFAULT_DST
    if is_within(dst, src) or is_within(src, dst):
        raise LeafError("--dst must be outside the source directory "
                        "and must not contain it")
    if os.path.exists(dst) and (not os.path.isdir(dst) or os.listdir(dst)):
        raise LeafError(f"{dst}: already exists, remove it or choose "
                        "another --dst")
    classes = find_classes(src)
    plan = balance_plan({c.label: c.files for c in classes},
                        target=args.target, seed=args.seed)
    jobs = []
    for cls in classes:
        rel = os.path.relpath(cls.path, src)
        out_dir = os.path.normpath(os.path.join(dst, rel))
        os.makedirs(out_dir, exist_ok=True)
        for path in cls.files:
            out = os.path.join(out_dir, os.path.basename(path))
            jobs.append((path, None, out, "", args.seed))
        for path, name in plan[cls.label]:
            out = os.path.join(out_dir, augmented_name(path, name))
            key = f"{rel}/{os.path.basename(path)}"
            jobs.append((path, name, out, key, args.seed))
    errors = [e for e in parallel_map(write_image, jobs, "Writing images")
              if e]
    for error in errors:
        warn(error)
    print(f"Balanced dataset written to {dst}")
    for cls in classes:
        print(f"  {cls.label}: {len(cls.files)} -> "
              f"{len(cls.files) + len(plan[cls.label])}")
    return 1 if errors else 0


def main():
    args = parse_args()
    if os.path.isdir(args.path):
        return balance_directory(args)
    if not os.path.exists(args.path):
        raise LeafError(f"{args.path}: no such file or directory")
    return augment_image(args)


if __name__ == "__main__":
    run(main)
