#!/usr/bin/env python3
"""Show or save PlantCV feature transformations of leaf images."""
import argparse
import os

from leaffliction.cli import LeafError, is_within, run, show, warn
from leaffliction.dataset import iter_images
from leaffliction.imageio import load_image
from leaffliction.parallel import parallel_map
from leaffliction.transform import save_transformations, transform

FLAGS = {
    "blur": ("Blur", "Gaussian blur of the thresholded leaf"),
    "mask": ("Mask", "leaf on a white background"),
    "roi": ("ROI", "objects inside the region of interest"),
    "analyze": ("Analyze", "shape analysis (outline, hull, centre)"),
    "pseudo": ("Pseudolandmarks", "landmarks along the leaf edges"),
    "hist": ("Histogram", "colour histograms (RGB, HSV, LAB)"),
}

EXAMPLES = """examples:
  Transformation.py "images/Apple_healthy/image (1).JPG"
  Transformation.py -src images/Apple_healthy -dst transformed
  Transformation.py -src images/Apple_healthy -dst transformed -mask -hist
"""


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Transformation.py", allow_abbrev=False, epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Display the transformations of one image, or save "
                    "them for every image of -src into -dst as "
                    "'<name>_<Transformation>'. Without transformation "
                    "flags, all of them are produced.")
    parser.add_argument("image", nargs="?", help="image to display")
    parser.add_argument("-src", metavar="PATH",
                        help="image or directory to process (recursive)")
    parser.add_argument("-dst", metavar="DIR",
                        help="where to save the transformations")
    group = parser.add_argument_group("transformations")
    for flag, (_, text) in FLAGS.items():
        group.add_argument(f"-{flag}", action="store_true", help=text)
    args = parser.parse_args()
    if args.image and args.src:
        parser.error("give either an image or -src, not both")
    if not args.image and not args.src:
        parser.error("an image or -src is required")
    if args.src and not args.dst:
        parser.error("-src requires -dst")
    args.names = [name for flag, (name, _) in FLAGS.items()
                  if getattr(args, flag)] or [n for n, _ in FLAGS.values()]
    return args


def display(path, names):
    import matplotlib.pyplot as plt

    from leaffliction.figures import histogram_figure, image_grid
    from leaffliction.histogram import color_histograms

    img = load_image(path)
    seg, views = transform(img, names)
    if not seg.mask.any():
        warn(f"{path}: no leaf found")
    title = os.path.basename(path)
    figures = []
    if views:
        panels = [("Original", img)] + [(t, v) for _, t, v in views]
        figures.append((image_grid(panels, title, cols=3, axes=True,
                                   figure=plt.figure), "transformations"))
    if "Histogram" in names:
        fig = histogram_figure(color_histograms(img, seg.mask),
                               f"{title} colour histogram", figure=plt.figure)
        figures.append((fig, "histogram"))
    show(figures)


def collect_jobs(src, dst, names):
    """One job per image; sub-folders of a directory are mirrored in dst."""
    if not os.path.exists(src):
        raise LeafError(f"{src}: no such file or directory")
    if os.path.exists(dst) and not os.path.isdir(dst):
        raise LeafError(f"{dst}: exists and is not a directory")
    if not os.path.isdir(src):
        return [(src, dst, names)]
    if is_within(dst, src):
        raise LeafError("-dst must be outside the -src directory")
    jobs = []
    for path in iter_images(src):
        rel = os.path.relpath(os.path.dirname(path), src)
        jobs.append((path, os.path.normpath(os.path.join(dst, rel)), names))
    if not jobs:
        raise LeafError(f"{src}: no images found")
    return jobs


def save_all(src, dst, names):
    jobs = collect_jobs(src, dst, names)
    errors = [e for e in parallel_map(save_transformations, jobs,
                                      "Transforming") if e]
    for error in errors:
        warn(error)
    print(f"{len(jobs) - len(errors)}/{len(jobs)} images transformed "
          f"into {dst} ({', '.join(names)})")
    return 1 if len(errors) == len(jobs) else 0


def main():
    args = parse_args()
    if args.src:
        return save_all(args.src, args.dst, args.names)
    if args.dst:
        return save_all(args.image, args.dst, args.names)
    return display(args.image, args.names)


if __name__ == "__main__":
    run(main)
