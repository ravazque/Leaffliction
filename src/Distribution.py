#!/usr/bin/env python3
"""Count the images of every class and chart them per plant type."""
import argparse

from leaffliction.cli import run, show
from leaffliction.dataset import find_classes


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Distribution.py",
        description="Show pie and bar charts of the images per class, "
                    "one figure per plant type. Every directory that "
                    "holds images is a class.")
    parser.add_argument("directory", help="dataset root or a class folder")
    return parser.parse_args()


def group_by_plant(classes):
    """{plant: {class label: image count}}, plants matched ignoring case."""
    plants = {}
    for cls in classes:
        name, counts = plants.setdefault(cls.plant.lower(), (cls.plant, {}))
        counts[cls.label] = len(cls.files)
    return dict(plants.values())


def print_table(plant, counts):
    total = sum(counts.values())
    width = max(len(label) for label in counts)
    print(f"{plant}: {len(counts)} classes, {total} images")
    for label, count in counts.items():
        print(f"  {label:<{width}}  {count:>6}  {count / total:6.1%}")


def main():
    import matplotlib.pyplot as plt

    from leaffliction.figures import distribution_figure

    args = parse_args()
    figures = []
    for plant, counts in group_by_plant(find_classes(args.directory)).items():
        print_table(plant, counts)
        fig = distribution_figure(plant, counts, figure=plt.figure)
        figures.append((fig, f"{plant}_distribution"))
    show(figures)


if __name__ == "__main__":
    run(main)
