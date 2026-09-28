#!/usr/bin/env python3
"""Predict the disease of a leaf image, or score a whole directory."""
import argparse
import os

import numpy as np
import torch

from leaffliction.cli import LeafError, run, show, warn
from leaffliction.data import leaf_only, prepare_file, to_array
from leaffliction.dataset import find_classes
from leaffliction.imageio import load_image
from leaffliction.learnings import load_learnings
from leaffliction.parallel import parallel_map
from leaffliction.training import pick_device, predict, report

MAX_LISTED = 20


def parse_args():
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description="Image: show it with its transformation and the "
                    "predicted class. Directory: predict every image and, "
                    "when folder names are known classes, print accuracy.")
    parser.add_argument("path", help="leaf image or directory of images")
    parser.add_argument("--learnings", default="learnings.zip",
                        help="learnings zip or extracted folder "
                             "(default: learnings.zip)")
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda or cuda:N (default: auto)")
    return parser.parse_args()


def predict_image(args, device):
    import matplotlib.pyplot as plt

    from leaffliction.figures import prediction_figure

    img = load_image(args.path)
    model, meta = load_learnings(args.learnings)
    leaf = leaf_only(img)
    x = torch.from_numpy(to_array(leaf, meta["img_size"])[None])
    probs = predict(model, x, device)[0]
    best = int(probs.argmax())
    label = meta["classes"][best]
    print(f"Class predicted: {label} ({probs[best]:.1%})")
    fig = prediction_figure(img, leaf, label, float(probs[best]),
                            figure=plt.figure)
    show([(fig, "prediction")])


def predict_directory(args, device):
    items = [(p, c.label) for c in find_classes(args.path) for p in c.files]
    model, meta = load_learnings(args.learnings)
    classes = meta["classes"]
    # Preprocess before the model reaches the GPU: workers are forked.
    arrays = parallel_map(prepare_file,
                          [(p, meta["img_size"]) for p, _ in items],
                          "Preparing images")
    good = []
    for (path, label), array in zip(items, arrays):
        if isinstance(array, str):
            warn(array)
        else:
            good.append((path, label, array))
    if not good:
        raise LeafError("no usable image")
    probs = predict(model, torch.from_numpy(np.stack([a for *_, a in good])),
                    device)
    preds = probs.argmax(1)
    known = [i for i, (_, label, _) in enumerate(good) if label in classes]
    if not known:
        for (path, _, _), k, p in zip(good, preds, probs):
            print(f"{path}: {classes[k]} ({p[k]:.1%})")
        return 0
    truth = [classes.index(good[i][1]) for i in known]
    _, text = report(preds[known], truth, classes)
    print(text)
    wrong = [i for i in known if classes[preds[i]] != good[i][1]]
    if wrong:
        print(f"\nMisclassified ({len(wrong)}):")
    for i in wrong[:MAX_LISTED]:
        print(f"  {good[i][0]}: predicted {classes[preds[i]]}")
    if len(wrong) > MAX_LISTED:
        print(f"  ... and {len(wrong) - MAX_LISTED} more")
    if len(known) < len(good):
        warn(f"{len(good) - len(known)} images are in folders that are "
             "not known classes and were not scored")
    return 0


def main():
    args = parse_args()
    device = pick_device(args.device)
    if os.path.isdir(args.path):
        return predict_directory(args, device)
    if not os.path.exists(args.path):
        raise LeafError(f"{args.path}: no such file or directory")
    return predict_image(args, device)


if __name__ == "__main__":
    run(main)
