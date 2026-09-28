#!/usr/bin/env python3
"""Train the leaf disease classifier and package the learnings."""
import argparse
import os
import shutil
import time
from collections import defaultdict

import numpy as np
import torch

from leaffliction.balance import augmented_name, balance_plan
from leaffliction.cli import LeafError, is_within, run, warn
from leaffliction.data import build_train_image, prepare_file
from leaffliction.dataset import find_classes, split_dataset
from leaffliction.learnings import (build_zip, prepare_workdir, save_model,
                                    write_signature)
from leaffliction.model import MIN_SIZE, LeafCNN
from leaffliction.parallel import parallel_map
from leaffliction.training import fit, pick_device, predict, report

MIN_VALIDATION, TARGET_ACCURACY = 100, 0.90


def parse_args():
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Split the dataset per leaf, balance and mask the "
                    "training part, train a CNN, measure it on the "
                    "validation part and zip everything with its sha1.")
    parser.add_argument("directory", help="dataset root (one folder per "
                        "class)")
    parser.add_argument("--out", default="learnings.zip",
                        help="zip to produce (default: learnings.zip)")
    parser.add_argument("--signature", default="signature.txt",
                        help="sha1 file (default: signature.txt)")
    parser.add_argument("--workdir", default="learnings",
                        help="folder zipped as the learnings "
                             "(default: learnings)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=128,
                        help="network input side in pixels (default: 128)")
    parser.add_argument("--lr", type=float, default=2e-3,
                        help="peak learning rate (default: 0.002)")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="share of leaves kept for validation")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda or cuda:N (default: auto)")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.img_size < MIN_SIZE:
        parser.error(f"--img-size must be at least {MIN_SIZE}")
    if not 0 < args.val_split < 1:
        parser.error("--val-split must be between 0 and 1")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    return args


def check_paths(args):
    work = args.workdir
    if is_within(work, args.directory) or is_within(args.directory, work):
        raise LeafError("--workdir must be outside the dataset and must "
                        "not contain it")
    for path in (args.out, args.signature):
        if is_within(path, work):
            raise LeafError(f"{path} must be outside --workdir")


def build_training_set(items, labels, args):
    """Balance the training split by augmentation and mask every image."""
    files = defaultdict(list)
    for path, label in items:
        files[label].append(path)
    plan = balance_plan(files, seed=args.seed)
    jobs, targets = [], []
    for index, label in enumerate(labels):
        out_dir = os.path.join(args.workdir, "train", label)
        os.makedirs(out_dir, exist_ok=True)
        todo = [(p, None, os.path.basename(p)) for p in files[label]]
        todo += [(p, n, augmented_name(p, n)) for p, n in plan[label]]
        for src, name, out in todo:
            key = f"{label}/{os.path.basename(src)}"
            jobs.append((src, name, os.path.join(out_dir, out),
                         args.img_size, key, args.seed))
            targets.append(index)
    arrays = parallel_map(build_train_image, jobs, "Preparing training set")
    return stack(arrays, targets)


def build_validation_set(items, labels, args):
    """Copy the untouched validation images and prepare them like predict."""
    for path, label in items:
        out_dir = os.path.join(args.workdir, "validation", label)
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy2(path, out_dir)
    arrays = parallel_map(prepare_file, [(p, args.img_size) for p, _ in items],
                          "Preparing validation set")
    return stack(arrays, [labels.index(label) for _, label in items])


def stack(arrays, targets):
    """Drop failed images (warning about them) and build tensors."""
    kept = []
    for array, target in zip(arrays, targets):
        if isinstance(array, str):
            warn(array)
        else:
            kept.append((array, target))
    if not kept:
        raise LeafError("no usable image")
    x = torch.from_numpy(np.stack([a for a, _ in kept]))
    return x, torch.tensor([t for _, t in kept], dtype=torch.long)


def main():
    args = parse_args()
    started = time.time()
    device = pick_device(args.device)
    check_paths(args)
    classes = find_classes(args.directory)
    if len(classes) < 2:
        raise LeafError("at least 2 classes are needed to train")
    labels = [c.label for c in classes]
    train_items, val_items = split_dataset(classes, args.val_split,
                                           args.seed)
    print(f"{len(labels)} classes: {len(train_items)} training and "
          f"{len(val_items)} validation images (split per leaf)")
    if len(val_items) < MIN_VALIDATION:
        warn(f"only {len(val_items)} validation images, fewer than "
             f"{MIN_VALIDATION}")
    prepare_workdir(args.workdir)
    x_train, y_train = build_training_set(train_items, labels, args)
    x_val, y_val = build_validation_set(val_items, labels, args)
    print(f"Training on {len(x_train)} balanced images with {device}")
    torch.manual_seed(args.seed)
    model = fit(LeafCNN(len(labels)), x_train, y_train, device,
                args.epochs, args.batch_size, args.lr, args.seed)
    probs = predict(model, x_val, device)
    accuracy, text = report(probs.argmax(1), y_val.numpy(), labels)
    print(f"\nValidation\n{text}")
    with open(os.path.join(args.workdir, "metrics.txt"), "w") as f:
        f.write(f"Validation\n{text}\n")
    if accuracy < TARGET_ACCURACY:
        warn(f"validation accuracy {accuracy:.2%} is below "
             f"{TARGET_ACCURACY:.0%}")
    save_model(args.workdir, model.cpu(), {
        "classes": labels, "img_size": args.img_size, "seed": args.seed,
        "epochs": args.epochs, "train_images": len(x_train),
        "validation_images": len(x_val),
        "validation_accuracy": round(accuracy, 4)})
    build_zip(args.workdir, args.out)
    digest = write_signature(args.out, args.signature)
    print(f"\nLearnings: {args.out} (sha1 {digest}), signature in "
          f"{args.signature}; done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    run(main)
