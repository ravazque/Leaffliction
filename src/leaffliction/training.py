"""Training loop, inference and validation report."""
import math
import time
import warnings

import numpy as np
import torch
from torch import nn

from .cli import LeafError
from .data import to_float

# GradScaler may skip the very first optimizer steps; that is expected.
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler")


def pick_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(name)
    except RuntimeError as err:
        raise LeafError(f"invalid device '{name}': {err}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise LeafError("CUDA requested but no CUDA device is available")
    return device


def _coin(n, gen, device):
    return (torch.rand(n, generator=gen) < 0.5).to(device).view(-1, 1, 1, 1)


def _random_flips(x, gen):
    """Per-sample flips and one random quarter turn per batch."""
    x = torch.where(_coin(len(x), gen, x.device), x.flip(3), x)
    x = torch.where(_coin(len(x), gen, x.device), x.flip(2), x)
    return torch.rot90(x, int(torch.randint(4, (1,), generator=gen)), (2, 3))


def fit(model, images, labels, device, epochs, batch_size, lr, seed,
        log=print):
    """AdamW with a one-cycle schedule, for a fixed number of epochs.

    images: uint8 tensor N x 3 x S x S, labels: int64 tensor N.
    """
    gen = torch.Generator().manual_seed(seed)
    model.to(device).train()
    images, labels = images.to(device), labels.to(device)
    n = len(images)
    steps = math.ceil(n / batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=epochs * steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    for epoch in range(1, epochs + 1):
        start, seen_loss, correct = time.time(), 0.0, 0
        order = torch.randperm(n, generator=gen).to(device)
        for i in range(steps):
            idx = order[i * batch_size:(i + 1) * batch_size]
            x = to_float(_random_flips(images[idx], gen))
            y = labels[idx]
            with torch.autocast(device.type, torch.float16, enabled=amp):
                out = model(x)
                loss = loss_fn(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            seen_loss += loss.item() * len(idx)
            correct += (out.argmax(1) == y).sum().item()
        log(f"epoch {epoch:>2}/{epochs}  loss {seen_loss / n:.4f}  "
            f"train accuracy {correct / n:.2%}  ({time.time() - start:.0f}s)")
    return model


@torch.no_grad()
def predict(model, images, device, batch_size=256):
    """Class probabilities (numpy N x C) for a uint8 image tensor."""
    model.to(device).eval()
    amp = device.type == "cuda"
    probs = []
    for i in range(0, len(images), batch_size):
        x = to_float(images[i:i + batch_size].to(device))
        with torch.autocast(device.type, torch.float16, enabled=amp):
            logits = model(x)
        probs.append(torch.softmax(logits.float(), 1).cpu())
    return torch.cat(probs).numpy()


def report(pred, truth, classes):
    """Return (accuracy, text with per-class scores and confusion matrix)."""
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (np.asarray(truth), np.asarray(pred)), 1)
    total, hits = int(cm.sum()), int(np.trace(cm))
    accuracy = hits / max(total, 1)
    width = max(len(c) for c in classes)
    lines = [f"Accuracy: {accuracy:.2%} ({hits}/{total} images)", "",
             "Per class:"]
    for i, name in enumerate(classes):
        right, count = cm[i, i], cm[i].sum()
        if count:
            lines.append(f"  {i:>2}  {name:<{width}}  {right / count:7.2%}"
                         f"  ({right}/{count})")
    lines += ["", "Confusion matrix (rows: true class, columns: predicted):",
              "    " + "".join(f"{j:>6}" for j in range(n))]
    lines += [f"  {i:>2}" + "".join(f"{v:>6}" for v in row)
              for i, row in enumerate(cm)]
    return accuracy, "\n".join(lines)
