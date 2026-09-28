"""Order-preserving process pool with a progress counter."""
import os
import signal
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2


def _init_worker():
    # Ctrl-C is handled once, by the parent, which cancels pending work.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cv2.setNumThreads(1)


def parallel_map(fn, items, label, workers=None):
    """Apply fn to every item across processes; results keep item order.

    fn must be a module-level function so it can be sent to workers.
    """
    items = list(items)
    total = len(items)
    workers = workers or os.cpu_count() or 1
    if workers == 1 or total < 64:
        return _track(map(fn, items), total, label)
    chunk = max(1, min(32, total // (workers * 4)))
    pool = ProcessPoolExecutor(workers, initializer=_init_worker)
    try:
        return _track(pool.map(fn, items, chunksize=chunk), total, label)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _track(results, total, label):
    out = []
    live = sys.stderr.isatty()
    step = max(1, total // 100)
    for done, result in enumerate(results, 1):
        out.append(result)
        if live and (done % step == 0 or done == total):
            print(f"\r{label}: {done}/{total}", end="", file=sys.stderr)
    if total:
        print(f"\r{label}: {total}/{total}", file=sys.stderr)
    return out
