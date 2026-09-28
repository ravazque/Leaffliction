"""Command-line plumbing: clean error reporting and figure display."""
import os
import sys

import matplotlib

NON_INTERACTIVE = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


class LeafError(Exception):
    """Expected, user-facing failure (bad path, bad input, ...)."""


def warn(message):
    print(f"Warning: {message}", file=sys.stderr)


def _fail(message):
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def run(main):
    """Run main() and turn every failure into a message and exit code."""
    try:
        code = main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except LeafError as err:
        _fail(err)
    except OSError as err:
        where = f": {err.filename}" if err.filename else ""
        _fail(f"{err.strerror or err}{where}")
    except MemoryError:
        _fail("out of memory")
    except Exception as err:
        _fail(f"unexpected {type(err).__name__}: {err}")
    sys.exit(code or 0)


def require_dir(path):
    if not os.path.exists(path):
        raise LeafError(f"{path}: no such file or directory")
    if not os.path.isdir(path):
        raise LeafError(f"{path}: not a directory")


def is_within(path, parent):
    """True when path is parent itself or lies somewhere inside it."""
    path, parent = os.path.realpath(path), os.path.realpath(parent)
    return os.path.commonpath([path, parent]) == parent


def interactive():
    return matplotlib.get_backend().lower() not in NON_INTERACTIVE


def show(figures):
    """Open (figure, name) pairs in windows, or save them as PNG files."""
    import matplotlib.pyplot as plt

    if interactive():
        plt.show()
    else:
        for fig, name in figures:
            path = f"{name.replace(' ', '_')}.png"
            fig.savefig(path, dpi=100, bbox_inches="tight")
            print(f"No display available, figure saved to {path}")
    plt.close("all")
