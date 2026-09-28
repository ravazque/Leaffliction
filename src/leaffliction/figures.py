"""Matplotlib figures used by the programs.

`figure` factory: Figure (default) works headless, pyplot.figure shows it.
"""
import math

from matplotlib.figure import Figure

from .imageio import to_rgb

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
DARK_SURFACE, DARK_INK, DARK_INK_2, GOOD = (
    "#1a1a19", "#ffffff", "#c3c2b7", "#0ca30c")


def bgr(color):
    """'#rrggbb' to an OpenCV BGR tuple."""
    return tuple(int(color[i:i + 2], 16) for i in (5, 3, 1))


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fold(counts):
    """Keep up to 8 classes; smaller extras fold into one 'Other' slice."""
    items = list(counts.items())
    if len(items) <= len(SERIES):
        return ([k for k, _ in items], [v for _, v in items],
                list(SERIES[:len(items)]))
    items.sort(key=lambda kv: -kv[1])
    keep, rest = items[:len(SERIES) - 1], items[len(SERIES) - 1:]
    labels = [k for k, _ in keep] + [f"Other ({len(rest)} classes)"]
    values = [v for _, v in keep] + [sum(v for _, v in rest)]
    return labels, values, list(SERIES[:len(keep)]) + [MUTED]


def distribution_figure(plant, counts, figure=Figure):
    """Pie (share) and bar (count) charts of one plant's classes."""
    labels, values, colors = _fold(counts)
    fig = figure(figsize=(13, 5.5), facecolor=SURFACE, layout="constrained")
    pie_ax, bar_ax = fig.subplots(1, 2, width_ratios=[1, 1.4])
    wedges, _, _ = pie_ax.pie(
        values, colors=colors, startangle=90, counterclock=False,
        autopct="%1.1f%%", pctdistance=1.17,
        wedgeprops={"edgecolor": SURFACE, "linewidth": 2},
        textprops={"color": INK_2, "fontsize": 9})
    pie_ax.legend(wedges, labels, loc="upper center", frameon=False,
                  bbox_to_anchor=(0.5, 0.02), ncols=2, labelcolor=INK_2)
    _style(bar_ax)
    bars = bar_ax.bar(labels, values, color=colors, width=0.6)
    bar_ax.bar_label(bars, padding=3, color=INK_2, fontsize=9)
    bar_ax.set_ylabel("Images", color=INK_2)
    bar_ax.tick_params(axis="x", labelrotation=20)
    for tick in bar_ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    fig.suptitle(f"{plant} class distribution ({sum(values)} images)",
                 color=INK, fontsize=14)
    return fig


def image_grid(images, title, cols=4, axes=False, size=3.2, figure=Figure):
    """Grid of (title, BGR or grayscale image) panels."""
    rows = math.ceil(len(images) / cols)
    fig = figure(figsize=(size * cols, (size + 0.3) * rows),
                 facecolor=SURFACE, layout="constrained")
    for i, ax in enumerate(fig.subplots(rows, cols, squeeze=False).flat):
        if i >= len(images):
            ax.set_visible(False)
            continue
        name, img = images[i]
        ax.imshow(to_rgb(img))
        ax.set_title(name, color=INK)
        if axes:
            ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8)
        else:
            ax.axis("off")
    fig.suptitle(title, color=INK, fontsize=13)
    return fig


def histogram_figure(histograms, title, figure=Figure):
    """One panel per colour space, three channel curves in each."""
    fig = figure(figsize=(15, 4.6), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, len(histograms), sharex=True)
    for ax, (space, channels) in zip(axes, histograms.items()):
        _style(ax)
        for color, (name, values) in zip(SERIES, channels.items()):
            ax.plot(values, color=color, linewidth=1.8, label=name)
        ax.set_title(space, color=INK)
        ax.set_xlim(0, 255)
        ax.set_xlabel("Pixel intensity", color=INK_2)
        ax.legend(frameon=False, labelcolor=INK_2, title="Channel",
                  title_fontproperties={"size": 9})
    axes[0].set_ylabel("Proportion of pixels (%)", color=INK_2)
    fig.suptitle(title, color=INK, fontsize=13)
    return fig


def prediction_figure(original, transformed, label, confidence,
                      figure=Figure):
    """Original and transformed leaf above the predicted class."""
    fig = figure(figsize=(8, 6), facecolor=DARK_SURFACE)
    grid = fig.add_gridspec(2, 2, height_ratios=[4, 1.6], hspace=0.05,
                            wspace=0.06, left=0.04, right=0.96,
                            top=0.97, bottom=0.03)
    for col, img in enumerate((original, transformed)):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(to_rgb(img))
        ax.axis("off")
    fig.text(0.5, 0.21, "===        DL classification        ===",
             ha="center", color=DARK_INK, fontsize=20)
    fig.text(0.5, 0.12, "Class predicted : ", ha="right",
             color=DARK_INK, fontsize=15)
    fig.text(0.5, 0.12, label, ha="left", color=GOOD, fontsize=15,
             fontweight="bold")
    fig.text(0.5, 0.05, f"confidence {confidence:.1%}", ha="center",
             color=DARK_INK_2, fontsize=10)
    return fig
