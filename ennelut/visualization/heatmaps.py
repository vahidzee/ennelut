from __future__ import annotations

import logging

import matplotlib.colors as mcolors
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ennelut.utils import Vector, numtensor, stats_description, stats_dict


def visualize_heatmaps(
    heatmaps: dict[str, Vector] | None = None,
    sources: dict[str, Vector] | None = None,
    color_heatmaps: dict[str, Vector] | None = None,
    color_heatmaps_cmap: str = "viridis",
    show_stats: bool = True,
    stats_fontsize: int = 9,
    colorbar: bool = True,
    scale_colorbar: float = 1.0,
    figsize: float = 10,
    title: str | None = None,
    warn: bool = False,
):
    # convert everything to numpy arrays
    heatmaps = {} if heatmaps is None else heatmaps
    heatmaps = {name: (numtensor(heatmap, tensor=False)) for name, heatmap in heatmaps.items()}

    if sources is not None:
        sources = {name: (numtensor(source, tensor=False, undo_normalize=True)) for name, source in sources.items()}

    # create a figure with the optional source images on the left
    # the optionally colored heatmaps in the middle
    # and heatmaps with optional colorbars (for each) on the right

    # create the figure
    num_heatmaps = len(heatmaps)
    num_subplots = (
        (len(sources) if sources is not None else 0)
        + (len(color_heatmaps) if color_heatmaps is not None else 0)
        + num_heatmaps
    )
    color_heatmaps_base_idx = len(sources) if sources is not None else 0
    heatmaps_base_idx = color_heatmaps_base_idx + (len(color_heatmaps) if color_heatmaps is not None else 0)
    fig, axes = plt.subplots(1, num_subplots, figsize=(figsize * num_subplots, figsize), constrained_layout=True)

    # plot the source images if provided
    for i, (source_name, source_image) in enumerate((sources or {}).items()):
        if (source_image.min() < 0 or source_image.max() > 1) and warn:
            logging.warning(
                f"Source image {source_name} is not in [0, 1], clipping - [{source_image.min()}, {source_image.max()}]"
            )
        source_image = np.clip(source_image, 0, 1)
        axes[i].imshow(source_image, vmin=0, vmax=1)
        axes[i].set_title(source_name, wrap=True)
        axes[i].axis("off")

    # plot the colored heatmaps if provided
    for i, (name, heatmap) in enumerate((color_heatmaps or {}).items()):
        visualize_colors_heatmap(
            heatmap=heatmap,
            cmap=color_heatmaps_cmap,
            ax=axes[i + color_heatmaps_base_idx],
            title=name if not show_stats else f"{name}\n{stats_description(stats_dict(heatmap))}",
            fontdict={"fontsize": stats_fontsize if show_stats else None},
        )

    # plot the heatmaps
    for i, (name, heatmap) in enumerate(heatmaps.items()):
        # Normalize the heatmap values to [0, 1]
        # get the max of (abs(min), abs(max)) to make sure the heatmap is centered around 0
        value_scale = max(abs(heatmap.min()), abs(heatmap.max()))
        norm = Normalize(vmin=-value_scale, vmax=value_scale)
        cmap = "bwr"
        # Use the "Reds" colormap if all values are positive, otherwise use "bwr"
        if heatmap.min() >= 0:
            cmap = "Reds"
            norm = Normalize(vmin=heatmap.min(), vmax=heatmap.max())
        img = axes[i + heatmaps_base_idx].imshow(heatmap, cmap=cmap, norm=norm, aspect="auto")
        axes[i + heatmaps_base_idx].set_title(
            name if not show_stats else f"{name}\n{stats_description(stats_dict(heatmap))}",
            wrap=True,
            fontdict={"fontsize": stats_fontsize if show_stats else None},
        )
        axes[i + heatmaps_base_idx].axis("off")

        # Add colorbar if required
        if colorbar:
            plt.colorbar(img, ax=axes[i + heatmaps_base_idx], shrink=scale_colorbar)

    # Add title if required
    if title is not None:
        fig.suptitle(title, wrap=True)
    return fig


def visualize_colors_heatmap(
    heatmap: np.array | torch.Tensor,
    cmap: str = "viridis",
    colorbar: bool = True,
    scale_colorbar: float = 1.0,
    figsize: float = 5,
    title: str | None = None,
    fontdict: dict[str, int] | None = None,
    ax: plt.Axes | None = None,
):
    heatmap = numtensor(heatmap, tensor=False)
    # If no colormap is provided, create a custom one from black to red to white
    if ax is None:
        fig, ax = plt.subplots(figsize=(figsize, figsize))
    cmap = plt.get_cmap(cmap)
    colors = [(0, 0, 0)] + cmap(np.linspace(0, 1, 255)).tolist()
    cm = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    img = ax.imshow(heatmap.squeeze(), cmap=cm, aspect="auto")

    ax.axis("off")
    if title is not None:
        ax.set_title(title, wrap=True, fontdict=fontdict)

    if colorbar:
        # Adjust colorbar to be the same height as the image
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(img, cax=cax, shrink=scale_colorbar)
        # add an empty axis to the right of the colorbar to allow for other subplots next to it
        divider = divider.append_axes("right", size="5%", pad=0.3)
        divider.axis("off")

    return ax if ax is not None else fig
