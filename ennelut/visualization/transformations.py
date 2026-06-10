from __future__ import annotations

import gc
from collections.abc import Iterable

import dypy
from matplotlib import pyplot as plt

from ennelut.utils import METRICS, Vector, metric_stats, numtensor, psnr, stats_description


def visualize_transformations(
    ground_truths: Iterable[Vector],
    predictions: Iterable[Vector],
    source: Vector | None = None,
    ground_truths_names: list[str] | str | None = None,
    predictions_names: list[str] | str | None = None,
    figsize: float = 10,
    title: str | None = None,
    colorbar: bool = True,
    scale_colorbar: float = 0.9,
    metrics: list[str] | None = None,
    show_stats: bool = True,
    stats_fontsize: int = 9,
    cmap: str = "gist_heat",
    axes: plt.axes | None = None,
    fig: plt.figure | None = None,
    num_pairs: int | None = None,
    return_metrics: bool = False,
):
    # Convert torch tensors to numpy arrays and undo zero centering if required
    source = numtensor(source, tensor=False, undo_normalize=True, quant=True) if source is not None else None
    ground_truths = numtensor(ground_truths, tensor=False, undo_normalize=True, quant=True)
    predictions = numtensor(predictions, tensor=False, undo_normalize=True, quant=True)

    # generate the names for the ground truth and predictions
    assert num_pairs is not None or len(ground_truths) == len(predictions), (
        "Number of ground truth images and predictions must be the same."
    )

    metrics = ["l1", "delta_e_lab"] if metrics is None else metrics
    metrics = {metric: METRICS[metric] for metric in metrics if metric in METRICS}
    num_pairs = len(ground_truths) if num_pairs is None else num_pairs
    num_cols = (2 + len(metrics)) * num_pairs + (1 if source is not None else 0)

    if ground_truths_names is None:
        ground_truths_names = [f"Ground Truth #{i + 1}" for i in range(num_pairs)]
    elif isinstance(ground_truths_names, str):
        ground_truths_names = [f"{ground_truths_names} #{i + 1}" for i in range(num_pairs)]
    else:
        assert len(ground_truths_names) == num_pairs, "Number of ground truth names must match the number of pairs."

    if predictions_names is None:
        predictions_names = [f"Prediction #{i + 1}" for i in range(num_pairs)]
    elif isinstance(predictions_names, str):
        predictions_names = [f"{predictions_names} #{i + 1}" for i in range(num_pairs)]
    else:
        assert len(predictions_names) == num_pairs, "Number of predictions names must match the number of pairs."

    if axes is not None:
        assert len(axes) == num_cols, "Number of axes must match the number of columns."
        assert fig is not None, "Figure must be provided if axes are provided."
    else:
        fig, axes = plt.subplots(
            1,
            num_cols,
            figsize=(figsize * num_cols, figsize),
            constrained_layout=True,
            gridspec_kw={"wspace": 0, "hspace": 0},
        )

    col_idx = 0
    results = {}
    # Plot the source image
    if source is not None:
        axes[col_idx].imshow(source)
        axes[col_idx].set_title("Source", wrap=True)
        axes[col_idx].axis("off")
        col_idx += 1
        del source
        gc.collect()
    for i, (gt, pred) in enumerate(zip(ground_truths, predictions, strict=False)):
        # Plot the ground truth transformation
        axes[col_idx].imshow(gt)
        axes[col_idx].set_title(ground_truths_names[i], wrap=True)
        axes[col_idx].axis("off")
        col_idx += 1

        # Plot the predicted transformation
        axes[col_idx].imshow(pred)
        psnr_value = psnr(pred=pred, target=gt)
        axes[col_idx].set_title(f"{predictions_names[i]}\npsnr: {psnr_value:02.04f}", wrap=True)
        axes[col_idx].axis("off")
        col_idx += 1
        if num_pairs > 1:
            results[predictions_names[i]] = dict()

        (results[predictions_names[i]] if num_pairs > 1 else results)["psnr"] = psnr_value.item()

        for metric, function in metrics.items():
            # Calculate the metric values between the ground truth and the prediction
            metric_values = dypy.functions.call_with_dynamic_args(function, target=gt, pred=pred, zero_centered=False)
            metric_results = metric_stats(metric_values, metric=metric)
            (results[predictions_names[i]] if num_pairs > 1 else results)[metric] = metric_results
            # Plot the difference
            header = f"Difference ({metric})"
            header = header if not show_stats else f"{header}\n{stats_description(stats=metric_results)}"

            img = axes[col_idx].imshow(metric_values, cmap=cmap)
            axes[col_idx].set_title(header, wrap=True, fontdict={"fontsize": stats_fontsize if show_stats else None})
            axes[col_idx].axis("off")
            col_idx += 1

            # Add colorbar if required
            if colorbar:
                # draw a colorbar with the same height as the image
                plt.colorbar(img, ax=axes[col_idx - 1], shrink=scale_colorbar)

            del metric_values
            gc.collect()

        del gt, pred
        gc.collect()

    # Add title if required
    if title is not None:
        fig.suptitle(title, wrap=True)

    return fig if not return_metrics else (fig, results)
