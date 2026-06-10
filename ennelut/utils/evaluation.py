from __future__ import annotations

import functools
from collections import defaultdict
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from colour.difference import delta_E_CIE2000
from torchmetrics.image import StructuralSimilarityIndexMeasure

from .vectors import rgb_to_lab, to_256_quantized, unnormalize

if TYPE_CHECKING:
    from ennelut.data.luts import LUT


def __prepare_values(values: torch.Tensor, undo_normalize: bool = False, quantize: bool = False) -> torch.Tensor:
    values = unnormalize(values) if undo_normalize else values
    values = to_256_quantized(values) if quantize else values
    return values


def l1(pred: torch.Tensor, target: torch.Tensor, undo_normalize: bool = False, quantize: bool = False) -> torch.Tensor:
    pred = __prepare_values(pred, undo_normalize=undo_normalize, quantize=quantize)
    target = __prepare_values(target, undo_normalize=undo_normalize, quantize=quantize)
    return torch.abs(pred.float() - target.float()).sum(-1)


def l2(pred: torch.Tensor, target: torch.Tensor, undo_normalize: bool = False, quantize: bool = False) -> torch.Tensor:
    pred = __prepare_values(pred, undo_normalize=undo_normalize, quantize=quantize)
    target = __prepare_values(target, undo_normalize=undo_normalize, quantize=quantize)
    return torch.sqrt(((pred.float() - target.float()).square()).sum(-1))


def delta_e(
    pred: torch.Tensor, target: torch.Tensor, undo_normalize: bool = False, quantize: bool = False, **kwargs
) -> torch.Tensor:
    pred = __prepare_values(pred, undo_normalize=undo_normalize, quantize=quantize)
    target = __prepare_values(target, undo_normalize=undo_normalize, quantize=quantize)
    return torch.sqrt((rgb_to_lab(pred) - rgb_to_lab(target)).square().sum(-1))


def delta_e_2000(
    pred: torch.Tensor, target: torch.Tensor, undo_normalize: bool = False, quantize: bool = False, **kwargs
):
    pred = rgb_to_lab(__prepare_values(pred, undo_normalize=undo_normalize, quantize=quantize)).detach().cpu().numpy()
    target = (
        rgb_to_lab(__prepare_values(target, undo_normalize=undo_normalize, quantize=quantize)).detach().cpu().numpy()
    )
    return torch.tensor(delta_E_CIE2000(pred, target))


def psnr(pred: torch.Tensor, target: torch.Tensor, quantize: bool = False) -> torch.Tensor:
    pred = __prepare_values(pred, quantize=quantize) * 255.0
    target = __prepare_values(target, quantize=quantize) * 255.0
    mse = torch.mean((pred - target).square())
    return 20 * torch.log10(255.0 / torch.sqrt(mse))


ssim_function = StructuralSimilarityIndexMeasure(1, data_range=1.0)


def ssim(pred: torch.Tensor, target: torch.Tensor, quantize: bool = False) -> torch.Tensor:
    pred = __prepare_values(pred, quantize=quantize)
    target = __prepare_values(target, quantize=quantize)
    return ssim_function(pred.unsqueeze(0).permute([0, 3, 1, 2]), target.unsqueeze(0).permute([0, 3, 1, 2]))


METRICS = {
    "l1": l1,
    "l2": l2,
    "delta_e": delta_e,
    "delta_e_normalize": functools.partial(delta_e, undo_normalize=True),
    "psnr_quantize": functools.partial(psnr, quantize=True),
    "delta_e_2000": delta_e_2000,
    "psnr": psnr,
    "ssim": ssim,
}


def stats_dict(
    values: torch.Tensor,
    prefix: str | None = None,
    stats: list[str] | None = None,  # ["min", "max", "std", "median", "mean"],
    quantiles: list[float] | None = None,
) -> dict[str, float]:
    values = values.float()
    prefix = "" if prefix is None else prefix
    stats = ["mean"] if stats is None else stats
    results = {f"{prefix}{stat}": getattr(values, stat)() for stat in stats}
    if quantiles is not None:
        quantile_values = values.quantile(torch.tensor(quantiles, dtype=torch.float, device=values.device) / 100)
        results.update(
            {f"{prefix}{q:.2f}th-quantile": value for q, value in zip(quantiles, quantile_values, strict=True)}
        )

    results = {k: v.detach().cpu().item() for k, v in results.items()}
    return results


def stats_description(stats: dict[str, float] | None = None, values: torch.Tensor | None = None) -> str:
    assert values is not None or stats_dict is not None, "Either values or stats_dict must be provided."
    stats = stats_dict(values) if stats is None else stats
    # show the percentage of consistency with two decimals in a separate line
    description = f"consistency: {stats['consistency']:02.2f}%\n" if "consistency" in stats else ""
    # show the other stats
    description += ", ".join([f"{k}: {v:02.4f}" for k, v in stats.items() if k != "consistency"])
    return description


def metric_stats(
    metric_values: torch.Tensor | None = None,
    pred: torch.Tensor | None = None,
    target: torch.Tensor | None = None,
    prefix: str | None = None,
    metric: Literal["l1", "l2", "delta_e", "psnr"] = "delta_e",
    quantiles: list[float] | None = None,
    stats: list[str] | None = None,
) -> dict[str, np.ndarray]:
    assert metric_values is not None or (pred is not None and target is not None), (
        "Either diff or source and reference must be provided"
    )
    stats = ["min", "max", "std", "median", "mean"] if stats is None else stats
    if metric_values is None:
        metric_values = METRICS[metric](pred, target)
    results = dict()
    if metric == "psnr":
        results[metric] = metric_values.detach().cpu().item()
    elif metric == "ssim":
        results[metric] = metric_values.detach().cpu().item()
    else:
        results.update(stats_dict(metric_values, quantiles=quantiles, stats=stats))
        results = {f"{metric}/{k}": v for k, v in results.items()}
    if prefix is not None:
        results = {f"{prefix}{k}": v for k, v in results.items()}
    return results


def evaluate_model(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    luts: list[LUT],
    metrics: list[str] | None = None,
    prefix: str | None = None,
    quantiles: list[float] | None = None,
    chunk_size: int | None = None,
) -> dict[str, list[float]]:
    torch.set_grad_enabled(False)
    metrics = ["delta_e", "psnr"] if metrics is None else metrics
    is_training = model.training
    model.eval()
    forward = model.forward if chunk_size is None else torch.vmap(model.forward, chunk_size=chunk_size)
    metric_values = defaultdict(list)
    original_shape = inputs.shape

    inputs = inputs.reshape(-1, 3)
    for idx, lut in enumerate(luts):
        outputs = to_256_quantized(unnormalize(lut(inputs)))
        inputs, outputs = (outputs, inputs) if model.reversed else (inputs, outputs)
        pred = forward(inputs if model.normalized else unnormalize(inputs), conditions=idx)
        pred = to_256_quantized(unnormalize(pred) if model.normalized else pred)
        for metric in metrics:
            metric_results = metric_stats(
                pred=pred.reshape(original_shape),
                target=outputs.reshape(original_shape),
                metric=metric,
                prefix=prefix,
                quantiles=quantiles,
            )
            for key, value in metric_results.items():
                metric_values[key].append(value)

    model.train(is_training)
    torch.set_grad_enabled(True)
    return metric_values
