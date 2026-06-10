from __future__ import annotations

import torch

from ennelut.model import ExactResFlow
from ennelut.utils import rgb_to_lab, unnormalize


def l2_objective_lab(model: ExactResFlow, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    source, target, lut_idx = batch
    if not model.normalized:
        source = unnormalize(source)
        target = unnormalize(target)

    if model.reversed:
        source, target = target, source
    prediction: torch.Tensor = model(source, conditions=lut_idx)
    if model.normalized:
        prediction = unnormalize(prediction)
        target = unnormalize(target)
    prediction = rgb_to_lab(prediction)
    target = rgb_to_lab(target)
    results = ((prediction - target).square()).sum(-1)
    return {"loss": results, "l2": results}


def l2_objective(model: ExactResFlow, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    source, target, lut_idx = batch
    if not model.normalized:
        source = unnormalize(source)
        target = unnormalize(target)

    if model.reversed:
        source, target = target, source
    prediction: torch.Tensor = model(source, conditions=lut_idx)
    results = ((prediction - target).square()).sum(-1)
    return {"loss": results, "l2": results}


def l2_blending(
    model: ExactResFlow,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    np: int | None = None,
    samples: int | None = None,
):
    source, target, lut_idx = batch
    if not model.normalized:
        source = unnormalize(source)
        target = unnormalize(target)

    if model.reversed:
        source, target = target, source
    prediction: torch.Tensor = model(source, conditions=lut_idx)
    results = ((prediction - target).square()).sum(-1)
    return {"loss": results, "l2": results}
