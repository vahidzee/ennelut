from __future__ import annotations

from collections.abc import Generator
from types import GeneratorType

import kornia
import numpy as np
import torch

Vector = np.ndarray | torch.Tensor
LEVEL = 0.1


def numtensor(
    value: list[Vector] | Generator[Vector, None, None] | Vector,
    tensor: bool = True,
    device: torch.device | None = None,
    float: bool = False,
    undo_normalize: bool = False,
    quant: int | bool = False,
) -> list[np.ndarray | torch.Tensor] | torch.Tensor | np.ndarray:
    if isinstance(value, (list, tuple)):
        return [numtensor(v, tensor=tensor, undo_normalize=undo_normalize, quant=quant) for v in value]
    if isinstance(value, GeneratorType):
        return (numtensor(v, tensor=tensor, undo_normalize=undo_normalize, quant=quant) for v in value)
    if tensor:
        results = torch.from_numpy(value).float() if isinstance(value, np.ndarray) else value
        results = results.float() if float else results
        results = results.to(device) if device is not None else results
    else:
        results = value.cpu().detach().numpy() if isinstance(value, torch.Tensor) else value
        results = results.astype(np.float32) if float else results
    results = results if not undo_normalize else unnormalize(results)
    if quant:
        depth = quant if not isinstance(quant, bool) else 8
        results = quantize(results, bits=depth)
    return results


def normalize(value: Vector):
    # assume value is in [-LEVEL, 1 + LEVEL]
    # put it in [-1, 1]
    return (value - 0.5) / (1 + 2 * LEVEL) * 2


def unnormalize(value: Vector):
    # assume values is in [-1, 1] put it in [-LEVEL, 1 + LEVEL]
    value = value * (1 + 2 * LEVEL) / 2 + 0.5
    # clamp values to [0, 1]
    return value.clamp(0, 1) if isinstance(value, torch.Tensor) else np.clip(value, 0, 1)


def to_256_quantized(x: torch.Tensor) -> torch.Tensor:
    return torch.floor(x * 255.0).to(torch.uint8).float() / 255.0


def rgb_to_lab(x):
    original_shape = x.shape
    x = kornia.color.rgb_to_lab(x.reshape(-1, 3).T.unsqueeze(0).unsqueeze(-1)).squeeze().reshape(3, -1).T
    return x.reshape(original_shape)


def quantize(value: Vector, bits: int = 8):
    if bits == 8:
        if isinstance(value, torch.Tensor):
            return (value * 255).clamp(0, 255).byte()
        return (value * 255).clip(0, 255).astype(np.uint8)
    max_val = 2**bits - 1
    if isinstance(value, torch.Tensor):
        return torch.round(value * max_val).to(torch.int32)
    return np.round(value * max_val).astype(np.uint32)
