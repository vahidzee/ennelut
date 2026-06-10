from __future__ import annotations

import torch


def trilinear_interpolation(inputs: torch.Tensor, grid: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    """
    Perform trilinear interpolation on a regular cubic lattice (the grid),
    with a given set of outputs (could be batched).
    """
    inputs = inputs.T

    grid = grid.contiguous()
    inputs = inputs.contiguous()

    idx = torch.clip(torch.searchsorted(grid, inputs, right=True), 1, len(grid) - 1)
    cube_size = grid[1] - grid[0]

    x_d = ((inputs[0] - grid[idx[0] - 1]) / cube_size).unsqueeze(1)
    y_d = ((inputs[1] - grid[idx[1] - 1]) / cube_size).unsqueeze(1)
    z_d = ((inputs[2] - grid[idx[2] - 1]) / cube_size).unsqueeze(1)

    if outputs.ndim == 4:
        outputs = outputs.unsqueeze(0)
    c00 = outputs[:, idx[0] - 1, idx[1] - 1, idx[2] - 1] * (1 - x_d) + outputs[:, idx[0], idx[1] - 1, idx[2] - 1] * x_d
    c01 = outputs[:, idx[0] - 1, idx[1] - 1, idx[2]] * (1 - x_d) + outputs[:, idx[0], idx[1] - 1, idx[2]] * x_d
    c10 = outputs[:, idx[0] - 1, idx[1], idx[2] - 1] * (1 - x_d) + outputs[:, idx[0], idx[1], idx[2] - 1] * x_d
    c11 = outputs[:, idx[0] - 1, idx[1], idx[2]] * (1 - x_d) + outputs[:, idx[0], idx[1], idx[2]] * x_d

    c0 = c00 * (1 - y_d) + c10 * y_d
    c1 = c01 * (1 - y_d) + c11 * y_d

    return (c0 * (1 - z_d) + c1 * z_d).squeeze()
