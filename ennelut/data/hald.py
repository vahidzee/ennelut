from __future__ import annotations

import functools
import math

import numpy as np
import torch

from ennelut.utils import normalize, unnormalize


@functools.cache
def generate_grid(grid_size: int = 32, meshgrid: bool = True, device: str = "numpy"):
    points = normalize(torch.linspace(0, 1.0, grid_size))
    results = torch.stack(torch.meshgrid(points, points, points, indexing="ij"), axis=-1) if meshgrid else points
    return results.numpy() if device == "numpy" else results.to(device)


def compute_hald_grid_size(hald_image: np.array) -> int:
    width, height, _ = hald_image.shape
    # Calculate the grid size using the formula pow(width * height, 1/3)
    grid_size = round(pow(width * height, 1 / 3))

    # Find the nearest power of two that is less than or equal to the rounded grid size
    power_of_two_grid_size = 2 ** math.floor(math.log2(grid_size))

    return power_of_two_grid_size


def compute_hald_square_size(grid_size: int) -> tuple[int, int, int]:
    # compute the closest width/height to keep the HALD image as close to a square as possible
    space_size = grid_size**3
    square_length = int(np.sqrt(space_size))

    # since the grid_size might not be a square, we might need to compensate
    width = (square_length // grid_size) * grid_size
    height = np.ceil(space_size // grid_size / width).astype(int) * grid_size

    return square_length, width, height


def unroll_grid(grid: np.array):
    # used for visualization purposes
    num_channels = grid.shape[-1]
    grid = grid.reshape(-1, num_channels)
    grid_size = int(np.ceil(pow(grid.shape[0], 1 / 3)))
    _, width, height = compute_hald_square_size(grid_size)

    image = np.zeros(width * height * num_channels).reshape(height, width, num_channels)

    # unroll the grid manually to create a HALD image
    # where as you go along x axis of the image (horizontal), first channel is is increased (red)
    # and vertically second channel (green) is increased, and as you traverse the slices from top left
    # to bottom right, third channel (blue) is increased.
    col = 0  # the column in the image (starting from left to right)
    for i in range(len(grid)):
        image[((i - col * grid_size * height) // grid_size) % height, col * grid_size + i % grid_size] = grid[i]
        if i == height * (col + 1) * grid_size - 1:
            col += 1
    return image


# @cached(images=True)
def generate_hald_image(grid_size: int = 32):
    return unroll_grid(generate_grid(grid_size, device="numpy", meshgrid=True))


def colors_hald_heatmap(
    colors: np.ndarray,
    grid_size: int = 256,
    dtype: np.dtype = np.uint8,
    unroll: bool = True,
):
    """
    Creates a hald heatmap of the given grid_size, highlighting the colors in the given colorspace.
    """
    # create the grid
    grid = np.zeros([grid_size] * colors.shape[-1], dtype=dtype)

    # convert colors to [0, 1] range
    colors = unnormalize(colors)
    # convert the colors to the given dtype (e.g. uint8 would be [0, 255])
    colors = (colors * np.iinfo(dtype).max).astype(dtype)

    # convert them to have grid_size quantization
    # values (e.g. [0, 1, 2, 3] for grid_size=4 and uint8)
    # to be able to index the grid
    indices = (colors // ((np.iinfo(dtype).max + 1) // grid_size)).reshape(-1, colors.shape[-1])

    # given that indices could be really large, simply doing grid[indices] would
    # create a huge array, so we opt for a loop instead
    for i in range(len(indices)):
        grid[tuple(indices[i])] += 1
    return unroll_grid(grid.transpose([2, 1, 0]).reshape(*grid.shape, 1)) if unroll else grid
