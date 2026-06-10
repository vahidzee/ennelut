from __future__ import annotations

import gc
import os
import re
from pathlib import Path

import cv2
import lightning.fabric as fabric
import numpy as np
import pandas as pd
import torch
import yaml
from lightning import pytorch as pl
from lightning.fabric.loggers import TensorBoardLogger as FabricTensorBoardLogger
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt

from .vectors import Vector, normalize, numtensor

NUMBERS_REGEX = re.compile(
    r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$"
)


def load_image(path: Path | str, float: bool = True) -> Vector:
    """
    Load an image from a path and return it as a numpy array. Depending on the value of `zero_centered`, the image
    will be returned in the range [0, 1] or [-1, 1].
    """
    image = cv2.imread(str(path), -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if float:
        image = image.astype(np.float32) / 255.0
    return normalize(image)


def load_cube(path: Path | str, device: str = "cpu") -> tuple[int, torch.Tensor]:
    grid_size = None
    with open(path) as f:
        for i, line in enumerate(f):
            if "LUT_3D_SIZE" in line:
                grid_size = int(line.split(" ")[1])
            # mark the start of the LUT data
            # check if the line contains 3 number in the form of f f f
            elif re.match(NUMBERS_REGEX, line) is not None:
                start_idx = i
                break
    outputs = np.loadtxt(path, skiprows=start_idx, usecols=(2, 1, 0))
    grid_size = int(np.ceil(pow(outputs.shape[0], 1 / 3))) if grid_size is None else grid_size

    assert outputs.shape[0] == grid_size**3, f"{path} LUT is not a cube"
    assert outputs.min() >= 0 and outputs.max() <= 1, (
        f"{path} LUT values are not in [0, 1], {outputs.min()}, {outputs.max()}"
    )
    outputs = torch.from_numpy(outputs.reshape(grid_size, grid_size, grid_size, -1)).float().to(device)
    return (grid_size, normalize(outputs))


def save_image(image: Vector, path: Path | str, undo_normalize: bool = True) -> None:
    """
    Save the provided image to the provided path. Depending on the value of `zero_centered`, the image will be
    assumed to be in the range [0, 1] or [-1, 1].
    """
    image: np.ndarray = numtensor(image, undo_normalize=undo_normalize, tensor=False)  # shape [H, W, 3]
    image = (image * 255.0).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image)


def log_figure(
    title: str,
    figure: plt.Figure,
    trainer: pl.Trainer = None,
    fabric: fabric.Fabric = None,
    prefix: str | None = None,
    save_root: str | None = None,
    step: int | None = None,
    logger=None,
):
    """
    Utility function used to log matplotlib figures to TensorBoard or Wandb. If `save_root` is provided, the figure
    will be saved to the provided path instead.
    """

    if logger is None:
        logger = trainer.logger if trainer is not None else fabric.logger
    step = step if step is not None else trainer.global_step
    assert logger is not None, "logger must be provided, either directly or through trainer or fabric"
    if save_root is not None:
        print("saving figure", os.path.join(save_root, f"{title}-{step}.png"))
        figure.savefig(os.path.join(save_root, f"{title}-{step}.png"))
        return
    if logger is None:
        return
    else:
        prefix = "" if prefix is None else f"{prefix}/"
        if isinstance(logger, (TensorBoardLogger, FabricTensorBoardLogger)):
            logger.experiment.add_figure(f"{prefix}{title}", figure, global_step=step)
            logger.experiment.flush()
        else:
            # otherwise assume wandb
            logger.log_image(key=f"{prefix}{title}", images=[figure], caption=[title], step=step)
    plt.clf()
    plt.close(figure)
    del figure
    gc.collect()


def results_yaml_to_csv(path: str, target: str = "hald-64") -> pd.DataFrame:
    with open(path) as file:
        new_data = yaml.safe_load(file)

    # Extract the data for hald-64 from the new file
    new_hald_64_data = new_data[target]

    # Initializing lists to hold extracted values for all metrics
    new_delta_e_lab_mean = []
    new_delta_e_lab_max = []
    new_delta_e_lab_min = []
    new_delta_e_lab_std = []
    new_delta_e_lab_consistency = []

    new_l1_mean = []
    new_l1_max = []
    new_l1_min = []
    new_l1_std = []
    new_l1_consistency = []

    new_psnr_values = []

    # Extracting values from each dictionary and appending to the lists
    for entry in new_hald_64_data:
        new_delta_e_lab_mean.append(entry["delta_e_lab"]["mean"])
        new_delta_e_lab_max.append(entry["delta_e_lab"]["max"])
        new_delta_e_lab_min.append(entry["delta_e_lab"]["min"])
        new_delta_e_lab_std.append(entry["delta_e_lab"]["std"])
        new_delta_e_lab_consistency.append(entry["delta_e_lab"]["consistency"])

        new_l1_mean.append(entry["l1"]["mean"])
        new_l1_max.append(entry["l1"]["max"])
        new_l1_min.append(entry["l1"]["min"])
        new_l1_std.append(entry["l1"]["std"])
        new_l1_consistency.append(entry["l1"]["consistency"])

        new_psnr_values.append(entry["psnr"])

    # Creating the dataframe using the extracted lists
    return pd.DataFrame(
        {
            "LUT-ID": range(0, len(new_hald_64_data)),
            "psnr": new_psnr_values,
            "l1_consistency": new_l1_consistency,
            "l1_mean": new_l1_mean,
            "l1_std": new_l1_std,
            "l1_max": new_l1_max,
            "l1_min": new_l1_min,
            "delta_e_lab_consistency": new_delta_e_lab_consistency,
            "delta_e_lab_mean": new_delta_e_lab_mean,
            "delta_e_lab_std": new_delta_e_lab_std,
            "delta_e_lab_max": new_delta_e_lab_max,
            "delta_e_lab_min": new_delta_e_lab_min,
        }
    )
