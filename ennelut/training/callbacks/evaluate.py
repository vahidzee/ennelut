from __future__ import annotations

import glob
import os

import torch
import wandb
import yaml

from ennelut.data import LUT, generate_grid
from ennelut.model import ExactResFlow
from ennelut.utils import evaluate_model, load_image


class FabricEvaluateCallback:
    def __init__(
        self,
        title: str = "evaluation",
        hald: int | None = 256,
        images: list[str] | None = None,
        metrics: list[str] | None = None,
        quantiles: list[float] | None = None,
        chunk_size: int = 128**3,
    ):
        self.title = title  # used for logging
        self.images = self.__process_images(images if images is not None else [])
        print("evaluation images", len(self.images))
        self.metrics = ["psnr", "delta_e"] if metrics is None else metrics
        self.quantiles = [75, 90, 95, 97.5, 99, 99.9] if quantiles is None else quantiles
        self.hald = hald
        self.chunk_size = chunk_size

        # set in setup/fit_start
        self.context = None
        self.root: str = self.title
        self.model: ExactResFlow = None
        self.device: torch.device = None
        self.luts: list[LUT] = None

    def __process_images(self, paths: list[str]) -> list[str]:
        if isinstance(paths, str):
            paths = [paths]
        processed_paths = []
        for path in paths:
            if "*" in path:
                processed_paths.extend(glob.glob(path))
            else:
                processed_paths.append(path)
        return processed_paths

    def setup(self, context, **kwargs):
        self.context = context
        # create checkpoint directory
        if "root" not in kwargs:
            self.root = os.path.join(wandb.run.dir, self.root)
        else:
            self.root = kwargs["root"]
        os.makedirs(self.root, exist_ok=True)

    def on_fit_start(self, **kwargs) -> None:
        self.model: ExactResFlow = self.context.model if "model" not in kwargs else kwargs["model"]
        self.device = self.context.fabric.device if "device" not in kwargs else kwargs["device"]
        self.luts = self.context.data.luts if "luts" not in kwargs else kwargs["luts"]

    def on_fit_end(self, **kwargs) -> None:
        name = "" if "name" not in kwargs else f"{kwargs['name']}-"
        if self.hald:
            print("Evaluating Hald", end="\r")
            inputs = generate_grid(self.hald, meshgrid=True, device=self.device)
            results = evaluate_model(
                model=self.model,
                inputs=inputs,
                luts=self.luts,
                metrics=self.metrics,
                quantiles=self.quantiles,
                chunk_size=self.chunk_size,
            )
            self.save_results(name=f"{name}hald-{self.hald}", results=results)
        for path in self.images:
            print(f"Evaluating {path}", end="\r")
            image = torch.from_numpy(load_image(path)).float().to(self.device)
            results = evaluate_model(
                model=self.model,
                inputs=image,
                luts=self.luts,
                metrics=self.metrics,
                quantiles=self.quantiles,
                chunk_size=self.chunk_size,
            )
            self.save_results(name=f"{name}{os.path.splitext(os.path.basename(path))[0]}", results=results)

    def save_results(self, results, name):
        path = os.path.join(self.root, f"{name}.yaml")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as file:
            yaml.dump(dict(results), file, default_flow_style=False)
