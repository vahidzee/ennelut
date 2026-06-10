from __future__ import annotations

import torch

from ennelut.data import generate_grid
from ennelut.model import ExactResFlow
from ennelut.utils.evaluation import evaluate_model, stats_dict


class FabricProgressTrackerCallback:
    def __init__(
        self,
        title: str = "progress",
        interval: int = 0,
        grid_size: int = 16,
        metrics: list[str] | None = None,
        quantiles: list[float] | None = None,
    ):
        self.title = title  # used for logging
        self.interval = interval

        self.grid_size = grid_size  # later updated if lut is not callable
        self.metrics = ["delta_e", "psnr"] if metrics is None else metrics
        self.quantiles = [75, 90, 95, 97.5, 99, 99.9] if quantiles is None else quantiles

        # set in setup/fit_start
        self.model = None
        self.logger = None
        self.luts = None
        self.device = None

    def is_active(self, step: int) -> bool:
        return step % self.interval == 0

    def setup(self, context, **kwargs):
        self.context = context

    def on_fit_start(self, **kwargs) -> None:
        self.model: ExactResFlow = self.context.model
        self.logger = self.context.fabric.logger
        self.device = self.context.fabric.device
        self.luts = self.context.data.luts

    def on_batch_end(self, step: int, **kwargs) -> None:
        if not self.is_active(step):
            return
        inputs = generate_grid(self.grid_size, device=self.device, meshgrid=True)
        metric_values = evaluate_model(
            model=self.model,
            inputs=inputs,
            luts=self.luts,
            metrics=self.metrics,
            prefix=f"{self.title}/",
            quantiles=self.quantiles,
        )
        for key, values in metric_values.items():
            self.logger.log_metrics(stats_dict(torch.tensor(values), prefix=f"{key}/", stats=["mean"]), step=step)
