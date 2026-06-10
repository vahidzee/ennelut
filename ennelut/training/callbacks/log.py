from __future__ import annotations

import torch


class FabricLogCallback:
    def __init__(self, interval: int = 16):
        self.interval = interval

    def setup(self, context, **kwargs):
        self.context = context

    def on_batch_end(self, step: int = None, results: dict | None = None, **kwargs) -> None:
        context = self.context
        logger = context.fabric.logger
        step = context.step if step is None else step

        if not self.interval or step % self.interval != 0:
            return
        if results is not None:
            logger.log_metrics(
                {item: value.mean() if isinstance(value, torch.Tensor) else value for item, value in results.items()},
                step=step,
            )

        for idx, optim in enumerate(context.optimizers):
            # logging learning rate
            logger.log_metrics({f"optim/lr/{idx}": optim.param_groups[0]["lr"]}, step=step)

        for idx, is_active in enumerate(context.optimizers_is_active):
            # logging learning rate
            logger.log_metrics({f"optim/is_active/{idx}": int(is_active)}, step=step)
