from __future__ import annotations

import gc

from ennelut.training.config import Config
from ennelut.training.sweep import Sweep


def run_experiment(sweep: Sweep | None = None, config: dict | None = None):
    ctx = Config.initialize(sweep=sweep, config=config)
    ctx.logger.log_hyperparams(ctx.config.hyperparameters)

    # initiate the fabric and manage devices
    ctx.config.initiate_fabric(ctx)

    if "setup" in ctx.config.stages:
        ctx.fabric.call("setup", context=ctx)

    train = "train" in ctx.config.stages

    if train:
        ctx.fabric.call("on_fit_start", context=ctx)
        ctx.data.reset_iterator()
        ctx.step = 0

    ctx.config.load_checkpoint(ctx)
    while train:
        batch = ctx.data.next()
        for idx, optim in enumerate(ctx.optimizers):
            if ctx.optimizers_frequency and not (ctx.step % ctx.optimizers_frequency):
                ctx.optimizers_is_active[idx] = not ctx.optimizers_is_active[idx]
            optim.zero_grad()
        results = ctx.objective(model=ctx.model, batch=batch)
        loss = results["loss"].sum(-1).mean()

        ctx.fabric.call("on_batch_end", step=ctx.step, results=results)

        ctx.fabric.backward(loss)
        # step optimizers
        for idx, optim in enumerate(ctx.optimizers):
            if ctx.optimizers_is_active[idx]:
                optim.step()

        # step lr schedulers
        if ctx.lr_scheduler is not None and ctx.optimizers_is_active[0]:
            ctx.lr_scheduler.step()
        if ctx.embeddings_lr_scheduler is not None and ctx.optimizers_is_active[1]:
            ctx.embeddings_lr_scheduler.step()

        # update progress bar
        ctx.step += 1
        if ctx.progress_bar is not None:
            ctx.progress_bar.update(1)
            ctx.progress_bar.set_postfix({"loss": loss.item()})

        if ctx.step == ctx.max_steps:
            break

    if "test" in ctx.config.stages:
        ctx.fabric.call("on_fit_end", context=ctx)

    del ctx  # clear memory
    gc.collect()
