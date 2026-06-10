from __future__ import annotations

import os

import lightning
import wandb


class FabricCheckpointingCallback:
    def __init__(self, interval: int = 1, save_last: bool = True, dir: str = "checkpoints"):
        self.interval = interval
        self.root = dir
        self.save_last = save_last

    def setup(self, context, **kwargs):
        self.context = context

        # create checkpoint directory
        self.root = os.path.join(wandb.run.dir, self.root)
        os.makedirs(self.root, exist_ok=True)

    def get_state(self):
        context = self.context
        state = {
            "step": context.step,
            "model": context.model,
            "optimizer": context.optimizer,
        }
        if context.lr_scheduler is not None:
            state["lr_scheduler"] = context.lr_scheduler
        if context.embeddings_optimizer is not None:
            state["embeddings_optimizer"] = context.embeddings_optimizer
        if context.embeddings_lr_scheduler is not None:
            state["embeddings_lr_scheduler"] = context.embeddings_lr_scheduler
        return state

    def on_batch_end(self, step: int = None, **kwargs) -> None:
        context = self.context
        step = context.step if step is None else step
        fabric: lightning.Fabric = context.fabric
        if step % self.interval == 0:
            fabric.save(os.path.join(self.root, f"{step}.ckpt"), state=self.get_state())
        if self.save_last:
            fabric.save(os.path.join(self.root, "last.ckpt"), state=self.get_state())
