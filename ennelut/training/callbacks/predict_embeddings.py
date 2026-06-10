from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import torch
import tqdm
import wandb

from ennelut.data import LUT
from ennelut.utils import metric_stats, numtensor, psnr, stats_dict


class FabricPredictEmbeddingsCallback:
    def __init__(
        self,
        only_remaining: bool = True,
        grid_size: int = 4,
        metrics: list | None = None,
        epochs: int = 2048,
        num_initializations: int = 512,
        init_std: float = 0.005,
        init_mean: float | None = None,
        # sgld
        noise_std: float | None = None,
        grad_clip: float | None = None,
        step_size: float | None = 0.01,
        half_every: int | None = 512,
        # perturbation
        perturb_std: float | None = None,
        perturb_threshold: float = 0.01,
        perturb_patience: int = 32,
        # logging
        title: str = "embeddings-prediction",
        progress_bar: bool = False,
    ):
        self.grid_size = grid_size
        self.metrics = ["delta_e_lab", "l1"] if metrics is None else metrics
        self.epochs = epochs
        self.num_initializations = num_initializations
        self.init_std = init_std
        self.init_mean = init_mean
        self.noise_std = noise_std
        self.grad_clip = grad_clip
        self.step_size = step_size
        self.half_every = half_every
        self.perturb_std = perturb_std
        self.perturb_threshold = perturb_threshold
        self.perturb_patience = perturb_patience
        self.save_root = title
        self.title = title
        self.only_remaining = only_remaining
        self.progress_bar = progress_bar

    def setup(self, context, **kwargs):
        self.context = context
        # create the results directory
        self.save_root = os.path.join(wandb.run.dir, self.save_root)
        os.makedirs(self.save_root, exist_ok=True)

        if not self.only_remaining:
            self.luts = context.data.luts
            return

        # find the rest of the luts
        luts_root = os.path.dirname(context.data.luts[0].path)

        all_ids = [
            int(os.path.splitext(os.path.basename(path))[0])
            for path in os.listdir(luts_root)
            if path.lower().endswith(".cube")
        ]

        training_ids = [int(os.path.splitext(os.path.basename(lut.path))[0]) for lut in context.data.luts]
        remaining_ids = sorted([id for id in all_ids if id not in training_ids])
        self.luts = [LUT(path=os.path.join(luts_root, f"{id}.cube")) for id in remaining_ids]

    def on_fit_end(self, **kwargs) -> None:
        context = self.context
        device = context.fabric.device
        model = context.model
        # freeze all model parameters
        for param in model.parameters():
            param.requires_grad = False

        model_embeddings = model.conditions_embedding

        init_std = abs(self.init_std) if self.init_std is not None else None
        init_mean = abs(self.init_mean) if self.init_mean is not None else 0
        noise_std = abs(self.noise_std) if self.noise_std is not None else None
        perturb_std = abs(self.perturb_std) if self.perturb_std is not None else None

        if model_embeddings is not None:
            init_std = abs(self.init_std) * model_embeddings.std(0) if self.init_std < 0 else abs(self.init_std)
            init_mean = model_embeddings.mean(0) if (self.init_mean is not None and self.init_mean < 0) else init_mean

            if self.noise_std is not None:
                noise_std = abs(self.noise_std) * model_embeddings.std(0) if self.noise_std < 0 else abs(self.noise_std)
            if self.perturb_std is not None:
                perturb_std = (
                    abs(self.perturb_std) * model_embeddings.std(0) if self.perturb_std < 0 else abs(self.perturb_std)
                )
        embedding_dimension = (
            model.conditions_embedding.shape[-1] if model.conditions_embedding is not None else model.conditions
        )
        assert embedding_dimension > 0, "model has no conditions embedding"

        predicted_embeddings = []
        psnrs = []
        metric_values = defaultdict(list)

        for idx, lut in enumerate(self.luts):
            inputs, outputs = lut.transform_hald(grid_size=self.grid_size)
            inputs = torch.from_numpy(inputs).float().to(device).reshape(-1, 3)
            outputs = torch.from_numpy(outputs).float().to(device).reshape(-1, 3)

            embeddings = torch.randn(
                self.num_initializations,
                embedding_dimension,
                device=device,
                requires_grad=True,
                dtype=torch.float32,
            )
            embeddings.data = embeddings.data * init_std + init_mean

            optimizer = torch.optim.Adam([embeddings], lr=self.step_size)
            lr_scheduler = (
                torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.half_every, gamma=0.5)
                if self.half_every is not None
                else None
            )

            losses_checkpoint = None
            losses_counter = torch.zeros(self.num_initializations, device=device, dtype=torch.int32)
            if self.progress_bar:
                pbar = tqdm.tqdm(total=self.epochs, desc=f"embedding lut {idx}", leave=True, position=0)
            else:
                print(f"embedding lut {idx}")
            for step in range(self.epochs):
                if (step + 1) % self.half_every == 0:
                    noise_std = noise_std * 0.5 if noise_std is not None else None
                    perturb_std = perturb_std * 0.5 if perturb_std is not None else None

                # perturb the embeddings
                if noise_std is not None:
                    embeddings.data.add_(torch.randn_like(embeddings) * noise_std)

                # forward pass (compute the energy)
                preds = model(
                    inputs.repeat(self.num_initializations, 1),
                    conditions=embeddings.repeat_interleave(inputs.shape[0], dim=0),
                    process_conditions=False,
                ).reshape(self.num_initializations, -1, 3)
                losses = (outputs - preds).pow(2).sum(-1).mean(-1)
                loss = losses.sum()
                loss.backward()
                # if an embedding is stuck in a local minimum, perturb it
                if perturb_std is not None:
                    if not step:
                        losses_checkpoint = losses.data.clone()
                    else:
                        changed = (losses_checkpoint - losses.data) >= self.perturb_threshold

                        # update counter
                        losses_counter[~changed] += 1
                        losses_counter[changed] = 0

                        # update checkpoint
                        losses_checkpoint[changed] = losses.data[changed]

                        # check if we need to perturb the embeddings
                        passed_patience = losses_counter >= self.perturb_patience
                        if passed_patience.any():
                            # perturb the embeddings
                            embeddings.data[passed_patience] += (
                                torch.randn_like(embeddings[passed_patience]) * perturb_std * (1 - step / self.epochs)
                            )
                            embeddings.grad.data[passed_patience] = (
                                0  # don't update this embedding at this step (except for the momentum)
                            )
                            # update the checkpoint and counter
                            losses_checkpoint[passed_patience] = losses.data[passed_patience]
                            losses_counter[passed_patience] = 0

                # # clip gradients
                if self.grad_clip is not None:
                    # For stabilizing and preventing too high gradients
                    embeddings.grad.data.clamp_(
                        *(self.grad_clip if isinstance(self.grad_clip, tuple) else (-self.grad_clip, self.grad_clip))
                    )

                # update the embeddings
                optimizer.step()

                # update the learning rate
                if lr_scheduler is not None:
                    lr_scheduler.step()
                # update progress bar
                if self.progress_bar:
                    pbar.update(1)
                    pbar.set_postfix({"loss": loss.item()})
                # reset gradients
                optimizer.zero_grad()

            # compute the loss one last time (without gradients)
            with torch.no_grad():
                preds = model(
                    inputs.repeat(self.num_initializations, 1),
                    conditions=embeddings.repeat_interleave(inputs.shape[0], 0),
                    process_conditions=False,
                ).reshape(self.num_initializations, -1, 3)
                losses = (outputs - preds).pow(2).sum(-1).mean(-1)

            best_embedding = embeddings[losses.argmin()]
            outputs = numtensor(outputs, undo_normalize=True, quant=True, tensor=False)
            pred = numtensor(preds[losses.argmin()], undo_normalize=True, quant=True, tensor=False)
            psnr_value = psnr(outputs, pred).item()
            if self.progress_bar:
                pbar.set_postfix({"loss": loss.item(), "best-loss": losses.min().item(), "best-psnr": psnr_value})
            else:
                print("\tloss", loss.item(), "best-loss", losses.min().item(), "best-psnr", psnr_value)
            predicted_embeddings.append(best_embedding.detach().cpu().numpy())
            psnrs.append(psnr_value)

            # compute metrics
            for metric in self.metrics:
                metric_results = metric_stats(
                    pred=pred, target=outputs, metric=metric, prefix=f"{self.title}/{metric}/"
                )
                for key, value in metric_results.items():
                    metric_values[key].append(value)

            # save the results to a file
            np.save(os.path.join(self.save_root, "embeddings.npy"), np.stack(predicted_embeddings, axis=0))
            np.save(os.path.join(self.save_root, "psnrs.npy"), np.array(psnrs))
            if self.progress_bar:
                pbar.close()

        # log the results
        psnrs = np.array(psnrs)
        context.fabric.logger.log_metrics(stats_dict(psnrs, prefix=f"{self.title}/psnr/"), step=context.step)
        for key, values in metric_values.items():
            values = np.array(values)
            context.fabric.logger.log_metrics(stats_dict(values, prefix=f"{key}/"), step=context.step)
