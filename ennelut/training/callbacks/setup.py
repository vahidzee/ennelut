from __future__ import annotations

import torch

from ennelut.model.layers import ActNorm


class FabricSetupCallback:
    def setup(self, context, **kwargs):
        self.context = context

        self._has_actnorm = False
        # loop through all the layers and if it has ActNorm, to initialize it with a batch
        for layer in context.model.modules():
            if isinstance(layer, ActNorm):
                self._has_actnorm = True
                break

    def on_fit_start(self, **kwargs) -> None:
        context = self.context
        logger = context.fabric.logger
        model = context.model
        if self._has_actnorm:
            # initialize the actnorm layer with a batch
            with torch.no_grad():
                context.data.reset_iterator()
                batch = context.data.next()
                source, _, lut_idx = batch
                _ = model(source, conditions=lut_idx)

        # compute parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        embedding_parameters = sum(
            p.numel() for name, p in model.named_parameters() if p.requires_grad and "embedding" in name
        )
        embedding_vectors = 0 if model.conditions_embedding is None else model.conditions_embedding.numel()
        lut_params = sum(lut.grid_size**3 for lut in context.data.luts)
        logger.log_metrics(
            {
                # log number of parameters
                "num_params/total": num_params,
                "num_params/network": num_params - embedding_parameters,
                "num_params/embedding_params": embedding_parameters - embedding_vectors,
                "num_params/embedding_vectors": embedding_vectors,
                "num_params/luts": lut_params,
                # log model details for convenience
                "network/conditions": model.conditions,
                "network/embedding": model.embeddings_dim,
                "network/num_flows": model.num_flows,
                "network/repeats": model.repeats,
                "network/num_hidden_layers": model.num_hidden_layers,
                "network/hidden_units": str(model.hidden_units),
            },
            step=0,
        )
        print(model)
