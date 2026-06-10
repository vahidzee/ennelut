from __future__ import annotations

import functools

import dypy
import torch
from torch.func import jacrev, vmap

from ennelut.model.layers import InvertibleResidualMLP


class ExactResFlow(torch.nn.Module):
    # exact resflow is a residual flow (https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/flows/residual.py)
    # with exact computation of the jacobian
    def __init__(
        self,
        dims: int = 3,
        conditions: int = 0,
        conditions_embedding: int = 0,  # 0 for no embedding
        conditions_embedding_tanh: bool = False,
        conditions_depth: int = -1,  # controls which layers should have the conditionining vectors
        share_flow_conditions: bool = False,
        num_flows: int = 2,  # the number of residual predictors
        share_flows: int = 0,  # how often to share the parameters of the residual predictors
        hidden_units: int | list[int] = 6,
        num_hidden_layers: int = 2,  # is ignored if hidden_units is a list
        bias: bool = True,
        device: torch.device | None = None,
        # post-processing configuration
        normalized: bool = True,  # whether inputs are in [-1, 1] rather than [0, 1]
        tanh_normalization: bool = True,
        # activation configuration
        activation: str | None = "ennelut.model.layers.LipSwish",
        activation_args: dict | None = None,
        first_bias: bool = False,  # whether to have a bias for the first linear layer
        first_activation: str | None = None,  # override the first activation
        first_activation_args: dict | None = None,
        # normalization configuration
        coefficient: float = 0.97,  # 0 for not bijective
        # invertibility configuration
        num_iterations: int = 5000,
        absolute_tolerance: float = 1e-6,
        relative_tolerance: float = 1e-6,
        # initialization configuration
        init_zeros: int = 0,  # after initialization divide the values by this number (0 for no division)
        # actnorm configs
        actnorm: str | None = "ennelut.model.layers.ActNorm",
        actnorm_args: dict | None = None,
        # residual configuration
        residual: bool = True,
        # external configs
        reversed: bool = False,  # swapping inputs and outputs to learn the inverse LUT
    ):
        super().__init__()
        self.num_flows = num_flows
        self.coefficient = coefficient
        self.conditions = conditions
        self.conditions_depth = conditions_depth if conditions_depth >= 0 else (num_flows + conditions_depth)
        self.normalized = normalized
        self.tanh_normalization = tanh_normalization and normalized  # if not normalized, tanh makes no sense
        layer_conditions = conditions_embedding if conditions and conditions_embedding else conditions
        self.reversed = reversed

        # share flows
        self.repeats = 0
        assert share_flows <= 0 or num_flows % share_flows == 0, "num_flows must be divisible by share_flows"
        if share_flows > 0:
            self.repeats = share_flows
            self.num_flows = num_flows // share_flows
        else:
            self.repeats = num_flows if share_flows < 0 else 1
            self.num_flows = 1 if share_flows < 0 else num_flows

        self.num_hidden_layers = num_hidden_layers
        self.hidden_units = [hidden_units] * num_hidden_layers if isinstance(hidden_units, int) else hidden_units
        self.transforms = torch.nn.ModuleDict()
        for i in range(self.num_flows):
            self.transforms[f"residual_{i}"] = InvertibleResidualMLP(
                dims=dims,
                hidden_features=self.hidden_units,
                conditions=layer_conditions if i <= self.conditions_depth and not share_flow_conditions else 0,
                bias=bias,
                device=device,
                # activation configuration
                activation=activation,
                activation_args=activation_args,
                first_bias=first_bias,
                first_activation=first_activation,
                first_activation_args=first_activation_args,
                # normalization configuration
                coefficient=coefficient,
                # invertibility configuration
                num_iterations=num_iterations,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                # initialization configuration
                init_zeros=init_zeros,
                # residual configuration
                residual=residual,
            )
            if actnorm is not None:
                act_norm = dypy.functions.call_with_dynamic_args(
                    dypy.eval(actnorm),
                    shape=(dims,),
                    normalized_shape=(dims,),
                    num_features=dims,
                    **(actnorm_args or {}),
                )
                self.transforms[f"actnorm_{i}"] = act_norm

        self.embeddings_dim = conditions_embedding

        self.conditions_embedding = (
            torch.nn.Parameter(torch.randn(conditions, conditions_embedding))
            if conditions and conditions_embedding
            else None
        )
        self.conditions_embedding_tanh = conditions_embedding_tanh

        self.flow_conditions_embedding = None
        if share_flow_conditions:
            self.flow_conditions_embedding = torch.nn.Parameter(torch.randn(layer_conditions, self.hidden_units[0]))

    @property
    def embeddings(self):
        if self.conditions_embedding is not None and self.conditions_embedding_tanh:
            return torch.tanh(self.conditions_embedding / 10) * 10
        return self.conditions_embedding

    @property
    def is_bijective(self):
        return self.coefficient != 0

    def process_conditions(
        self,
        batch_size: int,
        device: torch.device | None = None,
        conditions: torch.LongTensor | int | None = None,
        onehot: bool = True,
        embed: bool = True,
    ):
        if not self.conditions:
            return None
        assert conditions is not None, "condition must be provided"
        conditions = (
            conditions
            if isinstance(conditions, torch.Tensor)
            else torch.full(fill_value=conditions, size=(batch_size,), dtype=torch.long, device=device)
        )
        if onehot:
            # turn conditions into one-hot
            conditions = conditions.reshape(-1)
            conditions = torch.nn.functional.one_hot(conditions, num_classes=self.conditions).float()
        if embed and self.conditions_embedding is not None:
            conditions = torch.matmul(conditions, self.embeddings)
        conditions = conditions.unsqueeze(0) if conditions is not None and len(conditions.shape) == 1 else conditions
        conditions = (
            conditions @ self.flow_conditions_embedding if self.flow_conditions_embedding is not None else conditions
        )
        return conditions

    def forward(
        self,
        inputs,
        conditions: torch.LongTensor | int | torch.Tensor | None = None,
        recompute_normalization: bool = True,
        residual: bool | None = None,
        process_conditions: bool = True,
    ):
        x = inputs.reshape(-1, inputs.shape[-1])
        x = torch.arctanh(x) if self.tanh_normalization else x
        # process conditions
        if process_conditions:
            conditions = self.process_conditions(batch_size=x.shape[0], device=x.device, conditions=conditions)

        # forward
        for i in range(self.num_flows):
            for _ in range(self.repeats):
                x = getattr(self.transforms, f"residual_{i}")(
                    x,
                    recompute_normalization=recompute_normalization,
                    residual=residual,
                    conditions=conditions if i <= self.conditions_depth else None,
                )
                if hasattr(self.transforms, f"actnorm_{i}"):
                    actnorm = getattr(self.transforms, f"actnorm_{i}")(x)
                    x = actnorm if not isinstance(actnorm, tuple) else actnorm[0]  # drop log_det

        # post-processing
        if self.tanh_normalization:
            x = torch.tanh(x)
        else:
            x = x.clip(-1, 1) if self.normalized else x.clip(0, 1)
        return x.reshape(*inputs.shape[:-1], *x.shape[1:])

    def inverse(
        self,
        outputs,
        conditions: torch.LongTensor | int | None = None,
        recompute_normalization: bool = True,
    ):
        x = outputs
        x = x if not self.tanh_normalization else torch.arctanh(x)

        conditions = self.process_conditions(batch_size=x.shape[0], device=x.device, conditions=conditions)
        conditions = conditions.unsqueeze(0) if conditions is not None and len(conditions.shape) == 1 else conditions
        if conditions is not None and self.conditions_embedding is not None:
            conditions = torch.matmul(conditions, self.conditions_embedding)

        for i in range(self.num_flows - 1, -1, -1):
            for _ in range(self.repeats):
                if hasattr(self.transforms, f"actnorm_{i}"):
                    actnorm = getattr(self.transforms, f"actnorm_{i}").inverse(x)
                    x = actnorm if not isinstance(actnorm, tuple) else actnorm[0]  # drop log_det
                x = getattr(self.transforms, f"residual_{i}").inverse(
                    x,
                    recompute_normalization=recompute_normalization,
                    conditions=conditions if i <= self.conditions_depth else None,
                )
        x = x if not self.tanh_normalization else torch.tanh(x)
        return x.clip(0, 1) if not self.normalized else x.clip(-1, 1)

    def largest_singular_values(self, recompute: bool = False):
        return torch.stack(
            [
                getattr(self.transforms, f"residual_{i}").largest_singular_values(recompute=recompute)
                for i in range(self.num_flows)
            ]
        )

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @functools.cached_property
    def jacobian_function(self):
        return vmap(jacrev(functools.partial(self.forward, process_conditions=False)))

    def jacobian(
        self,
        inputs,
        chunk_size: int | None = None,
        conditions: int | torch.LongTensor | None = None,
        **kwargs,
    ):
        original_shape = inputs.shape
        inputs = inputs.reshape(-1, original_shape[-1])
        conditions = self.process_conditions(batch_size=inputs.shape[0], device=inputs.device, conditions=conditions)

        func = (
            self.jacobian_function
            if chunk_size is None and not kwargs
            else vmap(
                jacrev(functools.partial(self.forward, process_conditions=False)),
                chunk_size=chunk_size,
                **kwargs,
            )
        )
        results = func(inputs, conditions).squeeze(1) if conditions is not None else func(inputs).squeeze(1)
        return results.reshape(*original_shape[:-1], *results.shape[1:])

    @property
    def device(self):
        return next(self.parameters()).device

    def extra_repr(self) -> str:
        desc = []
        desc += ["normalized=True"] if self.normalized else []
        desc += ["tanh-normalization=True"] if self.tanh_normalization else []
        desc += [f"conditions={self.conditions}"] if self.conditions else []
        desc += [f"conditions_embedding={self.embeddings_dim}"] if self.conditions_embedding is not None else []
        desc += (
            [f"shared-flow-conditions={self.flow_conditions_embedding is not None}"]
            if self.flow_conditions_embedding is not None
            else []
        )
        desc += [f"reversed={self.reversed}"]
        return ", ".join(desc)
