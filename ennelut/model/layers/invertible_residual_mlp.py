from __future__ import annotations

from collections.abc import Iterable

import dypy
import torch

from ennelut.model.layers.induced_norm_linear import InducedNormLinear


class InvertibleResidualMLP(torch.nn.Module):
    def __init__(
        self,
        dims: int,
        conditions: int = 0,
        hidden_features: Iterable[int] | None = None,
        bias: bool = True,
        device: torch.device | None = None,
        # activation configuration
        first_activation: str | None = None,
        first_activation_args: dict | None = None,
        first_bias: bool = False,
        activation: str | None = "ennelut.model.layers.LipSwish",
        activation_args: dict | None = None,
        # normalization configuration
        coefficient: float = 0.97,  # 0 for no normalization
        # invertibility configuration
        num_iterations: int = 1000,
        absolute_tolerance: float = 1e-5,
        relative_tolerance: float = 1e-5,
        # initialization configuration
        init_zeros: int = 0,
        # residual configuration
        residual: bool = True,
    ):
        super().__init__()
        self.activation = (
            dypy.functions.call_with_dynamic_args(dypy.eval(activation), **(activation_args or {}), device=device)
            if activation
            else None
        )
        self.first_activation = None
        if first_activation is not None:
            self.first_activation = dypy.functions.call_with_dynamic_args(
                dypy.eval(first_activation), **(first_activation_args or {}), device=device
            )
        self.residual = residual
        self.conditions = conditions
        layers = [dims, *(hidden_features or []), dims]

        # conditions are a set of biases (counting to the number of conditions) that are added to the output
        self.embedding_weights = (
            torch.nn.Parameter(torch.randn(conditions, layers[1], device=device)) if conditions else None
        )

        self.num_layers = len(layers) - 1
        self.coefficient = coefficient

        for i in range(self.num_layers):
            linear = InducedNormLinear(
                layers[i],
                layers[i + 1],
                bias=first_bias if not i and self.conditions else bias,
                device=device,
                coefficient=coefficient,
                init_zeros=init_zeros,
            )
            setattr(self, f"linear_{i}", linear)

        self.num_iterations = num_iterations
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance

    def largest_singular_values(self, recompute: bool = False):
        return torch.stack(
            [getattr(self, f"linear_{i}").largest_singular_value(recompute=recompute) for i in range(self.num_layers)]
        )

    def forward(
        self,
        inputs,
        recompute_normalization: bool = True,
        residual: bool | None = True,
        conditions: torch.LongTensor | None = None,
    ):
        residual = residual if residual is not None else self.residual
        # assert self.conditions == 0 or conditions is not None, "conditions must be provided"
        if self.conditions and conditions is not None:
            conditions = conditions @ self.embedding_weights
        x = inputs
        for i in range(self.num_layers):
            x = getattr(self, f"linear_{i}")(x, recompute_normalization=recompute_normalization)
            if i == 0 and conditions is not None:
                x = x + conditions
            if i == 0 and self.first_activation is not None:
                x = self.first_activation(x)
            elif self.activation is not None and i != self.num_layers - 1:
                x = self.activation(x)
        x = (inputs + x) if residual else x
        return x

    def inverse(
        self,
        inputs,
        recompute_normalization: bool = True,
        num_iterations: int | None = None,
        absolute_tolerance: float | None = None,
        relative_tolerance: float | None = None,
        conditions: torch.LongTensor | None = None,
    ):
        x, x_prev = (
            inputs
            - self(inputs, residual=False, recompute_normalization=recompute_normalization, conditions=conditions),
            inputs,
        )
        atol = absolute_tolerance if absolute_tolerance is not None else self.absolute_tolerance
        rtol = relative_tolerance if relative_tolerance is not None else self.relative_tolerance
        num_iterations = num_iterations if num_iterations is not None else self.num_iterations
        i = 0
        tol = atol + inputs.abs() * rtol
        while not torch.all((x - x_prev) ** 2 / tol < 1) and i < num_iterations:
            x, x_prev = (
                inputs
                - self(x, residual=False, recompute_normalization=recompute_normalization, conditions=conditions),
                x,
            )
            i += 1
        return x

    def extra_repr(self):
        description = [f"residual={self.residual}"]
        description += [f"activation={self.activation}"]
        description += [f"coefficient={self.coefficient}"] if self.coefficient else []
        description += [f"conditions={self.conditions}"] if self.conditions else []
        return ",\n".join(description)
