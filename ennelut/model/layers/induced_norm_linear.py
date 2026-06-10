from __future__ import annotations

import math

import torch


class InducedNormLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        # normalization configuration
        coefficient: float = 0.97,  # 0 for no normalization
        # initialization configuration
        init_zeros: int = 0,
    ):
        self.init_zeros = init_zeros  # -1 for default initialization
        self.coefficient = coefficient

        super().__init__(in_features=in_features, out_features=out_features, bias=bias, device=device)

        # cache and fast inference variables
        self.cached_largest_singular_value = None

    def reset_parameters(self):
        if self.init_zeros < 0:
            return super().reset_parameters()
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.init_zeros:
            # normalize cannot handle zero weight in some cases.
            self.weight.data.div_(self.init_zeros)
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
            if self.init_zeros:
                self.bias.data.div_(self.init_zeros)

    def largest_singular_value(self, recompute: bool = False):
        if self.cached_largest_singular_value is not None and not recompute:
            return self.cached_largest_singular_value
        results = torch.linalg.matrix_norm(self.weight, ord=2)
        self.cached_largest_singular_value = results
        return results

    def forward(self, x, recompute_normalization: bool = True):
        factor = (
            1.0
            if not self.coefficient
            else self.largest_singular_value(recompute=recompute_normalization) / self.coefficient
        )
        return torch.nn.functional.linear(x, self.weight / factor, self.bias)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
            f"coefficient={self.coefficient}, "
            f"init_zeros={self.init_zeros}"
            f")"
        )
