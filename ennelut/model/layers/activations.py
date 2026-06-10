from __future__ import annotations

import torch


class LipSwish(torch.nn.Module):
    def __init__(self, device: torch.device | None = None):
        super().__init__()
        self.beta = torch.nn.Parameter(
            torch.nn.functional.softplus(torch.tensor([0.5], device=device)), requires_grad=False
        )

    def forward(self, x):
        return (x * torch.sigmoid_(x * self.beta)).div_(1.1)
