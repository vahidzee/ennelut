from __future__ import annotations

from normflows.flows import ActNorm


class ActNorm(ActNorm):
    def __init__(self, shape, **kwargs):
        super().__init__(shape, **kwargs)
