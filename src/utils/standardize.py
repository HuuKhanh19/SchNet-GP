"""
Per-feature standardizer (z-score) for CONAN-SchNet Step 2.

Used for the 128-d conformer embeddings: GP/SR trees operate on raw feature
values, so feeding standardized inputs (~N(0,1)) keeps tree constants and
operations well-scaled. Fit on TRAIN only; apply to valid/test.
"""

import torch
from torch import Tensor


class Standardizer:
    def __init__(self, eps: float = 1e-8):
        self.mean_: Tensor | None = None
        self.std_: Tensor | None = None
        self.eps = eps

    def fit(self, x: Tensor) -> "Standardizer":
        self.mean_ = x.mean(dim=0, keepdim=True)
        self.std_ = x.std(dim=0, keepdim=True).clamp(min=self.eps)
        return self

    def transform(self, x: Tensor) -> Tensor:
        assert self.mean_ is not None, "Standardizer not fitted"
        return (x - self.mean_.to(x.device)) / self.std_.to(x.device)

    def fit_transform(self, x: Tensor) -> Tensor:
        return self.fit(x).transform(x)

    def inverse(self, x: Tensor) -> Tensor:
        assert self.mean_ is not None, "Standardizer not fitted"
        return x * self.std_.to(x.device) + self.mean_.to(x.device)

    def state_dict(self) -> dict:
        return {"mean_": self.mean_, "std_": self.std_, "eps": self.eps}

    def load_state_dict(self, sd: dict) -> "Standardizer":
        self.mean_ = sd["mean_"]
        self.std_ = sd["std_"]
        self.eps = sd.get("eps", 1e-8)
        return self