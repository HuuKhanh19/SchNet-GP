"""Eggroll — low-rank (rank-1) Evolution Strategy cho fine-tune encoder.

Mỗi ma trận trọng số 2D M (m×n): perturbation E = a·bᵀ (a∈R^m, b∈R^n ~ N(0,1)),
weight perturbed = M + σE. Antithetic: dùng cặp ±E. Param 1D (bias) KHÔNG perturb.

Ước lượng ES gradient (OpenAI-style, rank utilities):
    fitness F_i = −loss_i ; u_i = centered-rank(F) ∈ [−0.5, 0.5]
    ĝ_M = (1/(Nσ)) Σ_i u_i E_i = (1/(Nσ)) (u·a)ᵀ b        (ascent direction trên F)
=> đặt p.grad = −ĝ rồi Adam.step() (Adam minimize loss = −F).

Tách biệt backend forward: caller cấp `eval_losses(stacked_overrides) -> (N,)` (vmap
hoặc loop) để module không phụ thuộc model. Xem scripts/run_exp3.py.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor


def centered_ranks(x: Tensor) -> Tensor:
    """Rank transform -> [-0.5, 0.5]. x cao (fitness tốt) -> utility cao."""
    n = x.numel()
    ranks = torch.empty_like(x)
    order = torch.argsort(x)
    ranks[order] = torch.arange(n, dtype=x.dtype, device=x.device)
    if n > 1:
        return ranks / (n - 1) - 0.5
    return torch.zeros_like(x)


class EggrollES:
    """Rank-1 antithetic ES trên các ma trận 2D của một module."""

    def __init__(self, module: torch.nn.Module, sigma: float = 0.03,
                 lr: float = 0.01, popsize: int = 256, seed: int = 0,
                 device=None):
        assert popsize % 2 == 0, "popsize phải chẵn (antithetic)"
        self.module = module
        self.sigma = sigma
        self.popsize = popsize
        self.device = device or next(module.parameters()).device
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(seed)

        # Chỉ perturb + optimize tham số 2D (weight matrices + embedding table).
        self.names = [n for n, p in module.named_parameters() if p.dim() == 2]
        self.params = [dict(module.named_parameters())[n] for n in self.names]
        self.opt = torch.optim.Adam(self.params, lr=lr)
        self._factors: Dict[str, Tuple[Tensor, Tensor]] = {}
        self._base: Dict[str, Tensor] = {}

    def sample(self) -> Dict[str, Tensor]:
        """Sinh stacked perturbed params {name: (N, m, n)} (antithetic ±E)."""
        N, half = self.popsize, self.popsize // 2
        stacked = {}
        self._factors, self._base = {}, {}
        for name, p in zip(self.names, self.params):
            m, n = p.shape
            base = p.detach()
            a_h = torch.randn(half, m, generator=self.gen).to(self.device)
            b_h = torch.randn(half, n, generator=self.gen).to(self.device)
            a = torch.cat([a_h, a_h], dim=0)            # (N, m)
            b = torch.cat([b_h, -b_h], dim=0)           # (N, n) -> [E; -E]
            E = a[:, :, None] * b[:, None, :]           # (N, m, n)
            stacked[name] = base[None] + self.sigma * E
            self._factors[name] = (a, b)
            self._base[name] = base
        return stacked

    def update(self, losses: Tensor) -> None:
        """Từ losses (N,) -> ES gradient -> Adam step. Gọi sau sample()."""
        losses = torch.as_tensor(losses, dtype=torch.float32, device=self.device)
        u = centered_ranks(-losses)                     # fitness = -loss
        self.opt.zero_grad(set_to_none=True)
        for name, p in zip(self.names, self.params):
            a, b = self._factors[name]                  # (N,m),(N,n)
            # ĝ = (1/(Nσ)) (u·a)ᵀ b
            ga = (u[:, None] * a).T @ b / (self.popsize * self.sigma)  # (m,n)
            p.grad = -ga.to(p.dtype)                     # ascent trên fitness
        self.opt.step()

    def state(self) -> Dict[str, Tensor]:
        return {n: p.detach().clone() for n, p in zip(self.names, self.params)}
