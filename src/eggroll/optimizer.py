"""§8 — Eggroll optimizer (evolution strategy, gradient-free).

θ = {2D: head W (+ adapter A_l,B_l ở P2)} ∪ {1D: head b}. Base KHÔNG trong θ. Mỗi step:
  - Sample N member (N/2 cặp antithetic). 2D M: perturbation rank-1 E_i=a_i⊗bf_i,
    a=[a_h;a_h], bf=[bf_h;−bf_h]; M_i=M+σ·E_i. 1D v: ξ=[ξ_h;−ξ_h]; v_i=v+σ·ξ_i.
    Lưu factors (CRN: không sinh lại ở update).
  - Eval mỗi member (full-batch train) -> loss_i = RMSE_train.
  - Shaping u=centered_ranks(−losses)∈[−0.5,0.5] (best->+0.5).
  - Grad: 2D ĝ_M=(1/(Nσ))(diag(u)·a)ᵀbf; 1D ĝ_v=(1/(Nσ))Σu_iξ_i. param.grad=−ĝ (ascent).
  - Adam.step + cosine decay es-lr.

Heaviside (head) không khả vi -> ES black-box trên fitness (kể cả ridge refit per member).
"""

from typing import Callable, Dict, List

import torch
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


def centered_ranks(x: Tensor) -> Tensor:
    """Utility ∈ [−0.5, 0.5]: x lớn hơn -> utility cao hơn (best→+0.5)."""
    n = x.numel()
    if n == 1:
        return torch.zeros_like(x)
    order = x.argsort()
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(n, device=x.device, dtype=x.dtype)
    return ranks / (n - 1) - 0.5


class Eggroll:
    """ES tối ưu tập param 2D (rank-1 antithetic) + 1D (antithetic) qua Adam."""

    def __init__(
        self,
        params2d: Dict[str, Tensor],
        params1d: Dict[str, Tensor],
        sigma2d: Dict[str, float],
        sigma1d: Dict[str, float],
        pop_size: int,
        es_lr: float,
        total_steps: int,
        gen_seed: int = 0,
    ):
        assert pop_size % 2 == 0, "pop_size phải chẵn (antithetic)."
        # leaf params requires_grad để Adam cập nhật; grad set tay (không backward).
        self.p2d = {k: v.clone().detach().requires_grad_(True) for k, v in params2d.items()}
        self.p1d = {k: v.clone().detach().requires_grad_(True) for k, v in params1d.items()}
        self.s2d = dict(sigma2d)
        self.s1d = dict(sigma1d)
        self.N = pop_size
        self.half = pop_size // 2
        dev = next(iter({**self.p2d, **self.p1d}.values())).device
        self.g = torch.Generator(device=dev).manual_seed(gen_seed)

        allp = list(self.p2d.values()) + list(self.p1d.values())
        self.opt = Adam(allp, lr=es_lr)
        self.sched = CosineAnnealingLR(self.opt, T_max=max(total_steps, 1))

    # -- param hiện tại (chưa perturb) để eval val / model selection --
    def current(self) -> Dict[str, Tensor]:
        return {**{k: v.detach() for k, v in self.p2d.items()},
                **{k: v.detach() for k, v in self.p1d.items()}}

    def _sample(self):
        fac2d = {}
        for name, M in self.p2d.items():
            m, n = M.shape
            a_h = torch.randn(self.half, m, generator=self.g, device=M.device, dtype=M.dtype)
            bf_h = torch.randn(self.half, n, generator=self.g, device=M.device, dtype=M.dtype)
            a = torch.cat([a_h, a_h], dim=0)        # (N, m)
            bf = torch.cat([bf_h, -bf_h], dim=0)    # (N, n)  antithetic
            fac2d[name] = (a, bf)
        fac1d = {}
        for name, v in self.p1d.items():
            n = v.shape[0]
            xi_h = torch.randn(self.half, n, generator=self.g, device=v.device, dtype=v.dtype)
            xi = torch.cat([xi_h, -xi_h], dim=0)    # (N, n)
            fac1d[name] = xi
        return fac2d, fac1d

    def _member(self, i: int, fac2d, fac1d) -> Dict[str, Tensor]:
        theta = {}
        for name, M in self.p2d.items():
            a, bf = fac2d[name]
            theta[name] = M.detach() + self.s2d[name] * torch.outer(a[i], bf[i])
        for name, v in self.p1d.items():
            theta[name] = v.detach() + self.s1d[name] * fac1d[name][i]
        return theta

    @torch.no_grad()
    def step(self, eval_fn: Callable[[Dict[str, Tensor]], Tensor]) -> Tensor:
        """Một bước ES. eval_fn(theta)->0-dim loss (RMSE_train). Trả losses (N,)."""
        fac2d, fac1d = self._sample()
        dev = next(iter({**self.p2d, **self.p1d}.values())).device
        losses = torch.empty(self.N, device=dev)
        for i in range(self.N):
            losses[i] = eval_fn(self._member(i, fac2d, fac1d))

        u = centered_ranks(-losses)                  # (N,)
        for name, M in self.p2d.items():
            a, bf = fac2d[name]
            g = (a * u.unsqueeze(1)).t() @ bf / (self.N * self.s2d[name])   # (m, n)
            M.grad = -g
        for name, v in self.p1d.items():
            xi = fac1d[name]
            g = (u.unsqueeze(1) * xi).sum(dim=0) / (self.N * self.s1d[name])  # (n,)
            v.grad = -g
        self.opt.step()
        self.sched.step()
        return losses

    @property
    def lr(self) -> float:
        return self.opt.param_groups[0]["lr"]
