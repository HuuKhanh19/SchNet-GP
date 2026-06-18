"""§10 — Diagnostics. Stage D: canary drift + adapter norm. (Mở rộng đầy đủ ở Stage E.)

Canary (P2 bắt buộc): linear-probe trên e_pooled mỗi step — phải giữ ~ floor (~0.99). Leo
= adapter đang phá encoder -> giảm σ_adapter / r.
"""

from typing import Dict

import torch
from torch import Tensor

from .readout import linear_probe


def canary_probe(e_train: Tensor, y_train: Tensor, e_eval: Tensor, y_eval: Tensor,
                 lam: float) -> float:
    """Linear-probe RMSE trên e_pooled hiện tại (đo chất lượng encoder, KHÔNG qua head)."""
    rmse, _ = linear_probe(e_train, y_train, e_eval, y_eval, lam)
    return rmse


def adapter_norm(A: Dict[str, Tensor], B: Dict[str, Tensor], r: int, alpha: float) -> float:
    """‖Δ‖ tổng (Frobenius) của các adapter (α/r)·B·A — đo độ lệch khỏi base."""
    scale = alpha / r
    total = 0.0
    for wname in A:
        delta = scale * (B[wname] @ A[wname])
        total += float((delta ** 2).sum())
    return total ** 0.5
