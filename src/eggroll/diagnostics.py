"""§10 — Diagnostics. Stage D: canary drift + adapter norm. (Mở rộng đầy đủ ở Stage E.)

Canary (P2 bắt buộc): linear-probe trên e_pooled mỗi step — phải giữ ~ floor (~0.99). Leo
= adapter đang phá encoder -> giảm σ_adapter / r.
"""

from typing import Dict

import torch
from torch import Tensor

from .readout import linear_probe


def canary_probe(e_train: Tensor, y_train: Tensor, e_eval: Tensor, y_eval: Tensor,
                 lam: float, task_type: str = "regression") -> float:
    """Linear-probe cost trên e_pooled hiện tại (đo chất lượng encoder, KHÔNG qua head)."""
    c, _ = linear_probe(e_train, y_train, e_eval, y_eval, lam, task_type)
    return c


def adapter_norm(A: Dict[str, Tensor], B: Dict[str, Tensor], r: int, alpha: float) -> float:
    """‖Δ‖ tổng (Frobenius) của các adapter (α/r)·B·A — đo độ lệch khỏi base."""
    scale = alpha / r
    total = 0.0
    for wname in A:
        delta = scale * (B[wname] @ A[wname])
        total += float((delta ** 2).sum())
    return total ** 0.5


def ridge_cond(c_std_train: Tensor, lam: float) -> float:
    """Số điều kiện của (CᵀC + λI) — health của ridge T2 (counts đã chuẩn hoá)."""
    d = c_std_train.shape[1]
    A = c_std_train.t() @ c_std_train + lam * torch.eye(
        d, device=c_std_train.device, dtype=c_std_train.dtype)
    return float(torch.linalg.cond(A).item())


def head_drift(W: Tensor, W0: Tensor) -> float:
    """‖ΔW‖/‖W0‖ — head đã đi xa warm-start bao nhiêu (tương đối)."""
    return float((W - W0).norm() / W0.norm().clamp_min(1e-12))


def floor_flag(best_cost: float, y_val: Tensor, task_type: str = "regression") -> bool:
    """True nếu model tệ hơn baseline tầm thường (báo động).

    regression: cost(RMSE) > std(val_y) (tệ hơn dự đoán hằng số).
    classification: cost(1−AUC) > 0.5 (AUC < 0.5, tệ hơn ngẫu nhiên).
    """
    if task_type == "classification":
        return best_cost > 0.5
    return best_cost > float(y_val.std())
