"""§6 — Readout closed-form (ridge). Stage A: chỉ linear-probe (sàn tầng-1).

Ridge giải dạng đóng trên torch (chạy được trên GPU, refit nhanh mỗi member ES).
Stage B sẽ mở rộng thành delta 2 tầng (T1 e_pooled, T2 counts) + toggle counts-only.
"""

from typing import Tuple

import torch
from torch import Tensor


def ridge_fit(X: Tensor, y: Tensor, lam: float) -> Tensor:
    """Giải ridge closed-form (có cột bias).

    X: (n, d), y: (n,) hoặc (n, t). Trả coef (d+1,) hoặc (d+1, t) với hàng cuối là bias.
    coef = (X̃ᵀX̃ + λI)⁻¹ X̃ᵀy, X̃ = [X | 1]. Không phạt cột bias (đặt λ=0 cho hàng cuối).
    """
    n, d = X.shape
    ones = torch.ones(n, 1, device=X.device, dtype=X.dtype)
    Xb = torch.cat([X, ones], dim=1)                 # (n, d+1)
    A = Xb.T @ Xb                                     # (d+1, d+1)
    reg = lam * torch.eye(d + 1, device=X.device, dtype=X.dtype)
    reg[d, d] = 0.0                                   # không phạt bias
    A = A + reg
    b = Xb.T @ y
    coef = torch.linalg.solve(A, b)
    return coef


def ridge_predict(X: Tensor, coef: Tensor) -> Tensor:
    """Dự đoán với coef từ ridge_fit (có bias)."""
    n = X.shape[0]
    ones = torch.ones(n, 1, device=X.device, dtype=X.dtype)
    Xb = torch.cat([X, ones], dim=1)
    return Xb @ coef


def rmse(pred: Tensor, target: Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2)).item())


def linear_probe(
    e_train: Tensor, y_train: Tensor,
    e_eval: Tensor, y_eval: Tensor,
    lam: float = 1.0,
) -> Tuple[float, Tensor]:
    """Ridge linear-probe trên embedding -> RMSE (đơn vị gốc, vd. log-S).

    Standardize y theo train (mean/std), fit ridge e->y_std, predict, un-standardize.
    Trả (rmse_eval, coef). Dùng cho gate Stage A (e_pooled phải đạt ~0.99 trên ESOL).
    """
    y_mean = y_train.mean()
    y_std = y_train.std().clamp_min(1e-6)
    yt = (y_train - y_mean) / y_std

    coef = ridge_fit(e_train, yt, lam)
    pred_std = ridge_predict(e_eval, coef)
    pred = pred_std * y_std + y_mean
    return rmse(pred, y_eval), coef
