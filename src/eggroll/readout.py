"""§6 — Readout delta 2 tầng, closed-form (ridge). KHÔNG eggroll, refit mỗi member.

T1 (sàn): β=ridge(e_pooled, y_std); base=e_pooled@β; resid=y_std−base. -> sàn = linear-probe.
T2 (head): c=ridge(c_std, resid); pred=base + c_std@c. Head chỉ học dư phi tuyến.
Un-standardize pred -> RMSE (đơn vị gốc). Toggle counts_only bỏ T1.

fit_delta/predict_delta tách rời để dùng lại làm fitness eggroll (Stage C): fit ridge
trên train, eval RMSE trên train cho từng member. Ridge giải dạng đóng torch (GPU, nhanh).
"""

from typing import Dict, Tuple

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


def rmse_tensor(pred: Tensor, target: Tensor) -> Tensor:
    """RMSE dạng 0-dim tensor (không .item() -> tránh sync mỗi member trong vòng ES)."""
    return torch.sqrt(torch.mean((pred - target) ** 2))


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


# =============================================================================
# Delta readout 2 tầng (§6)
# =============================================================================

def fit_delta(
    e_train: Tensor, c_train: Tensor, y_train: Tensor,
    lam: float = 1.0, counts_only: bool = False,
) -> Dict:
    """Fit delta readout trên TRAIN. Trả model dict (stats + coef ridge).

    e_train: e_pooled (n, hidden). c_train: counts (n, H). y_train: (n,) đơn vị gốc.
    """
    y_mean = y_train.mean()
    y_std = y_train.std().clamp_min(1e-6)
    yt = (y_train - y_mean) / y_std

    c_mean = c_train.mean(dim=0)
    c_sd = c_train.std(dim=0).clamp_min(1e-6)
    c_std_train = (c_train - c_mean) / c_sd

    if counts_only:
        beta = None
        base_train = torch.zeros_like(yt)
    else:
        beta = ridge_fit(e_train, yt, lam)
        base_train = ridge_predict(e_train, beta)

    resid = yt - base_train
    coef_c = ridge_fit(c_std_train, resid, lam)

    return {
        "y_mean": y_mean, "y_std": y_std,
        "c_mean": c_mean, "c_sd": c_sd,
        "beta": beta, "coef_c": coef_c,
        "counts_only": counts_only,
    }


def predict_delta(model: Dict, e_eval: Tensor, c_eval: Tensor) -> Tensor:
    """Dự đoán (đơn vị gốc) trên eval set từ model fit_delta."""
    c_std_eval = (c_eval - model["c_mean"]) / model["c_sd"]
    if model["counts_only"]:
        base = torch.zeros(e_eval.shape[0], device=e_eval.device, dtype=e_eval.dtype)
    else:
        base = ridge_predict(e_eval, model["beta"])
    head = ridge_predict(c_std_eval, model["coef_c"])
    pred_std = base + head
    return pred_std * model["y_std"] + model["y_mean"]


def delta_rmse(
    e_train: Tensor, c_train: Tensor, y_train: Tensor,
    e_eval: Tensor, c_eval: Tensor, y_eval: Tensor,
    lam: float = 1.0, counts_only: bool = False,
) -> Tuple[float, Dict]:
    """Tiện ích: fit trên train, trả RMSE trên eval + model."""
    model = fit_delta(e_train, c_train, y_train, lam, counts_only)
    pred = predict_delta(model, e_eval, c_eval)
    return rmse(pred, y_eval), model
