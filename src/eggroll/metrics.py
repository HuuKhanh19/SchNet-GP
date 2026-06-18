"""§11 — Metric/fitness theo task. Classification: fitness = AUC trực tiếp (cửa outperform).

Quy ước thống nhất: `cost` = càng THẤP càng tốt cho cả 2 task -> eggroll minimize, model
selection = min(cost) dùng chung 1 đường code.
  - regression:     cost = RMSE
  - classification: cost = 1 − AUC   (maximize AUC)
Hiển thị: to_metric(cost) -> RMSE (giữ nguyên) hoặc AUC (= 1 − cost).
"""

import torch
from torch import Tensor


def auc_tensor(score: Tensor, y: Tensor) -> Tensor:
    """AUC (Mann-Whitney U) dạng 0-dim tensor. y∈{0,1}, score cao -> dự đoán dương.

    Rank-based, không sync .item() (dùng được trong vòng ES). Ties hiếm với score liên tục
    (ridge) nên xấp xỉ rank đơn giản là đủ cho fitness.
    """
    pos = y > 0.5
    n = y.numel()
    n_pos = pos.sum()
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:          # suy biến -> AUC 0.5
        return torch.full((), 0.5, device=score.device, dtype=score.dtype)
    order = torch.argsort(score)
    ranks = torch.empty(n, device=score.device, dtype=score.dtype)
    ranks[order] = torch.arange(1, n + 1, device=score.device, dtype=score.dtype)
    sum_pos = ranks[pos].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def cost(pred: Tensor, y: Tensor, task_type: str) -> Tensor:
    """Cost (lower=better) dạng tensor cho fitness eggroll."""
    if task_type == "classification":
        return 1.0 - auc_tensor(pred, y)
    return torch.sqrt(torch.mean((pred - y) ** 2))


def cost_float(pred: Tensor, y: Tensor, task_type: str) -> float:
    return float(cost(pred, y, task_type).item())


def to_metric(cost_val: float, task_type: str) -> float:
    """Đổi cost -> metric hiển thị (RMSE hoặc AUC)."""
    return (1.0 - cost_val) if task_type == "classification" else cost_val


def metric_name(task_type: str) -> str:
    return "AUC" if task_type == "classification" else "RMSE"
