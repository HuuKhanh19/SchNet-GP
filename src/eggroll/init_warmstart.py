"""§7 — Hyperplane warm-start cho head (từ base embedding, trước P1).

w_1 = β (hướng linear-probe trên e_pooled); w_2..w_H = top-(H−1) PCA của per-atom h
(toàn atom train) + nhiễu nhỏ; b_h theo percentile của w_h·h_i để ~50% atom fire/rule.
-> eggroll bắt đầu gần signal thay vì từ rule ngẫu nhiên.

Lưu ý: dùng CÙNG không gian h mà head sẽ ăn (đã standardize per-dim ở curriculum) cho cả
β, PCA và bias -> nhất quán.
"""

from typing import Tuple

import torch
from torch import Tensor

from .readout import ridge_fit


def init_head_warmstart(
    h_train: Tensor, e_train: Tensor, y_train: Tensor, H: int,
    lam: float = 1.0, fire_rate: float = 0.5, noise: float = 0.01, seed: int = 0,
) -> Tuple[Tensor, Tensor]:
    """Trả (W (H,hidden), b (H,)) khởi tạo warm-start. h_train/e_train ở không gian đã chuẩn hoá."""
    device, dtype = h_train.device, h_train.dtype
    hidden = h_train.shape[1]
    g = torch.Generator(device=device).manual_seed(seed)
    W = torch.zeros(H, hidden, device=device, dtype=dtype)

    # w_1 = hướng linear-probe (ridge trên e_pooled, bỏ cột bias)
    y_mean = y_train.mean()
    y_sd = y_train.std().clamp_min(1e-6)
    beta = ridge_fit(e_train, (y_train - y_mean) / y_sd, lam)   # (hidden+1,)
    w1 = beta[:hidden]
    W[0] = w1 / w1.norm().clamp_min(1e-12)

    # w_2..w_H = top-(H-1) PCA của per-atom h
    n_pc = min(H - 1, hidden)
    if n_pc > 0:
        hc = h_train - h_train.mean(dim=0, keepdim=True)
        q = min(n_pc, min(hc.shape))
        _, _, V = torch.pca_lowrank(hc, q=q)        # V: (hidden, q)
        pcs = V.t()                                  # (q, hidden)
        k = min(n_pc, pcs.shape[0])
        W[1:1 + k] = pcs[:k]
        filled = 1 + k
    else:
        filled = 1

    # nếu H-1 > hidden: phần dư khởi tạo ngẫu nhiên
    if filled < H:
        W[filled:] = torch.randn(H - filled, hidden, generator=g, device=device, dtype=dtype)

    # nhiễu nhỏ + chuẩn hoá đơn vị mỗi hàng
    W = W + noise * torch.randn(H, hidden, generator=g, device=device, dtype=dtype)
    W = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)

    # b_h = -percentile_{1-fire_rate}(w_h·h_i) -> ~fire_rate atom fire mỗi rule
    proj = h_train @ W.t()                            # (n_atoms, H)
    thresh = torch.quantile(proj, 1.0 - fire_rate, dim=0)
    b = -thresh
    return W, b
