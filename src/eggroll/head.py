"""§5 — Head per-atom hard-count (Heaviside rules).

Mỗi rule h: z_{h,i} = step(w_h·h_i + b_h) ∈ {0,1} cho từng atom i; count_h =
scatter_add(z_h, batch) = số atom thoả rule h trong phân tử. Per-atom-rồi-sum (KHÔNG
pool-rồi-step): extensive + atomwise (đúng readout SchNet gốc) và landscape trơn cho
eggroll (đổi w_h chỉ lật vài atom -> count đổi ±1, không lật cả phân tử).

Heaviside không khả vi -> head tối ưu bằng eggroll (ES, Stage C), không backprop.
"""

from typing import Tuple

import torch
from torch import Tensor

from .hooks import scatter_add


def compute_counts(
    W: Tensor, b: Tensor, h: Tensor, batch_idx: Tensor, num_mols: int
) -> Tensor:
    """counts (num_mols, H) từ head {W (H,hidden), b (H,)} và per-atom h (n_atoms,hidden)."""
    logits = h @ W.t() + b            # (n_atoms, H)
    z = (logits > 0).to(h.dtype)      # Heaviside step {0,1}
    return scatter_add(z, batch_idx, num_mols)


def init_head_random(
    h_train: Tensor, H: int, fire_rate: float = 0.5, seed: int = 0,
) -> Tuple[Tensor, Tensor]:
    """Init head dùng cho Stage B (machinery check): hướng ngẫu nhiên + bias percentile.

    w_h ~ N(0,1) chuẩn hoá đơn vị; b_h = -quantile_{1-fire_rate}(w_h·h_i) trên toàn atom
    train -> ~fire_rate atom fire mỗi rule (tránh rule chết/bão hoà). Stage C
    (init_warmstart) thay hướng bằng β + PCA cho khởi đầu gần signal.
    """
    device, dtype = h_train.device, h_train.dtype
    hidden = h_train.shape[1]
    g = torch.Generator(device=device).manual_seed(seed)
    W = torch.randn(H, hidden, generator=g, device=device, dtype=dtype)
    W = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
    proj = h_train @ W.t()                                  # (n_atoms, H)
    thresh = torch.quantile(proj, 1.0 - fire_rate, dim=0)   # (H,)
    b = -thresh
    return W, b


def count_diagnostics(counts: Tensor, n_atoms: int) -> dict:
    """Sanity counts: fire-rate per rule (frac atom fire), #dead/#sat, range."""
    fire = counts.sum(dim=0) / max(n_atoms, 1)             # (H,) frac atom fire/rule
    return {
        "fire_mean": float(fire.mean()),
        "fire_min": float(fire.min()),
        "fire_max": float(fire.max()),
        "n_dead": int((fire < 0.01).sum()),                # rule gần như không fire
        "n_sat": int((fire > 0.99).sum()),                 # rule fire gần hết
        "count_max": float(counts.max()),
        "count_mean": float(counts.mean()),
    }
