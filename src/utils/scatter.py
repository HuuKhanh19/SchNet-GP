"""
Pure-torch segment reductions for CONAN-SchNet Step 2.

Implemented without torch_scatter so there is no extra CUDA-extension
dependency (which is risky on Blackwell sm_120). Only standard torch ops.

All functions reduce along dim 0 by an integer `index` of shape (N,).
"""

from typing import Optional
import torch
from torch import Tensor


def _resolve_dim_size(index: Tensor, dim_size: Optional[int]) -> int:
    if dim_size is not None:
        return int(dim_size)
    if index.numel() == 0:
        return 0
    return int(index.max().item()) + 1


def scatter_add(src: Tensor, index: Tensor, dim_size: Optional[int] = None) -> Tensor:
    """Sum `src` rows into segments defined by `index`.

    src:   (N,) or (N, D)
    index: (N,) int64
    returns (dim_size,) or (dim_size, D)
    """
    dim_size = _resolve_dim_size(index, dim_size)
    out = src.new_zeros((dim_size,) + tuple(src.shape[1:]))
    idx = index
    if src.dim() > 1:
        idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out.scatter_add_(0, idx, src)
    return out


def scatter_mean(src: Tensor, index: Tensor, dim_size: Optional[int] = None) -> Tensor:
    """Mean of `src` rows per segment. Empty segments -> 0."""
    dim_size = _resolve_dim_size(index, dim_size)
    summed = scatter_add(src, index, dim_size)
    count = src.new_zeros(dim_size)
    count.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    count = count.clamp(min=1.0)
    if src.dim() > 1:
        count = count.view(-1, *([1] * (src.dim() - 1)))
    return summed / count


def scatter_softmax(src: Tensor, index: Tensor, dim_size: Optional[int] = None) -> Tensor:
    """Softmax of the 1-D `src` *within* each segment (numerically stable).

    src:   (N,) float
    index: (N,) int64
    returns (N,) -- weights that sum to 1 inside each segment.
    """
    assert src.dim() == 1, "scatter_softmax expects 1-D src"
    dim_size = _resolve_dim_size(index, dim_size)
    seg_max = src.new_full((dim_size,), float("-inf"))
    seg_max.scatter_reduce_(0, index, src.detach(), reduce="amax", include_self=True)
    # guard segments that received nothing (shouldn't happen for conf->mol);
    # detach: the max is a stabilization constant, not part of the softmax gradient
    seg_max = torch.nan_to_num(seg_max, neginf=0.0).detach()
    shifted = (src - seg_max[index]).exp()
    seg_sum = scatter_add(shifted, index, dim_size)
    return shifted / (seg_sum[index] + 1e-12)