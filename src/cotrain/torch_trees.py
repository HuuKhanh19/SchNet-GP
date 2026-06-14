"""Differentiable torch interpreter cho DEAP PrimitiveTree — CHỈ dùng cho ablation
backprop-qua-trees (Nhánh A, function set khả vi).

Eggroll KHÔNG cần module này (chỉ cần forward -> dùng numpy assemble_phi của gp_head).
Backprop cần: encoder(torch) -> emb -> trees(torch, khả vi) -> ridge(torch.linalg.solve,
khả vi) -> loss -> backward vào encoder.

Op torch trùng TÊN + ngữ nghĩa với protected ops numpy trong src/gp/gp_head.py (funcset
khả vi). Toán tử non-diff (gt/ifte/step...) KHÔNG có ở đây: backprop không chạy được
với chúng -> đó chính là lý do nguyên tắc để dùng eggroll (xem spec mục 0/6 nhánh B).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from deap import gp
from torch import Tensor

_EPS = 1e-6


def _t_div(a, b):
    safe = b.abs() > _EPS
    return torch.where(safe, a / torch.where(safe, b, torch.ones_like(b)),
                       torch.ones_like(a))


# name -> (callable torch, arity). Trùng FUNCSET khả vi của gp_head.
TORCH_OPS = {
    "add": (lambda a, b: a + b, 2),
    "sub": (lambda a, b: a - b, 2),
    "mul": (lambda a, b: a * b, 2),
    "pdiv": (_t_div, 2),
    "sin": (torch.sin, 1),
    "cos": (torch.cos, 1),
    "plog": (lambda a: torch.log(a.abs() + _EPS), 1),
    "psqrt": (lambda a: torch.sqrt(a.abs() + 1e-12), 1),
    "tanh": (torch.tanh, 1),
}


def eval_tree_torch(tree: gp.PrimitiveTree, emb: Tensor, cols) -> Tensor:
    """Interpret 1 cây (prefix) -> tensor (N,) khả vi theo emb.

    emb: (N, D) tensor; cols: chỉ số global dim của khối cây này (arg x{k} -> emb[:,cols[k]]).
    """
    n = emb.shape[0]
    pos = 0

    def rec() -> Tensor:
        nonlocal pos
        node = tree[pos]
        pos += 1
        if isinstance(node, gp.Primitive):
            args = [rec() for _ in range(node.arity)]
            fn, _ = TORCH_OPS[node.name]
            return fn(*args)
        # Terminal: arg 'x{k}' hoặc hằng số (ERC).
        val = node.value
        if isinstance(val, str) and val.startswith("x"):
            return emb[:, int(cols[int(val[1:])])]
        return torch.full((n,), float(val), dtype=emb.dtype, device=emb.device)

    return rec()


def assemble_phi_torch(trees, emb: Tensor, partition) -> Tensor:
    """Φ (N, q) khả vi: cây j eval trên khối partition[j] của emb. nan/inf -> 0."""
    cols_phi = [eval_tree_torch(trees[j], emb, partition[j]) for j in range(len(trees))]
    phi = torch.stack(cols_phi, dim=1)
    return torch.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)


def ridge_torch(phi: Tensor, t: Tensor, alpha: float) -> Tuple[Tensor, Tensor, Tensor]:
    """Ridge closed-form khả vi (torch.linalg.solve). Trả (pred, w, b)."""
    b = t.mean()
    tc = t - b
    p = phi.shape[1]
    A = phi.T @ phi + alpha * torch.eye(p, dtype=phi.dtype, device=phi.device)
    w = torch.linalg.solve(A, phi.T @ tc)
    return phi @ w + b, w, b
