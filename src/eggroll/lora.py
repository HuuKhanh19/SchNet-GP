"""§4 — LoRA adapter trên các Linear trong interaction block.

W_eff = W_base + (α/r)·B·A, A∈R^{r×in}~N(0,nhỏ), B∈R^{out×r}=0 (adapter=0 lúc start ->
encoder=base). Chỉ A,B trainable (base frozen). KHÔNG adapt embedding atom-type, KHÔNG
output-net gốc (lin1/lin2 — vốn không dùng).

Target = mọi nn.Linear dưới 'interactions.' (mlp.0, mlp.2, conv.lin1, conv.lin2, lin).
named_modules() dedup nên conv.nn (== mlp) không bị tính trùng.

P2 forward: build_override(...) -> dict {weight_name -> W_eff} cho torch.func.functional_call
(không mutate in-place).
"""

from contextlib import contextmanager
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def discover_lora_targets(model) -> Dict[str, Tuple[int, int]]:
    """Trả {weight_name -> (out, in)} cho mọi Linear trong interaction block."""
    targets = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.startswith("interactions."):
            targets[name + ".weight"] = tuple(mod.weight.shape)   # (out, in)
    return targets


def init_lora_params(
    targets: Dict[str, Tuple[int, int]], r: int, init_std: float,
    seed: int, device, dtype,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """A ~ N(0, init_std) (r×in); B = 0 (out×r). Hiệu lực ban đầu = 0 vì B=0."""
    g = torch.Generator(device=device).manual_seed(seed)
    A, B = {}, {}
    for wname, (out, inn) in targets.items():
        A[wname] = torch.randn(r, inn, generator=g, device=device, dtype=dtype) * init_std
        B[wname] = torch.zeros(out, r, device=device, dtype=dtype)
    return A, B


def build_override(model, A: Dict[str, Tensor], B: Dict[str, Tensor],
                   r: int, alpha: float) -> Dict[str, Tensor]:
    """override[wname] = W_base + (α/r)·B·A. W_base lấy từ model (frozen)."""
    scale = alpha / r
    override = {}
    for wname in A:
        base = model.get_parameter(wname).detach()
        override[wname] = base + scale * (B[wname] @ A[wname])
    return override


@contextmanager
def lora_weights(model, override: Dict[str, Tensor]):
    """Tạm hoán .data của các weight target = W_eff cho forward, rồi khôi phục nguyên trạng.

    Thay cho torch.func.functional_call: SchNet chia sẻ module mlp (== conv.nn) -> tied
    weights khiến functional_call KHÔNG restore đúng (weight thành Tensor, mất Parameter).
    Hoán .data giữ nguyên Parameter, module chia sẻ tự nhất quán (cùng một Parameter), và
    finally đảm bảo base về đúng kể cả khi forward lỗi. (Không phải mutate gradient -> an toàn.)
    """
    saved = {}
    try:
        for wname, weff in override.items():
            p = model.get_parameter(wname)
            saved[wname] = p.data
            p.data = weff
        yield
    finally:
        for wname, orig in saved.items():
            model.get_parameter(wname).data = orig
