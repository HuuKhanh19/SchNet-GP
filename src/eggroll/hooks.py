"""§3 — SchNet hook: per-atom features + batch index (atom->molecule).

Lấy h ∈ R^{n_atoms×hidden} SAU interaction block cuối, TRƯỚC output/readout MLP gốc
(readout gốc của SchNet bị thay bằng head hard-count + delta readout). Hook đã có sẵn
trong SchNet.forward(return_atom_emb_only=True) (src/models/schnet.py) — module này chỉ
bọc lại + dựng batch index và các phép pooling, KHÔNG sửa model.

Bất biến alignment: nhãn `target` đi cùng dict batch (do collate_multi_conformer tạo).
Không bao giờ index nhãn theo vị trí so với một mảng riêng.
"""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from src.data.data_loader import collate_multi_conformer


# =============================================================================
# Pooling (scatter) — tự implement để khỏi phụ thuộc torch_scatter
# =============================================================================

def scatter_add(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Sum-pool `src` (N, ...) theo `index` (N,) -> (dim_size, ...)."""
    out = src.new_zeros((dim_size,) + src.shape[1:])
    out.index_add_(0, index, src)
    return out


def scatter_mean(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Mean-pool `src` (N, ...) theo `index` (N,) -> (dim_size, ...)."""
    summed = scatter_add(src, index, dim_size)
    ones = torch.ones(src.shape[0], device=src.device, dtype=src.dtype)
    count = scatter_add(ones, index, dim_size).clamp_(min=1.0)
    if summed.dim() > 1:
        count = count.view((-1,) + (1,) * (summed.dim() - 1))
    return summed / count


# =============================================================================
# Full-train batch + per-atom feature extraction
# =============================================================================

def build_full_batch(dataset) -> Dict[str, Tensor]:
    """Gộp toàn bộ dataset thành MỘT batch (full-batch fitness của eggroll).

    pred & label co-index tuyệt đối vì cùng một collate, cùng thứ tự -> triệt tiêu
    misalignment (spec §2).
    """
    items = [dataset[i] for i in range(len(dataset))]
    return collate_multi_conformer(items)


def atom_to_mol_index(inputs: Dict[str, Tensor]) -> Tensor:
    """batch index atom->molecule. K=1: atom->conf->mol compose lại."""
    atom_to_conf = inputs["_idx_atom_to_conf"]
    conf_to_mol = inputs["_idx_conf_to_mol"]
    return conf_to_mol[atom_to_conf]


@torch.no_grad()
def extract_atom_features(
    model,
    batch: Dict[str, Tensor],
    device: torch.device,
    param_override: Optional[Dict[str, Tensor]] = None,
) -> Tuple[Tensor, Tensor, int]:
    """Trả (h, batch_idx, num_mols).

    h: per-atom features (n_atoms, hidden) sau interaction cuối.
    batch_idx: (n_atoms,) ánh xạ atom -> molecule.
    num_mols: số phân tử trong batch.

    param_override (dùng ở P2/LoRA): dict {tên param -> tensor} truyền vào
    torch.func.functional_call để chạy encoder với adapter mà KHÔNG mutate base in-place.
    """
    model.eval()
    inputs = {k: v.to(device) for k, v in batch.items() if k != "target"}

    if param_override is None:
        out = model(inputs, return_atom_emb_only=True)
    else:
        from torch.func import functional_call
        out = functional_call(
            model, param_override, args=(inputs,),
            kwargs={"return_atom_emb_only": True},
        )

    h = out["atom_embeddings"]                       # (n_atoms, hidden)
    batch_idx = atom_to_mol_index(inputs)            # (n_atoms,)
    num_mols = int(inputs["_idx_conf_to_mol"].max().item()) + 1
    return h, batch_idx, num_mols


def pooled_embedding(h: Tensor, batch_idx: Tensor, num_mols: int) -> Tensor:
    """e_pooled = scatter_mean(h, batch) -> (num_mols, hidden)."""
    return scatter_mean(h, batch_idx, num_mols)
