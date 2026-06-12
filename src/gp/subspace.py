"""Subspace cố định cho cây feature (KHÔNG evolve).

q = num_emb_trees + num_desc3d_trees slot, định nghĩa MỘT lần cho mỗi seed:
- Slot embedding: gắn cứng d chiều random từ hidden (128). Các slot ĐỘC LẬP nhau
  -> CHO PHÉP chồng lấn chiều giữa slot (task quy định). Trong một slot, d chiều
  phân biệt.
- Slot desc-3D: gắn một tập 3D descriptor (chia từ danh sách desc3d trong config).

Subspace GIỐNG NHAU ở mọi bin và mọi option cờ tree_sharing -> 3 option chỉ khác ở
việc biểu thức cây có share hay không (clean ablation). Seed hoá theo seed run nên
trong cùng một seed, cả 6 cell dùng đúng một subspace.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Subspace:
    hidden_dim: int                 # 128
    emb_dim: int                    # d
    emb_slots: List[np.ndarray]     # num_emb_trees x (d,) chỉ số chiều trong [0, hidden)
    desc3d_slots: List[List[int]]   # num_desc3d_trees x list index vào desc3d_names
    desc3d_names: List[str]         # tên 3D descriptor (theo config)

    @property
    def num_emb_trees(self) -> int:
        return len(self.emb_slots)

    @property
    def num_desc3d_trees(self) -> int:
        return len(self.desc3d_slots)

    @property
    def q(self) -> int:
        """Tổng số cây feature mỗi bin."""
        return self.num_emb_trees + self.num_desc3d_trees

    def slot_dim(self, slot_idx: int) -> int:
        """Số terminal (chiều) của cây feature slot `slot_idx` (emb trước, desc3d sau)."""
        if slot_idx < self.num_emb_trees:
            return self.emb_dim
        return len(self.desc3d_slots[slot_idx - self.num_emb_trees])


def build_subspace(
    hidden_dim: int,
    emb_dim: int,
    num_emb_trees: int,
    num_desc3d_trees: int,
    desc3d_names: List[str],
    seed: int,
) -> Subspace:
    """Lấy mẫu subspace cố định, seed hoá hoàn toàn theo `seed`."""
    rng = np.random.default_rng(seed)

    if emb_dim > hidden_dim:
        raise ValueError(f"emb_dim={emb_dim} > hidden_dim={hidden_dim}")

    # Slot embedding: mỗi slot d chiều phân biệt; slot độc lập -> overlap tự nhiên.
    emb_slots = [
        np.sort(rng.choice(hidden_dim, size=emb_dim, replace=False))
        for _ in range(num_emb_trees)
    ]

    # Slot desc-3D: xáo trộn rồi chia danh sách desc3d thành num_desc3d_trees nhóm
    # gần đều (mỗi slot >= 1 descriptor). Mỗi nhóm là terminal của cây feature đó.
    n3 = len(desc3d_names)
    if num_desc3d_trees > n3:
        raise ValueError(
            f"num_desc3d_trees={num_desc3d_trees} > số 3D descriptor={n3}"
        )
    perm = rng.permutation(n3)
    desc3d_slots = [list(map(int, chunk)) for chunk in np.array_split(perm, num_desc3d_trees)]

    return Subspace(
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        emb_slots=emb_slots,
        desc3d_slots=desc3d_slots,
        desc3d_names=list(desc3d_names),
    )
