"""STEP 2 — toàn bộ descriptor 2D RDKit (mức phân tử), KHÔNG chọn lọc.

Khác `descriptors.py` (12 descriptor chọn tay cho GP head): ở đây lấy HẾT bộ
`Descriptors._descList` của RDKit (~217 descriptor 2D) để đo "sức mạnh" thực sự
của desc2d. Việc giảm trọng số / loại bỏ descriptor vô dụng để RidgeCV (co hệ số)
hoặc cây GP (không tham chiếu biến) tự lo — không lọc trước.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

# Toàn bộ descriptor 2D RDKit. Thứ tự cố định -> cache reproduce được.
_DESC_LIST = list(Descriptors._descList)
FULL_DESC2D_NAMES: List[str] = [name for name, _ in _DESC_LIST]
_FUNCS = [fn for _, fn in _DESC_LIST]


def compute_full_desc2d(mol: Chem.Mol) -> np.ndarray:
    """Vector toàn bộ descriptor 2D cho 1 mol (đã parse). Lỗi/đơn lẻ -> NaN.

    NaN/inf để bước standardize xử lý (điền median train), không nhét 0 ở đây để
    tránh tạo giá trị giả 0 có nghĩa với một số descriptor.
    """
    out = np.empty(len(_FUNCS), dtype=np.float64)
    for i, fn in enumerate(_FUNCS):
        try:
            out[i] = float(fn(mol))
        except Exception:
            out[i] = np.nan
    return out


def build_desc2d_matrix(smiles: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Trả (X, ok): X shape (N, D) toàn bộ desc2d; ok (N,) bool = parse SMILES được.

    SMILES trong split CSV đã được validate ở khâu preprocess nên ok hầu hết True;
    mol fail (hiếm) -> hàng NaN, ok=False (caller tự quyết loại bỏ).
    """
    D = len(_FUNCS)
    X = np.full((len(smiles), D), np.nan, dtype=np.float64)
    ok = np.zeros(len(smiles), dtype=bool)
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        X[i] = compute_full_desc2d(m)
        ok[i] = True
    return X, ok
