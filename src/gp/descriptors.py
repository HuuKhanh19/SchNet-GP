"""Descriptor RDKit cho GP head.

- 2D (mức phân tử, bất biến conformer): chọn num_2d theo |corr| với target TRÊN TRAIN
  (chống leak — chỉ dùng train để chọn, áp cùng index cho valid/test).
- 3D (mức conformer): tính trực tiếp từ hình học conformer (atomic_numbers + positions
  đã cache, energy-ranked). Dùng cho slot desc-3D của subspace.
"""

from typing import List, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Descriptors3D
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdApp.*")


# =============================================================================
# 2D — mức phân tử
# =============================================================================

def _get_2d_fn(name: str):
    fn = getattr(Descriptors, name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"RDKit Descriptors không có 2D descriptor: '{name}'")
    return fn


def compute_2d_descriptors(
    smiles_list: Sequence[str], names: Sequence[str]
) -> np.ndarray:
    """(N, len(names)) float32. Lỗi/giá trị không hợp lệ -> NaN (xử lý ở chỗ chọn)."""
    fns = [_get_2d_fn(n) for n in names]
    out = np.full((len(smiles_list), len(names)), np.nan, dtype=np.float64)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for j, fn in enumerate(fns):
            try:
                v = fn(mol)
                if v is not None and np.isfinite(v):
                    out[i, j] = float(v)
            except Exception:
                pass
    return out.astype(np.float32)


def select_2d_by_corr(
    desc2d_train: np.ndarray,
    targets_train: np.ndarray,
    names: Sequence[str],
    k: int,
) -> Tuple[List[int], List[str]]:
    """Chọn k descriptor 2D có |Pearson corr| với target lớn nhất, TRÊN TRAIN.

    Cột toàn NaN/hằng số (std=0) bị loại. Trả (indices vào `names`, tên đã chọn),
    sắp theo |corr| giảm dần. Index này áp lại cho valid/test (chống leak).
    """
    y = np.asarray(targets_train, dtype=np.float64)
    n_feat = desc2d_train.shape[1]
    corrs = np.zeros(n_feat, dtype=np.float64)
    for j in range(n_feat):
        col = desc2d_train[:, j].astype(np.float64)
        mask = np.isfinite(col)
        if mask.sum() < 3:
            corrs[j] = 0.0
            continue
        c, yv = col[mask], y[mask]
        if np.std(c) < 1e-12 or np.std(yv) < 1e-12:
            corrs[j] = 0.0
            continue
        corrs[j] = abs(float(np.corrcoef(c, yv)[0, 1]))
    corrs = np.nan_to_num(corrs, nan=0.0)
    k = int(min(k, n_feat))
    order = np.argsort(-corrs)[:k]
    idx = [int(j) for j in order]
    return idx, [names[j] for j in idx]


def impute_columns(x: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """Thay NaN theo từng cột bằng `fill` (vd median train). `fill` shape (n_cols,)."""
    x = x.copy()
    inds = np.where(~np.isfinite(x))
    if len(inds[0]):
        x[inds] = np.take(fill, inds[1])
    return x


# =============================================================================
# 3D — mức conformer
# =============================================================================

def _get_3d_fn(name: str):
    fn = getattr(Descriptors3D, name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"RDKit Descriptors3D không có 3D descriptor: '{name}'")
    return fn


def mol_from_z_coords(atomic_numbers: np.ndarray, coords: np.ndarray) -> Chem.Mol:
    """Dựng RDKit Mol (không bond) từ atomic numbers + toạ độ 3D.

    Đủ cho các 3D descriptor dựa trên tensor mô-men quán tính (PMI, RG, NPR,
    Asphericity, ...): chúng chỉ cần khối lượng nguyên tử + toạ độ, không cần bond.
    """
    rw = Chem.RWMol()
    for z in atomic_numbers:
        rw.AddAtom(Chem.Atom(int(z)))
    n = len(atomic_numbers)
    conf = Chem.Conformer(n)
    conf.Set3D(True)
    for i in range(n):
        x, y, z = coords[i]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol = rw.GetMol()
    mol.AddConformer(conf, assignId=True)
    return mol


def compute_3d_descriptors(
    atomic_numbers: np.ndarray, coords: np.ndarray, names: Sequence[str]
) -> np.ndarray:
    """(len(names),) float32 cho MỘT conformer. Lỗi -> 0.0 (an toàn cho GP)."""
    fns = [_get_3d_fn(n) for n in names]
    out = np.zeros(len(names), dtype=np.float64)
    try:
        mol = mol_from_z_coords(atomic_numbers, coords)
    except Exception:
        return out.astype(np.float32)
    for j, fn in enumerate(fns):
        try:
            v = fn(mol)
            if v is not None and np.isfinite(v):
                out[j] = float(v)
        except Exception:
            pass
    return out.astype(np.float32)
