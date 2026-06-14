"""RDKit 2D descriptor featurizer (toàn bộ ~200 descriptor qua Descriptors.descList).

Dùng cho Exp 0 (baseline 2D) và tái dùng cho các thí nghiệm sau. Featurization là
độc lập theo phân tử; phần *làm sạch* (impute median, VarianceThreshold) phải FIT TRÊN
TRAIN rồi áp cho valid/test để không rò rỉ thông tin.

Pipeline làm sạch (fit trên train):
  1. inf -> NaN.
  2. Impute NaN bằng MEDIAN của train (descriptor nào fail toàn bộ train -> drop cột).
  3. VarianceThreshold(0): bỏ cột near-constant trên train.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# =============================================================================
# Tính descriptor thô
# =============================================================================

def _descriptor_functions():
    """Trả về (names, funcs) cho toàn bộ RDKit 2D descriptor."""
    from rdkit.Chem import Descriptors
    names = [name for name, _ in Descriptors.descList]
    funcs = [fn for _, fn in Descriptors.descList]
    return names, funcs


def compute_raw_descriptors(smiles_list: List[str]):
    """Tính ma trận descriptor thô (N, D) cho danh sách SMILES.

    Descriptor nào raise -> điền NaN cho phân tử đó. SMILES không parse được ->
    cả hàng NaN. Trả về (X, names).
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    names, funcs = _descriptor_functions()
    n, d = len(smiles_list), len(funcs)
    X = np.full((n, d), np.nan, dtype=np.float64)

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        for j, fn in enumerate(funcs):
            try:
                v = fn(mol)
                X[i, j] = float(v)
            except Exception:
                X[i, j] = np.nan

    RDLogger.EnableLog("rdApp.*")
    return X, names


# =============================================================================
# Featurizer có trạng thái (fit trên train, transform mọi split)
# =============================================================================

@dataclass
class Descriptor2DFeaturizer:
    """Tính + làm sạch RDKit 2D descriptor. Fit trên train, transform valid/test.

    Sau fit, `feature_names` là tên các cột còn lại (sau khi drop NaN-toàn-train +
    near-constant). `transform` trả về ma trận đã impute median-train + lọc cột,
    KHÔNG standardize (để model tự xử lý: Ridge cần StandardScaler, GBT thì không).
    """

    variance_threshold: float = 0.0
    # Trạng thái học từ train:
    _all_names: List[str] = field(default_factory=list, repr=False)
    _keep_idx: Optional[np.ndarray] = field(default=None, repr=False)
    _medians: Optional[np.ndarray] = field(default=None, repr=False)
    feature_names: List[str] = field(default_factory=list)

    def fit(self, smiles_train: List[str]) -> "Descriptor2DFeaturizer":
        X, names = compute_raw_descriptors(smiles_train)
        self._all_names = names
        X = np.where(np.isfinite(X), X, np.nan)

        # Median train (bỏ qua NaN). Cột all-NaN -> median NaN -> sẽ bị drop.
        with np.errstate(all="ignore"):
            medians = np.nanmedian(X, axis=0)
        valid_col = np.isfinite(medians)

        # Impute để tính variance sau impute.
        Xi = X.copy()
        med_fill = np.where(valid_col, medians, 0.0)
        inds = np.where(np.isnan(Xi))
        Xi[inds] = np.take(med_fill, inds[1])

        # VarianceThreshold trên train (sau impute).
        var = Xi.var(axis=0)
        keep_mask = valid_col & (var > self.variance_threshold)

        self._keep_idx = np.where(keep_mask)[0]
        self._medians = medians[self._keep_idx]
        self.feature_names = [names[i] for i in self._keep_idx]
        return self

    def transform(self, smiles_list: List[str]) -> np.ndarray:
        if self._keep_idx is None:
            raise RuntimeError("Gọi fit() trước transform().")
        X, _ = compute_raw_descriptors(smiles_list)
        X = np.where(np.isfinite(X), X, np.nan)
        X = X[:, self._keep_idx]
        # Impute bằng median TRAIN.
        inds = np.where(np.isnan(X))
        X[inds] = np.take(self._medians, inds[1])
        return X

    def fit_transform(self, smiles_train: List[str]) -> np.ndarray:
        self.fit(smiles_train)
        return self.transform(smiles_train)
