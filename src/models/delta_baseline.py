"""
Delta-learning baseline for CONAN-SchNet Step 2.

A cheap, conformer-independent baseline y_base = f(molecule descriptors) so the
GP head only has to learn the residual Delta = y - y_base. Fit on TRAIN only.

Descriptor options:
    'rdkit2d' : ~200 RDKit 2D descriptors (topological, one vector per molecule)
    'morgan'  : Morgan/ECFP count fingerprint
    'none'    : disabled -> y_base = 0, Delta = y
"""

from typing import List, Optional

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


_RDKIT_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


class DeltaBaseline:
    def __init__(self, kind: str = 'rdkit2d', morgan_radius: int = 2,
                 morgan_bits: int = 2048, alphas=_RDKIT_ALPHAS):
        self.kind = kind
        self.morgan_radius = morgan_radius
        self.morgan_bits = morgan_bits
        self.alphas = list(alphas)

        self.scaler: Optional[StandardScaler] = None
        self.ridge: Optional[RidgeCV] = None
        self.keep_cols: Optional[np.ndarray] = None   # bool mask over raw descriptors
        self.desc_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    def _featurize(self, smiles_list: List[str]) -> np.ndarray:
        if self.kind == 'morgan':
            return self._featurize_morgan(smiles_list)
        return self._featurize_rdkit2d(smiles_list)

    def _featurize_rdkit2d(self, smiles_list: List[str]) -> np.ndarray:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors
        RDLogger.DisableLog('rdApp.*')

        if self.desc_names is None:
            self.desc_names = [name for name, _ in Descriptors.descList]
        funcs = [fn for _, fn in Descriptors.descList]

        rows = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rows.append([np.nan] * len(funcs))
                continue
            vals = []
            for fn in funcs:
                try:
                    vals.append(float(fn(mol)))
                except Exception:
                    vals.append(np.nan)
            rows.append(vals)
        return np.asarray(rows, dtype=np.float64)

    def _featurize_morgan(self, smiles_list: List[str]) -> np.ndarray:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem
        RDLogger.DisableLog('rdApp.*')

        rows = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rows.append(np.zeros(self.morgan_bits, dtype=np.float64))
                continue
            fp = AllChem.GetHashedMorganFingerprint(
                mol, self.morgan_radius, nBits=self.morgan_bits)
            arr = np.zeros(self.morgan_bits, dtype=np.float64)
            for idx, cnt in fp.GetNonzeroElements().items():
                arr[idx] = cnt
            rows.append(arr)
        return np.asarray(rows, dtype=np.float64)

    # ------------------------------------------------------------------
    def fit(self, smiles_train: List[str], y_train: np.ndarray) -> np.ndarray:
        """Fit baseline; return y_base for the training molecules."""
        if self.kind == 'none':
            return np.zeros(len(smiles_train), dtype=np.float64)

        X = self._featurize(smiles_train)
        # drop columns with any non-finite value on TRAIN
        finite_col = np.isfinite(X).all(axis=0)
        if not finite_col.any():
            raise RuntimeError("All descriptor columns invalid on train set")
        self.keep_cols = finite_col
        X = X[:, finite_col]

        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)

        self.ridge = RidgeCV(alphas=self.alphas).fit(Xs, np.asarray(y_train))
        y_base = self.ridge.predict(Xs)
        print(f"  Delta baseline ({self.kind}): kept {int(finite_col.sum())} features, "
              f"alpha={self.ridge.alpha_:.3g}")
        return y_base

    def predict(self, smiles_list: List[str]) -> np.ndarray:
        if self.kind == 'none' or self.ridge is None:
            return np.zeros(len(smiles_list), dtype=np.float64)
        X = self._featurize(smiles_list)[:, self.keep_cols]
        Xs = self.scaler.transform(X)
        # any leftover non-finite (a test mol failing a kept descriptor) -> train mean (0)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        return self.ridge.predict(Xs)