"""
Delta-learning baseline for CONAN-SchNet Step 2.

REGRESSION (unchanged):
    y_base = f(molecule descriptors), so the GP head only learns the residual
    Delta = y - y_base. Fit on TRAIN only.

CLASSIFICATION (binary, e.g. BACE):
    z_base = a *logit* baseline from descriptors (LogisticRegressionCV) -- or,
    when descriptors are disabled, the class-prior log-odds (the standard
    gradient-boosting F0 init). The GP head then learns the functional-gradient
    pseudo-residual  r = y - sigmoid(z_base)  (computed in the trainer), and the
    final probability is
        p = sigmoid(z_base + gamma * Delta_hat).
    i.e. classification is ONE round of logit-space gradient boosting on top of
    the exact same correlation-driven MFC machinery used for regression.

`fit` / `predict` return the additive baseline term in the model's working space:
    regression     -> y_base   (target units)
    classification -> z_base   (logits)
`predict_proba` is provided for convenience (classification only).

Descriptor options:
    'rdkit2d' : ~200 RDKit 2D descriptors (one vector per molecule)
    'morgan'  : Morgan/ECFP count fingerprint
    'none'    : disabled -> regression y_base = 0 ;
                            classification z_base = prior log-odds (intercept only)
"""

from typing import List, Optional

import numpy as np
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler


_RDKIT_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
_LOGIT_CS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)   # inverse-reg grid for logistic CV


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class DeltaBaseline:
    def __init__(self, kind: str = 'rdkit2d', task: str = 'regression',
                 morgan_radius: int = 2, morgan_bits: int = 2048,
                 alphas=_RDKIT_ALPHAS, logit_Cs=_LOGIT_CS):
        self.kind = kind
        self.task = task
        self.morgan_radius = morgan_radius
        self.morgan_bits = morgan_bits
        self.alphas = list(alphas)
        self.logit_Cs = list(logit_Cs)

        self.scaler: Optional[StandardScaler] = None
        self.ridge = None                               # RidgeCV (reg) | LogisticRegressionCV (clf)
        self.keep_cols: Optional[np.ndarray] = None     # bool mask over raw descriptors
        self.desc_names: Optional[List[str]] = None
        self.intercept_logit: Optional[float] = None    # clf prior log-odds (none / fallback)

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
        """Fit baseline; return the baseline term for the training molecules
        (y_base for regression, z_base logits for classification)."""
        y_train = np.asarray(y_train, dtype=np.float64)
        if self.task == 'classification':
            return self._fit_classification(smiles_train, y_train)
        return self._fit_regression(smiles_train, y_train)

    # ---- regression (unchanged behaviour) ----
    def _fit_regression(self, smiles_train: List[str], y_train: np.ndarray) -> np.ndarray:
        if self.kind == 'none':
            return np.zeros(len(smiles_train), dtype=np.float64)

        X = self._featurize(smiles_train)
        finite_col = np.isfinite(X).all(axis=0)
        if not finite_col.any():
            raise RuntimeError("All descriptor columns invalid on train set")
        self.keep_cols = finite_col
        X = X[:, finite_col]

        self.scaler = StandardScaler().fit(X)
        Xs = np.clip(self.scaler.transform(X), -10.0, 10.0)
        self.ridge = RidgeCV(alphas=self.alphas).fit(Xs, y_train)
        y_base = self.ridge.predict(Xs)
        print(f"  Delta baseline ({self.kind}): kept {int(finite_col.sum())} features, "
              f"alpha={self.ridge.alpha_:.3g}")
        return y_base

    # ---- classification (logit baseline) ----
    def _fit_classification(self, smiles_train: List[str], y_train: np.ndarray) -> np.ndarray:
        # prior log-odds: used for kind='none' and as a degenerate single-class fallback
        p = float(np.clip(y_train.mean(), 1e-6, 1.0 - 1e-6))
        self.intercept_logit = float(np.log(p / (1.0 - p)))

        if self.kind == 'none':
            print(f"  Delta baseline (none, clf): prior log-odds = {self.intercept_logit:.3f}")
            return np.full(len(smiles_train), self.intercept_logit, dtype=np.float64)

        X = self._featurize(smiles_train)
        finite_col = np.isfinite(X).all(axis=0)
        if not finite_col.any():
            raise RuntimeError("All descriptor columns invalid on train set")
        self.keep_cols = finite_col
        X = X[:, finite_col]

        self.scaler = StandardScaler().fit(X)
        Xs = np.clip(self.scaler.transform(X), -10.0, 10.0)

        if len(np.unique(y_train)) < 2:
            print("  Delta baseline (clf): single-class train -> intercept-only baseline")
            self.ridge = None
            return np.full(len(smiles_train), self.intercept_logit, dtype=np.float64)

        self.ridge = LogisticRegressionCV(
            Cs=self.logit_Cs, cv=5, scoring='neg_log_loss', max_iter=2000,
        ).fit(Xs, y_train.astype(int))
        z_base = self.ridge.decision_function(Xs)
        print(f"  Delta baseline ({self.kind}, clf): kept {int(finite_col.sum())} features, "
              f"C={float(self.ridge.C_[0]):.3g}")
        return np.asarray(z_base, dtype=np.float64)

    # ------------------------------------------------------------------
    def predict(self, smiles_list: List[str]) -> np.ndarray:
        """Regression -> y_base ; classification -> z_base (logits)."""
        if self.task == 'classification':
            return self._predict_logit(smiles_list)
        return self._predict_regression(smiles_list)

    def _predict_regression(self, smiles_list: List[str]) -> np.ndarray:
        if self.kind == 'none' or self.ridge is None:
            return np.zeros(len(smiles_list), dtype=np.float64)
        X = self._featurize(smiles_list)[:, self.keep_cols]
        Xs = self.scaler.transform(X)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        Xs = np.clip(Xs, -7.0, 7.0)
        return self.ridge.predict(Xs)

    def _predict_logit(self, smiles_list: List[str]) -> np.ndarray:
        if self.kind == 'none' or self.ridge is None:
            val = self.intercept_logit if self.intercept_logit is not None else 0.0
            return np.full(len(smiles_list), val, dtype=np.float64)
        X = self._featurize(smiles_list)[:, self.keep_cols]
        Xs = self.scaler.transform(X)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        Xs = np.clip(Xs, -7.0, 7.0)
        return np.asarray(self.ridge.decision_function(Xs), dtype=np.float64)

    def predict_proba(self, smiles_list: List[str]) -> np.ndarray:
        """Classification convenience: sigmoid(z_base)."""
        return _sigmoid(self._predict_logit(smiles_list))