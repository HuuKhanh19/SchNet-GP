"""
Energy-based MoE gate for CONAN-SchNet Step 2.

Routes each conformer to an expert by its relative MMFF energy dE.
Bins are GLOBAL quantiles fit on TRAIN conformers only, then frozen and
applied unchanged to valid/test (no leakage).

    num_experts = 1  -> single expert (MoE off): every conformer -> bin 0
    num_experts = B  -> B equal-count bins by dE quantile

dE is per-molecule-relative (each molecule's lowest conformer has dE = 0), so
bin 0 captures "near the ground state of the ensemble" and higher bins capture
progressively higher-energy regimes.
"""

from typing import List, Optional

import numpy as np


class EnergyGate:
    def __init__(self, num_experts: int = 2,
                 energy_clip='train_max', binning: str = 'quantile'):
        assert num_experts >= 1
        assert binning == 'quantile', "only quantile binning implemented"
        self.num_experts = num_experts
        self.energy_clip = energy_clip
        self.boundaries: Optional[np.ndarray] = None   # (num_experts-1,)
        self.clip_val: Optional[float] = None

    # ------------------------------------------------------------------
    def _clip_value(self, dE_train: np.ndarray) -> float:
        if isinstance(self.energy_clip, (int, float)):
            return float(self.energy_clip)
        finite = dE_train[np.isfinite(dE_train)]
        if finite.size == 0:
            return 25.0
        return float(finite.max())

    def clamp(self, dE) -> np.ndarray:
        dE = np.asarray(dE, dtype=np.float64).copy()
        cv = self.clip_val if self.clip_val is not None else 25.0
        dE[~np.isfinite(dE)] = cv
        dE[dE > cv] = cv
        return dE

    # ------------------------------------------------------------------
    def fit(self, dE_train: np.ndarray) -> "EnergyGate":
        self.clip_val = self._clip_value(np.asarray(dE_train, dtype=np.float64))
        dE = self.clamp(dE_train)
        if self.num_experts == 1:
            self.boundaries = np.array([], dtype=np.float64)
        else:
            qs = [100.0 * j / self.num_experts for j in range(1, self.num_experts)]
            self.boundaries = np.percentile(dE, qs).astype(np.float64)
        return self

    def route(self, dE) -> np.ndarray:
        """Return integer bin id in {0..num_experts-1} per conformer."""
        assert self.boundaries is not None, "EnergyGate not fitted"
        dE = self.clamp(dE)
        if self.num_experts == 1:
            return np.zeros(len(dE), dtype=np.int64)
        return np.digitize(dE, self.boundaries).astype(np.int64)

    # ------------------------------------------------------------------
    def bin_ranges(self, dE) -> List[str]:
        """Human-readable dE range (kcal/mol) and count per bin, for logging."""
        dE = self.clamp(dE)
        bins = self.route(dE)
        out = []
        for b in range(self.num_experts):
            m = bins == b
            if m.any():
                out.append(f"bin{b}: [{dE[m].min():.2f}, {dE[m].max():.2f}] "
                           f"kcal/mol  n={int(m.sum())}")
            else:
                out.append(f"bin{b}: empty")
        return out