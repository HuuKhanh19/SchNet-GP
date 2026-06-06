"""
CONAN head for Step 2: assembles the fitted pieces and runs end-to-end
prediction.

    route confs -> per-expert per-conf score s_c
    aggregate    -> alpha_c = softmax(-dE_c / tau) per molecule ; S_hat = sum_c alpha_c s_c
    output:
        regression     -> y_hat = y_base + S_hat
        classification -> p_hat = sigmoid(z_base + logit_scale * S_hat)

For classification, s_c is the per-conformer functional-gradient increment
(ridge prediction of the pseudo-residual r = y - sigmoid(z_base)); the additive
structure is identical to regression but lives in logit space, with an optional
global logit_scale (gamma) fitted in Phase 2.

This is an inference container; fitting lives in Step2Trainer, which populates it.
"""

from typing import List, Optional

import numpy as np
import torch
from torch import Tensor

from src.utils.scatter import scatter_add, scatter_mean, scatter_softmax


def predict_per_conf_delta(experts: List, emb_std: Tensor, bins: np.ndarray) -> np.ndarray:
    """Run each conformer through its assigned expert. Returns s: (total_confs,).

    For regression s is the predicted Delta; for classification it is the predicted
    pseudo-residual (a logit increment before the global logit_scale).
    """
    s = np.zeros(emb_std.shape[0], dtype=np.float64)
    bins = np.asarray(bins)
    for b, expert in enumerate(experts):
        idx = np.where(bins == b)[0]
        if idx.size == 0:
            continue
        s[idx] = expert.predict(emb_std[idx])
    return s


def aggregate(s: Tensor, dE: Tensor, conf2mol: Tensor, n_mol: int,
              agg: str = 'learned_softmax', tau: float = 0.593) -> Tensor:
    """Aggregate per-conf score into per-molecule score.

    agg='mean'            : uniform weights (ensemble mean)
    agg='boltzmann'       : softmax(-dE/tau) with fixed tau
    agg='learned_softmax' : same form, tau learned in Phase 2 (passed in here)
    """
    if agg == 'mean':
        return scatter_mean(s, conf2mol, dim_size=n_mol)
    w = scatter_softmax(-dE / max(tau, 1e-4), conf2mol, dim_size=n_mol)
    return scatter_add(w * s, conf2mol, dim_size=n_mol)


class ConanHead:
    def __init__(self, delta_baseline, emb_standardizer, gate, experts,
                 agg: str = 'learned_softmax', tau: float = 0.593,
                 task: str = 'regression', logit_scale: float = 1.0,
                 device: str = "cuda:0"):
        self.delta_baseline = delta_baseline
        self.emb_standardizer = emb_standardizer
        self.gate = gate
        self.experts = experts
        self.agg = agg
        self.tau = tau
        self.task = task
        self.logit_scale = float(logit_scale)
        self.device = device

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _aggregate_score(self, emb_raw: Tensor, dE: Tensor, conf2mol: Tensor) -> np.ndarray:
        """Per-molecule aggregated score S_hat (numpy)."""
        n_mol = int(conf2mol.max().item()) + 1 if conf2mol.numel() else 0
        emb_std = (self.emb_standardizer.transform(emb_raw)
                   if self.emb_standardizer is not None else emb_raw)
        bins = self.gate.route(dE.cpu().numpy())
        s = predict_per_conf_delta(self.experts, emb_std, bins)
        dE_clamped = torch.tensor(self.gate.clamp(dE.cpu().numpy()), dtype=torch.float32)
        s_t = torch.tensor(s, dtype=torch.float32)
        return aggregate(s_t, dE_clamped, conf2mol.long(), n_mol,
                         agg=self.agg, tau=self.tau).cpu().numpy()

    @torch.no_grad()
    def predict(self, emb_raw: Tensor, dE: Tensor, conf2mol: Tensor,
                smiles_per_mol: List[str]) -> np.ndarray:
        """emb_raw: (C,128) un-standardized; dE,(C,); conf2mol,(C,); per-mol smiles.

        Returns regression y_hat (n_mol,) or classification probability (n_mol,).
        """
        n_mol = len(smiles_per_mol)
        emb_std = (self.emb_standardizer.transform(emb_raw)
                   if self.emb_standardizer is not None else emb_raw)
        bins = self.gate.route(dE.cpu().numpy())
        s = predict_per_conf_delta(self.experts, emb_std, bins)

        dE_clamped = torch.tensor(self.gate.clamp(dE.cpu().numpy()), dtype=torch.float32)
        s_t = torch.tensor(s, dtype=torch.float32)
        s_hat = aggregate(s_t, dE_clamped, conf2mol.long(), n_mol,
                          agg=self.agg, tau=self.tau).cpu().numpy()

        base = self.delta_baseline.predict(smiles_per_mol)   # y_base (reg) | z_base (clf)
        if self.task == 'classification':
            z = base + self.logit_scale * s_hat
            return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        return base + s_hat

    @torch.no_grad()
    def predict_logit(self, emb_raw: Tensor, dE: Tensor, conf2mol: Tensor,
                      smiles_per_mol: List[str]) -> np.ndarray:
        """Classification only: returns the pre-sigmoid logit z = z_base + gamma*S_hat."""
        s_hat = self._aggregate_score(emb_raw, dE, conf2mol)
        base = self.delta_baseline.predict(smiles_per_mol)
        return base + self.logit_scale * s_hat

    # ------------------------------------------------------------------
    def to_state(self) -> dict:
        return {
            "delta_baseline": self.delta_baseline,
            "emb_standardizer_sd": (self.emb_standardizer.state_dict()
                                    if self.emb_standardizer is not None else None),
            "gate": self.gate,
            "experts": [e.to_state() for e in self.experts],
            "agg": self.agg,
            "tau": self.tau,
            "task": self.task,
            "logit_scale": self.logit_scale,
        }