"""
Step 2 Trainer for CONAN-SchNet: frozen SchNet encoder + energy-routed MoE of
MFC (EvoGP) experts + Boltzmann conformer aggregation.

Pipeline (regression only; BACE/classification deferred):
    PHASE 0  extract per-conformer embeddings from the frozen encoder + cache
    PHASE A  delta-learning baseline (RDKit desc + ridge) -> y_base, Delta
    PHASE B  energy gate (quantile bins on train dE)
    PHASE 1  train one MFC expert per energy bin (target = per-conf Delta)
    PHASE 2  freeze experts, fit tau for softmax(-dE/tau) aggregation
    EVAL     aggregate per-conf predictions -> y_hat ; RMSE/MAE
"""

import json
import math
import os
import time
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.data_loader import collate_multi_conformer
from src.models.embedding_extractor import extract_conf_embeddings
from src.models.delta_baseline import DeltaBaseline
from src.models.energy_gate import EnergyGate
from src.models.mfc_expert import MFCExpert
from src.models.conan_head import ConanHead, predict_per_conf_delta, aggregate
from src.utils.standardize import Standardizer
from src.utils.scatter import scatter_add, scatter_softmax


class Step2Trainer:
    def __init__(self, config: Dict[str, Any], device: torch.device, experiment_dir: str):
        self.config = config
        self.device = device
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)

        self.task_type = config['dataset']['task_type']
        if self.task_type != 'regression':
            raise NotImplementedError(
                "Step 2 currently supports regression only (BACE/classification deferred)."
            )
        self.s2 = config['step2']

    # ------------------------------------------------------------------
    def _extract(self, dataset, encoder):
        """Forward a dataset (no shuffle) so conf->mol index aligns with dataset.smiles."""
        loader = DataLoader(
            dataset, batch_size=self.config['training']['batch_size'],
            shuffle=False, collate_fn=collate_multi_conformer,
            num_workers=0, pin_memory=False,
        )
        emb, conf2mol, dE, y = extract_conf_embeddings(
            encoder, loader, self.device, pool=self.s2['emb_pool'])
        return emb, conf2mol, dE, y, list(dataset.smiles)

    # ------------------------------------------------------------------
    def _fit_tau(self, s, dE, conf2mol, y, ybase, gate) -> float:
        agg = self.s2['agg']
        if agg == 'mean':
            return float('inf')
        if agg == 'boltzmann':
            return float(self.s2['tau_init'])

        n_mol = len(y)
        dEc = torch.tensor(gate.clamp(dE), dtype=torch.float32)
        s_t = torch.tensor(np.asarray(s), dtype=torch.float32)
        c2m = conf2mol.long()
        y_t = torch.tensor(np.asarray(y), dtype=torch.float32)
        yb_t = torch.tensor(np.asarray(ybase), dtype=torch.float32)

        ti = max(float(self.s2['tau_init']), 1e-3)
        rho = torch.tensor([math.log(math.expm1(ti))], requires_grad=True)  # softplus^-1(ti)
        opt = torch.optim.Adam([rho], lr=0.05)
        for _ in range(300):
            tau = torch.nn.functional.softplus(rho) + 1e-4
            w = scatter_softmax(-dEc / tau, c2m, dim_size=n_mol)
            dh = scatter_add(w * s_t, c2m, dim_size=n_mol)
            loss = torch.nn.functional.mse_loss(yb_t + dh, y_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        tau = float(torch.nn.functional.softplus(rho).item() + 1e-4)
        print(f"  Phase 2: learned tau = {tau:.3f} kcal/mol  (RT = 0.593)")
        return tau

    # ------------------------------------------------------------------
    def _eval(self, head: ConanHead, emb, dE, conf2mol, smiles, y) -> Dict[str, float]:
        yhat = head.predict(emb, dE, conf2mol, smiles)
        y = np.asarray(y)
        return {
            'rmse': float(np.sqrt(np.mean((yhat - y) ** 2))),
            'mae': float(np.mean(np.abs(yhat - y))),
            'n': int(len(y)),
        }

    # ------------------------------------------------------------------
    def train(self, train_loader, valid_loader, test_loader, encoder) -> Dict[str, Any]:
        t0 = time.time()
        print("\n" + "=" * 70)
        print("STEP 2: frozen SchNet + energy-MoE of MFC experts + Boltzmann agg")
        print("=" * 70)
        self._print_config()

        # ---------- PHASE 0: embeddings ----------
        print("\n[PHASE 0] Extracting frozen-encoder embeddings ...")
        emb_tr, c2m_tr, dE_tr, y_tr, smi_tr = self._extract(train_loader.dataset, encoder)
        emb_va, c2m_va, dE_va, y_va, smi_va = self._extract(valid_loader.dataset, encoder)
        emb_te, c2m_te, dE_te, y_te, smi_te = self._extract(test_loader.dataset, encoder)

        std = Standardizer().fit(emb_tr) if self.s2['standardize_emb'] else None
        emb_tr_s = std.transform(emb_tr) if std else emb_tr

        # ---------- PHASE A: delta baseline ----------
        print("\n[PHASE A] Delta-learning baseline ...")
        kind = self.s2['descriptors'] if self.s2['delta_learning'] else 'none'
        db = DeltaBaseline(kind=kind)
        ybase_tr = db.fit(smi_tr, y_tr.numpy())
        delta_tr_mol = y_tr.numpy() - ybase_tr
        delta_tr_conf = delta_tr_mol[c2m_tr.numpy()]
        print(f"  Delta range (train): [{delta_tr_mol.min():.3f}, {delta_tr_mol.max():.3f}]")

        # ---------- PHASE B: gate ----------
        print("\n[PHASE B] Energy gate ...")
        gate = EnergyGate(
            num_experts=self.s2['gate']['num_experts'],
            energy_clip=self.s2['gate']['energy_clip'],
            binning=self.s2['gate']['binning'],
        ).fit(dE_tr.numpy())
        bin_lines = gate.bin_ranges(dE_tr.numpy())
        for line in bin_lines:
            print("   ", line)
        bins_tr = gate.route(dE_tr.numpy())

        # ---------- PHASE 1: experts ----------
        print("\n[PHASE 1] Training MFC experts ...")
        experts: List[MFCExpert] = []
        for b in range(gate.num_experts):
            idx = np.where(bins_tr == b)[0]
            print(f"  Expert {b}: {idx.size} train confs")
            exp = MFCExpert(self.s2['expert'], device=str(self.device))
            if idx.size > 0:
                Xb = emb_tr_s[idx]
                db_b = torch.tensor(delta_tr_conf[idx], dtype=torch.float32)
                exp.fit(Xb, db_b, gp_seed=self.s2['expert']['gp_seed'],
                        min_samples=self.s2['gate']['min_conf_per_bin'])
            else:
                # empty bin: degenerate ridge on a single zero so predict() returns ~0
                exp._fit_ridge(torch.zeros(2, emb_tr_s.shape[1]), torch.zeros(2))
            experts.append(exp)

        # ---------- PHASE 2: tau ----------
        print("\n[PHASE 2] Fitting aggregation tau ...")
        s_tr = predict_per_conf_delta(experts, emb_tr_s, bins_tr)
        tau = self._fit_tau(s_tr, dE_tr.numpy(), c2m_tr, y_tr.numpy(), ybase_tr, gate)

        head = ConanHead(db, std, gate, experts, agg=self.s2['agg'],
                         tau=tau, device=str(self.device))

        # ---------- EVAL ----------
        print("\n[EVAL]")
        valid_metrics = self._eval(head, emb_va, dE_va, c2m_va, smi_va, y_va.numpy())
        test_metrics = self._eval(head, emb_te, dE_te, c2m_te, smi_te, y_te.numpy())
        print(f"  Valid : RMSE={valid_metrics['rmse']:.4f}  MAE={valid_metrics['mae']:.4f}")
        print(f"  Test  : RMSE={test_metrics['rmse']:.4f}  MAE={test_metrics['mae']:.4f}")

        total_time = time.time() - t0
        print(f"\nStep 2 done in {total_time:.1f}s ({total_time/60:.1f}min)")

        self._save(head, experts, valid_metrics, test_metrics, tau, bin_lines, total_time)
        return {
            'step': 2,
            'tau': tau,
            'valid_metrics': valid_metrics,
            'test_metrics': test_metrics,
            'expert_modes': [e.mode for e in experts],
            'total_time_s': total_time,
        }

    # ------------------------------------------------------------------
    def _save(self, head, experts, valid_metrics, test_metrics, tau, bin_lines, total_time):
        # interpretable constructed-feature expressions
        expr_path = os.path.join(self.experiment_dir, 'cf_expressions.txt')
        with open(expr_path, 'w') as f:
            for b, e in enumerate(experts):
                f.write(f"=== Expert {b} (mode={e.mode}) ===\n")
                for k, ex in enumerate(e.export_expressions()):
                    f.write(f"  CF{k}: {ex}\n")
                f.write("\n")

        # full head (note: experts contain pickled evogp forests; loads where evogp is installed)
        torch.save(head.to_state(), os.path.join(self.experiment_dir, 'conan_head.pt'))

        results = {
            'step': 2,
            'tau': tau,
            'expert_modes': [e.mode for e in experts],
            'bin_ranges': bin_lines,
            'valid_metrics': valid_metrics,
            'test_metrics': test_metrics,
            'total_time_s': total_time,
            'config': self.config,
        }
        with open(os.path.join(self.experiment_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved results + head + CF expressions to: {self.experiment_dir}")

    # ------------------------------------------------------------------
    def _print_config(self):
        s2 = self.s2
        print(f"  emb_pool={s2['emb_pool']}  standardize={s2['standardize_emb']}  "
              f"delta_learning={s2['delta_learning']} ({s2['descriptors']})")
        print(f"  gate: num_experts={s2['gate']['num_experts']}  "
              f"binning={s2['gate']['binning']}  clip={s2['gate']['energy_clip']}")
        e = s2['expert']
        print(f"  expert: q={e['q']} n_best={e['n_best']} pop={e['pop_size']} "
              f"gens={e['generation_limit']} depth={e['max_layer_cnt']} seed={e['gp_seed']}")
        print(f"  agg={s2['agg']}  tau_init={s2['tau_init']}")