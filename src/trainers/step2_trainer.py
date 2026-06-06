"""
Step 2 Trainer for CONAN-SchNet: frozen SchNet encoder + energy-routed MoE of
MFC (EvoGP) experts + Boltzmann conformer aggregation.

Supports BOTH regression and binary classification through one pipeline.

    PHASE 0  extract per-conformer embeddings from the frozen encoder + cache
    PHASE A  baseline:
               regression     -> y_base = ridge(desc) ;  target = Delta = y - y_base
               classification -> z_base = logit(desc)  ;  target = r = y - sigmoid(z_base)
                                 (functional-gradient pseudo-residual; GBM-style)
    PHASE B  energy gate (quantile bins on train dE)
    PHASE 1  train one MFC expert per energy bin (per-conf target = Delta or r)
    PHASE 2  freeze experts, fit aggregation:
               regression     -> tau for softmax(-dE/tau)
               classification -> tau (if learned_softmax) AND a global logit-scale
                                 gamma, both under BCE-with-logits
    EVAL     regression -> RMSE / MAE ;  classification -> AUC / ACC / logloss
                p = sigmoid(z_base + gamma * S_hat)

The GP feature-construction, the q-feature greedy decorrelation, the ridge
combiner and the energy gate are IDENTICAL across tasks: classification simply
swaps the per-conformer target (pseudo-residual), the final activation (sigmoid),
the Phase-2 loss (BCE) and the metric (AUC). This keeps the conformer-energy
physics and the interpretable CF expressions intact for both tasks.

Logging additions (vs the original "single-shot" version):
    * PHASE 1  : per-expert in-sample fit RMSE(target) + Pearson r on its bin.
    * after P1 : a train+val checkpoint at the provisional tau (RMSE or AUC).
    * PHASE 2  : train AND val metric printed every `tau_log_every` Adam steps
                 (val is logging-only, it never enters the loss).
    * EVAL     : train metrics printed next to valid/test.
    * everything is also appended to self.history and dumped to history.json.
"""

import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, log_loss

from src.data.data_loader import collate_multi_conformer
from src.models.embedding_extractor import extract_conf_embeddings
from src.models.delta_baseline import DeltaBaseline
from src.models.energy_gate import EnergyGate
from src.models.mfc_expert import MFCExpert
from src.models.conan_head import ConanHead, predict_per_conf_delta, aggregate
from src.utils.standardize import Standardizer
from src.utils.scatter import scatter_add, scatter_mean, scatter_softmax


def _sigmoid_np(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=np.float64), -30.0, 30.0)))


class Step2Trainer:
    def __init__(self, config: Dict[str, Any], device: torch.device, experiment_dir: str):
        self.config = config
        self.device = device
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)

        self.task_type = config['dataset']['task_type']
        self.is_cls = (self.task_type == 'classification')
        if self.task_type not in ('regression', 'classification'):
            raise NotImplementedError(
                f"Step 2 supports regression and classification, got '{self.task_type}'."
            )
        self.s2 = config['step2']
        self.logit_scale_init = float(self.s2.get('logit_scale_init', 1.0))
        # per-phase / per-step train+val log (dumped to history.json in _save)
        self.history: List[Dict[str, Any]] = []

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
    @staticmethod
    def _auc_safe(y: np.ndarray, score: np.ndarray) -> float:
        """ROC-AUC that won't crash on a single-class slice (returns nan instead)."""
        y = np.asarray(y)
        try:
            return float(roc_auc_score(y, np.asarray(score)))
        except ValueError:
            return float('nan')

    # ------------------------------------------------------------------
    @staticmethod
    def _rmse_at_tau(tau_val, s_t, dEc, c2m, yb_t, y_t, n_mol) -> float:
        """End-to-end RMSE (y_base + aggregated Delta vs y) at a given tau.

        Cheap: reuses already-computed per-conf Delta `s_t` and baseline `yb_t`;
        only the softmax weights depend on tau, so this is fine to call inside
        the Phase-2 loop for both train and val.
        """
        w = scatter_softmax(-dEc / max(float(tau_val), 1e-4), c2m, dim_size=n_mol)
        dh = scatter_add(w * s_t, c2m, dim_size=n_mol)
        return float(torch.sqrt(torch.mean((yb_t + dh - y_t) ** 2)))

    # ------------------------------------------------------------------
    def _fit_tau(self, s, dE, conf2mol, y, ybase, gate,
                 val_pack: Optional[Tuple] = None) -> float:
        """Fit aggregation temperature tau (Phase 2, REGRESSION).

        val_pack, if given, is (s_va, dE_va, conf2mol_va, y_va, ybase_va). When
        present, train AND val RMSE are logged every `tau_log_every` Adam steps
        so the aggregation can be watched converging instead of only seeing the
        final number. The validation tensors are used for LOGGING ONLY.
        """
        agg = self.s2['agg']
        if agg == 'mean':
            return float('inf')
        if agg == 'boltzmann':
            return float(self.s2['tau_init'])

        # ---- train tensors (these define the loss) ----
        n_mol = len(y)
        dEc = torch.tensor(gate.clamp(dE), dtype=torch.float32)
        s_t = torch.tensor(np.asarray(s), dtype=torch.float32)
        c2m = conf2mol.long() if torch.is_tensor(conf2mol) else torch.as_tensor(conf2mol).long()
        y_t = torch.tensor(np.asarray(y), dtype=torch.float32)
        yb_t = torch.tensor(np.asarray(ybase), dtype=torch.float32)

        # ---- optional val tensors (logging only) ----
        have_val = val_pack is not None
        if have_val:
            s_va, dE_va, c2m_va, y_va, yb_va = val_pack
            n_mol_va = len(y_va)
            dEc_va = torch.tensor(gate.clamp(np.asarray(dE_va)), dtype=torch.float32)
            s_va_t = torch.tensor(np.asarray(s_va), dtype=torch.float32)
            c2m_va_t = (c2m_va.long() if torch.is_tensor(c2m_va)
                        else torch.as_tensor(c2m_va).long())
            y_va_t = torch.tensor(np.asarray(y_va), dtype=torch.float32)
            yb_va_t = torch.tensor(np.asarray(yb_va), dtype=torch.float32)

        n_steps = int(self.s2.get('tau_steps', 300))
        log_every = int(self.s2.get('tau_log_every', 50))

        ti = max(float(self.s2['tau_init']), 1e-3)
        rho = torch.tensor([math.log(math.expm1(ti))], requires_grad=True)  # softplus^-1(ti)
        opt = torch.optim.Adam([rho], lr=0.05)

        header = f"  {'step':>5s} | {'tau':>7s} | {'train_rmse':>10s}"
        if have_val:
            header += f" | {'val_rmse':>8s}"
        print(header)

        for step in range(1, n_steps + 1):
            tau = torch.nn.functional.softplus(rho) + 1e-4
            w = scatter_softmax(-dEc / tau, c2m, dim_size=n_mol)
            dh = scatter_add(w * s_t, c2m, dim_size=n_mol)
            loss = torch.nn.functional.mse_loss(yb_t + dh, y_t)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step == 1 or step % log_every == 0 or step == n_steps:
                with torch.no_grad():
                    tau_now = float(torch.nn.functional.softplus(rho).item() + 1e-4)
                    tr_rmse = self._rmse_at_tau(tau_now, s_t, dEc, c2m, yb_t, y_t, n_mol)
                    rec: Dict[str, Any] = {'phase': 'tau', 'step': step,
                                           'tau': tau_now, 'train_rmse': tr_rmse}
                    line = f"  {step:5d} | {tau_now:7.3f} | {tr_rmse:10.4f}"
                    if have_val:
                        va_rmse = self._rmse_at_tau(
                            tau_now, s_va_t, dEc_va, c2m_va_t, yb_va_t, y_va_t, n_mol_va)
                        rec['val_rmse'] = va_rmse
                        line += f" | {va_rmse:8.4f}"
                    self.history.append(rec)
                    print(line)

        tau = float(torch.nn.functional.softplus(rho).item() + 1e-4)
        print(f"  Phase 2: learned tau = {tau:.3f} kcal/mol  (RT = 0.593)")
        return tau

    # ------------------------------------------------------------------
    def _fit_agg_cls(self, s, dE, conf2mol, y, zbase, gate,
                     val_pack: Optional[Tuple] = None) -> Tuple[float, float]:
        """Fit aggregation for CLASSIFICATION (Phase 2).

        Optimises the global logit-scale gamma (always) and -- only when
        agg='learned_softmax' -- the temperature tau, under BCE-with-logits on
        z = z_base + gamma * S_hat. For agg in {mean, boltzmann} the conformer
        weights are fixed and only gamma is fitted (calibration). Returns
        (tau, gamma); tau = +inf for mean, tau_init for boltzmann.

        val_pack (logging only) = (s_va, dE_va, conf2mol_va, y_va, zbase_va).
        Train+val BCE and AUC are logged every `tau_log_every` steps.
        """
        agg = self.s2['agg']
        learn_tau = (agg == 'learned_softmax')
        fixed_tau = float(self.s2['tau_init'])

        n_mol = len(y)
        dEc = torch.tensor(gate.clamp(dE), dtype=torch.float32)
        s_t = torch.tensor(np.asarray(s), dtype=torch.float32)
        c2m = conf2mol.long() if torch.is_tensor(conf2mol) else torch.as_tensor(conf2mol).long()
        y_t = torch.tensor(np.asarray(y), dtype=torch.float32)
        zb_t = torch.tensor(np.asarray(zbase), dtype=torch.float32)

        have_val = val_pack is not None
        if have_val:
            s_va, dE_va, c2m_va, y_va, zb_va = val_pack
            n_mol_va = len(y_va)
            dEc_va = torch.tensor(gate.clamp(np.asarray(dE_va)), dtype=torch.float32)
            s_va_t = torch.tensor(np.asarray(s_va), dtype=torch.float32)
            c2m_va_t = (c2m_va.long() if torch.is_tensor(c2m_va)
                        else torch.as_tensor(c2m_va).long())
            y_va_t = torch.tensor(np.asarray(y_va), dtype=torch.float32)
            zb_va_t = torch.tensor(np.asarray(zb_va), dtype=torch.float32)
            y_va_np = np.asarray(y_va)

        n_steps = int(self.s2.get('tau_steps', 300))
        log_every = int(self.s2.get('tau_log_every', 50))

        gamma = torch.tensor([self.logit_scale_init], requires_grad=True)
        params = [gamma]
        rho = None
        if learn_tau:
            ti = max(fixed_tau, 1e-3)
            rho = torch.tensor([math.log(math.expm1(ti))], requires_grad=True)
            params.append(rho)
        opt = torch.optim.Adam(params, lr=0.05)
        bce = torch.nn.functional.binary_cross_entropy_with_logits

        def _tau_value() -> float:
            if learn_tau:
                return float(torch.nn.functional.softplus(rho).item() + 1e-4)
            return float('inf') if agg == 'mean' else fixed_tau

        def _dh(s_tensor, dEc_tensor, c2m_tensor, nm) -> torch.Tensor:
            if agg == 'mean':
                return scatter_mean(s_tensor, c2m_tensor, dim_size=nm)
            if learn_tau:
                tau = torch.nn.functional.softplus(rho) + 1e-4
            else:
                tau = torch.tensor(fixed_tau)
            w = scatter_softmax(-dEc_tensor / tau, c2m_tensor, dim_size=nm)
            return scatter_add(w * s_tensor, c2m_tensor, dim_size=nm)

        def _bce_auc(s_tensor, dEc_tensor, c2m_tensor, y_tensor, zb_tensor, nm, y_np):
            with torch.no_grad():
                dh = _dh(s_tensor, dEc_tensor, c2m_tensor, nm)
                z = zb_tensor + gamma * dh
                loss_v = float(bce(z, y_tensor).item())
                auc_v = self._auc_safe(y_np, z.cpu().numpy())
            return loss_v, auc_v

        y_np = np.asarray(y)
        header = f"  {'step':>5s} | {'tau':>7s} | {'gamma':>6s} | {'tr_bce':>7s} | {'tr_auc':>7s}"
        if have_val:
            header += f" | {'va_bce':>7s} | {'va_auc':>7s}"
        print(header)

        for step in range(1, n_steps + 1):
            dh = _dh(s_t, dEc, c2m, n_mol)
            z = zb_t + gamma * dh
            loss = bce(z, y_t)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step == 1 or step % log_every == 0 or step == n_steps:
                tau_now = _tau_value()
                g_now = float(gamma.item())
                tr_bce, tr_auc = _bce_auc(s_t, dEc, c2m, y_t, zb_t, n_mol, y_np)
                rec: Dict[str, Any] = {'phase': 'agg_cls', 'step': step,
                                       'tau': tau_now, 'gamma': g_now,
                                       'train_bce': tr_bce, 'train_auc': tr_auc}
                tau_str = "inf" if math.isinf(tau_now) else f"{tau_now:7.3f}"
                line = f"  {step:5d} | {tau_str:>7s} | {g_now:6.3f} | {tr_bce:7.4f} | {tr_auc:7.4f}"
                if have_val:
                    va_bce, va_auc = _bce_auc(
                        s_va_t, dEc_va, c2m_va_t, y_va_t, zb_va_t, n_mol_va, y_va_np)
                    rec['val_bce'] = va_bce
                    rec['val_auc'] = va_auc
                    line += f" | {va_bce:7.4f} | {va_auc:7.4f}"
                self.history.append(rec)
                print(line)

        tau_final = _tau_value()
        gamma_final = float(gamma.item())
        tau_disp = "inf" if math.isinf(tau_final) else f"{tau_final:.3f}"
        print(f"  Phase 2: learned gamma = {gamma_final:.3f}  | tau = {tau_disp} "
              f"(RT = 0.593)")
        return tau_final, gamma_final

    # ------------------------------------------------------------------
    def _eval(self, head: ConanHead, emb, dE, conf2mol, smiles, y) -> Dict[str, float]:
        out = head.predict(emb, dE, conf2mol, smiles)
        y = np.asarray(y)
        if self.is_cls:
            p = np.asarray(out, dtype=np.float64)
            auc = self._auc_safe(y, p)
            acc = float(np.mean(((p > 0.5).astype(np.float64) == y).astype(np.float64)))
            try:
                ll = float(log_loss(y, np.clip(p, 1e-7, 1.0 - 1e-7), labels=[0, 1]))
            except ValueError:
                ll = float('nan')
            return {'auc': auc, 'acc': acc, 'logloss': ll, 'n': int(len(y))}
        return {
            'rmse': float(np.sqrt(np.mean((out - y) ** 2))),
            'mae': float(np.mean(np.abs(out - y))),
            'n': int(len(y)),
        }

    # ------------------------------------------------------------------
    def train(self, train_loader, valid_loader, test_loader, encoder) -> Dict[str, Any]:
        t0 = time.time()
        print("\n" + "=" * 70)
        print(f"STEP 2 [{self.task_type}]: frozen SchNet + energy-MoE of MFC experts "
              "+ Boltzmann agg")
        print("=" * 70)
        self._print_config()

        # ---------- PHASE 0: embeddings ----------
        print("\n[PHASE 0] Extracting frozen-encoder embeddings ...")
        emb_tr, c2m_tr, dE_tr, y_tr, smi_tr = self._extract(train_loader.dataset, encoder)
        emb_va, c2m_va, dE_va, y_va, smi_va = self._extract(valid_loader.dataset, encoder)
        emb_te, c2m_te, dE_te, y_te, smi_te = self._extract(test_loader.dataset, encoder)

        std = Standardizer().fit(emb_tr) if self.s2['standardize_emb'] else None
        emb_tr_s = std.transform(emb_tr) if std else emb_tr

        # ---------- PHASE A: baseline ----------
        print("\n[PHASE A] Baseline ...")
        kind = self.s2['descriptors'] if self.s2['delta_learning'] else 'none'
        db = DeltaBaseline(kind=kind, task=self.task_type)
        base_tr = db.fit(smi_tr, y_tr.numpy())               # y_base (reg) | z_base (clf)

        if self.is_cls:
            pbase_tr = _sigmoid_np(base_tr)
            target_tr_mol = y_tr.numpy() - pbase_tr           # functional-gradient pseudo-residual
            base_auc = self._auc_safe(y_tr.numpy(), base_tr)
            print(f"  Pseudo-residual range (train): "
                  f"[{target_tr_mol.min():.3f}, {target_tr_mol.max():.3f}]  "
                  f"| baseline train AUC = {base_auc:.4f}")
        else:
            target_tr_mol = y_tr.numpy() - base_tr            # Delta
            print(f"  Delta range (train): [{target_tr_mol.min():.3f}, {target_tr_mol.max():.3f}]")
        target_tr_conf = target_tr_mol[c2m_tr.numpy()]

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
                tgt_b = target_tr_conf[idx]
                db_b = torch.tensor(tgt_b, dtype=torch.float32)
                exp.fit(Xb, db_b, gp_seed=self.s2['expert']['gp_seed'],
                        min_samples=self.s2['gate']['min_conf_per_bin'])

                # in-sample fit quality on this bin (so PHASE 1 isn't a black box)
                pred_b = np.asarray(exp.predict(Xb))
                rmse_b = float(np.sqrt(np.mean((pred_b - tgt_b) ** 2)))
                if pred_b.std() > 1e-12 and tgt_b.std() > 1e-12:
                    r_b = float(np.corrcoef(pred_b, tgt_b)[0, 1])
                else:
                    r_b = float('nan')
                print(f"    -> mode={exp.mode}  train_rmse(target)={rmse_b:.4f}  "
                      f"pearson_r={r_b:.3f}")
                self.history.append({'phase': 'expert', 'bin': int(b),
                                     'n': int(idx.size), 'mode': exp.mode,
                                     'train_rmse_target': rmse_b,
                                     'train_pearson_r': r_b})
            else:
                # empty bin: degenerate ridge on a single zero so predict() returns ~0
                exp._fit_ridge(torch.zeros(2, emb_tr_s.shape[1]), torch.zeros(2))
                self.history.append({'phase': 'expert', 'bin': int(b), 'n': 0,
                                     'mode': exp.mode})
            experts.append(exp)

        # per-conf scores (do NOT change with tau/gamma -> compute once)
        s_tr = predict_per_conf_delta(experts, emb_tr_s, bins_tr)

        emb_va_s = std.transform(emb_va) if std else emb_va
        bins_va = gate.route(dE_va.numpy())
        s_va = predict_per_conf_delta(experts, emb_va_s, bins_va)
        base_va = db.predict(smi_va)                          # y_base (reg) | z_base (clf)

        # ---------- checkpoint: where do we stand BEFORE tuning the aggregation? ----------
        prov_tau = float(self.s2['tau_init'])
        prov_head = ConanHead(db, std, gate, experts, agg=self.s2['agg'],
                              tau=prov_tau, task=self.task_type,
                              logit_scale=self.logit_scale_init, device=str(self.device))
        ck_tr = self._eval(prov_head, emb_tr, dE_tr, c2m_tr, smi_tr, y_tr.numpy())
        ck_va = self._eval(prov_head, emb_va, dE_va, c2m_va, smi_va, y_va.numpy())
        if self.is_cls:
            print(f"  [after PHASE 1 @ agg={self.s2['agg']}, tau={prov_tau:.3f}, "
                  f"gamma={self.logit_scale_init:.3f}] "
                  f"train_auc={ck_tr['auc']:.4f}  val_auc={ck_va['auc']:.4f}")
            self.history.append({'phase': 'after_experts', 'tau': prov_tau,
                                 'gamma': self.logit_scale_init,
                                 'train_auc': ck_tr['auc'], 'val_auc': ck_va['auc']})
        else:
            print(f"  [after PHASE 1 @ agg={self.s2['agg']}, tau={prov_tau:.3f}] "
                  f"train_rmse={ck_tr['rmse']:.4f}  val_rmse={ck_va['rmse']:.4f}")
            self.history.append({'phase': 'after_experts', 'tau': prov_tau,
                                 'train_rmse': ck_tr['rmse'], 'val_rmse': ck_va['rmse']})

        # ---------- PHASE 2: aggregation ----------
        print("\n[PHASE 2] Fitting aggregation ...")
        if self.is_cls:
            val_pack = (s_va, dE_va.numpy(), c2m_va, y_va.numpy(), base_va)
            tau, gamma = self._fit_agg_cls(s_tr, dE_tr.numpy(), c2m_tr, y_tr.numpy(),
                                           base_tr, gate, val_pack=val_pack)
        else:
            val_pack = (s_va, dE_va.numpy(), c2m_va, y_va.numpy(), base_va)
            tau = self._fit_tau(s_tr, dE_tr.numpy(), c2m_tr, y_tr.numpy(), base_tr,
                                gate, val_pack=val_pack)
            gamma = 1.0

        head = ConanHead(db, std, gate, experts, agg=self.s2['agg'],
                         tau=tau, task=self.task_type, logit_scale=gamma,
                         device=str(self.device))

        # ---------- EVAL ----------
        print("\n[EVAL]")
        train_metrics = self._eval(head, emb_tr, dE_tr, c2m_tr, smi_tr, y_tr.numpy())
        valid_metrics = self._eval(head, emb_va, dE_va, c2m_va, smi_va, y_va.numpy())
        test_metrics = self._eval(head, emb_te, dE_te, c2m_te, smi_te, y_te.numpy())

        if self.is_cls:
            zbase_te = db.predict(smi_te)
            y_te_np = y_te.numpy()
            base_te_auc = self._auc_safe(y_te_np, zbase_te)
            print(f"  [decomp] baseline-alone AUC = {base_te_auc:.4f}"
                  f"  | full Step2 AUC = {test_metrics['auc']:.4f}")
            print(f"  Train : AUC={train_metrics['auc']:.4f}  ACC={train_metrics['acc']:.4f}")
            print(f"  Valid : AUC={valid_metrics['auc']:.4f}  ACC={valid_metrics['acc']:.4f}")
            print(f"  Test  : AUC={test_metrics['auc']:.4f}  ACC={test_metrics['acc']:.4f}"
                  f"  logloss={test_metrics['logloss']:.4f}")
            self.history.append({'phase': 'eval', 'tau': tau, 'gamma': gamma,
                                 'train_auc': train_metrics['auc'],
                                 'val_auc': valid_metrics['auc'],
                                 'test_auc': test_metrics['auc'],
                                 'train_acc': train_metrics['acc'],
                                 'val_acc': valid_metrics['acc'],
                                 'test_acc': test_metrics['acc'],
                                 'test_logloss': test_metrics['logloss']})
        else:
            ybase_te = db.predict(smi_te)
            y_te_np = y_te.numpy()
            base_rmse = float(np.sqrt(np.mean((ybase_te - y_te_np) ** 2)))
            mean_floor = float(np.sqrt(np.mean((y_tr.numpy().mean() - y_te_np) ** 2)))
            print(f"  [decomp] train-mean floor = {mean_floor:.4f}"
                  f"  | 2D-baseline-alone = {base_rmse:.4f}"
                  f"  | full Step2 = {test_metrics['rmse']:.4f}")
            print(f"  Train : RMSE={train_metrics['rmse']:.4f}  MAE={train_metrics['mae']:.4f}")
            print(f"  Valid : RMSE={valid_metrics['rmse']:.4f}  MAE={valid_metrics['mae']:.4f}")
            print(f"  Test  : RMSE={test_metrics['rmse']:.4f}  MAE={test_metrics['mae']:.4f}")
            self.history.append({'phase': 'eval', 'tau': tau,
                                 'train_rmse': train_metrics['rmse'],
                                 'val_rmse': valid_metrics['rmse'],
                                 'test_rmse': test_metrics['rmse'],
                                 'train_mae': train_metrics['mae'],
                                 'val_mae': valid_metrics['mae'],
                                 'test_mae': test_metrics['mae']})

        total_time = time.time() - t0
        print(f"\nStep 2 done in {total_time:.1f}s ({total_time/60:.1f}min)")

        self._save(head, experts, train_metrics, valid_metrics, test_metrics,
                   tau, gamma, bin_lines, total_time)
        return {
            'step': 2,
            'task': self.task_type,
            'tau': tau,
            'gamma': gamma,
            'train_metrics': train_metrics,
            'valid_metrics': valid_metrics,
            'test_metrics': test_metrics,
            'expert_modes': [e.mode for e in experts],
            'history': self.history,
            'total_time_s': total_time,
        }

    # ------------------------------------------------------------------
    def _save(self, head, experts, train_metrics, valid_metrics, test_metrics,
              tau, gamma, bin_lines, total_time):
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

        # per-phase / per-step train+val log
        with open(os.path.join(self.experiment_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

        results = {
            'step': 2,
            'task': self.task_type,
            'tau': tau,
            'gamma': gamma,
            'expert_modes': [e.mode for e in experts],
            'bin_ranges': bin_lines,
            'train_metrics': train_metrics,
            'valid_metrics': valid_metrics,
            'test_metrics': test_metrics,
            'history': self.history,
            'total_time_s': total_time,
            'config': self.config,
        }
        with open(os.path.join(self.experiment_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved results + head + CF expressions + history to: {self.experiment_dir}")

    # ------------------------------------------------------------------
    def _print_config(self):
        s2 = self.s2
        print(f"  task={self.task_type}  metric={'auc' if self.is_cls else 'rmse'}")
        print(f"  emb_pool={s2['emb_pool']}  standardize={s2['standardize_emb']}  "
              f"delta_learning={s2['delta_learning']} ({s2['descriptors']})")
        print(f"  gate: num_experts={s2['gate']['num_experts']}  "
              f"binning={s2['gate']['binning']}  clip={s2['gate']['energy_clip']}")
        e = s2['expert']
        print(f"  expert: q={e['q']} n_best={e['n_best']} pop={e['pop_size']} "
              f"gens={e['generation_limit']} depth={e['max_layer_cnt']} seed={e['gp_seed']}")
        if self.is_cls:
            print(f"  agg={s2['agg']}  tau_init={s2['tau_init']}  "
                  f"logit_scale_init={self.logit_scale_init}")
        else:
            print(f"  agg={s2['agg']}  tau_init={s2['tau_init']}")