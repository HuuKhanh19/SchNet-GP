"""§9 — Curriculum driver: P1 (head warm-up) -> P2 (gentle co-adapt LoRA).

P1: adapter freeze (=0) -> encoder = base. Precompute h, e_pooled, standardize h per-dim
một lần. eggroll tối ưu CHỈ head trên embedding cố định -> nhanh, N lớn được.
P2: unfreeze {A_l,B_l}. eggroll {adapter+head}, forward per-member qua lora_weights
(σ_adapter nhẹ, phase ngắn). Canary drift mỗi step (linear-probe e_pooled phải giữ ~floor).
Model selection xuyên suốt 2 phase = best-val checkpoint.

Task-aware (§11): cost (lower=better) = RMSE (regression) hoặc 1−AUC (classification).
eggroll minimize cost; selection = min(cost). Hiển thị to_metric(cost) (RMSE / AUC).
"""

from typing import Dict, Tuple

import torch

from .head import compute_counts, count_diagnostics
from .hooks import forward_atom_features, pooled_embedding
from .init_warmstart import init_head_warmstart
from .lora import discover_lora_targets, init_lora_params, build_override, lora_weights
from .diagnostics import canary_probe, adapter_norm, ridge_cond, head_drift, floor_flag
from .optimizer import Eggroll
from .readout import fit_delta, predict_delta, linear_probe
from .metrics import cost, cost_float, to_metric, metric_name

LORA_INIT_STD = 0.01   # std init A (B=0 -> hiệu lực ban đầu = 0)


def _standardize_splits(splits: Dict) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
    """Standardize per-atom h per-dim (stats atom-train); recompute e_pooled từ h chuẩn hoá."""
    h_mean = splits["train"]["h"].mean(dim=0, keepdim=True)
    h_sd = splits["train"]["h"].std(dim=0, keepdim=True).clamp_min(1e-6)
    for s in splits.values():
        s["hs"] = (s["h"] - h_mean) / h_sd
        s["es"] = pooled_embedding(s["hs"], s["batch_idx"], s["num_mols"])
    return splits, h_mean, h_sd


def run_curriculum(model, splits: Dict, eg: Dict, device, log_every: int = 20) -> Dict:
    lam, co, H = eg["ridge_lambda"], eg["counts_only"], eg["H"]
    seed_train = eg["seed_train"]
    r, alpha = eg["lora_r"], eg["lora_alpha"]
    task = eg["task_type"]
    mname = metric_name(task)

    splits, h_mean, h_sd = _standardize_splits(splits)
    tr, va, te = splits["train"], splits["valid"], splits["test"]

    floor_te, _ = linear_probe(tr["es"], tr["y"], te["es"], te["y"], lam, task)
    floor_va, _ = linear_probe(tr["es"], tr["y"], va["es"], va["y"], lam, task)
    print(f"[T1 floor std] linear-probe {mname}: valid={to_metric(floor_va, task):.4f} "
          f"test={to_metric(floor_te, task):.4f}")

    # =====================================================================
    # P1 — head only (precomputed hs/es, encoder frozen)
    # =====================================================================
    def _counts_pre(W, b, s):
        return compute_counts(W, b, s["hs"], s["batch_idx"], s["num_mols"])

    def _eval_pre(W, b, target) -> float:
        c_tr = _counts_pre(W, b, tr)
        c_s = _counts_pre(W, b, target)
        m = fit_delta(tr["es"], c_tr, tr["y"], lam, co)
        return cost_float(predict_delta(m, target["es"], c_s), target["y"], task)

    def _eval_train_pre(theta):
        c = _counts_pre(theta["head_W"], theta["head_b"], tr)
        m = fit_delta(tr["es"], c, tr["y"], lam, co)
        return cost(predict_delta(m, tr["es"], c), tr["y"], task)

    W0, b0 = init_head_warmstart(tr["hs"], tr["es"], tr["y"], H, lam,
                                 fire_rate=0.5, noise=0.01, seed=seed_train)
    W0_init = W0.clone()
    d0 = count_diagnostics(_counts_pre(W0, b0, tr), n_atoms=tr["hs"].shape[0])
    print(f"[warm-start] fire_rate mean={d0['fire_mean']:.3f} dead={d0['n_dead']} "
          f"sat={d0['n_sat']}")

    egg = Eggroll({"head_W": W0}, {"head_b": b0},
                  {"head_W": eg["sigma_head"]}, {"head_b": eg["sigma_head"]},
                  eg["pop_size"], eg["es_lr"], eg["p1_epochs"], gen_seed=seed_train)

    cur = egg.current()
    best_cost = _eval_pre(cur["head_W"], cur["head_b"], va)
    best = {"head_W": cur["head_W"].clone(), "head_b": cur["head_b"].clone(),
            "A": None, "B": None}
    best_step, best_phase = 0, "P1"
    print(f"[P1] init val={to_metric(best_cost, task):.4f}")

    for step in range(1, eg["p1_epochs"] + 1):
        losses = egg.step(_eval_train_pre)
        cur = egg.current()
        vc = _eval_pre(cur["head_W"], cur["head_b"], va)
        if vc < best_cost:
            best_cost, best_step, best_phase = vc, step, "P1"
            best = {"head_W": cur["head_W"].clone(), "head_b": cur["head_b"].clone(),
                    "A": None, "B": None}
        if step % log_every == 0 or step == 1:
            lo, hi = float(losses.min()), float(losses.max())
            print(f"[P1] step {step:4d} | fit best={to_metric(lo, task):.4f} "
                  f"worst={to_metric(hi, task):.4f} spread={abs(hi-lo):.4f} | "
                  f"val={to_metric(vc, task):.4f} "
                  f"best_val={to_metric(best_cost, task):.4f}@{best_step} | lr={egg.lr:.2e}")

    print(f"[P1 done] best_val={to_metric(best_cost, task):.4f}@{best_step} | "
          f"test(best head)={to_metric(_eval_pre(best['head_W'], best['head_b'], te), task):.4f}")

    # =====================================================================
    # P2 — gentle co-adapt (LoRA + head), forward per-member
    # =====================================================================
    eval_full = None
    if eg["p2_epochs"] > 0:
        targets = discover_lora_targets(model)
        A0, B0 = init_lora_params(targets, r, LORA_INIT_STD, seed_train,
                                  device, tr["hs"].dtype)
        params2d = {"head_W": best["head_W"].clone()}
        sigma2d = {"head_W": eg["sigma_head"]}
        for w in targets:
            params2d["A::" + w], params2d["B::" + w] = A0[w], B0[w]
            sigma2d["A::" + w] = sigma2d["B::" + w] = eg["sigma_adapter"]
        params1d = {"head_b": best["head_b"].clone()}
        sigma1d = {"head_b": eg["sigma_head"]}

        egg2 = Eggroll(params2d, params1d, sigma2d, sigma1d,
                       eg["pop_size"], eg["es_lr"], eg["p2_epochs"], gen_seed=seed_train + 1)

        def _split_AB(theta):
            return ({w: theta["A::" + w] for w in targets},
                    {w: theta["B::" + w] for w in targets})

        def _fwd(inputs, override):
            with lora_weights(model, override):
                h, bi, nm = forward_atom_features(model, inputs)
            hs = (h - h_mean) / h_sd
            return hs, pooled_embedding(hs, bi, nm), bi, nm

        def _eval_train_p2(theta):
            A, B = _split_AB(theta)
            ov = build_override(model, A, B, r, alpha)
            hs, e, bi, nm = _fwd(tr["inputs"], ov)
            c = compute_counts(theta["head_W"], theta["head_b"], hs, bi, nm)
            m = fit_delta(e, c, tr["y"], lam, co)
            return cost(predict_delta(m, e, c), tr["y"], task)

        def eval_full(W, b, A, B, target):
            """Forward encoder+adapter cho train+target -> (cost, canary_cost)."""
            ov = build_override(model, A, B, r, alpha)
            hs_tr, e_tr, bi, nm = _fwd(tr["inputs"], ov)
            c_tr = compute_counts(W, b, hs_tr, bi, nm)
            hs_s, e_s, bis, nms = _fwd(target["inputs"], ov)
            c_s = compute_counts(W, b, hs_s, bis, nms)
            m = fit_delta(e_tr, c_tr, tr["y"], lam, co)
            cst = cost_float(predict_delta(m, e_s, c_s), target["y"], task)
            can = canary_probe(e_tr, tr["y"], e_s, target["y"], lam, task)
            return cst, can

        cur = egg2.current()
        A, B = _split_AB(cur)
        v0, can0 = eval_full(cur["head_W"], cur["head_b"], A, B, va)
        print(f"[P2] init val={to_metric(v0, task):.4f} "
              f"canary={to_metric(can0, task):.4f} (floor_valid={to_metric(floor_va, task):.4f})")

        for step in range(1, eg["p2_epochs"] + 1):
            losses = egg2.step(_eval_train_p2)
            cur = egg2.current()
            A, B = _split_AB(cur)
            vc, can = eval_full(cur["head_W"], cur["head_b"], A, B, va)
            if vc < best_cost:
                best_cost, best_step, best_phase = vc, step, "P2"
                best = {"head_W": cur["head_W"].clone(), "head_b": cur["head_b"].clone(),
                        "A": {w: A[w].clone() for w in targets},
                        "B": {w: B[w].clone() for w in targets}}
            if step % log_every == 0 or step == 1:
                lo, hi = float(losses.min()), float(losses.max())
                nrm = adapter_norm(A, B, r, alpha)
                print(f"[P2] step {step:4d} | fit best={to_metric(lo, task):.4f} "
                      f"spread={abs(hi-lo):.4f} | val={to_metric(vc, task):.4f} "
                      f"best_val={to_metric(best_cost, task):.4f}@{best_step}({best_phase}) | "
                      f"canary={to_metric(can, task):.4f} | ‖Δadapter‖={nrm:.3f} | lr={egg2.lr:.2e}")

    # =====================================================================
    # Model selection -> test (branch theo có adapter hay không)
    # =====================================================================
    if best["A"] is not None:
        test_c, _ = eval_full(best["head_W"], best["head_b"], best["A"], best["B"], te)
        valid_c, _ = eval_full(best["head_W"], best["head_b"], best["A"], best["B"], va)
        train_c, _ = eval_full(best["head_W"], best["head_b"], best["A"], best["B"], tr)
        ov = build_override(model, best["A"], best["B"], r, alpha)
        hs_tr, es_tr, bi, nm = _fwd(tr["inputs"], ov)
        c_tr = compute_counts(best["head_W"], best["head_b"], hs_tr, bi, nm)
    else:
        test_c = _eval_pre(best["head_W"], best["head_b"], te)
        valid_c = _eval_pre(best["head_W"], best["head_b"], va)
        train_c = _eval_pre(best["head_W"], best["head_b"], tr)
        es_tr, c_tr = tr["es"], _counts_pre(best["head_W"], best["head_b"], tr)

    print(f"[BEST {best_phase}@step {best_step}] {mname}: train={to_metric(train_c, task):.4f} "
          f"valid={to_metric(valid_c, task):.4f} test={to_metric(test_c, task):.4f}")

    # --- diagnostics cuối (§10) ---
    m_best = fit_delta(es_tr, c_tr, tr["y"], lam, co)
    c_std = (c_tr - m_best["c_mean"]) / m_best["c_sd"]
    dfin = count_diagnostics(c_tr, n_atoms=tr["hs"].shape[0])
    flag = floor_flag(best_cost, va["y"], task)
    print(f"[diag] fire={dfin['fire_mean']:.3f}[{dfin['fire_min']:.3f},{dfin['fire_max']:.3f}] "
          f"dead={dfin['n_dead']} sat={dfin['n_sat']} count_mean={dfin['count_mean']:.2f} | "
          f"head_drift={head_drift(best['head_W'], W0_init):.3f} "
          f"‖coef_c‖={float(m_best['coef_c'].norm()):.3f} ridge_cond={ridge_cond(c_std, lam):.1f}"
          f"{'  FLOOR_FLAG!!' if flag else ''}")

    return {
        "metric_name": mname,
        "floor_test": to_metric(floor_te, task), "floor_valid": to_metric(floor_va, task),
        "best_phase": best_phase, "best_step": best_step,
        "best_val": to_metric(best_cost, task),
        "test_metric": to_metric(test_c, task), "valid_metric": to_metric(valid_c, task),
        "best": best,
    }
