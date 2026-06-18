"""§9 — Curriculum driver. Stage C: chỉ P1 (head warm-up, encoder frozen).

P1: adapter freeze (=0) -> encoder = base. Precompute h, e_pooled, standardize h per-dim
một lần (encoder cố định). eggroll tối ưu CHỈ head trên embedding cố định -> nhanh, N lớn
được. Model selection = best-val. (P2 gentle co-adapt LoRA: thêm ở Stage D.)
"""

from typing import Dict

import torch

from .head import compute_counts, count_diagnostics
from .hooks import pooled_embedding
from .init_warmstart import init_head_warmstart
from .optimizer import Eggroll
from .readout import fit_delta, predict_delta, rmse, rmse_tensor, linear_probe


def _standardize_splits(splits: Dict, device):
    """Standardize per-atom h per-dim bằng stats atom-train; recompute e_pooled từ h chuẩn hoá."""
    h_mean = splits["train"]["h"].mean(dim=0, keepdim=True)
    h_sd = splits["train"]["h"].std(dim=0, keepdim=True).clamp_min(1e-6)
    for s in splits.values():
        s["hs"] = (s["h"] - h_mean) / h_sd
        s["es"] = pooled_embedding(s["hs"], s["batch_idx"], s["num_mols"])
    return splits


def run_curriculum(splits: Dict, eg: Dict, device, log_every: int = 20) -> Dict:
    """Chạy P1 (Stage C). Trả dict kết quả + best head."""
    lam = eg["ridge_lambda"]
    co = eg["counts_only"]
    H = eg["H"]
    seed_train = eg["seed_train"]

    splits = _standardize_splits(splits, device)
    tr, va, te = splits["train"], splits["valid"], splits["test"]

    # T1 floor tham chiếu (trên e_pooled chuẩn hoá)
    floor_test, _ = linear_probe(tr["es"], tr["y"], te["es"], te["y"], lam)
    floor_valid, _ = linear_probe(tr["es"], tr["y"], va["es"], va["y"], lam)
    print(f"[T1 floor std] linear-probe RMSE: valid={floor_valid:.4f} test={floor_test:.4f}")

    # Warm-start head (§7)
    W0, b0 = init_head_warmstart(tr["hs"], tr["es"], tr["y"], H, lam,
                                 fire_rate=0.5, noise=0.01, seed=seed_train)
    diag0 = count_diagnostics(
        compute_counts(W0, b0, tr["hs"], tr["batch_idx"], tr["num_mols"]),
        n_atoms=tr["hs"].shape[0])
    print(f"[warm-start] fire_rate mean={diag0['fire_mean']:.3f} "
          f"dead={diag0['n_dead']} sat={diag0['n_sat']}")

    # --- eval helpers ---
    def _eval_train(theta):
        c = compute_counts(theta["head_W"], theta["head_b"],
                           tr["hs"], tr["batch_idx"], tr["num_mols"])
        model = fit_delta(tr["es"], c, tr["y"], lam, co)
        return rmse_tensor(predict_delta(model, tr["es"], c), tr["y"])

    def _eval_split(W, b, s) -> float:
        c_tr = compute_counts(W, b, tr["hs"], tr["batch_idx"], tr["num_mols"])
        c_s = compute_counts(W, b, s["hs"], s["batch_idx"], s["num_mols"])
        model = fit_delta(tr["es"], c_tr, tr["y"], lam, co)
        return rmse(predict_delta(model, s["es"], c_s), s["y"])

    # --- P1 eggroll ---
    egg = Eggroll(
        params2d={"head_W": W0}, params1d={"head_b": b0},
        sigma2d={"head_W": eg["sigma_head"]}, sigma1d={"head_b": eg["sigma_head"]},
        pop_size=eg["pop_size"], es_lr=eg["es_lr"],
        total_steps=eg["p1_epochs"], gen_seed=seed_train,
    )

    cur = egg.current()
    best_val = _eval_split(cur["head_W"], cur["head_b"], va)
    best = {k: v.clone() for k, v in cur.items()}
    best_step = 0
    print(f"[P1] init val={best_val:.4f}")

    for step in range(1, eg["p1_epochs"] + 1):
        losses = egg.step(_eval_train)
        cur = egg.current()
        val = _eval_split(cur["head_W"], cur["head_b"], va)
        if val < best_val:
            best_val, best_step = val, step
            best = {k: v.clone() for k, v in cur.items()}

        if step % log_every == 0 or step == 1:
            lo = float(losses.min()); me = float(losses.mean()); hi = float(losses.max())
            print(f"[P1] step {step:4d} | train_fit best={lo:.4f} mean={me:.4f} "
                  f"worst={hi:.4f} spread={hi-lo:.4f} | val={val:.4f} "
                  f"best_val={best_val:.4f}@{best_step} | lr={egg.lr:.2e}")

    # --- model selection -> test ---
    Wb, bb = best["head_W"], best["head_b"]
    test_rmse = _eval_split(Wb, bb, te)
    valid_rmse = _eval_split(Wb, bb, va)
    train_rmse = _eval_split(Wb, bb, tr)
    final_diag = count_diagnostics(
        compute_counts(Wb, bb, tr["hs"], tr["batch_idx"], tr["num_mols"]),
        n_atoms=tr["hs"].shape[0])
    print(f"[P1 best @step {best_step}] train={train_rmse:.4f} valid={valid_rmse:.4f} "
          f"test={test_rmse:.4f} | fire_rate={final_diag['fire_mean']:.3f} "
          f"dead={final_diag['n_dead']} sat={final_diag['n_sat']}")

    return {
        "floor_test": floor_test,
        "p1_best_val": best_val,
        "p1_best_step": best_step,
        "test_rmse": test_rmse,
        "valid_rmse": valid_rmse,
        "best_head": best,
    }
