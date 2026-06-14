#!/usr/bin/env python
"""Exp 0 — Baseline 2D descriptor (reference RMSE).

Mục tiêu: đo phần phương sai mà RDKit 2D descriptor nắm được, và sinh prediction
baseline cho MỌI phân tử / MỌI split để Exp 1 & 2 dùng làm residual (đọc lại từ đĩa,
KHÔNG tính lại baseline mỗi thí nghiệm -> tránh lệch).

Hai model (chạy cả hai):
  - Ridge:  StandardScaler fit train, alpha grid logspace(-3,3,13), chọn theo val RMSE.
  - GBT:    HistGradientBoostingRegressor, tune learning_rate ∈ {0.03,0.05,0.1} và
            max_leaf_nodes, n_estimators (early stopping trên VAL). Trees không scale.

Protocol: mỗi split trong 5 (seed-split 0..4): fit train, chọn HP trên val, report
test RMSE (log-S). So mốc baseline 0.8994 ± 0.0946.

Output (experiments/exp0_baseline2d/<ds>/<split>/seed_<seed>/):
  - ridge_pred.csv, gbt_pred.csv : cột smiles,target,split,pred,is_oof.
      * train  -> pred = OUT-OF-FOLD (KFold trên train, không rò rỉ) -> Exp 1 dùng
                  cho residual train. is_oof=True.
      * valid/test -> pred = model fit trên FULL train. is_oof=False.
  - metrics.json : RMSE + hyperparam đã chọn + danh sách feature.
Tổng hợp: experiments/exp0_baseline2d/<ds>/<split>/summary.json + in mean±std/per-split.

Chỉ cần CPU + sklearn + rdkit (không cần GPU). Ví dụ:
    python scripts/run_exp0.py --dataset esol --seed-split 0 1 2 3 4
"""

import argparse
import json
import os
import statistics
import sys

import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config import DATASETS
from src.data.data_loader import prepare_dataset, save_splits
from src.baselines.descriptors2d import Descriptor2DFeaturizer

BASELINE_REF = "0.8994 ± 0.0946"
N_OOF_FOLDS = 5


# =============================================================================
# Tiện ích
# =============================================================================

def rmse(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def get_splits(dataset, split_method, seed, raw_dir, processed_dir):
    """Lấy train/valid/test ĐÚNG như SchNet/GP (chia sẻ cache để residual khớp).

    Đọc cache .../<ds>/<split_method>/seed_<seed>/{train,valid,test}.csv nếu có;
    nếu chưa có thì prepare_dataset (chỉ cần RDKit) rồi lưu lại cache đó.
    """
    ds_dir = os.path.join(processed_dir, dataset, split_method, f"seed_{seed}")
    paths = {s: os.path.join(ds_dir, f"{s}.csv") for s in ("train", "valid", "test")}
    if all(os.path.exists(p) for p in paths.values()):
        print(f"  Dùng cache split: {ds_dir}")
        return (pd.read_csv(paths["train"]), pd.read_csv(paths["valid"]),
                pd.read_csv(paths["test"]))

    print(f"  Chưa có cache split, tạo mới tại {ds_dir}")
    config = {
        "dataset": DATASETS[dataset],
        "data": {
            "raw_dir": raw_dir, "processed_dir": processed_dir,
            "split_method": split_method, "random_seed_split": seed,
        },
    }
    train_df, valid_df, test_df = prepare_dataset(config)
    save_splits(train_df, valid_df, test_df, ds_dir)
    return train_df, valid_df, test_df


# =============================================================================
# Ridge
# =============================================================================

def run_ridge(X_tr, y_tr, X_va, y_va, X_te, seed):
    """StandardScaler + Ridge; chọn alpha theo val RMSE. Trả về preds + info."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold

    alphas = np.logspace(-3, 3, 13)

    def fit_pred(Xt, yt, Xs, alpha):
        sc = StandardScaler().fit(Xt)
        m = Ridge(alpha=alpha).fit(sc.transform(Xt), yt)
        return m.predict(sc.transform(Xs))

    best_alpha, best_val = None, float("inf")
    for a in alphas:
        r = rmse(y_va, fit_pred(X_tr, y_tr, X_va, a))
        if r < best_val:
            best_val, best_alpha = r, a

    # Full-train model -> valid/test.
    pred_va = fit_pred(X_tr, y_tr, X_va, best_alpha)
    pred_te = fit_pred(X_tr, y_tr, X_te, best_alpha)

    # OOF train (KFold, alpha đã chọn) -> residual không rò rỉ cho Exp 1.
    oof = np.zeros_like(y_tr, dtype=np.float64)
    kf = KFold(n_splits=N_OOF_FOLDS, shuffle=True, random_state=seed)
    for tr_idx, va_idx in kf.split(X_tr):
        oof[va_idx] = fit_pred(X_tr[tr_idx], y_tr[tr_idx], X_tr[va_idx], best_alpha)

    info = {"alpha": float(best_alpha), "val_rmse": float(best_val)}
    return oof, pred_va, pred_te, info


# =============================================================================
# GBT (HistGradientBoostingRegressor)
# =============================================================================

def _gbt_es(X_tr, y_tr, X_va, y_va, lr, mln, seed, step=25, max_iter=1000, patience=8):
    """Fit HGB tăng dần (warm_start), early stopping trên VAL. Trả về (best_val, best_n)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(
        learning_rate=lr, max_leaf_nodes=mln, max_iter=step,
        warm_start=True, early_stopping=False, random_state=seed,
    )
    best_val, best_n, no_improve, n = float("inf"), step, 0, step
    while n <= max_iter:
        m.max_iter = n
        m.fit(X_tr, y_tr)
        r = rmse(y_va, m.predict(X_va))
        if r < best_val - 1e-6:
            best_val, best_n, no_improve = r, n, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
        n += step
    return best_val, best_n


def _gbt_fit_pred(X_tr, y_tr, Xs, lr, mln, n_iter, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(
        learning_rate=lr, max_leaf_nodes=mln, max_iter=n_iter,
        early_stopping=False, random_state=seed,
    ).fit(X_tr, y_tr)
    return m.predict(Xs)


def run_gbt(X_tr, y_tr, X_va, y_va, X_te, seed):
    """Tune (lr, max_leaf_nodes, n_estimators) trên val; preds + info."""
    from sklearn.model_selection import KFold

    lr_grid = [0.03, 0.05, 0.1]
    mln_grid = [15, 31, 63]

    best = {"val": float("inf"), "lr": None, "mln": None, "n": None}
    for lr in lr_grid:
        for mln in mln_grid:
            val_r, n_it = _gbt_es(X_tr, y_tr, X_va, y_va, lr, mln, seed)
            if val_r < best["val"]:
                best = {"val": val_r, "lr": lr, "mln": mln, "n": n_it}

    lr, mln, n_it = best["lr"], best["mln"], best["n"]
    pred_va = _gbt_fit_pred(X_tr, y_tr, X_va, lr, mln, n_it, seed)
    pred_te = _gbt_fit_pred(X_tr, y_tr, X_te, lr, mln, n_it, seed)

    # OOF train với HP đã chọn.
    oof = np.zeros_like(y_tr, dtype=np.float64)
    kf = KFold(n_splits=N_OOF_FOLDS, shuffle=True, random_state=seed)
    for tr_idx, va_idx in kf.split(X_tr):
        oof[va_idx] = _gbt_fit_pred(
            X_tr[tr_idx], y_tr[tr_idx], X_tr[va_idx], lr, mln, n_it, seed)

    info = {"learning_rate": lr, "max_leaf_nodes": mln,
            "n_estimators": int(n_it), "val_rmse": float(best["val"])}
    return oof, pred_va, pred_te, info


# =============================================================================
# Lưu prediction
# =============================================================================

def build_pred_df(train_df, valid_df, test_df, oof_tr, pred_va, pred_te):
    """Gộp prediction mọi split thành 1 DataFrame (train=OOF, valid/test=full-train)."""
    rows = []
    for df, preds, split, is_oof in [
        (train_df, oof_tr, "train", True),
        (valid_df, pred_va, "valid", False),
        (test_df, pred_te, "test", False),
    ]:
        rows.append(pd.DataFrame({
            "smiles": df["smiles"].values,
            "target": df["target"].values.astype(np.float64),
            "split": split,
            "pred": np.asarray(preds, dtype=np.float64),
            "is_oof": is_oof,
        }))
    return pd.concat(rows, ignore_index=True)


# =============================================================================
# Một split
# =============================================================================

def run_one_seed(dataset, split_method, seed, raw_dir, processed_dir, out_root):
    train_df, valid_df, test_df = get_splits(
        dataset, split_method, seed, raw_dir, processed_dir)
    print(f"  Data: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

    # Featurize: fit làm sạch trên TRAIN, transform mọi split.
    feat = Descriptor2DFeaturizer().fit(train_df["smiles"].tolist())
    X_tr = feat.transform(train_df["smiles"].tolist())
    X_va = feat.transform(valid_df["smiles"].tolist())
    X_te = feat.transform(test_df["smiles"].tolist())
    y_tr = train_df["target"].values.astype(np.float64)
    y_va = valid_df["target"].values.astype(np.float64)
    y_te = test_df["target"].values.astype(np.float64)
    print(f"  2D descriptor: {len(feat.feature_names)} feature (sau impute+VarThresh)")

    out_dir = os.path.join(out_root, dataset, split_method, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for name, runner in [("ridge", run_ridge), ("gbt", run_gbt)]:
        oof_tr, pred_va, pred_te, info = runner(X_tr, y_tr, X_va, y_va, X_te, seed)
        test_r = rmse(y_te, pred_te)
        oof_r = rmse(y_tr, oof_tr)
        info.update({"test_rmse": test_r, "train_oof_rmse": oof_r})
        print(f"  [{name.upper()}] test RMSE={test_r:.4f} | val={info['val_rmse']:.4f} "
              f"| train-OOF={oof_r:.4f} | {('alpha=%.3g' % info['alpha']) if name=='ridge' else info}")

        df = build_pred_df(train_df, valid_df, test_df, oof_tr, pred_va, pred_te)
        df.to_csv(os.path.join(out_dir, f"{name}_pred.csv"), index=False)
        results[name] = info

    metrics = {
        "dataset": dataset, "split_method": split_method, "seed": seed,
        "n_features": len(feat.feature_names),
        "feature_names": feat.feature_names,
        "ridge": results["ridge"], "gbt": results["gbt"],
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return results["ridge"]["test_rmse"], results["gbt"]["test_rmse"]


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Exp 0 — Baseline 2D descriptor (Ridge + GBT).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="esol", choices=list(DATASETS))
    p.add_argument("--seed-split", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   dest="seed_split", help="Các seed chia split (mặc định 0..4).")
    p.add_argument("--split-method", default="random_scaffold",
                   choices=["random_scaffold", "random"], dest="split_method")
    p.add_argument("--raw-dir", default="data/raw", dest="raw_dir")
    p.add_argument("--processed-dir", default="data/processed", dest="processed_dir")
    p.add_argument("--output-dir", default="experiments/exp0_baseline2d",
                   dest="output_dir")
    args = p.parse_args()

    ridge_scores, gbt_scores = {}, {}
    for seed in args.seed_split:
        print(f"\n{'#'*60}\n# {args.dataset} | split seed {seed}\n{'#'*60}")
        r, g = run_one_seed(args.dataset, args.split_method, seed,
                            args.raw_dir, args.processed_dir, args.output_dir)
        ridge_scores[seed], gbt_scores[seed] = r, g

    def summarize(name, scores):
        vals = list(scores.values())
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"\n{name}-2D test RMSE:")
        for s, v in scores.items():
            print(f"  seed {s}: {v:.4f}")
        print(f"  mean ± std = {mean:.4f} ± {std:.4f}")
        return {"per_split": scores, "mean": mean, "std": std}

    print(f"\n{'='*60}\nEXP 0 — Baseline 2D ({args.dataset}, {args.split_method})")
    print(f"Mốc tham chiếu (SchNet K=1): {BASELINE_REF}\n{'='*60}")
    summary = {
        "dataset": args.dataset, "split_method": args.split_method,
        "baseline_ref": BASELINE_REF,
        "ridge": summarize("Ridge", ridge_scores),
        "gbt": summarize("GBT", gbt_scores),
    }
    out_dir = os.path.join(args.output_dir, args.dataset, args.split_method)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nĐã lưu summary + prediction vào {out_dir}/")


if __name__ == "__main__":
    main()
