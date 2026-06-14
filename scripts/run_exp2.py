#!/usr/bin/env python
"""Exp 2 — Frozen encoder + multi-tree GP head trên ESOL (control).

DECOUPLED hoàn toàn: đọc ma trận embedding 128-dim đã trích sẵn
(scripts/run_exp2_extract.py) -> GP+ridge chạy thuần trên ma trận đó, KHÔNG có
SchNet trong vòng lặp. Rất nhanh, chạy CPU.

Target:
  - 2A (--mode delta, PRIMARY): residual trên GBT-2D đóng băng (Exp 0).
        r_train=OOF, r_val/r_test=full-GBT. final = gbt_pred + ridge(Φ).
  - 2B (--mode raw): raw y (head-vs-head với MLP head 0.8994). final = ridge(Φ).
Target được standardize bằng stats train; denorm khi báo cáo (đơn vị log-S).

Head: q cây (mặc định 8), partition random disjoint 128->q khối; ridge merge
closed-form trong fitness; model selection theo VAL; α cuối tune val logspace(-2,4)
=> graceful floor. Safeguard (2A): GP không vượt residual=0 trên val -> xuất baseline.

Mốc so sánh ESOL: 0.8994 (raw SchNet/MLP) | 0.7953 (GBT-2D) | 0.861 (Exp1 delta-MLP).

Ví dụ (sau khi đã trích embedding + chạy exp0):
    python scripts/run_exp2.py --dataset esol --seed-split 0 1 2 3 4 --mode delta
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

from src.gp.gp_head import GPConfig, run_gp_head, rmse


def load_embeddings(emb_dir, dataset, split_method, seed):
    d = os.path.join(emb_dir, dataset, split_method, f"seed_{seed}")
    if not os.path.exists(os.path.join(d, "emb_train.npy")):
        raise FileNotFoundError(
            f"Chưa có embedding: {d}\n  -> chạy scripts/run_exp2_extract.py trước.")
    out = {}
    for s in ("train", "valid", "test"):
        out[s] = (np.load(os.path.join(d, f"emb_{s}.npy")),
                  pd.read_csv(os.path.join(d, f"meta_{s}.csv")))
    return out


def load_gbt_baseline(exp0_dir, dataset, split_method, seed, meta_by_split):
    """Trả dict split -> baseline pred (căn theo smiles của embedding meta)."""
    path = os.path.join(exp0_dir, dataset, split_method, f"seed_{seed}", "gbt_pred.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không thấy baseline Exp 0: {path}\n  -> chạy scripts/run_exp0.py trước.")
    base = pd.read_csv(path)
    out = {}
    for s in ("train", "valid", "test"):
        sub = base[base["split"] == s]
        smi2pred = dict(zip(sub["smiles"], sub["pred"]))
        meta = meta_by_split[s]
        pred = meta["smiles"].map(smi2pred)
        if pred.isna().any():
            raise ValueError(f"  {int(pred.isna().sum())} smiles split '{s}' "
                             f"không khớp baseline Exp 0.")
        out[s] = pred.values.astype(np.float64)
    return out


def run_one_seed(args, seed):
    sm = args.split_method
    print(f"\n{'#'*60}\n# {args.dataset} | seed {seed} | mode={args.mode} | "
          f"q={args.q}\n{'#'*60}")
    emb = load_embeddings(args.emb_dir, args.dataset, sm, seed)
    meta = {s: emb[s][1] for s in emb}
    Xtr, Xva, Xte = emb["train"][0], emb["valid"][0], emb["test"][0]
    y = {s: meta[s]["y"].values.astype(np.float64) for s in meta}

    # --- Target theo mode ---
    if args.mode == "delta":
        base = load_gbt_baseline(args.exp0_dir, args.dataset, sm, seed, meta)
        tgt = {s: y[s] - base[s] for s in y}     # residual
        base_te = base["test"]
    else:  # raw
        tgt = {s: y[s] for s in y}
        base_te = np.zeros_like(y["test"])

    t_tr = tgt["train"]
    print(f"  Data: train={len(t_tr)}, valid={len(tgt['valid'])}, test={len(tgt['test'])}")
    print(f"  target({args.mode}) train: min={t_tr.min():.3f} max={t_tr.max():.3f} "
          f"mean={t_tr.mean():.4f} std={t_tr.std():.4f}")

    # --- Standardize target (train stats) ---
    tm, ts = float(t_tr.mean()), float(t_tr.std())
    ts = ts if ts > 1e-8 else 1.0
    std = {s: (tgt[s] - tm) / ts for s in tgt}

    # --- GP head ---
    cfg = GPConfig(q=args.q, emb_dim=Xtr.shape[1], pop_size=args.pop_size,
                   generations=args.generations, es_patience=args.es_patience,
                   ridge_alpha=args.ridge_alpha, seed=args.gp_seed)
    res = run_gp_head(Xtr, std["train"], Xva, std["valid"], Xte, std["test"], cfg)

    # --- Denorm về đơn vị target ---
    denorm = lambda p: p * ts + tm
    pred_va = denorm(res["pred_va_std"])
    pred_te = denorm(res["pred_te_std"])

    # --- Safeguard sàn baseline (chỉ mode delta) ---
    r_va, r_te = tgt["valid"], tgt["test"]
    gp_val_rmse = rmse(pred_va, r_va)
    chosen = "gp"
    if args.mode == "delta":
        base_val_rmse = rmse(np.zeros_like(r_va), r_va)   # residual=0 = baseline val RMSE
        if not (gp_val_rmse < base_val_rmse):
            chosen = "baseline(resid=0)"
            pred_te = np.zeros_like(r_te)
        print(f"  [safeguard] val: baseline={base_val_rmse:.4f} | gp={gp_val_rmse:.4f} "
              f"-> chọn {chosen}")

    # --- Eval + diagnostic ---
    final_pred = base_te + pred_te
    y_te = y["test"]
    final_rmse = rmse(final_pred, y_te)

    if args.mode == "delta":
        base_test_rmse = rmse(np.zeros_like(r_te), r_te)   # = GBT-2D test RMSE
        gp_resid_rmse = rmse(pred_te, r_te) if chosen == "gp" else base_test_rmse
        delta = base_test_rmse - rmse(denorm(res["pred_te_std"]), r_te)
        print(f"  [diag] test: baseline(GBT-2D)={base_test_rmse:.4f} | "
              f"residual-pred RMSE={rmse(denorm(res['pred_te_std']), r_te):.4f} "
              f"(Δ={delta:+.4f}) | alpha={res['alpha']:.2g}")
        harm = max(0.0, final_rmse - base_test_rmse)
        print(f"  final RMSE={final_rmse:.4f} [{chosen}] | hại so baseline={harm:+.4f}")
        info = {"seed": seed, "mode": args.mode, "final_rmse": final_rmse,
                "baseline_test_rmse": base_test_rmse, "gp_resid_test_rmse": gp_resid_rmse,
                "gp_val_rmse": gp_val_rmse, "chosen": chosen, "alpha": res["alpha"],
                "harm": harm}
    else:
        print(f"  final RMSE={final_rmse:.4f} (raw y; so MLP head 0.8994) | "
              f"alpha={res['alpha']:.2g}")
        info = {"seed": seed, "mode": args.mode, "final_rmse": final_rmse,
                "gp_val_rmse_std": res["val_rmse_std"], "alpha": res["alpha"]}

    if args.save:
        out_dir = os.path.join(args.output_dir, "exp2_gp", args.dataset, sm,
                               f"seed_{seed}_{args.mode}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "trees.json"), "w") as f:
            json.dump({"trees": res["trees_str"],
                       "partition": [p.tolist() for p in res["partition"]],
                       "alpha": res["alpha"], "info": info}, f, indent=2)
        pd.DataFrame({"smiles": meta["test"]["smiles"], "y_true": y_te,
                      "baseline": base_te, "resid_or_y_pred": pred_te,
                      "final_pred": final_pred}).to_csv(
            os.path.join(out_dir, "final_pred_test.csv"), index=False)
        print(f"  Đã lưu: {out_dir}/")
    return info


def main():
    p = argparse.ArgumentParser(
        description="Exp 2 — frozen encoder + multi-tree GP head (control).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="esol")
    p.add_argument("--seed-split", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   dest="seed_split")
    p.add_argument("--split-method", default="random_scaffold", dest="split_method",
                   choices=["random_scaffold", "random"])
    p.add_argument("--mode", default="delta", choices=["delta", "raw"],
                   help="delta=2A residual trên GBT-2D (primary); raw=2B y thô (head-vs-head).")
    p.add_argument("--q", type=int, default=8, help="Số cây/individual (sweep {4,8,16}).")
    p.add_argument("--pop-size", type=int, default=1000, dest="pop_size")
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--es-patience", type=int, default=20, dest="es_patience")
    p.add_argument("--ridge-alpha", type=float, default=1.0, dest="ridge_alpha",
                   help="α cố định trong evolution (α cuối tune trên val).")
    p.add_argument("--gp-seed", type=int, default=0, dest="gp_seed",
                   help="Seed cho partition + GP (cố định để lặp lại).")
    p.add_argument("--emb-dir", default="experiments/exp2_embeddings", dest="emb_dir")
    p.add_argument("--exp0-dir", default="experiments/exp0_baseline2d", dest="exp0_dir")
    p.add_argument("--output-dir", default="experiments", dest="output_dir")
    p.add_argument("--save", action="store_true", help="Lưu cây + final pred.")
    args = p.parse_args()

    infos = {s: run_one_seed(args, s) for s in args.seed_split}

    finals = [infos[s]["final_rmse"] for s in args.seed_split]
    mean = statistics.mean(finals)
    std = statistics.stdev(finals) if len(finals) > 1 else 0.0
    print(f"\n{'='*60}")
    print(f"EXP 2 — GP head ({args.dataset}, mode={args.mode}, q={args.q})")
    print(f"{'='*60}")
    for s in args.seed_split:
        i = infos[s]
        extra = (f" (baseline={i['baseline_test_rmse']:.4f}, {i['chosen']}, "
                 f"hại={i['harm']:+.4f})" if args.mode == "delta" else "")
        print(f"  seed {s}: final={i['final_rmse']:.4f}{extra}")
    print(f"  final RMSE = {mean:.4f} ± {std:.4f}")
    if args.mode == "delta":
        worst_harm = max(i["harm"] for i in infos.values())
        print(f"  hại seed-tệ-nhất = {worst_harm:+.4f}  (Exp1 delta-MLP: 0.16–0.21)")
        print(f"  mốc: 0.8994 (MLP) | 0.7953 (GBT-2D) | 0.861 (Exp1)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
