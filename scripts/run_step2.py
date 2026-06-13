#!/usr/bin/env python
"""STEP 2 — đo sức mạnh descriptor 2D (KHÔNG dùng embedding, KHÔNG conformer).

Học thẳng TOÀN BỘ descriptor 2D RDKit (~217, không chọn lọc) -> target. Hai mode:
  - ridge : RidgeCV (sklearn), co hệ số L2 tự giảm trọng số descriptor vô dụng.
  - tree  : MỘT cây GP symbolic (DEAP), biến không tham chiếu = bị loại.
Quét nhiều split seed -> in RMSE mean ± std (đơn vị gốc), so baseline SchNet 1-conf
0.8994 ± 0.0946. Split DÙNG LẠI đúng CSV như run_gp (cùng seed -> cùng phân chia).

Nhẹ + CPU-friendly: desc2d là 2D (mức đồ thị) nên không cần sinh conformer/encoder.
Cache ma trận desc2d thô ở data/processed/<ds>/<split>/seed_<seed>/step2_desc2d_full.pkl
(độc lập mode/hyper); --force-extract để tính lại.

Ví dụ:
    python scripts/run_step2.py --dataset esol --seed-split 0 1 2 3 4
    python scripts/run_step2.py --dataset esol --seed-split 0 1 2 3 4 --mode ridge
    python scripts/run_step2.py --dataset esol --seed-split 0 1 2 3 4 --mode tree --generations 200
"""

import argparse
import json
import os
import pickle
import statistics
import sys
import time

import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config import DATASETS
from src.data.data_loader import prepare_dataset, save_splits
from src.gp.desc2d_full import FULL_DESC2D_NAMES, build_desc2d_matrix
from src.gp.step2 import (
    GPTreeConfig, fit_standardizer, run_gp_tree, run_ridge,
)

BASELINE = "0.8994 ± 0.0946"


# =============================================================================
# Argparse
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="STEP 2 — desc2d (full RDKit 2D) -> target qua RidgeCV / cây GP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("Global")
    g.add_argument("--dataset", default="esol", choices=list(DATASETS))
    g.add_argument("--seed-split", type=int, nargs="+", default=[0], dest="seed_split",
                   help="Các split seed (vd 0 1 2 3 4) -> in RMSE mean ± std.")
    g.add_argument("--split-method", default="random_scaffold",
                   choices=["random_scaffold", "random"], dest="split_method")
    g.add_argument("--mode", default="both", choices=["ridge", "tree", "both"],
                   help="ridge=RidgeCV, tree=một cây GP, both=cả hai.")
    g.add_argument("--seed-train", type=int, default=0, dest="seed_train",
                   help="Seed randomness của cây GP.")
    g.add_argument("--clip", type=float, default=10.0,
                   help="Clip z-score descriptor vào [-clip,clip] (ghìm outlier như Ipc).")
    g.add_argument("--force-extract", action="store_true",
                   help="Tính lại ma trận desc2d dù cache đã có.")

    t = p.add_argument_group("Cây GP (mode tree)")
    t.add_argument("--pop", type=int, default=500, dest="pop")
    t.add_argument("--mu", type=int, default=500)
    t.add_argument("--lam", "--lambda", type=int, default=500, dest="lam")
    t.add_argument("--generations", type=int, default=120)
    t.add_argument("--height", type=int, default=17, help="Giới hạn chiều cao cây.")
    t.add_argument("--max-len", type=int, default=6000, dest="max_len",
                   help="Trần số node cây (vượt thì revert về cha).")
    t.add_argument("--cxpb", type=float, default=0.7)
    t.add_argument("--mutpb", type=float, default=0.25)
    t.add_argument("--tree-free", action="store_true", dest="tree_free",
                   help="Cây GP TỰ CHỌN biến (cũ). Mặc định TẮT = MỘT cây to dùng MỌI "
                        "biến desc2d (repair coverage, không tự loại biến).")

    o = p.add_argument_group("Output")
    o.add_argument("--save", action=argparse.BooleanOptionalAction, default=True,
                   help="Lưu kết quả vào experiments/step2/. Mặc định BẬT.")
    o.add_argument("--output-dir", default="experiments", dest="output_dir")
    return p


# =============================================================================
# Data: dùng lại đúng split CSV như run_gp
# =============================================================================

def _load_dfs(dataset: str, split_method: str, seed_split: int) -> dict:
    config = {
        "dataset_name": dataset,
        "dataset": DATASETS[dataset],
        "data": {
            "raw_dir": "data/raw", "processed_dir": "data/processed",
            "split_method": split_method, "random_seed_split": seed_split,
        },
    }
    ds_dir = f"data/processed/{dataset}/{split_method}/seed_{seed_split}"
    if os.path.exists(os.path.join(ds_dir, "train.csv")):
        return {n: pd.read_csv(os.path.join(ds_dir, f"{n}.csv"))
                for n in ("train", "valid", "test")}
    tr, va, te = prepare_dataset(config)
    save_splits(tr, va, te, ds_dir)
    return {"train": tr, "valid": va, "test": te}


def _desc2d_cache_path(dataset: str, split_method: str, seed_split: int) -> str:
    return (f"data/processed/{dataset}/{split_method}/seed_{seed_split}"
            "/step2_desc2d_full.pkl")


def _get_desc2d(dataset, split_method, seed_split, force: bool) -> dict:
    """Trả {'train','valid','test': (X, y)} với X = toàn bộ desc2d thô (NaN-able)."""
    path = _desc2d_cache_path(dataset, split_method, seed_split)
    if os.path.exists(path) and not force:
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"  [desc2d] dùng lại cache: {path}")
        return data

    dfs = _load_dfs(dataset, split_method, seed_split)
    data = {}
    for name in ("train", "valid", "test"):
        smiles = dfs[name]["smiles"].tolist()
        y = dfs[name]["target"].values.astype(np.float64)
        X, ok = build_desc2d_matrix(smiles)
        if not ok.all():
            X, y = X[ok], y[ok]
            print(f"  [desc2d] {name}: bỏ {int((~ok).sum())} mol parse fail")
        data[name] = (X, y)
    data["names"] = FULL_DESC2D_NAMES
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  [desc2d] tính {len(FULL_DESC2D_NAMES)} descriptor, lưu cache: {path}")
    return data


# =============================================================================
# Per-seed
# =============================================================================

def run_one_seed(args, seed_split: int) -> dict:
    data = _get_desc2d(args.dataset, args.split_method, seed_split, args.force_extract)
    names = data["names"]
    Xtr_raw, ytr = data["train"]
    Xva_raw, yva = data["valid"]
    Xte_raw, yte = data["test"]

    st = fit_standardizer(Xtr_raw, ytr, names, clip=args.clip)
    Xtr, Xva, Xte = (st.transform(Xtr_raw), st.transform(Xva_raw), st.transform(Xte_raw))
    print(f"  [prep] {Xtr.shape[1]}/{len(names)} descriptor giữ lại (bỏ cột hằng), "
          f"train={Xtr.shape[0]} val={Xva.shape[0]} test={Xte.shape[0]}")

    out = {"seed_split": seed_split, "n_desc_kept": int(Xtr.shape[1])}

    if args.mode in ("ridge", "both"):
        r = run_ridge(Xtr, ytr, Xva, yva, Xte, yte, st.names)
        print(f"  [ridge] test RMSE = {r.test_rmse:.4f} (val {r.val_rmse:.4f}, "
              f"train {r.train_rmse:.4f}, alpha={r.alpha:g})")
        out["ridge"] = {
            "test_rmse": r.test_rmse, "val_rmse": r.val_rmse,
            "train_rmse": r.train_rmse, "alpha": r.alpha, "top_coef": r.top_coef,
        }

    if args.mode in ("tree", "both"):
        ytr_z = (ytr - st.target_mean) / st.target_std
        cfg = GPTreeConfig(
            pop_size=args.pop, mu=args.mu, lam=args.lam, generations=args.generations,
            cxpb=args.cxpb, mutpb=args.mutpb, height=args.height, max_len=args.max_len,
            full_coverage=not args.tree_free, seed=args.seed_train,
        )
        g = run_gp_tree(Xtr, ytr_z, Xva, yva, Xte, yte,
                        st.target_mean, st.target_std, st.names, cfg, verbose=True)
        print(f"  [tree ] test RMSE = {g.test_rmse:.4f} (val {g.val_rmse:.4f}, "
              f"train {g.train_rmse:.4f}, size={g.size}, "
              f"#desc dùng={len(g.used_descriptors)}/{Xtr.shape[1]})")
        out["tree"] = {
            "test_rmse": g.test_rmse, "val_rmse": g.val_rmse, "train_rmse": g.train_rmse,
            "size": g.size, "formula": g.formula,
            "used_descriptors": g.used_descriptors, "history": g.history,
        }
    return out


# =============================================================================
# Main
# =============================================================================

def _summary(tag: str, scores: dict) -> tuple:
    vals = [v for v in scores.values() if v is not None]
    mean = statistics.mean(vals) if vals else float("nan")
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"\n  [{tag}] test RMSE từng seed:")
    for s, v in scores.items():
        print(f"    seed {s}: {v:.4f}")
    if len(vals) > 1:
        print(f"  [{tag}] Trung bình = {mean:.4f} ± {std:.4f}")
    return mean, std


def main():
    args = build_parser().parse_args()
    run_dir = os.path.join(args.output_dir, "step2", args.dataset, args.split_method,
                           time.strftime("%Y%m%d_%H%M%S"))

    per_seed = {}
    ridge_scores, tree_scores = {}, {}
    for i, seed in enumerate(args.seed_split):
        print(f"\n{'#'*64}\n# split seed {seed} ({i+1}/{len(args.seed_split)})\n{'#'*64}")
        t0 = time.time()
        res = run_one_seed(args, seed)
        per_seed[seed] = res
        if "ridge" in res:
            ridge_scores[seed] = res["ridge"]["test_rmse"]
        if "tree" in res:
            tree_scores[seed] = res["tree"]["test_rmse"]
        print(f"  (seed {seed} xong trong {time.time()-t0:.1f}s)")

    print(f"\n{'='*64}")
    print(f"STEP 2 — {args.dataset} ({args.split_method}) | mode={args.mode} | "
          f"desc2d=full RDKit 2D ({len(FULL_DESC2D_NAMES)})")
    summary = {"dataset": args.dataset, "split_method": args.split_method,
               "mode": args.mode, "args": vars(args), "baseline_1conf": BASELINE,
               "per_seed": per_seed}
    if ridge_scores:
        m, s = _summary("ridge", ridge_scores)
        summary["ridge_mean"], summary["ridge_std"] = m, s
    if tree_scores:
        m, s = _summary("tree", tree_scores)
        summary["tree_mean"], summary["tree_std"] = m, s
    print(f"\n  (baseline SchNet 1-conf: {BASELINE})")
    print(f"{'='*64}")

    if args.save:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Kết quả lưu ở: {run_dir}")


if __name__ == "__main__":
    main()
