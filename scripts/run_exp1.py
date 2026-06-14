#!/usr/bin/env python
"""Exp 1 — SchNet + delta/residual learning (giữ MLP head, K=1) trên ESOL.

Ý tưởng: chỉ đổi TARGET của SchNet từ y thô sang residual của baseline 2D
(GBT từ Exp 0). Mọi thứ khác (kiến trúc, K=1, optimizer, LR schedule, epochs,
early-stopping, 5 split) giữ y hệt run baseline -> cô lập đúng tác dụng của delta.

Tính chất khoá: final_pred = baseline + resid_pred và y = baseline + r, nên
    RMSE(final, y) ≡ RMSE(resid_pred, r).
=> Cho SchNet học target = residual thì val/test RMSE mà trainer báo CHÍNH LÀ
   RMSE cuối ở đơn vị log-S. Diff so với baseline là tối thiểu.

Baseline đóng băng từ Exp 0: đọc thẳng prediction đã lưu (gbt_pred.csv), KHÔNG
tune lại. Mỗi split có 3 bộ:
  - train -> OOF (5-fold trong train, is_oof=True): nhãn residual "honest".
  - valid/test -> GBT fit full-train (is_oof=False): không leak.

Cơ chế neo baseline: set_normalization(mean(r_train), std(r_train)) + lin2 zero-init
=> prediction ban đầu = mean(r_train) ≈ 0 => final ban đầu ≈ baseline (RMSE ~ 0.795).
SchNet chỉ đi xuống nếu học được residual thật.

Safeguard (bước 5): nếu trong suốt training SchNet KHÔNG vượt được "residual=0"
trên val (= RMSE val của baseline) thì xuất thẳng baseline (resid=0) cho split đó
=> delta-SchNet không bao giờ tệ hơn baseline.

Ví dụ:
    python scripts/run_exp1.py --dataset esol --seed-split 0 1 2 3 4 --deterministic
(cần chạy scripts/run_exp0.py trước để có experiments/exp0_baseline2d/.../gbt_pred.csv)
"""

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import pandas as pd
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # để import run_step1

from run_step1 import build_parser, print_overrides
from src.config import build_config
from src.data.data_loader import prepare_dataset, save_splits, create_dataloaders
from src.models.schnet import build_schnet_model
from src.trainers.step1_trainer import Step1Trainer
from src.utils.utils import seed_everything

RESID_OUTLIER_THRESH = 5.0  # |residual| lớn hơn ngưỡng này -> cảnh báo (bẫy của ridge)


def rmse(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


# =============================================================================
# Load split + baseline -> residual
# =============================================================================

def load_split_csvs(processed_dir, dataset, split_method, seed, config):
    """Lấy train/valid/test ĐÚNG như baseline (cache .../<split_method>/seed_<seed>)."""
    ds_dir = os.path.join(processed_dir, dataset, split_method, f"seed_{seed}")
    paths = {s: os.path.join(ds_dir, f"{s}.csv") for s in ("train", "valid", "test")}
    if all(os.path.exists(p) for p in paths.values()):
        return (pd.read_csv(paths["train"]), pd.read_csv(paths["valid"]),
                pd.read_csv(paths["test"]))
    print(f"  Chưa có split cache, tạo mới tại {ds_dir}")
    train_df, valid_df, test_df = prepare_dataset(config)
    save_splits(train_df, valid_df, test_df, ds_dir)
    return train_df, valid_df, test_df


def load_baseline_preds(exp0_dir, dataset, split_method, seed, baseline):
    """Đọc prediction baseline đóng băng từ Exp 0 (gbt_pred.csv / ridge_pred.csv)."""
    path = os.path.join(exp0_dir, dataset, split_method, f"seed_{seed}",
                        f"{baseline}_pred.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không thấy baseline Exp 0: {path}\n"
            f"  -> chạy: python scripts/run_exp0.py --dataset {dataset} "
            f"--seed-split {seed}")
    return pd.read_csv(path)


def attach_residual(split_df, base_df, split_name):
    """Gắn baseline (map theo smiles) + target=residual. Giữ y_true, baseline để báo cáo."""
    sub = base_df[base_df["split"] == split_name]
    smi2pred = dict(zip(sub["smiles"], sub["pred"]))
    base = split_df["smiles"].map(smi2pred)
    missing = int(base.isna().sum())
    if missing:
        raise ValueError(
            f"  {missing} smiles trong split '{split_name}' không có baseline pred "
            f"(split Exp 0 và split hiện tại không khớp?).")
    out = split_df.copy()
    out["y_true"] = split_df["target"].values.astype(np.float64)
    out["baseline"] = base.values.astype(np.float64)
    out["target"] = out["y_true"] - out["baseline"]  # <-- residual = nhãn mới
    return out


# =============================================================================
# Một split
# =============================================================================

def run_one_seed(args, device, seed):
    config = build_config(args)
    config["data"]["random_seed_split"] = seed

    seed_everything(config["random_seed_train"], deterministic=config["deterministic"])
    print(f"\n{'#'*60}\n# {args.dataset} | split seed {seed} | "
          f"baseline={args.baseline.upper()} (Exp 0, frozen)\n{'#'*60}")

    # 1) Split + baseline -> residual
    train_df, valid_df, test_df = load_split_csvs(
        config["data"]["processed_dir"], args.dataset,
        config["data"]["split_method"], seed, config)
    base_df = load_baseline_preds(args.exp0_dir, args.dataset,
                                  config["data"]["split_method"], seed, args.baseline)
    train_r = attach_residual(train_df, base_df, "train")
    valid_r = attach_residual(valid_df, base_df, "valid")
    test_r = attach_residual(test_df, base_df, "test")

    r_train = train_r["target"].values.astype(np.float64)
    print(f"  Data: train={len(train_r)}, valid={len(valid_r)}, test={len(test_r)}")

    # 3) Sanity check residual train (trước khi train)
    lo, hi = float(r_train.min()), float(r_train.max())
    mu, sd = float(r_train.mean()), float(r_train.std())
    print(f"  r_train: min={lo:.3f} max={hi:.3f} mean={mu:.4f} std={sd:.4f}")
    if max(abs(lo), abs(hi)) > args.resid_outlier_thresh:
        print(f"  ⚠️  CẢNH BÁO: residual có outlier |.|>{args.resid_outlier_thresh} "
              f"-> baseline có thể extrapolate hỏng (bẫy đã giết ridge). Kiểm tra lại!")

    # 4) Train SchNet trên residual (diff tối thiểu: chỉ đổi target + normalization)
    train_loader, valid_loader, test_loader = create_dataloaders(
        config, train_r, valid_r, test_r)
    model = build_schnet_model(config)
    model.set_normalization(mu, sd)  # net học residual chuẩn hoá; lin2 zero-init -> resid≈0

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.output_dir, "exp1_residual", args.dataset,
                           config["data"]["split_method"], f"seed_{seed}", timestamp)
    trainer = Step1Trainer(model, config, device, exp_dir)
    trainer.train(train_loader, valid_loader, test_loader)

    # 5) Predictions residual + safeguard sàn baseline
    resid_val, r_val = trainer.predict(valid_loader)
    resid_test, r_test = trainer.predict(test_loader)

    base_val_rmse = rmse(np.zeros_like(r_val), r_val)    # residual=0 trên val = baseline val RMSE
    schnet_val_rmse = rmse(resid_val, r_val)             # = best val RMSE (đã load best ckpt)
    use_schnet = schnet_val_rmse < base_val_rmse

    base_test_rmse = rmse(np.zeros_like(r_test), r_test)  # = baseline (GBT-2D) test RMSE
    schnet_test_rmse = rmse(resid_test, r_test)           # final RMSE nếu dùng SchNet
    if use_schnet:
        final_test_rmse, chosen = schnet_test_rmse, "schnet"
    else:
        final_test_rmse, chosen = base_test_rmse, "baseline(resid=0)"
        resid_test = np.zeros_like(r_test)

    # 6+7) Báo cáo + diagnostic
    print(f"\n  [safeguard] val: baseline={base_val_rmse:.4f} | "
          f"schnet={schnet_val_rmse:.4f} -> chọn {chosen}")
    print(f"  [diag] test: baseline(GBT-2D)={base_test_rmse:.4f} | "
          f"residual-pred RMSE={schnet_test_rmse:.4f}")
    delta = base_test_rmse - schnet_test_rmse
    band = 5e-3  # chênh nhỏ hơn ~0.005 RMSE coi như noise (neutral)
    if delta > band:
        verdict = f"WIN: SchNet rút được tín hiệu 3D thật (giảm {delta:.4f})"
    elif delta >= -band:
        verdict = f"NEUTRAL: residual ≈ 0/noise -> combined ≈ baseline (Δ={delta:+.4f})"
    else:
        verdict = f"HẠI: SchNet fit noise (tăng {-delta:.4f}); safeguard chặn ở test bằng baseline"
    print(f"  [diag] {verdict}")
    print(f"\n{args.dataset} (seed {seed}): final RMSE={final_test_rmse:.4f} [{chosen}]")

    info = {
        "seed": seed, "baseline": args.baseline,
        "baseline_val_rmse": base_val_rmse, "schnet_val_rmse": schnet_val_rmse,
        "baseline_test_rmse": base_test_rmse, "schnet_resid_test_rmse": schnet_test_rmse,
        "final_test_rmse": final_test_rmse, "chosen": chosen,
        "r_train_stats": {"min": lo, "max": hi, "mean": mu, "std": sd},
    }

    if args.save:
        os.makedirs(exp_dir, exist_ok=True)
        # Lưu final prediction (log-S) căn theo smiles của dataset (shuffle=False).
        smi = list(test_loader.dataset.smiles)
        smi2base = dict(zip(base_df[base_df.split == "test"]["smiles"],
                            base_df[base_df.split == "test"]["pred"]))
        base_te = np.array([smi2base[s] for s in smi], dtype=np.float64)
        out = pd.DataFrame({
            "smiles": smi,
            "y_true": base_te + r_test,
            "baseline": base_te,
            "resid_pred": resid_test,
            "final_pred": base_te + resid_test,
        })
        out.to_csv(os.path.join(exp_dir, "final_pred_test.csv"), index=False)
        with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
            json.dump(info, f, indent=2)
        print(f"  Đã lưu: {exp_dir}/")

    return info


# =============================================================================
# Main
# =============================================================================

def main():
    parser = build_parser()
    parser.description = "Exp 1 — SchNet + delta/residual learning (baseline 2D đóng băng)."
    g = parser.add_argument_group("Exp 1 (residual)")
    g.add_argument("--exp0-dir", default="experiments/exp0_baseline2d", dest="exp0_dir",
                   help="Thư mục output Exp 0 (chứa <baseline>_pred.csv mỗi split).")
    g.add_argument("--baseline", default="gbt", choices=["gbt", "ridge"],
                   help="Baseline đóng băng dùng làm residual. GBT (mặc định): OOF "
                        "sạch, bị chặn trong khoảng target. Ridge: dễ extrapolate hỏng.")
    g.add_argument("--resid-outlier-thresh", type=float, default=RESID_OUTLIER_THRESH,
                   dest="resid_outlier_thresh",
                   help="Ngưỡng |residual| để cảnh báo outlier ở sanity check.")
    args = parser.parse_args()
    print_overrides(parser, args)

    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    seeds = args.seed_split
    infos = {}
    for seed in seeds:
        infos[seed] = run_one_seed(args, device, seed)

    # Tổng kết 5 seed
    finals = [infos[s]["final_test_rmse"] for s in seeds]
    bases = [infos[s]["baseline_test_rmse"] for s in seeds]
    mean = statistics.mean(finals)
    std = statistics.stdev(finals) if len(finals) > 1 else 0.0
    base_mean = statistics.mean(bases)
    base_std = statistics.stdev(bases) if len(bases) > 1 else 0.0
    print(f"\n{'='*60}")
    print(f"EXP 1 — SchNet + delta ({args.dataset}, {args.baseline.upper()} baseline)")
    print(f"{'='*60}")
    for s in seeds:
        i = infos[s]
        print(f"  seed {s}: final={i['final_test_rmse']:.4f} "
              f"(baseline={i['baseline_test_rmse']:.4f}, {i['chosen']})")
    print(f"  Baseline 2D test RMSE   = {base_mean:.4f} ± {base_std:.4f}")
    print(f"  Exp1 final  test RMSE   = {mean:.4f} ± {std:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
