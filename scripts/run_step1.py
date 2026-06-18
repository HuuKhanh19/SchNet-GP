#!/usr/bin/env python
"""Step 1: SchNet Baseline (Adam optimizer).

Chạy SchNet gốc (K=1 conformer) hoặc bản mở rộng K conformer cho dự đoán tính
chất phân tử. Cấu hình qua argparse — xem `python scripts/run_step1.py -h`.

Ví dụ:
    # SchNet gốc, ESOL, 1 split seed
    python scripts/run_step1.py --dataset esol --seed-split 0

    # Quét 5 split seed trong 1 lệnh -> in RMSE trung bình ± std
    python scripts/run_step1.py --dataset esol --seed-split 0 1 2 3 4

    # Mở rộng K=10 conformer, lưu output, chạy deterministic
    python scripts/run_step1.py --dataset esol --num-conformers 10 --save --deterministic
"""

import argparse
import os
import statistics
import sys
import time

import pandas as pd
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config import DATASETS, build_config
from src.data.data_loader import prepare_dataset, save_splits, create_dataloaders
from src.models.schnet import build_schnet_model
from src.trainers.step1_trainer import Step1Trainer
from src.utils.utils import seed_everything


# =============================================================================
# Argument parser
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Step 1: SchNet baseline (Adam).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Global ---
    g = p.add_argument_group("Global")
    g.add_argument("--dataset", default="esol", choices=list(DATASETS),
                   help="Tên dataset (định nghĩa trong src/config.py).")
    g.add_argument("--gpu", type=int, default=0,
                   help="Chỉ số GPU dùng. Đặt -1 để chạy CPU. "
                        "Trên server: card 0 thường rảnh, card 1 hay bận.")
    g.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Bật thuật toán CUDA deterministic để cùng seed -> cùng "
                        "kết quả (scatter/atomic của message passing vốn "
                        "non-deterministic). Mặc định TẮT (nhanh hơn, kết quả lệch "
                        "nhẹ giữa các lần). Bật bằng --deterministic khi cần lặp lại.")

    # --- Seeds ---
    s = p.add_argument_group("Seeds")
    s.add_argument("--seed-train", type=int, default=0, dest="seed_train",
                   help="Seed cho khởi tạo model + thứ tự batch. Cố định cho cả "
                        "5 split seed để chỉ thay đổi cách chia dữ liệu.")
    s.add_argument("--seed-split", type=int, nargs="+", default=[0],
                   dest="seed_split",
                   help="Seed chia split. Có thể truyền nhiều seed trong 1 lệnh "
                        "(vd. --seed-split 0 1 2 3 4) -> chạy lần lượt rồi in RMSE "
                        "trung bình ± std.")
    s.add_argument("--seed-gen", type=int, default=42, dest="seed_gen",
                   help="Seed sinh conformer (RDKit ETKDGv3).")

    # --- Data / split ---
    d = p.add_argument_group("Data / Split")
    d.add_argument("--split-method", default="random_scaffold",
                   choices=["random_scaffold", "random"], dest="split_method",
                   help="Cách chia train/valid/test (~81/9/10). scaffold khó hơn "
                        "random; kết quả public của project này dùng scaffold.")
    d.add_argument("--raw-dir", default="data/raw", dest="raw_dir",
                   help="Thư mục chứa CSV gốc.")
    d.add_argument("--processed-dir", default="data/processed", dest="processed_dir",
                   help="Thư mục cache split + conformer.")

    # --- Conformer ---
    c = p.add_argument_group("Conformer")
    c.add_argument("--num-conformers", "-K", type=int, default=1,
                   dest="num_conformers",
                   help="Số conformer mỗi phân tử (K). K=1 = SchNet gốc; K>1 = mở rộng.")
    c.add_argument("--max-attempts", type=int, default=500, dest="max_attempts",
                   help="Số vòng embed tối đa của RDKit.")
    c.add_argument("--prune-rms-thresh", type=float, default=0.0,
                   dest="prune_rms_thresh",
                   help="Ngưỡng RMS để loại conformer trùng (0 = không loại).")
    c.add_argument("--use-random-coords", action=argparse.BooleanOptionalAction,
                   default=False, dest="use_random_coords",
                   help="Cho phép RDKit dùng toạ độ ngẫu nhiên khi embed thất bại.")
    c.add_argument("--optimize-mmff", action=argparse.BooleanOptionalAction,
                   default=True, dest="optimize_mmff",
                   help="Tối ưu hoá hình học bằng MMFF/UFF (dùng để sắp xếp theo năng lượng).")

    # --- Model (SchNet) ---
    m = p.add_argument_group("Model (SchNet)")
    m.add_argument("--n-atom-basis", type=int, default=128, dest="n_atom_basis",
                   help="Số chiều embedding ẩn (hidden_channels).")
    m.add_argument("--n-interactions", type=int, default=6, dest="n_interactions",
                   help="Số interaction block.")
    m.add_argument("--n-rbf", type=int, default=50, dest="n_rbf",
                   help="Số hàm Gaussian khai triển khoảng cách.")
    m.add_argument("--n-filters", type=int, default=128, dest="n_filters",
                   help="Số filter trong CFConv.")
    m.add_argument("--cutoff", type=float, default=10.0,
                   help="Bán kính cutoff (Å) dựng đồ thị. SchNet gốc dùng 10.0 "
                        "(cutoff=5.0 cho RMSE kém hơn rõ rệt).")
    m.add_argument("--conf-readout", default="mean", choices=["mean", "add"],
                   dest="conf_readout",
                   help="Gộp K conformer: 'mean' = trung bình ensemble (bất biến số "
                        "conformer); 'add' = tổng (prediction phụ thuộc số conformer). "
                        "Với K=1 hai cái như nhau.")

    # --- Training ---
    t = p.add_argument_group("Training (Adam)")
    t.add_argument("--epochs", type=int, default=300, help="Số epoch tối đa.")
    t.add_argument("--batch-size", type=int, default=32, dest="batch_size",
                   help="Kích thước batch (số phân tử).")
    t.add_argument("--learning-rate", "--lr", type=float, default=1e-3,
                   dest="learning_rate", help="Learning rate của Adam.")
    t.add_argument("--weight-decay", type=float, default=1e-5, dest="weight_decay",
                   help="Weight decay (L2) của Adam.")
    t.add_argument("--scheduler-patience", type=int, default=25,
                   dest="scheduler_patience",
                   help="Patience của ReduceLROnPlateau (epoch không cải thiện -> giảm LR).")
    t.add_argument("--scheduler-factor", type=float, default=0.5,
                   dest="scheduler_factor", help="Hệ số nhân LR khi giảm.")
    t.add_argument("--early-stopping-patience", type=int, default=100,
                   dest="early_stopping_patience",
                   help="Số epoch không cải thiện val thì dừng sớm.")
    t.add_argument("--gradient-clip", type=float, default=1.0, dest="gradient_clip",
                   help="Ngưỡng clip norm gradient (<=0 = tắt).")

    # --- Experiment / output ---
    e = p.add_argument_group("Experiment")
    e.add_argument("--output-dir", default="experiments", dest="output_dir",
                   help="Thư mục gốc lưu kết quả.")
    e.add_argument("--save", action=argparse.BooleanOptionalAction, default=False,
                   help="Có lưu checkpoint + results.json + config vào experiments/ "
                        "hay không. Mặc định KHÔNG lưu (chỉ chạy & in metric, best "
                        "model giữ trong RAM). Bật bằng --save khi muốn giữ kết quả.")
    e.add_argument("--verbose", action="store_true",
                   help="In thêm shape của từng tham số model.")
    e.add_argument("--encoder-out", default=None, dest="encoder_out",
                   help="Xuất encoder state_dict (best model) ra path này để warm-start "
                        "eggroll. Có thể chứa '{seed}' -> thay bằng split seed "
                        "(per-seed, tránh leak), vd. pretrained/esol/seed_{seed}.pt. "
                        "Mặc định không xuất.")
    return p


# =============================================================================
# Run
# =============================================================================

def run_step1(config: dict, device: torch.device) -> dict:
    dataset_name = config["dataset_name"]

    # -- Seed everything FIRST --
    train_seed = config["random_seed_train"]
    deterministic = config.get("deterministic", True)
    seed_everything(train_seed, deterministic=deterministic)
    print(f"random_seed_train={train_seed}, deterministic={deterministic}")

    print(f"\n{'='*60}")
    print(f"Step 1: SchNet Baseline - {dataset_name.upper()}")
    print(f"{'='*60}")

    # -- Load or prepare data (cache key gồm cả split_method) --
    base_dir = config["data"]["processed_dir"]
    split_method = config["data"]["split_method"]
    split_seed = config["data"]["random_seed_split"]
    ds_dir = f"{base_dir}/{dataset_name}/{split_method}/seed_{split_seed}"

    if os.path.exists(os.path.join(ds_dir, "train.csv")):
        print(f"Loading preprocessed data from {ds_dir}")
        train_df = pd.read_csv(os.path.join(ds_dir, "train.csv"))
        valid_df = pd.read_csv(os.path.join(ds_dir, "valid.csv"))
        test_df = pd.read_csv(os.path.join(ds_dir, "test.csv"))
    else:
        print("Preprocessed data not found, running preprocessing...")
        train_df, valid_df, test_df = prepare_dataset(config)
        save_splits(train_df, valid_df, test_df, ds_dir)

    print(f"Data: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

    train_loader, valid_loader, test_loader = create_dataloaders(
        config, train_df, valid_df, test_df
    )

    # -- Build model --
    model = build_schnet_model(config)

    # -- Standardize regression targets from training data --
    # Net dự đoán (target - mean) / std; forward denormalize. Cách canonical SchNet
    # xử lý scale target (giữ gradient ổn định giữa các dataset).
    if config["dataset"]["task_type"] == "regression":
        mean_target = float(train_df["target"].mean())
        std_target = float(train_df["target"].std())
        model.set_normalization(mean_target, std_target)

    if config["experiment"].get("verbose", False):
        print("\n" + "=" * 60)
        print("MODEL PARAMETER SHAPES")
        print("=" * 60)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  {name:<50s} | {str(list(param.shape)):<20s} | {param.numel():,}")
        print("=" * 60)

    # -- Train --
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(
        config["experiment"]["output_dir"],
        f"step1/{dataset_name}/{split_method}/seed_{split_seed}/{timestamp}",
    )

    trainer = Step1Trainer(
        model=model, config=config, device=device, experiment_dir=exp_dir
    )
    results = trainer.train(train_loader, valid_loader, test_loader)
    if config["experiment"].get("save", True):
        print(f"\nResults saved to: {exp_dir}")

    # Xuất encoder warm-start (best model đã được trainer restore vào model). Độc lập
    # với --save: per-seed path khớp split seed để eggroll nạp đúng (tránh leak).
    encoder_out = config["experiment"].get("encoder_out")
    if encoder_out:
        encoder_path = encoder_out.format(seed=split_seed)
        os.makedirs(os.path.dirname(encoder_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), encoder_path)
        print(f"Encoder warm-start saved to: {encoder_path}")

    return results


def print_overrides(parser: argparse.ArgumentParser, args: argparse.Namespace):
    """In gọn các hyper được truyền khác giá trị default."""
    defaults = parser.parse_args([])
    changed = {k: v for k, v in vars(args).items() if getattr(defaults, k) != v}
    if changed:
        items = ", ".join(f"{k}={v}" for k, v in changed.items())
        print(f"Hyper khác default: {items}")
    else:
        print("Hyper: tất cả default")


def _metric_of(test_metrics: dict):
    """Lấy (tên_metric, giá_trị) chính từ test_metrics."""
    if "rmse" in test_metrics:
        return "RMSE", test_metrics["rmse"]
    if "auc" in test_metrics:
        return "AUC", test_metrics["auc"]
    return None, None


def main():
    parser = build_parser()
    args = parser.parse_args()
    print_overrides(parser, args)

    # Device
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    seeds = args.seed_split  # luôn là list (nargs='+')
    dataset_name = args.dataset
    scores = {}        # seed -> giá trị metric
    metric_name = None

    for i, seed in enumerate(seeds):
        config = build_config(args)
        config["data"]["random_seed_split"] = seed
        if len(seeds) > 1:
            print(f"\n{'#'*60}\n# split seed {seed} ({i+1}/{len(seeds)})\n{'#'*60}")
        results = run_step1(config, device)
        name, val = _metric_of(results.get("test_metrics", {}))
        if val is not None:
            scores[seed] = val
            metric_name = name
            print(f"\n{dataset_name} (seed {seed}): {name}={val:.4f}")

    # Tổng kết trung bình khi quét nhiều seed
    if len(seeds) > 1 and scores:
        vals = list(scores.values())
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"\n{'='*60}")
        print(f"Tổng kết {len(vals)} seed — {dataset_name}:")
        for s, v in scores.items():
            print(f"  seed {s}: {metric_name}={v:.4f}")
        print(f"  Trung bình {metric_name} = {mean:.4f} ± {std:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
