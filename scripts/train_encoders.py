#!/usr/bin/env python
"""Train + lưu encoder SchNet ĐÓNG BĂNG cho GP head (Cờ 1: encoder_conformers).

Tái dùng nguyên training code của Step 1 (build_config, data pipeline, SchNet,
Step1Trainer). Lưu checkpoint deterministic để run_gp.py nạp lại:

    <ckpt_dir>/<dataset>/<single|multi>/seed_<seed>.pt

- single: train K=1 conformer (chính là baseline 0.8994).
- multi : train K conformer (mặc định K=10), conf_readout='mean'.

Ví dụ:
    python scripts/train_encoders.py --dataset esol --encoder both \
        --k-multi 10 --seeds 0 1 2 3 4 --gpu 0
"""

import argparse
import os
import sys

import pandas as pd
import torch

# Console Windows mặc định cp1252 -> print tiếng Việt có dấu sẽ crash. Ép UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from scripts.run_step1 import build_parser as build_step1_parser
from src.config import build_config
from src.data.data_loader import prepare_dataset, save_splits, create_dataloaders
from src.models.schnet import build_schnet_model
from src.trainers.step1_trainer import Step1Trainer
from src.utils.utils import seed_everything


def train_one(dataset: str, encoder_conformers: str, k_multi: int, seed: int,
              gpu: int, epochs: int, split_method: str, ckpt_dir: str,
              processed_dir: str, raw_dir: str, deterministic: bool = False) -> str:
    """Train 1 encoder cho (encoder_conformers, seed), lưu checkpoint, trả path."""
    K = 1 if encoder_conformers == "single" else int(k_multi)

    # Lấy default Step 1 rồi override -> đảm bảo encoder train ĐÚNG như baseline.
    args = build_step1_parser().parse_args([])
    args.dataset = dataset
    args.num_conformers = K
    args.seed_split = [seed]
    args.seed_train = 0
    args.split_method = split_method
    args.processed_dir = processed_dir
    args.raw_dir = raw_dir
    args.epochs = epochs
    args.gpu = gpu
    args.save = False
    args.conf_readout = "mean"      # multi: mean ensemble; với K=1 vô hại

    config = build_config(args)
    config["data"]["random_seed_split"] = seed

    seed_everything(args.seed_train, deterministic=deterministic)
    device = (torch.device(f"cuda:{gpu}")
              if torch.cuda.is_available() and gpu >= 0 else torch.device("cpu"))

    print(f"\n{'='*64}\nEncoder '{encoder_conformers}' (K={K}) | {dataset} | seed {seed} "
          f"| {device}\n{'='*64}")

    # Data (tái dùng cache split + conformer của baseline nếu có).
    ds_dir = f"{processed_dir}/{dataset}/{split_method}/seed_{seed}"
    if os.path.exists(os.path.join(ds_dir, "train.csv")):
        train_df = pd.read_csv(os.path.join(ds_dir, "train.csv"))
        valid_df = pd.read_csv(os.path.join(ds_dir, "valid.csv"))
        test_df = pd.read_csv(os.path.join(ds_dir, "test.csv"))
    else:
        train_df, valid_df, test_df = prepare_dataset(config)
        save_splits(train_df, valid_df, test_df, ds_dir)

    train_loader, valid_loader, test_loader = create_dataloaders(
        config, train_df, valid_df, test_df
    )

    model = build_schnet_model(config)
    if config["dataset"]["task_type"] == "regression":
        model.set_normalization(float(train_df["target"].mean()),
                                float(train_df["target"].std()))

    trainer = Step1Trainer(model=model, config=config, device=device,
                           experiment_dir="_tmp_encoder")
    trainer.save = False
    results = trainer.train(train_loader, valid_loader, test_loader)
    # Step1Trainer.train đã restore best_state vào model -> giờ model = encoder tốt nhất.

    path = os.path.join(ckpt_dir, dataset, encoder_conformers, f"seed_{seed}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "schnet_config": config["schnet"],
        "task_type": config["dataset"]["task_type"],
        "encoder_conformers": encoder_conformers,
        "num_conformers": K,
        "seed": seed,
        "split_method": split_method,
        "deterministic": deterministic,
        "target_mean": float(model.target_mean),
        "target_std": float(model.target_std),
        "test_metrics": results.get("test_metrics", {}),
        "best_epoch": results.get("best_epoch", -1),
    }, path)
    print(f"Saved frozen encoder -> {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train + save frozen SchNet encoders cho GP.")
    p.add_argument("--dataset", default="esol")
    p.add_argument("--encoder", default="both", choices=["single", "multi", "both"],
                   help="Encoder nào cần train.")
    p.add_argument("--k-multi", type=int, default=10, dest="k_multi")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Bật torch deterministic khi train encoder -> re-train ra "
                        "cùng weights (chậm hơn; scatter-add GPU warn_only nên không "
                        "đảm bảo tuyệt đối). Mặc định TẮT (giống baseline). KHÔNG cần "
                        "cho độ ổn định lúc extract (đã eval+no_grad+cache).")
    p.add_argument("--split-method", default="random_scaffold", dest="split_method")
    p.add_argument("--ckpt-dir", default="pretrained", dest="ckpt_dir")
    p.add_argument("--processed-dir", default="data/processed", dest="processed_dir")
    p.add_argument("--raw-dir", default="data/raw", dest="raw_dir")
    return p


def main():
    args = build_parser().parse_args()
    encoders = ["single", "multi"] if args.encoder == "both" else [args.encoder]
    for enc in encoders:
        for seed in args.seeds:
            train_one(args.dataset, enc, args.k_multi, seed, args.gpu, args.epochs,
                      args.split_method, args.ckpt_dir, args.processed_dir, args.raw_dir,
                      deterministic=args.deterministic)


if __name__ == "__main__":
    main()
