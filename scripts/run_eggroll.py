#!/usr/bin/env python
"""Eggroll: hard-count head trên SchNet warm-start (pure-encoder).

Xem plan + spec §0-§12. Build theo stage:
    Stage A (HIỆN TẠI): nạp encoder per-seed -> e_pooled -> linear-probe (gate ~0.99).
    Stage B+: hard-count head, delta readout, eggroll ES, LoRA, curriculum, diagnostics.

Ví dụ:
    # Gate Stage A trên server (cần encoder đã xuất ở Step 0):
    python scripts/run_eggroll.py --dataset esol --seed-split 0 1 2 3 4 \
        --encoder pretrained/esol/seed_{seed}.pt

    # Smoke test cấu trúc trên CPU (subset nhỏ, encoder random nếu chưa có ckpt):
    python scripts/run_eggroll.py --dataset esol --seed-split 0 --smoke
"""

import argparse
import os
import statistics
import sys

import pandas as pd
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config import DATASETS, build_eggroll_config
from src.data.data_loader import prepare_dataset, save_splits, SchNetMolDataset
from src.models.schnet import build_schnet_model
from src.eggroll.hooks import build_full_batch, extract_atom_features, pooled_embedding
from src.eggroll.readout import linear_probe


# =============================================================================
# Argument parser (toàn bộ surface, kể cả field cho stage sau)
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Eggroll hard-count head trên SchNet warm-start.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Global")
    g.add_argument("--dataset", default="esol", choices=list(DATASETS))
    g.add_argument("--gpu", type=int, default=0, help="GPU index; -1 = CPU.")
    g.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                   default=False)
    g.add_argument("--smoke", action="store_true",
                   help="Smoke test cấu trúc: subset nhỏ, ES rút gọn, encoder random "
                        "nếu thiếu ckpt. Chỉ kiểm không crash + shape (chạy CPU).")

    s = p.add_argument_group("Seeds")
    s.add_argument("--seed-train", type=int, default=0, dest="seed_train")
    s.add_argument("--seed-split", type=int, nargs="+", default=[0], dest="seed_split",
                   help="Một/nhiều split seed; encoder nạp theo từng seed (per-seed).")
    s.add_argument("--seed-gen", type=int, default=42, dest="seed_gen")

    d = p.add_argument_group("Data / Split")
    d.add_argument("--split-method", default="random_scaffold",
                   choices=["random_scaffold", "random"], dest="split_method")
    d.add_argument("--raw-dir", default="data/raw", dest="raw_dir")
    d.add_argument("--processed-dir", default="data/processed", dest="processed_dir")

    c = p.add_argument_group("Conformer")
    c.add_argument("--num-conformers", "-K", type=int, default=1, dest="num_conformers")
    c.add_argument("--max-attempts", type=int, default=500, dest="max_attempts")
    c.add_argument("--prune-rms-thresh", type=float, default=0.0, dest="prune_rms_thresh")
    c.add_argument("--use-random-coords", action=argparse.BooleanOptionalAction,
                   default=False, dest="use_random_coords")
    c.add_argument("--optimize-mmff", action=argparse.BooleanOptionalAction,
                   default=True, dest="optimize_mmff")

    m = p.add_argument_group("Model (SchNet) — phải khớp encoder Step 0")
    m.add_argument("--n-atom-basis", type=int, default=128, dest="n_atom_basis")
    m.add_argument("--n-interactions", type=int, default=6, dest="n_interactions")
    m.add_argument("--n-rbf", type=int, default=50, dest="n_rbf")
    m.add_argument("--n-filters", type=int, default=128, dest="n_filters")
    m.add_argument("--cutoff", type=float, default=10.0)
    m.add_argument("--conf-readout", default="mean", choices=["mean", "add"],
                   dest="conf_readout")
    m.add_argument("--batch-size", type=int, default=32, dest="batch_size")

    e = p.add_argument_group("Eggroll")
    e.add_argument("--encoder", default="pretrained/{dataset}/seed_{seed}.pt",
                   help="Path template encoder warm-start; {dataset},{seed} được thay.")
    e.add_argument("--H", type=int, default=64, help="Số rule hard-count head.")
    e.add_argument("--ridge-lambda", type=float, default=1.0, dest="ridge_lambda")
    e.add_argument("--counts-only", action="store_true", dest="counts_only",
                   help="Bỏ tầng-1 (sàn linear trên e_pooled), chỉ dùng counts.")
    e.add_argument("--pop-size", "-N", type=int, default=256, dest="pop_size")
    e.add_argument("--lora-r", type=int, default=2, dest="lora_r")
    e.add_argument("--lora-alpha", type=float, default=2.0, dest="lora_alpha")
    e.add_argument("--sigma-adapter", type=float, default=0.005, dest="sigma_adapter")
    e.add_argument("--sigma-head", type=float, default=0.03, dest="sigma_head")
    e.add_argument("--es-lr", type=float, default=0.005, dest="es_lr")
    e.add_argument("--p1-epochs", type=int, default=200, dest="p1_epochs")
    e.add_argument("--p2-epochs", type=int, default=50, dest="p2_epochs")
    return p


# =============================================================================
# Data
# =============================================================================

def _load_dataframes(config):
    """Đọc split CSV đã cache (hoặc tạo mới). Giống logic run_step1.run_step1."""
    base_dir = config["data"]["processed_dir"]
    ds = config["dataset_name"]
    split_method = config["data"]["split_method"]
    split_seed = config["data"]["random_seed_split"]
    ds_dir = f"{base_dir}/{ds}/{split_method}/seed_{split_seed}"

    if os.path.exists(os.path.join(ds_dir, "train.csv")):
        print(f"Loading preprocessed data from {ds_dir}")
        train_df = pd.read_csv(os.path.join(ds_dir, "train.csv"))
        valid_df = pd.read_csv(os.path.join(ds_dir, "valid.csv"))
        test_df = pd.read_csv(os.path.join(ds_dir, "test.csv"))
    else:
        print("Preprocessed data not found, running preprocessing...")
        train_df, valid_df, test_df = prepare_dataset(config)
        save_splits(train_df, valid_df, test_df, ds_dir)
    return train_df, valid_df, test_df


def load_datasets(config, smoke: bool):
    """Trả dict {split: SchNetMolDataset}. smoke -> subset nhỏ, không cache."""
    train_df, valid_df, test_df = _load_dataframes(config)

    if smoke:
        train_df = train_df.head(32).reset_index(drop=True)
        valid_df = valid_df.head(16).reset_index(drop=True)
        test_df = test_df.head(16).reset_index(drop=True)
        print(f"[smoke] subset: train={len(train_df)} valid={len(valid_df)} "
              f"test={len(test_df)} (conformer cache OFF)")

    ds = config["dataset_name"]
    split_method = config["data"]["split_method"]
    split_seed = config["data"]["random_seed_split"]
    n_conf = config["conformer"]["num_conformers"]
    cache_dir = os.path.join(config["data"]["processed_dir"], ds, split_method,
                             f"seed_{split_seed}", f"{n_conf}_conformers")

    datasets = {}
    for name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        cache_path = None if smoke else os.path.join(cache_dir, f"{name}.pkl")
        datasets[name] = SchNetMolDataset(config=config, df=df, cache_path=cache_path)
    return datasets


# =============================================================================
# Encoder
# =============================================================================

def load_encoder(config, seed: int, device, smoke: bool):
    """Dựng SchNet + nạp encoder warm-start per-seed. smoke: cho phép random nếu thiếu."""
    model = build_schnet_model(config)
    path = config["eggroll"]["encoder"].format(dataset=config["dataset_name"], seed=seed)
    if os.path.exists(path):
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded encoder warm-start: {path}")
    else:
        if not smoke:
            raise FileNotFoundError(
                f"Không thấy encoder warm-start: {path}. Chạy Step 0 trước:\n"
                f"  python scripts/run_step1.py --dataset {config['dataset_name']} "
                f"--seed-split {seed} --save --encoder-out "
                f"{config['eggroll']['encoder']}")
        print(f"[smoke] encoder không tồn tại ({path}) -> dùng SchNet random.")
    return model.to(device)


def embed_split(model, dataset, device):
    """Trả (e_pooled, y) cho một split (full-batch)."""
    batch = build_full_batch(dataset)
    h, batch_idx, num_mols = extract_atom_features(model, batch, device)
    e = pooled_embedding(h, batch_idx, num_mols)
    y = batch["target"].to(device)
    return e, y


# =============================================================================
# Run một seed (Stage A: linear-probe gate)
# =============================================================================

def run_seed(config, seed: int, device, smoke: bool) -> dict:
    config["data"]["random_seed_split"] = seed
    print(f"\n{'='*60}\nEggroll Stage A — {config['dataset_name'].upper()} seed {seed}"
          f"\n{'='*60}")

    datasets = load_datasets(config, smoke)
    model = load_encoder(config, seed, device, smoke)

    e_train, y_train = embed_split(model, datasets["train"], device)
    e_valid, y_valid = embed_split(model, datasets["valid"], device)
    e_test, y_test = embed_split(model, datasets["test"], device)
    print(f"e_pooled: train={tuple(e_train.shape)} valid={tuple(e_valid.shape)} "
          f"test={tuple(e_test.shape)}")

    lam = config["eggroll"]["ridge_lambda"]
    rmse_train, _ = linear_probe(e_train, y_train, e_train, y_train, lam)
    rmse_valid, _ = linear_probe(e_train, y_train, e_valid, y_valid, lam)
    rmse_test, _ = linear_probe(e_train, y_train, e_test, y_test, lam)
    print(f"linear-probe RMSE: train={rmse_train:.4f} valid={rmse_valid:.4f} "
          f"test={rmse_test:.4f}")
    return {"seed": seed, "probe_rmse_test": rmse_test, "probe_rmse_valid": rmse_valid}


def main():
    args = build_parser().parse_args()

    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    config = build_eggroll_config(args)
    scores = []
    for seed in args.seed_split:
        res = run_seed(config, seed, device, args.smoke)
        scores.append(res["probe_rmse_test"])

    if len(scores) > 1:
        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        print(f"\n{'='*60}\nlinear-probe test RMSE: {mean:.4f} ± {std:.4f} "
              f"({len(scores)} seed)\n{'='*60}")


if __name__ == "__main__":
    main()
