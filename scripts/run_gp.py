#!/usr/bin/env python
"""SchNet (freeze, encoder 1-conf) -> DEAP multi-tree GP head cho ESOL.

Một lệnh chạy trọn pipeline cho từng split seed:
  1. Train encoder SchNet SINGLE-conformer (K=1) bằng training code sẵn có -> freeze.
  2. Extract conf-embedding (mean-pool) + descriptor 2D/3D + energy trên K conformer,
     standardize theo train -> cache (data/processed/<ds>/<split>/seed_<seed>/gp_K<K>).
  3. Chạy DEAP GP head -> RMSE test (denormalize). Quét nhiều seed -> in mean ± std.

Lần chạy đầu sẽ train encoder + extract (chậm, cần GPU cho encoder). Các lần sau tinh
chỉnh hyper GP dùng lại cache (bỏ qua encoder) bằng cách KHÔNG truyền --force-extract.

Ví dụ (server):
    python scripts/run_gp.py --dataset esol --seed-split 0 1 2 3 4 --gpu 0
    # tinh chỉnh GP, dùng lại cache:
    python scripts/run_gp.py --dataset esol --seed-split 0 1 2 3 4 --warmup 10 --pop 300
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

from src.config import DATASETS
from src.data.data_loader import prepare_dataset, save_splits, create_dataloaders
from src.models.schnet import build_schnet_model
from src.trainers.step1_trainer import Step1Trainer
from src.utils.utils import seed_everything
from src.gp.features import (
    build_feature_cache, save_feature_cache, load_feature_cache, feature_cache_dir,
)
from src.gp.gp_head import run_gp, GPConfig


# =============================================================================
# Argparse
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SchNet freeze -> DEAP GP head.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("Global")
    g.add_argument("--dataset", default="esol", choices=list(DATASETS))
    g.add_argument("--gpu", type=int, default=0, help="GPU index; -1 = CPU.")
    g.add_argument("--seed-split", type=int, nargs="+", default=[0], dest="seed_split",
                   help="Các split seed (vd 0 1 2 3 4) -> in RMSE mean ± std.")
    g.add_argument("--seed-train", type=int, default=0, dest="seed_train",
                   help="Seed train encoder + GP randomness.")
    g.add_argument("--seed-gen", type=int, default=42, dest="seed_gen",
                   help="Seed sinh conformer (RDKit).")
    g.add_argument("--split-method", default="random_scaffold",
                   choices=["random_scaffold", "random"], dest="split_method")
    g.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                   default=False)

    # --- Encoder (SchNet K=1) ---
    e = p.add_argument_group("Encoder (SchNet, K=1)")
    e.add_argument("--encoder-epochs", type=int, default=300, dest="encoder_epochs")
    e.add_argument("--cutoff", type=float, default=10.0)
    e.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    e.add_argument("--force-extract", action="store_true",
                   help="Train encoder + extract lại dù cache đã có.")

    # --- Feature ---
    f = p.add_argument_group("Feature")
    f.add_argument("-K", "--num-conformers", type=int, default=8, dest="K",
                   help="Số conformer/phân tử cho GP head.")
    f.add_argument("--num-2d", type=int, default=8, dest="num_2d",
                   help="Số descriptor 2D chọn theo |corr| target trên TRAIN (5–15).")

    # --- GP ---
    gp_ = p.add_argument_group("GP head (DEAP)")
    gp_.add_argument("--num-emb", type=int, default=7, dest="num_emb")
    gp_.add_argument("--num-desc3d", type=int, default=2, dest="num_desc3d")
    gp_.add_argument("--d", type=int, default=16, help="Số chiều mỗi subspace embedding.")
    gp_.add_argument("--pop", type=int, default=300, dest="pop")
    gp_.add_argument("--generations", type=int, default=100)
    gp_.add_argument("--warmup", type=int, default=0,
                     help="0 = joint ngay; >0 = w thế hệ đầu chỉ tiến hóa L1.")
    gp_.add_argument("--cxpb", type=float, default=0.7)
    gp_.add_argument("--mutpb", type=float, default=0.2)
    gp_.add_argument("--seed-prob", type=float, default=0.15, dest="seed_prob")
    return p


# =============================================================================
# Encoder train (K=1) + dfs
# =============================================================================

def _encoder_config(args, seed_split: int) -> dict:
    """Config dict (cấu trúc cũ) để train encoder SchNet K=1 (không lưu đĩa)."""
    return {
        "dataset_name": args.dataset,
        "random_seed_train": args.seed_train,
        "gpu": args.gpu,
        "deterministic": args.deterministic,
        "experiment": {"output_dir": "experiments", "verbose": False, "save": False},
        "dataset": DATASETS[args.dataset],
        "data": {
            "raw_dir": "data/raw", "processed_dir": "data/processed",
            "split_method": args.split_method, "random_seed_split": seed_split,
        },
        "conformer": {
            "num_conformers": 1, "max_attempts": 500, "prune_rms_thresh": 0.0,
            "use_random_coords": False, "optimize_mmff": True,
            "random_seed_gen": args.seed_gen,
        },
        "schnet": {
            "n_atom_basis": 128, "n_interactions": 6, "n_rbf": 50,
            "n_filters": 128, "cutoff": args.cutoff, "atomref": None,
            "conf_readout": "mean",
        },
        "training": {
            "epochs": args.encoder_epochs, "batch_size": args.batch_size,
            "learning_rate": 1e-3, "weight_decay": 1e-5,
            "scheduler": "reduce_on_plateau", "scheduler_patience": 25,
            "scheduler_factor": 0.5, "early_stopping_patience": 100,
            "save_checkpoints": False, "gradient_clip": 1.0,
        },
    }


def _load_dfs(config: dict, seed_split: int):
    base = config["data"]["processed_dir"]
    sm = config["data"]["split_method"]
    ds = config["dataset_name"]
    ds_dir = f"{base}/{ds}/{sm}/seed_{seed_split}"
    if os.path.exists(os.path.join(ds_dir, "train.csv")):
        dfs = {n: pd.read_csv(os.path.join(ds_dir, f"{n}.csv"))
               for n in ("train", "valid", "test")}
    else:
        tr, va, te = prepare_dataset(config)
        save_splits(tr, va, te, ds_dir)
        dfs = {"train": tr, "valid": va, "test": te}
    return dfs


def train_encoder(config: dict, dfs: dict, device: torch.device) -> torch.nn.Module:
    """Train SchNet K=1 -> trả model đã nạp best_state (freeze)."""
    seed_everything(config["random_seed_train"], deterministic=config["deterministic"])
    train_loader, valid_loader, test_loader = create_dataloaders(
        config, dfs["train"], dfs["valid"], dfs["test"]
    )
    model = build_schnet_model(config)
    mean_t = float(dfs["train"]["target"].mean())
    std_t = float(dfs["train"]["target"].std())
    model.set_normalization(mean_t, std_t)

    trainer = Step1Trainer(model=model, config=config, device=device,
                           experiment_dir="experiments/_gp_encoder_tmp")
    res = trainer.train(train_loader, valid_loader, test_loader)
    print(f"  [encoder] single-conf test RMSE = "
          f"{res.get('test_metrics', {}).get('rmse', float('nan')):.4f}")
    # Step1Trainer đã load best_state vào model.
    for pm in model.parameters():
        pm.requires_grad_(False)
    model.eval()
    return model


# =============================================================================
# Per-seed pipeline
# =============================================================================

def run_one_seed(args, seed_split: int, device: torch.device) -> float:
    cfg = _encoder_config(args, seed_split)
    cache_dir = feature_cache_dir("data/processed", args.dataset,
                                  args.split_method, seed_split, args.K)

    cache = None if args.force_extract else load_feature_cache(cache_dir)
    if cache is None:
        print(f"  [extract] no cache -> train encoder + extract (K={args.K})")
        dfs = _load_dfs(cfg, seed_split)
        encoder = train_encoder(cfg, dfs, device)
        cache = build_feature_cache(
            encoder, dfs, seed_gen=args.seed_gen, K=args.K, num_2d=args.num_2d,
            device=device, optimize_mmff=True, verbose=True,
        )
        save_feature_cache(cache, cache_dir)
    else:
        print(f"  [extract] dùng lại cache: {cache_dir}")

    gcfg = GPConfig(
        K=args.K, num_emb=args.num_emb, num_desc3d=args.num_desc3d, d=args.d,
        num_2d=args.num_2d, pop_size=args.pop, generations=args.generations,
        cxpb=args.cxpb, mutpb=args.mutpb, seed_prob=args.seed_prob,
        warmup=args.warmup, seed=args.seed_train,
    )
    res = run_gp(cache, gcfg, verbose=True)
    print(f"  [GP] seed {seed_split}: test RMSE = {res.test_rmse:.4f} "
          f"(val {res.val_rmse:.4f}, train {res.train_rmse:.4f})")
    return res.test_rmse


def main():
    args = build_parser().parse_args()
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    scores = {}
    for i, seed in enumerate(args.seed_split):
        print(f"\n{'#'*64}\n# split seed {seed} ({i+1}/{len(args.seed_split)})\n{'#'*64}")
        t0 = time.time()
        scores[seed] = run_one_seed(args, seed, device)
        print(f"  (seed {seed} xong trong {time.time()-t0:.1f}s)")

    print(f"\n{'='*64}")
    print(f"GP head — {args.dataset} (K={args.K}, num_emb={args.num_emb}, "
          f"num_desc3d={args.num_desc3d}, d={args.d}, warmup={args.warmup})")
    vals = list(scores.values())
    for s, v in scores.items():
        print(f"  seed {s}: test RMSE = {v:.4f}")
    if len(vals) > 1:
        mean = statistics.mean(vals)
        std = statistics.stdev(vals)
        print(f"  Trung bình test RMSE = {mean:.4f} ± {std:.4f}")
        print(f"  (mốc baseline SchNet 1-conf: 0.8994 ± 0.0946)")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
