#!/usr/bin/env python
"""SchNet (freeze, encoder 1-conf) -> DEAP multi-tree GP head cho ESOL.

Một lệnh chạy trọn pipeline cho từng split seed:
  1. Train encoder SchNet SINGLE-conformer (K=1) bằng training code sẵn có -> freeze.
  2. Extract conf-embedding (mean-pool) + descriptor 2D/3D + energy trên K conformer,
     standardize theo train -> cache (data/processed/<ds>/<split>/seed_<seed>/gp_K<K>).
  3. Chạy DEAP GP head -> RMSE test (denormalize). Quét nhiều seed -> in mean ± std.

Encoder PRETRAIN deterministic + lưu MỘT lần (key theo ds/split/seed, độc lập K). Lần đầu
train encoder + extract; các lần sau DÙNG LẠI checkpoint encoder (không train lại) — kể cả
khi đổi K/num-2d (feature cache mới) thì cũng chỉ extract lại, encoder giữ nguyên.

Ví dụ (server):
    python scripts/run_gp.py --dataset esol --seed-split 0 1 2 3 4 --gpu 0
    # tinh chỉnh GP, dùng lại cache (encoder + feature):
    python scripts/run_gp.py --dataset esol --seed-split 0 1 2 3 4 --warmup 10 --pop 300
    # đổi K -> extract lại nhưng KHÔNG train lại encoder:
    python scripts/run_gp.py --dataset esol --seed-split 0 1 2 3 4 -K 12 --force-extract
"""

import argparse
import json
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
    encoder_ckpt_dir,
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

    # --- Encoder (SchNet K=1) ---
    e = p.add_argument_group("Encoder (SchNet, K=1)")
    e.add_argument("--encoder-epochs", type=int, default=300, dest="encoder_epochs")
    e.add_argument("--cutoff", type=float, default=10.0)
    e.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    e.add_argument("--encoder-deterministic", action=argparse.BooleanOptionalAction,
                   default=True, dest="encoder_deterministic",
                   help="Pretrain encoder deterministic để cố định/lặp lại (mặc định BẬT). "
                        "Encoder chỉ train MỘT lần rồi lưu checkpoint dùng lại.")
    e.add_argument("--force-encoder", action="store_true",
                   help="Train lại encoder dù checkpoint đã có (kéo theo extract lại).")
    e.add_argument("--force-extract", action="store_true",
                   help="Extract lại feature dù cache đã có (encoder vẫn dùng lại checkpoint).")

    # --- Feature ---
    f = p.add_argument_group("Feature")
    f.add_argument("-K", "--num-conformers", type=int, default=10, dest="K",
                   help="Số conformer/phân tử cho GP head.")
    f.add_argument("--num-2d", type=int, default=8, dest="num_2d",
                   help="Số descriptor 2D chọn theo |corr| target trên TRAIN (5–15).")

    # --- GP ---
    gp_ = p.add_argument_group("GP head (DEAP)")
    gp_.add_argument("--num-emb", type=int, default=8, dest="num_emb")
    gp_.add_argument("--num-desc3d", type=int, default=2, dest="num_desc3d")
    gp_.add_argument("--d", type=int, default=16, help="Số chiều mỗi subspace embedding.")
    gp_.add_argument("--pop", type=int, default=1000, dest="pop",
                     help="Cỡ quần thể KHỞI TẠO (đa dạng ban đầu).")
    gp_.add_argument("--mu", type=int, default=300,
                     help="(μ+λ): số cha mẹ giữ lại mỗi thế hệ.")
    gp_.add_argument("--lam", "--lambda", type=int, default=300, dest="lam",
                     help="(μ+λ): số con sinh ra mỗi thế hệ.")
    gp_.add_argument("--generations", type=int, default=200)
    gp_.add_argument("--warmup", type=int, default=0,
                     help="0 = joint ngay; >0 = w thế hệ đầu chỉ tiến hóa L1.")
    gp_.add_argument("--cxpb", type=float, default=0.7)
    gp_.add_argument("--mutpb", type=float, default=0.2)
    gp_.add_argument("--seed-prob", type=float, default=0.15, dest="seed_prob")

    # --- Output ---
    o = p.add_argument_group("Output")
    o.add_argument("--save", action=argparse.BooleanOptionalAction, default=True,
                   help="Lưu kết quả + CÔNG THỨC học được (mọi cây GP, subspace, stats "
                        "denorm) vào experiments/gp/. Mặc định BẬT (--no-save để tắt).")
    o.add_argument("--output-dir", default="experiments", dest="output_dir")
    return p


# =============================================================================
# Encoder train (K=1) + dfs
# =============================================================================

def _encoder_config(args, seed_split: int) -> dict:
    """Config dict (cấu trúc cũ) để train encoder SchNet K=1.

    Pretrain deterministic (mặc định) để cố định/lặp lại; checkpoint lưu rời (xem
    get_encoder) nên KHÔNG cần Step1Trainer ghi đĩa (save=False)."""
    return {
        "dataset_name": args.dataset,
        "random_seed_train": args.seed_train,
        "gpu": args.gpu,
        "deterministic": args.encoder_deterministic,
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
    test_rmse = res.get("test_metrics", {}).get("rmse", float("nan"))
    print(f"  [encoder] single-conf test RMSE = {test_rmse:.4f}")
    # Step1Trainer đã load best_state vào model.
    _freeze(model)
    return model, float(test_rmse)


def _freeze(model: torch.nn.Module) -> None:
    for pm in model.parameters():
        pm.requires_grad_(False)
    model.eval()


def get_encoder(args, config: dict, dfs_provider, seed_split: int,
                device: torch.device) -> torch.nn.Module:
    """Trả encoder freeze: NẠP checkpoint nếu có (train MỘT lần), ngược lại train + lưu.

    Checkpoint key theo (ds, split, seed) — độc lập K, nên đổi K không train lại encoder.
    """
    ck_dir = encoder_ckpt_dir("data/processed", args.dataset, args.split_method, seed_split)
    ck_path = os.path.join(ck_dir, "best_model.pt")
    meta_path = os.path.join(ck_dir, "meta.json")

    if os.path.exists(ck_path) and not args.force_encoder:
        model = build_schnet_model(config)
        state = torch.load(ck_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        _freeze(model)
        cutoff = json.load(open(meta_path)).get("cutoff") if os.path.exists(meta_path) else None
        if cutoff is not None and abs(float(cutoff) - args.cutoff) > 1e-9:
            print(f"  [encoder] CẢNH BÁO: checkpoint cutoff={cutoff} != --cutoff={args.cutoff}; "
                  f"dùng lại checkpoint (xoá thư mục encoder_1conf nếu muốn train lại).")
        print(f"  [encoder] dùng lại checkpoint: {ck_path}")
        return model

    print(f"  [encoder] train MỘT lần (deterministic={config['deterministic']}) "
          f"-> lưu {ck_path}")
    model, test_rmse = train_encoder(config, dfs_provider(), device)
    os.makedirs(ck_dir, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ck_path)
    with open(meta_path, "w") as f:
        json.dump({
            "cutoff": args.cutoff, "n_atom_basis": 128, "n_interactions": 6,
            "n_rbf": 50, "n_filters": 128, "encoder_epochs": args.encoder_epochs,
            "deterministic": config["deterministic"], "seed_train": args.seed_train,
            "single_conf_test_rmse": test_rmse,
        }, f, indent=2)
    print(f"  [encoder] đã lưu checkpoint + meta vào {ck_dir}")
    return model


# =============================================================================
# Per-seed pipeline
# =============================================================================

def run_one_seed(args, seed_split: int, device: torch.device, run_dir: str) -> float:
    cfg = _encoder_config(args, seed_split)
    cache_dir = feature_cache_dir("data/processed", args.dataset,
                                  args.split_method, seed_split, args.K)

    cache = None if (args.force_extract or args.force_encoder) else load_feature_cache(cache_dir)
    if cache is None:
        print(f"  [extract] cache thiếu/buộc làm lại -> extract (K={args.K})")
        # Lazy dfs: chỉ đọc khi cần (encoder train hoặc extract).
        _dfs_cache = {}
        def dfs_provider():
            if not _dfs_cache:
                _dfs_cache.update(_load_dfs(cfg, seed_split))
            return _dfs_cache
        encoder = get_encoder(args, cfg, dfs_provider, seed_split, device)
        cache = build_feature_cache(
            encoder, dfs_provider(), seed_gen=args.seed_gen, K=args.K,
            num_2d=args.num_2d, device=device, optimize_mmff=True, verbose=True,
        )
        save_feature_cache(cache, cache_dir)
    else:
        print(f"  [extract] dùng lại feature cache: {cache_dir}")

    gcfg = GPConfig(
        K=args.K, num_emb=args.num_emb, num_desc3d=args.num_desc3d, d=args.d,
        num_2d=args.num_2d, pop_size=args.pop, mu=args.mu, lam=args.lam,
        generations=args.generations, cxpb=args.cxpb, mutpb=args.mutpb,
        seed_prob=args.seed_prob, warmup=args.warmup, seed=args.seed_train,
    )
    res = run_gp(cache, gcfg, verbose=True)
    print(f"  [GP] seed {seed_split}: test RMSE = {res.test_rmse:.4f} "
          f"(val {res.val_rmse:.4f}, train {res.train_rmse:.4f})")

    if args.save:
        os.makedirs(run_dir, exist_ok=True)
        out = {
            "seed_split": seed_split,
            "test_rmse": res.test_rmse,
            "val_rmse": res.val_rmse,
            "train_rmse": res.train_rmse,
            "best_val_rmse_std": res.best_rmse_std_val,
            "formula": res.formula,          # công thức học được (mọi cây + subspace + stats)
            "history": res.history,
            "encoder_ckpt": os.path.join(
                encoder_ckpt_dir("data/processed", args.dataset,
                                 args.split_method, seed_split), "best_model.pt"),
        }
        path = os.path.join(run_dir, f"seed_{seed_split}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"  [save] kết quả + công thức -> {path}")
    return res.test_rmse


def main():
    args = build_parser().parse_args()
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    run_dir = os.path.join(args.output_dir, "gp", args.dataset, args.split_method,
                           time.strftime("%Y%m%d_%H%M%S"))

    scores = {}
    for i, seed in enumerate(args.seed_split):
        print(f"\n{'#'*64}\n# split seed {seed} ({i+1}/{len(args.seed_split)})\n{'#'*64}")
        t0 = time.time()
        scores[seed] = run_one_seed(args, seed, device, run_dir)
        print(f"  (seed {seed} xong trong {time.time()-t0:.1f}s)")

    print(f"\n{'='*64}")
    print(f"GP head — {args.dataset} (K={args.K}, num_emb={args.num_emb}, "
          f"num_desc3d={args.num_desc3d}, d={args.d}, warmup={args.warmup})")
    vals = list(scores.values())
    for s, v in scores.items():
        print(f"  seed {s}: test RMSE = {v:.4f}")
    mean = statistics.mean(vals) if vals else float("nan")
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    if len(vals) > 1:
        print(f"  Trung bình test RMSE = {mean:.4f} ± {std:.4f}")
        print(f"  (mốc baseline SchNet 1-conf: 0.8994 ± 0.0946)")
    print(f"{'='*64}")

    if args.save:
        os.makedirs(run_dir, exist_ok=True)
        summary = {
            "dataset": args.dataset, "split_method": args.split_method,
            "args": vars(args),
            "scores": {str(s): v for s, v in scores.items()},
            "test_rmse_mean": mean, "test_rmse_std": std,
            "baseline_1conf": "0.8994 ± 0.0946",
        }
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Kết quả lưu ở: {run_dir}")


if __name__ == "__main__":
    main()
