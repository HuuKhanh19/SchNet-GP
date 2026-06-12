#!/usr/bin/env python
"""Phase 3 entrypoint: lưới GP head 3 (tree_sharing) x 2 (encoder_conformers) x N seed.

Quy trình mỗi (encoder_conformers, seed):
  1. Nạp encoder SchNet ĐÓNG BĂNG (train sẵn bằng scripts/train_encoders.py).
  2. Dựng dataset K=extract.k conformer (tái dùng cache split + conformer của baseline).
  3. Extract + cache embedding conformer + 3D descriptor (Phase 2).
  4. 2D descriptor: tính pool trên SMILES, chọn num_2d theo |corr| TRÊN TRAIN (chống leak).
  5. Standardize emb / desc3d / desc2d / target theo TRAIN.
  6. Subspace cố định (seed) -> với mỗi tree_sharing: evolve GP, eval test, log CSV.

Config: configs/gp.yaml. Override: --set gp.population=200 --set seeds=[0].
"""

import argparse
import ast
import csv
import os
import sys
import time

# Console Windows mặc định cp1252 -> print tiếng Việt có dấu sẽ crash. Ép UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from scripts.run_step1 import build_parser as build_step1_parser
from src.config import build_config
from src.data.data_loader import prepare_dataset, save_splits, SchNetMolDataset
from src.models.schnet import build_schnet_model
import pandas as pd
import random

from src.gp.descriptors import compute_2d_descriptors, select_2d_by_corr, impute_columns
from src.gp.embeddings_cache import get_split_features
from src.gp.subspace import build_subspace
from src.gp.gp_head import build_layout, evolve, predict, GPProblem
from src.gp.fitness import rmse, mae, r2

CSV_COLUMNS = [
    "encoder_conformers", "tree_sharing", "seed", "K", "q",
    "num_emb_trees", "num_desc3d_trees", "d", "num_2d",
    "rmse", "mae", "r2", "genotype_tree_count", "avg_tree_size", "train_time",
]


# =============================================================================
# Config + CLI override
# =============================================================================

def load_config(path: str, overrides: list) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"--set sai cú pháp (key=value): {ov}")
        key, val = ov.split("=", 1)
        try:
            val = ast.literal_eval(val)        # số / list / bool
        except (ValueError, SyntaxError):
            pass                               # giữ string
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return cfg


# =============================================================================
# Dataset + encoder
# =============================================================================

def make_dataset_config(cfg: dict, seed: int, k: int) -> dict:
    args = build_step1_parser().parse_args([])
    args.dataset = cfg["dataset"]
    args.num_conformers = k
    args.seed_split = [seed]
    args.seed_train = cfg.get("seed_train", 0)
    args.split_method = cfg["split_method"]
    args.processed_dir = cfg["processed_dir"]
    args.raw_dir = cfg["raw_dir"]
    config = build_config(args)
    config["data"]["random_seed_split"] = seed
    return config


def load_frozen_encoder(cfg: dict, encoder_conformers: str, seed: int,
                        device: torch.device) -> torch.nn.Module:
    ckpt_path = os.path.join(cfg["encoder"]["ckpt_dir"], cfg["dataset"],
                             encoder_conformers, f"seed_{seed}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Thiếu encoder frozen: {ckpt_path}\n"
            f"Chạy trước: python scripts/train_encoders.py --dataset {cfg['dataset']} "
            f"--encoder {encoder_conformers} --seeds {seed}"
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_schnet_model({"schnet": ckpt["schnet_config"],
                                "dataset": {"task_type": ckpt["task_type"]}})
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_or_prepare_splits(config: dict):
    ds = config["dataset_name"]
    split_method = config["data"]["split_method"]
    seed = config["data"]["random_seed_split"]
    ds_dir = f"{config['data']['processed_dir']}/{ds}/{split_method}/seed_{seed}"
    if os.path.exists(os.path.join(ds_dir, "train.csv")):
        return (pd.read_csv(os.path.join(ds_dir, "train.csv")),
                pd.read_csv(os.path.join(ds_dir, "valid.csv")),
                pd.read_csv(os.path.join(ds_dir, "test.csv")))
    tr, va, te = prepare_dataset(config)
    save_splits(tr, va, te, ds_dir)
    return tr, va, te


def build_sch_dataset(config: dict, df, split: str, k: int) -> SchNetMolDataset:
    ds = config["dataset_name"]
    split_method = config["data"]["split_method"]
    seed = config["data"]["random_seed_split"]
    cache_dir = os.path.join(config["data"]["processed_dir"], ds, split_method,
                             f"seed_{seed}", f"{k}_conformers")
    return SchNetMolDataset(config=config, df=df,
                            cache_path=os.path.join(cache_dir, f"{split}.pkl"))


# =============================================================================
# Standardize features
# =============================================================================

def _standardize_fit(x_train, axis):
    mean = x_train.mean(axis=axis, keepdims=True)
    std = x_train.std(axis=axis, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def build_problems(feats: dict, cfg: dict, desc3d_names, subspace, seed):
    """feats = {'train':{emb,desc3d,targets,smiles}, 'valid':..., 'test':...}."""
    tr = feats["train"]

    # --- target standardization (theo TRAIN) ---
    y_mean = float(np.mean(tr["targets"]))
    y_std = float(np.std(tr["targets"]))
    y_std = y_std if y_std > 1e-8 else 1.0

    # --- 2D descriptor: tính pool + chọn theo |corr| TRÊN TRAIN ---
    pool = cfg["descriptors"]["pool_2d"]
    num_2d = int(cfg["descriptors"]["num_2d"])
    d2_tr_full = compute_2d_descriptors(tr["smiles"], pool)
    sel_idx, sel_names = select_2d_by_corr(d2_tr_full, tr["targets"], pool, num_2d)
    fill = np.nanmedian(np.where(np.isfinite(d2_tr_full), d2_tr_full, np.nan), axis=0)
    fill = np.nan_to_num(fill[sel_idx], nan=0.0)

    # --- standardize stats (TRAIN) ---
    emb_mean, emb_std = _standardize_fit(tr["emb"], axis=(0, 1))      # (1,1,H)
    d3_mean, d3_std = _standardize_fit(tr["desc3d"], axis=(0, 1))     # (1,1,n3d)
    d2_tr_sel = impute_columns(d2_tr_full[:, sel_idx], fill)
    d2_mean, d2_std = _standardize_fit(d2_tr_sel, axis=0)             # (1,num_2d)

    problems = {}
    for split in ("train", "valid", "test"):
        fs = feats[split]
        emb = (fs["emb"] - emb_mean) / emb_std
        d3 = (fs["desc3d"] - d3_mean) / d3_std
        d2_full = (d2_tr_full if split == "train"
                   else compute_2d_descriptors(fs["smiles"], pool))
        d2 = impute_columns(d2_full[:, sel_idx], fill)
        d2 = (d2 - d2_mean) / d2_std
        target_std = (fs["targets"] - y_mean) / y_std
        problems[split] = GPProblem(
            emb=emb.astype(np.float32), desc3d=d3.astype(np.float32),
            desc2d=d2.astype(np.float32), target_std=target_std.astype(np.float32),
            subspace=subspace,
        )
    meta = {"y_mean": y_mean, "y_std": y_std, "sel_2d": sel_names, "num_2d": len(sel_idx)}
    return problems, meta


def append_csv(path: str, row: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="GP head grid runner.")
    ap.add_argument("--config", default="configs/gp.yaml")
    ap.add_argument("--set", action="append", default=[], dest="overrides",
                    help="Override config: --set gp.population=200 --set seeds=[0]")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    cfg = load_config(a.config, a.overrides)
    device = (torch.device(f"cuda:{a.gpu}")
              if torch.cuda.is_available() and a.gpu >= 0 else torch.device("cpu"))
    print(f"Device: {device}")

    k = int(cfg["extract"]["k"])
    desc3d_names = cfg["descriptors"]["desc3d"]
    ops = cfg["gp"]["operators"]
    eph = tuple(cfg["gp"]["ephemeral"])
    csv_path = cfg["eval"]["csv"]

    for enc in cfg["encoder"]["conformers"]:
        for seed in cfg["seeds"]:
            print(f"\n{'#'*64}\n# encoder={enc} seed={seed}\n{'#'*64}")
            model = load_frozen_encoder(cfg, enc, seed, device)

            ds_config = make_dataset_config(cfg, seed, k)
            tr_df, va_df, te_df = load_or_prepare_splits(ds_config)

            feats = {}
            for split, df in (("train", tr_df), ("valid", va_df), ("test", te_df)):
                sch = build_sch_dataset(ds_config, df, split, k)
                feats[split] = get_split_features(
                    model, sch, desc3d_names, k, device,
                    cache_dir=cfg["extract"]["cache_dir"], dataset=cfg["dataset"],
                    split_method=cfg["split_method"], encoder_conformers=enc,
                    seed=seed, split=split, smiles=list(sch.smiles),
                )

            subspace = build_subspace(
                hidden_dim=int(model.hidden_channels),
                emb_dim=int(cfg["subspace"]["emb_dim"]),
                num_emb_trees=int(cfg["subspace"]["num_emb_trees"]),
                num_desc3d_trees=int(cfg["subspace"]["num_desc3d_trees"]),
                desc3d_names=desc3d_names, seed=seed,
            )
            problems, meta = build_problems(feats, cfg, desc3d_names, subspace, seed)

            for mode in cfg["gp"]["tree_sharing"]:
                t0 = time.time()
                rng = random.Random(seed)
                layout = build_layout(mode, k, subspace, meta["num_2d"], ops, eph, rng)
                best, train_fit, stats = evolve(problems["train"], layout, cfg["gp"], seed)
                train_time = time.time() - t0

                pred_std = predict(best, problems["test"], layout)
                pred = pred_std * meta["y_std"] + meta["y_mean"]
                y_test = feats["test"]["targets"]
                row = {
                    "encoder_conformers": enc, "tree_sharing": mode, "seed": seed,
                    "K": k, "q": subspace.q,
                    "num_emb_trees": subspace.num_emb_trees,
                    "num_desc3d_trees": subspace.num_desc3d_trees,
                    "d": subspace.emb_dim, "num_2d": meta["num_2d"],
                    "rmse": round(rmse(pred, y_test), 4),
                    "mae": round(mae(pred, y_test), 4),
                    "r2": round(r2(pred, y_test), 4),
                    "genotype_tree_count": stats["genotype_tree_count"],
                    "avg_tree_size": round(stats["avg_tree_size"], 2),
                    "train_time": round(train_time, 1),
                }
                append_csv(csv_path, row)
                print(f"  [{mode}] test RMSE={row['rmse']} MAE={row['mae']} "
                      f"R2={row['r2']} | trees={row['genotype_tree_count']} "
                      f"avg_size={row['avg_tree_size']} | {train_time:.1f}s")

    print(f"\nXong. Kết quả: {csv_path}")


if __name__ == "__main__":
    main()
