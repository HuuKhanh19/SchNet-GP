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
from src.eggroll.hooks import (
    build_full_batch, prepare_inputs, forward_atom_features, pooled_embedding,
)
from src.eggroll.curriculum import run_curriculum


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
    """Trả dict {h, batch_idx, num_mols, e, y, inputs} cho một split (full-batch).

    Giữ `inputs` để P2 re-forward encoder với adapter (functional_call).
    """
    batch = build_full_batch(dataset)
    inputs = prepare_inputs(batch, device)
    h, batch_idx, num_mols = forward_atom_features(model, inputs)
    e = pooled_embedding(h, batch_idx, num_mols)
    y = batch["target"].to(device)
    return {"h": h, "batch_idx": batch_idx, "num_mols": num_mols, "e": e, "y": y,
            "inputs": inputs}


# =============================================================================
# Run một seed (Stage A: linear-probe gate)
# =============================================================================

def _apply_smoke(eg: dict):
    """Rút gọn ES cho smoke CPU: N nhỏ (chẵn), ít epoch."""
    eg["pop_size"] = min(eg["pop_size"], 8)
    if eg["pop_size"] % 2:
        eg["pop_size"] -= 1
    eg["p1_epochs"] = min(eg["p1_epochs"], 5)
    eg["p2_epochs"] = min(eg["p2_epochs"], 3)


def run_seed(config, seed: int, device, smoke: bool) -> dict:
    config["data"]["random_seed_split"] = seed
    eg = dict(config["eggroll"])  # copy để smoke override không rò sang seed sau
    if smoke:
        _apply_smoke(eg)
    print(f"\n{'='*60}\nEggroll Stage D (P1+P2) — {config['dataset_name'].upper()} seed {seed}"
          f"\n{'='*60}")

    datasets = load_datasets(config, smoke)
    model = load_encoder(config, seed, device, smoke)

    splits = {name: embed_split(model, datasets[name], device)
              for name in ("train", "valid", "test")}
    print(f"e_pooled: train={tuple(splits['train']['e'].shape)} "
          f"valid={tuple(splits['valid']['e'].shape)} "
          f"test={tuple(splits['test']['e'].shape)}")

    log_every = 1 if smoke else 20
    res = run_curriculum(model, splits, eg, device, log_every=log_every)
    return {"seed": seed, "floor_test": res["floor_test"],
            "test_metric": res["test_metric"], "metric_name": res["metric_name"]}


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Guard chống leak per-seed: quét nhiều seed mà template encoder thiếu '{seed}' ->
    # mọi seed nạp CÙNG một encoder (sai split -> leak test). Hay gặp khi PowerShell nuốt
    # '{seed}' do không quote. (smoke bỏ qua: cho phép encoder random/dùng chung.)
    if (not args.smoke and len(args.seed_split) > 1
            and "{seed}" not in args.encoder):
        parser.error(
            f"--encoder='{args.encoder}' thiếu '{{seed}}' khi quét "
            f"{len(args.seed_split)} seed -> mọi seed nạp cùng 1 encoder (leak). "
            f"Dùng path có '{{seed}}' và QUOTE trên PowerShell, vd:\n"
            f'  --encoder "pretrained/esol/seed_{{seed}}.pt"')

    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    config = build_eggroll_config(args)
    floor_scores, test_scores, mname = [], [], "RMSE"
    for seed in args.seed_split:
        res = run_seed(config, seed, device, args.smoke)
        floor_scores.append(res["floor_test"])
        test_scores.append(res["test_metric"])
        mname = res["metric_name"]

    if len(args.seed_split) > 1:
        def _ms(xs):
            return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)
        fm, fs = _ms(floor_scores)
        tm, ts = _ms(test_scores)
        print(f"\n{'='*60}\nTest {mname} ({len(args.seed_split)} seed):")
        print(f"  T1 floor (linear-probe): {fm:.4f} ± {fs:.4f}")
        print(f"  Eggroll (hard-count):    {tm:.4f} ± {ts:.4f}")
        # Tham chiếu vanilla SchNet nội bộ (deterministic, random_scaffold, 5 seed cùng split)
        ref = {"esol": (0.8994, 0.0946), "bace": (0.7820, 0.0575)}.get(args.dataset)
        if ref:
            print(f"  vanilla SchNet ref:      {ref[0]:.4f} ± {ref[1]:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
