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
from src.eggroll.head import compute_counts, init_head_random, count_diagnostics
from src.eggroll.readout import linear_probe, delta_rmse


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
    """Trả dict {h, batch_idx, num_mols, e, y} cho một split (full-batch)."""
    batch = build_full_batch(dataset)
    h, batch_idx, num_mols = extract_atom_features(model, batch, device)
    e = pooled_embedding(h, batch_idx, num_mols)
    y = batch["target"].to(device)
    return {"h": h, "batch_idx": batch_idx, "num_mols": num_mols, "e": e, "y": y}


# =============================================================================
# Run một seed (Stage A: linear-probe gate)
# =============================================================================

def run_seed(config, seed: int, device, smoke: bool) -> dict:
    config["data"]["random_seed_split"] = seed
    eg = config["eggroll"]
    print(f"\n{'='*60}\nEggroll Stage B — {config['dataset_name'].upper()} seed {seed}"
          f"\n{'='*60}")

    datasets = load_datasets(config, smoke)
    model = load_encoder(config, seed, device, smoke)

    tr = embed_split(model, datasets["train"], device)
    va = embed_split(model, datasets["valid"], device)
    te = embed_split(model, datasets["test"], device)
    print(f"e_pooled: train={tuple(tr['e'].shape)} valid={tuple(va['e'].shape)} "
          f"test={tuple(te['e'].shape)}")

    lam = eg["ridge_lambda"]

    # --- T1 floor: linear-probe trên e_pooled (Stage A reference) ---
    probe_test, _ = linear_probe(tr["e"], tr["y"], te["e"], te["y"], lam)
    probe_valid, _ = linear_probe(tr["e"], tr["y"], va["e"], va["y"], lam)
    print(f"[T1 floor] linear-probe RMSE: valid={probe_valid:.4f} test={probe_test:.4f}")

    # --- Stage B: hard-count head (RANDOM init) + delta readout (machinery check) ---
    H = eg["H"]
    W, b = init_head_random(tr["h"], H, fire_rate=0.5, seed=eg["seed_train"])
    c_tr = compute_counts(W, b, tr["h"], tr["batch_idx"], tr["num_mols"])
    c_va = compute_counts(W, b, va["h"], va["batch_idx"], va["num_mols"])
    c_te = compute_counts(W, b, te["h"], te["batch_idx"], te["num_mols"])

    diag = count_diagnostics(c_tr, n_atoms=tr["h"].shape[0])
    print(f"[counts] H={H} fire_rate mean={diag['fire_mean']:.3f} "
          f"[{diag['fire_min']:.3f},{diag['fire_max']:.3f}] "
          f"dead={diag['n_dead']} sat={diag['n_sat']} "
          f"count_max={diag['count_max']:.0f} mean={diag['count_mean']:.2f}")

    co = eg["counts_only"]
    delta_test, _ = delta_rmse(tr["e"], c_tr, tr["y"], te["e"], c_te, te["y"], lam, co)
    delta_valid, _ = delta_rmse(tr["e"], c_tr, tr["y"], va["e"], c_va, va["y"], lam, co)
    delta_train, _ = delta_rmse(tr["e"], c_tr, tr["y"], tr["e"], c_tr, tr["y"], lam, co)
    tag = "counts-only" if co else "delta(T1+T2)"
    print(f"[Stage B] {tag} readout (RANDOM head): train={delta_train:.4f} "
          f"valid={delta_valid:.4f} test={delta_test:.4f}")

    return {"seed": seed, "probe_test": probe_test, "delta_test": delta_test}


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
    probe_scores, delta_scores = [], []
    for seed in args.seed_split:
        res = run_seed(config, seed, device, args.smoke)
        probe_scores.append(res["probe_test"])
        delta_scores.append(res["delta_test"])

    if len(args.seed_split) > 1:
        def _ms(xs):
            return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)
        pm, ps = _ms(probe_scores)
        dm, ds = _ms(delta_scores)
        print(f"\n{'='*60}\nTest RMSE ({len(args.seed_split)} seed):")
        print(f"  T1 floor (linear-probe): {pm:.4f} ± {ps:.4f}")
        print(f"  Stage B delta (RANDOM head): {dm:.4f} ± {ds:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
