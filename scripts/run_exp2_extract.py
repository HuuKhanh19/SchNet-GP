#!/usr/bin/env python
"""Exp 2 — Bước 0+1: trích embedding 128-dim từ encoder SchNet ĐÓNG BĂNG.

Decoupled hoàn toàn: encoder chỉ forward MỘT lần để sinh ma trận feature 128-dim cố
định cho train/val/test mỗi split; GP/ridge ở bước sau chạy thuần trên ma trận này
(không có SchNet trong vòng lặp GP) -> Exp 2 rất nhanh.

Encoder = checkpoint SchNet baseline (E_raw, đã cho 0.8994), đóng băng, đúng từng split.
KHÔNG dùng checkpoint Exp 1 (E_delta) trên ESOL (overfit noise residual).

Embedding (B,128):
  - forward tới sau interaction blocks -> h (N,128) qua hook return_atom_emb_only=True.
  - mean-pool atom->conf (scatter-mean theo _idx_atom_to_conf) rồi conf->mol
    (theo _idx_conf_to_mol; K=1 -> conf->mol là identity). ESOL intensive -> mean.
  - StandardScaler fit TRÊN TRAIN, apply val/test (128 chiều).

Lưu mỗi split: experiments/exp2_embeddings/<ds>/<split_method>/seed_<seed>/
    emb_{train,valid,test}.npy  (đã standardize)
    meta_{train,valid,test}.csv (smiles,y) — căn hàng với .npy
    scaler.json (mean/scale 128-dim) + info.json
=> tái dùng cho cả head GP (Exp 2) lẫn Exp 3.

Ví dụ (server, sau khi đã --save baseline để có best_model.pt):
    python scripts/run_exp2_extract.py --dataset esol --seed-split 0 1 2 3 4 \
        --ckpt-root experiments/step1
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import scatter

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_step1 import build_parser, print_overrides
from src.config import build_config
from src.data.data_loader import prepare_dataset, save_splits, create_dataloaders
from src.models.schnet import build_schnet_model
from src.utils.utils import seed_everything


def find_checkpoint(ckpt_root, dataset, split_method, seed):
    """Tìm best_model.pt baseline cho split này (lấy bản mới nhất nếu nhiều)."""
    pat = os.path.join(ckpt_root, dataset, split_method, f"seed_{seed}", "*",
                       "best_model.pt")
    hits = sorted(glob.glob(pat))
    return hits[-1] if hits else None


@torch.no_grad()
def extract_split(model, loader, device):
    """Forward đóng băng -> (emb (N,128), smiles list) căn theo thứ tự dataset."""
    model.eval()
    embs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        h = model(batch, return_atom_emb_only=True)["atom_embeddings"]  # (N_atoms,128)
        # atom -> conf (mean) -> mol (mean). K=1: conf->mol là identity.
        conf = scatter(h, batch["_idx_atom_to_conf"], dim=0, reduce="mean")
        mol = scatter(conf, batch["_idx_conf_to_mol"], dim=0, reduce="mean")
        embs.append(mol.cpu().numpy())
    emb = np.concatenate(embs, axis=0).astype(np.float64)
    smiles = list(loader.dataset.smiles)
    targets = np.asarray(loader.dataset.targets, dtype=np.float64)
    assert len(smiles) == emb.shape[0] == len(targets)
    return emb, smiles, targets


def run_one_seed(args, device, seed):
    config = build_config(args)
    config["data"]["random_seed_split"] = seed
    seed_everything(config["random_seed_train"], deterministic=config["deterministic"])

    sm = config["data"]["split_method"]
    print(f"\n{'#'*60}\n# {args.dataset} | split seed {seed} | trích embedding 128-dim\n{'#'*60}")

    # Split (đúng cache baseline) + dataloaders (conformer cache dùng chung).
    ds_dir = os.path.join(config["data"]["processed_dir"], args.dataset, sm, f"seed_{seed}")
    paths = {s: os.path.join(ds_dir, f"{s}.csv") for s in ("train", "valid", "test")}
    if all(os.path.exists(p) for p in paths.values()):
        train_df, valid_df, test_df = (pd.read_csv(paths["train"]),
                                       pd.read_csv(paths["valid"]),
                                       pd.read_csv(paths["test"]))
    else:
        train_df, valid_df, test_df = prepare_dataset(config)
        save_splits(train_df, valid_df, test_df, ds_dir)
    train_loader, valid_loader, test_loader = create_dataloaders(
        config, train_df, valid_df, test_df)

    # Encoder đóng băng.
    model = build_schnet_model(config).to(device)
    ckpt = args.ckpt or find_checkpoint(args.ckpt_root, args.dataset, sm, seed)
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"  Encoder checkpoint: {ckpt}")
    elif args.allow_random:
        print("  ⚠️  KHÔNG có checkpoint -> dùng encoder NGẪU NHIÊN (chỉ để smoke test!)")
    else:
        raise FileNotFoundError(
            f"Không thấy checkpoint baseline cho seed {seed} dưới {args.ckpt_root}.\n"
            f"  -> chạy baseline với --save trước, hoặc --ckpt <path>, "
            f"hoặc --allow-random để smoke test.")

    # Trích + standardize (fit train).
    emb_tr, smi_tr, y_tr = extract_split(model, train_loader, device)
    emb_va, smi_va, y_va = extract_split(model, valid_loader, device)
    emb_te, smi_te, y_te = extract_split(model, test_loader, device)

    mean = emb_tr.mean(axis=0)
    scale = emb_tr.std(axis=0)
    scale[scale < 1e-8] = 1.0  # dim hằng -> không chia 0
    norm = lambda X: (X - mean) / scale

    out_dir = os.path.join(args.output_dir, "exp2_embeddings", args.dataset, sm,
                           f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    for split, emb, smi, y in [("train", emb_tr, smi_tr, y_tr),
                               ("valid", emb_va, smi_va, y_va),
                               ("test", emb_te, smi_te, y_te)]:
        np.save(os.path.join(out_dir, f"emb_{split}.npy"), norm(emb))
        pd.DataFrame({"smiles": smi, "y": y}).to_csv(
            os.path.join(out_dir, f"meta_{split}.csv"), index=False)
    with open(os.path.join(out_dir, "scaler.json"), "w") as f:
        json.dump({"mean": mean.tolist(), "scale": scale.tolist()}, f)
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump({"dataset": args.dataset, "split_method": sm, "seed": seed,
                   "dim": int(emb_tr.shape[1]), "checkpoint": ckpt,
                   "n_train": len(smi_tr), "n_valid": len(smi_va),
                   "n_test": len(smi_te)}, f, indent=2)
    print(f"  Lưu embedding {emb_tr.shape[1]}-dim: {out_dir}/ "
          f"(train={len(smi_tr)}, valid={len(smi_va)}, test={len(smi_te)})")


def main():
    parser = build_parser()
    parser.description = "Exp 2 — trích embedding 128-dim từ encoder SchNet đóng băng."
    g = parser.add_argument_group("Exp 2 (extract)")
    g.add_argument("--ckpt", default=None,
                   help="Checkpoint encoder cụ thể (ghi đè auto-find). Dùng khi chạy 1 seed.")
    g.add_argument("--ckpt-root", default="experiments/step1", dest="ckpt_root",
                   help="Gốc để auto-find best_model.pt theo <ds>/<split>/seed_<seed>/*/.")
    g.add_argument("--allow-random", action="store_true", dest="allow_random",
                   help="Cho phép encoder ngẫu nhiên nếu thiếu checkpoint (CHỈ smoke test).")
    args = parser.parse_args()
    print_overrides(parser, args)

    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    for seed in args.seed_split:
        run_one_seed(args, device, seed)


if __name__ == "__main__":
    main()
