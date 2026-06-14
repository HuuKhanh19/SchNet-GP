#!/usr/bin/env python
"""Exp 3 — Co-train encoder + GP head (delta, K=1) bằng two-timescale.

THIẾT KẾ (xem spec mục 0): DEAP đầy đủ trong mỗi bước ES là bất khả thi -> two-timescale:
  - FAST (mỗi bước): trees CỐ ĐỊNH; cập nhật ENCODER. Ridge refit closed-form.
      * method=eggroll : low-rank ES (forward-only) -> dùng được CẢ funcset non-diff.
      * method=backprop: ablation — gradient qua trees khả vi + ridge (torch) vào encoder.
  - SLOW (mỗi --slow-every bước): DEAP re-evolve trees trên embedding hiện tại,
      warm-start từ best cũ.

Hai nhánh function set:
  - diff   (Nhánh A): so eggroll vs backprop head-to-head (cùng head khả vi).
  - nondiff(Nhánh B): thêm gt/ifte/min/max/step -> backprop chết -> chỉ eggroll
      (đây là đóng góp không thể thay thế của eggroll).

Encoder warm-start từ checkpoint baseline (KHÔNG train from scratch). Target = residual
trên GBT-2D (Exp 0). Model-selection theo val final RMSE + floor safeguard (worst case
= baseline). ESOL là control: kỳ vọng ở lại ~baseline ổn định (win để dành FreeSolv/QM7).

Ví dụ (server):
  python scripts/run_exp3.py --dataset esol --seed-split 0 --save \
      --method eggroll --funcset diff --fast-steps 300 --slow-every 50
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.func import functional_call, vmap
from torch_geometric.utils import scatter

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_step1 import build_parser, print_overrides
from src.config import build_config
from src.data.data_loader import prepare_dataset, save_splits, create_dataloaders
from src.models.schnet import build_schnet_model
from src.utils.utils import seed_everything
from src.gp.gp_head import (GPConfig, run_gp_head, make_partition, make_psets,
                            assemble_phi, fit_ridge, ridge_predict, Compiler, rmse)
from src.cotrain.eggroll import EggrollES
from src.cotrain.torch_trees import assemble_phi_torch, ridge_torch

ALPHA_GRID = np.logspace(-2, 4, 13)


# =============================================================================
# Data / embedding helpers
# =============================================================================

def one_batch(loader, device):
    """Gộp toàn dataset thành 1 batch tensor (ESOL nhỏ -> full-batch)."""
    batch = next(iter(loader))
    return {k: v.to(device) for k, v in batch.items()}, list(loader.dataset.smiles)


def pool(h, batch):
    """mean-pool atom->conf->mol. h: (...,A,128) -> (...,B,128)."""
    conf = scatter(h, batch["_idx_atom_to_conf"], dim=-2, reduce="mean")
    return scatter(conf, batch["_idx_conf_to_mol"], dim=-2, reduce="mean")


def emb_of(model, batch):
    h = model(batch, return_atom_emb_only=True)["atom_embeddings"]
    return pool(h, batch)


def load_gbt_residual(exp0_dir, dataset, sm, seed, smiles_by_split, y_by_split):
    path = os.path.join(exp0_dir, dataset, sm, f"seed_{seed}", "gbt_pred.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Thiếu baseline Exp 0: {path} -> chạy run_exp0.py.")
    base = pd.read_csv(path)
    out = {}
    for s in ("train", "valid", "test"):
        sub = base[base["split"] == s]
        smi2pred = dict(zip(sub["smiles"], sub["pred"]))
        bp = np.array([smi2pred[x] for x in smiles_by_split[s]], dtype=np.float64)
        out[s] = {"baseline": bp, "resid": y_by_split[s] - bp}
    return out


# =============================================================================
# Eval một snapshot (encoder + trees) -> final RMSE (đơn vị log-S, raw residual)
# =============================================================================

def eval_snapshot(model, batches, resid, best_ind, partition, psets, compiler):
    with torch.no_grad():
        e = {s: emb_of(model, batches[s]).cpu().numpy() for s in ("train", "valid", "test")}
    mean, std = e["train"].mean(0), e["train"].std(0)
    std[std < 1e-8] = 1.0
    z = {s: (e[s] - mean) / std for s in e}
    phi = {s: assemble_phi(best_ind, z[s], partition, psets, compiler) for s in z}
    r_tr, r_va, r_te = resid["train"]["resid"], resid["valid"]["resid"], resid["test"]["resid"]
    best_a, best_v = ALPHA_GRID[0], float("inf")
    for a in ALPHA_GRID:
        w, b = fit_ridge(phi["train"], r_tr, a)
        v = rmse(ridge_predict(phi["valid"], w, b), r_va)
        if v < best_v:
            best_v, best_a = v, float(a)
    w, b = fit_ridge(phi["train"], r_tr, best_a)
    pred_va = ridge_predict(phi["valid"], w, b)
    pred_te = ridge_predict(phi["test"], w, b)
    return {"val_rmse": rmse(pred_va, r_va), "test_rmse": rmse(pred_te, r_te),
            "alpha": best_a, "scaler": (mean, std), "ridge": (w, b)}


# =============================================================================
# Fast loop: eggroll
# =============================================================================

def _forward_pool_np(model, params, batch_tr):
    h = functional_call(model, params, (batch_tr,), {"return_atom_emb_only": True})
    return pool(h["atom_embeddings"], batch_tr).cpu().numpy()


def fast_eggroll(model, es, batch_tr, r_tr, best_ind, partition, psets, compiler,
                 use_vmap, chunk, device):
    """1 bước ES: forward N perturbation -> ridge per-perturbation -> losses.

    Mặc định loop tuần tự (robust: functional_call+vmap+radius_graph fragile ở full
    scale — xem spec mục 2 fallback). --vmap bật batched (nhanh hơn trên GPU, có thể
    lỗi functorch tùy version).
    """
    stacked2d = es.sample()
    N = es.popsize
    base_all = {k: v.detach() for k, v in model.named_parameters()}
    losses = np.empty(N, dtype=np.float64)

    def loss_from_emb(emb_np):
        z = (emb_np - SCALER[0]) / SCALER[1]
        phi = assemble_phi(best_ind, z, partition, psets, compiler)
        w, b = fit_ridge(phi, r_tr, es_alpha)
        return rmse(ridge_predict(phi, w, b), r_tr)

    with torch.no_grad():
        if use_vmap:
            def f(p):
                h = functional_call(model, p, (batch_tr,), {"return_atom_emb_only": True})
                return pool(h["atom_embeddings"], batch_tr)
            for c0 in range(0, N, chunk):
                c1 = min(c0 + chunk, N)
                m = c1 - c0
                sp = {k: (stacked2d[k][c0:c1] if k in stacked2d
                          else v.unsqueeze(0).expand(m, *v.shape).contiguous())
                      for k, v in base_all.items()}
                emb = vmap(f)(sp).cpu().numpy()
                for i in range(m):
                    losses[c0 + i] = loss_from_emb(emb[i])
        else:
            for i in range(N):
                p = {k: (stacked2d[k][i] if k in stacked2d else base_all[k])
                     for k in base_all}
                losses[i] = loss_from_emb(_forward_pool_np(model, p, batch_tr))
    es.update(torch.tensor(losses, device=device))
    return float(losses.min())


# scaler + α cố định trong pha fast (refit ở slow step) — set bởi run_one_seed.
SCALER = (0.0, 1.0)
es_alpha = 1.0


# =============================================================================
# Fast loop: backprop-qua-trees (ablation, chỉ funcset diff)
# =============================================================================

def fast_backprop(model, opt, batch_tr, r_tr_t, best_ind, partition, device):
    model.train()
    opt.zero_grad()
    emb = emb_of(model, batch_tr)                          # (B,128) torch, grad
    z = (emb - SCALER_T[0]) / SCALER_T[1]
    phi = assemble_phi_torch(best_ind, z, partition)
    pred, _, _ = ridge_torch(phi, r_tr_t, es_alpha)
    loss = ((pred - r_tr_t) ** 2).mean()
    loss.backward()
    opt.step()
    return float(loss.detach().sqrt())


SCALER_T = (None, None)


# =============================================================================
# Một split
# =============================================================================

def run_one_seed(args, device, seed):
    global SCALER, SCALER_T, es_alpha
    es_alpha = args.ridge_alpha
    config = build_config(args)
    config["data"]["random_seed_split"] = seed
    config["training"]["batch_size"] = 10 ** 9  # full-batch
    seed_everything(config["random_seed_train"], deterministic=config["deterministic"])
    sm = config["data"]["split_method"]
    print(f"\n{'#'*60}\n# {args.dataset} seed {seed} | method={args.method} | "
          f"funcset={args.funcset} | q={args.q}\n{'#'*60}")
    if args.method == "backprop" and args.funcset == "nondiff":
        raise SystemExit("backprop KHÔNG chạy với funcset nondiff (op không khả vi).")

    # Split + loaders (full-batch) + residual baseline.
    ds_dir = os.path.join(config["data"]["processed_dir"], args.dataset, sm, f"seed_{seed}")
    if all(os.path.exists(os.path.join(ds_dir, f"{s}.csv")) for s in ("train", "valid", "test")):
        dfs = {s: pd.read_csv(os.path.join(ds_dir, f"{s}.csv")) for s in ("train", "valid", "test")}
    else:
        tr, va, te = prepare_dataset(config); save_splits(tr, va, te, ds_dir)
        dfs = {"train": tr, "valid": va, "test": te}
    loaders = dict(zip(("train", "valid", "test"),
                       create_dataloaders(config, dfs["train"], dfs["valid"], dfs["test"])))
    batches, smiles = {}, {}
    for s in ("train", "valid", "test"):
        batches[s], smiles[s] = one_batch(loaders[s], device)
    y_by = {s: np.asarray(loaders[s].dataset.targets, np.float64) for s in batches}
    resid = load_gbt_residual(args.exp0_dir, args.dataset, sm, seed, smiles, y_by)
    r_tr = resid["train"]["resid"]
    base_te = resid["test"]["baseline"]
    base_test_rmse = rmse(np.zeros_like(resid["test"]["resid"]), resid["test"]["resid"])
    print(f"  Data: " + ", ".join(f"{s}={len(smiles[s])}" for s in batches) +
          f" | GBT-2D test RMSE={base_test_rmse:.4f}")

    # Encoder warm-start từ checkpoint baseline.
    model = build_schnet_model(config).to(device)
    if args.ckpt and os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
        print(f"  Encoder checkpoint: {args.ckpt}")
    elif not args.allow_random:
        raise FileNotFoundError("Cần --ckpt (baseline) hoặc --allow-random (smoke).")
    else:
        print("  ⚠️  Encoder NGẪU NHIÊN (smoke test).")

    gpcfg = GPConfig(q=args.q, emb_dim=model.hidden_channels, pop_size=args.gp_pop,
                     generations=args.slow_gens, es_patience=max(5, args.slow_gens // 2),
                     ridge_alpha=args.ridge_alpha, seed=args.gp_seed, funcset=args.funcset)
    partition = make_partition(gpcfg)
    psets = make_psets(gpcfg)
    compiler = Compiler()

    def refit_scaler():
        global SCALER, SCALER_T
        with torch.no_grad():
            e = emb_of(model, batches["train"]).cpu().numpy()
        mean, std = e.mean(0), e.std(0); std[std < 1e-8] = 1.0
        SCALER = (mean, std)
        SCALER_T = (torch.tensor(mean, dtype=torch.float32, device=device),
                    torch.tensor(std, dtype=torch.float32, device=device))
        return e, (e - mean) / std

    # SLOW step 0: evolve trees lần đầu trên embedding baseline.
    _, z_tr = refit_scaler()
    with torch.no_grad():
        z_va = (emb_of(model, batches["valid"]).cpu().numpy() - SCALER[0]) / SCALER[1]
        z_te = (emb_of(model, batches["test"]).cpu().numpy() - SCALER[0]) / SCALER[1]
    res0 = run_gp_head(z_tr, r_tr, z_va, resid["valid"]["resid"], z_te,
                       resid["test"]["resid"], gpcfg, verbose=False)
    best_ind = res0["best_ind"]
    print(f"  slow#0: trees evolved (val_rmse_std={res0['val_rmse_std']:.4f})")

    # Optimizer cho fast loop.
    if args.method == "eggroll":
        es = EggrollES(model, sigma=args.sigma, lr=args.es_lr, popsize=args.es_pop,
                       seed=seed, device=device)
        opt = None
    else:
        es = None
        opt = torch.optim.Adam([p for p in model.parameters() if p.dim() == 2], lr=args.bp_lr)
    r_tr_t = torch.tensor(r_tr, dtype=torch.float32, device=device)

    # Snapshot ban đầu = baseline-head (floor).
    snap = eval_snapshot(model, batches, resid, best_ind, partition, psets, compiler)
    best = {"val": snap["val_rmse"], "test": snap["test_rmse"], "alpha": snap["alpha"],
            "encoder": copy.deepcopy(model.state_dict()), "trees": copy.deepcopy(best_ind),
            "step": 0}
    val_curve = [(0, snap["val_rmse"])]
    print(f"  init: val={snap['val_rmse']:.4f} test={snap['test_rmse']:.4f} "
          f"(GBT-2D val={rmse(np.zeros_like(resid['valid']['resid']), resid['valid']['resid']):.4f})")

    # ---- Two-timescale loop ----
    for step in range(1, args.fast_steps + 1):
        if args.method == "eggroll":
            tr_rmse = fast_eggroll(model, es, batches["train"], r_tr, best_ind,
                                   partition, psets, compiler, args.vmap, args.chunk, device)
        else:
            tr_rmse = fast_backprop(model, opt, batches["train"], r_tr_t, best_ind,
                                    partition, device)

        if step % args.slow_every == 0:  # SLOW: re-evolve trees (warm-start)
            _, z_tr = refit_scaler()
            with torch.no_grad():
                z_va = (emb_of(model, batches["valid"]).cpu().numpy() - SCALER[0]) / SCALER[1]
                z_te = (emb_of(model, batches["test"]).cpu().numpy() - SCALER[0]) / SCALER[1]
            r = run_gp_head(z_tr, r_tr, z_va, resid["valid"]["resid"], z_te,
                            resid["test"]["resid"], gpcfg,
                            seed_individuals=[best_ind], verbose=False)
            best_ind = r["best_ind"]

        if step % args.eval_every == 0 or step == args.fast_steps:
            snap = eval_snapshot(model, batches, resid, best_ind, partition, psets, compiler)
            val_curve.append((step, snap["val_rmse"]))
            if snap["val_rmse"] < best["val"] - 1e-9:
                best.update(val=snap["val_rmse"], test=snap["test_rmse"], alpha=snap["alpha"],
                            encoder=copy.deepcopy(model.state_dict()),
                            trees=copy.deepcopy(best_ind), step=step)
            print(f"  step {step:4d} | train_resid_rmse={tr_rmse:.4f} | "
                  f"val={snap['val_rmse']:.4f} | best_val={best['val']:.4f}@{best['step']}")

    # Floor safeguard (val-based): không vượt residual=0 trên val -> baseline.
    base_val_rmse = rmse(np.zeros_like(resid["valid"]["resid"]), resid["valid"]["resid"])
    if best["val"] < base_val_rmse:
        final_rmse, chosen = best["test"], "cotrain"
    else:
        final_rmse, chosen = base_test_rmse, "baseline(resid=0)"
    harm = max(0.0, final_rmse - base_test_rmse)
    print(f"  [floor] val: baseline={base_val_rmse:.4f} | best_cotrain={best['val']:.4f} "
          f"-> {chosen}")
    print(f"  FINAL test RMSE={final_rmse:.4f} [{chosen}] | hại so baseline={harm:+.4f} "
          f"| best@step{best['step']}")

    info = {"seed": seed, "method": args.method, "funcset": args.funcset,
            "final_rmse": final_rmse, "baseline_test_rmse": base_test_rmse,
            "cotrain_test_rmse": best["test"], "cotrain_val_rmse": best["val"],
            "chosen": chosen, "harm": harm, "best_step": best["step"],
            "val_curve": val_curve, "trees": [str(t) for t in best["trees"]]}
    if args.save:
        out = os.path.join(args.output_dir, "exp3_cotrain", args.dataset, sm,
                           f"seed_{seed}_{args.method}_{args.funcset}")
        os.makedirs(out, exist_ok=True)
        torch.save(best["encoder"], os.path.join(out, "encoder_best.pt"))
        with open(os.path.join(out, "info.json"), "w") as f:
            json.dump(info, f, indent=2)
        print(f"  Đã lưu: {out}/")
    return info


def main():
    parser = build_parser()
    parser.description = "Exp 3 — co-train encoder + GP head (two-timescale)."
    g = parser.add_argument_group("Exp 3 (co-train)")
    g.add_argument("--method", default="eggroll", choices=["eggroll", "backprop"])
    g.add_argument("--funcset", default="diff", choices=["diff", "nondiff"])
    g.add_argument("--q", type=int, default=8)
    g.add_argument("--fast-steps", type=int, default=300, dest="fast_steps")
    g.add_argument("--slow-every", type=int, default=50, dest="slow_every")
    g.add_argument("--slow-gens", type=int, default=30, dest="slow_gens")
    g.add_argument("--eval-every", type=int, default=10, dest="eval_every")
    g.add_argument("--gp-pop", type=int, default=300, dest="gp_pop")
    g.add_argument("--gp-seed", type=int, default=0, dest="gp_seed")
    g.add_argument("--ridge-alpha", type=float, default=1.0, dest="ridge_alpha")
    # eggroll
    g.add_argument("--sigma", type=float, default=0.03)
    g.add_argument("--es-lr", type=float, default=0.01, dest="es_lr")
    g.add_argument("--es-pop", type=int, default=64, dest="es_pop")
    g.add_argument("--vmap", action="store_true",
                   help="Batched forward bằng vmap (nhanh hơn trên GPU; mặc định TẮT "
                        "vì functional_call+vmap+radius_graph có thể lỗi functorch).")
    g.add_argument("--chunk", type=int, default=16, help="Số perturbation/chunk khi --vmap.")
    # backprop
    g.add_argument("--bp-lr", type=float, default=1e-3, dest="bp_lr")
    # IO
    g.add_argument("--exp0-dir", default="experiments/exp0_baseline2d", dest="exp0_dir")
    g.add_argument("--ckpt", default=None, help="Checkpoint encoder baseline.")
    g.add_argument("--allow-random", action="store_true", dest="allow_random")
    args = parser.parse_args()
    print_overrides(parser, args)

    device = torch.device(f"cuda:{args.gpu}") if (torch.cuda.is_available() and args.gpu >= 0) \
        else torch.device("cpu")
    print(f"Using {device}")

    infos = {s: run_one_seed(args, device, s) for s in args.seed_split}
    finals = [infos[s]["final_rmse"] for s in args.seed_split]
    import statistics
    mean = statistics.mean(finals)
    std = statistics.stdev(finals) if len(finals) > 1 else 0.0
    print(f"\n{'='*60}\nEXP 3 — co-train ({args.dataset}, {args.method}/{args.funcset})\n{'='*60}")
    for s in args.seed_split:
        i = infos[s]
        print(f"  seed {s}: final={i['final_rmse']:.4f} (baseline={i['baseline_test_rmse']:.4f}, "
              f"{i['chosen']}, hại={i['harm']:+.4f})")
    print(f"  final RMSE = {mean:.4f} ± {std:.4f}")
    print(f"  mốc: 0.7917 (Exp2 frozen-GP) | 0.7953 (GBT-2D) | 0.861 (Exp1) | 0.8994 (MLP)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
