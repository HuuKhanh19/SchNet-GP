#!/usr/bin/env python
"""Step 2: frozen SchNet + energy-MoE of MFC (EvoGP) experts.

Regression AND binary classification share one pipeline (task is read from the
dataset config). Classification uses a logit baseline + functional-gradient
pseudo-residual + sigmoid head + BCE/AUC; see src/trainers/step2_trainer.py.

Usage (same data conventions as run_step1.py):
    python scripts/run_step2.py dataset_name=esol
    python scripts/run_step2.py dataset_name=lipo step2.gate.num_experts=3
    python scripts/run_step2.py dataset_name=freesolv step2.delta_learning=false
    python scripts/run_step2.py dataset_name=bace                 # classification (AUC)

NOTE on BACE / classification: with conformer.num_conformers=1 the energy gate
and Boltzmann aggregation are inert (one conformer -> dE=0 -> single active
expert, trivial aggregation). For a clean run keep step2.gate.num_experts=1, or
raise conformer.num_conformers to actually exercise the energy-MoE on classification.

Per-generation GP logging (PHASE 1) -- print convergence every N generations:
    python scripts/run_step2.py dataset_name=esol step2.expert.log_gp_every=10

Multi-seed in ONE command (mean +/- std over data splits). NOTE: each split
needs its OWN Step-1 checkpoint under
    experiments/step1/{dataset}/seed_{split}/.../best_model.pt
Missing checkpoints are skipped (and excluded from the average):
    python scripts/run_step2.py dataset_name=bace +split_seeds=[0,1,2,3,4]

If a `split_seeds:` key is not in configs/base.yaml, pass it with a leading '+'
(as above) to add it, or add `split_seeds: null` to base.yaml.

If a `step2:` block is absent from configs/base.yaml, the built-in DEFAULT_STEP2
below is used (and can be overridden from the CLI as shown).
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
import time
import copy
import json

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import torch
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.data_loader import prepare_dataset, save_splits, create_dataloaders
from src.models.embedding_extractor import load_frozen_encoder, resolve_checkpoint
from src.trainers.step2_trainer import Step2Trainer
from src.utils.utils import seed_everything, set_determinism


DEFAULT_STEP2 = {
    'encoder_checkpoint': 'auto',
    'emb_pool': 'mean',            # mean | add
    'standardize_emb': True,
    'delta_learning': True,
    'descriptors': 'rdkit2d',      # rdkit2d | morgan | none
    'gate': {
        'num_experts': 3,          # 1 (=MoE off) | 2 | 3
        'gate_by': 'energy',
        'binning': 'quantile',
        'energy_clip': 'train_max',
        'min_conf_per_bin': 100,
    },
    'expert': {
        'q': 6,
        'n_best': 64,
        'pop_size': 1000,
        'generation_limit': 100,               # sr_test uses 100
        'max_tree_len': 128,                   # sr_test
        'max_layer_cnt': 5,                    # generation depth (sr_test)
        'mutation_max_layer_cnt': 3,           # mutation subtree depth (sr_test, shallower)
        'mutation_rate': 0.2,                  # sr_test
        'survival_rate': 0.3,                  # sr_test
        'elite_rate': 0.01,
        'const_prob': 0.5,
        'out_prob': 0.5,
        'layer_leaf_prob': 0.2,
        'using_funcs': ['+', '-', '*', '/', 'sin', 'cos'],   # add 'exp','log' to ablate
        'const_range': [-3.0, 3.0],            # standardized embeddings ~ +/-3 sigma
        'sample_cnt': 8,                       # # of constant candidates (sr_test)
        'ridge_alphas': [0.001, 0.01, 0.1, 1.0, 10.0],
        'gp_seed': 0,
        'log_gp_every': 0,                     # 0 = off; N = log GP convergence every N gens
    },
    'agg': 'learned_softmax',      # mean | boltzmann | learned_softmax
    'tau_init': 0.593,             # RT (kcal/mol)
    'tau_steps': 300,              # Phase-2 Adam steps
    'tau_log_every': 50,           # log train/val metric every N tau steps
    'logit_scale_init': 1.0,       # classification: initial global logit scale (gamma)
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _pick_device(cfg: DictConfig) -> torch.device:
    gpu = cfg.get('gpu', 0)
    if torch.cuda.is_available() and gpu >= 0:
        device = torch.device(f"cuda:{gpu}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def run_step2(config: dict, device: torch.device):
    dataset_name = config['dataset']['name']

    train_seed = config['random_seed_train']
    seed_everything(train_seed)

    split_seed = config['data']['random_seed_split']
    base_dir = config['data']['processed_dir']
    ds_dir = f"{base_dir}/{dataset_name}/seed_{split_seed}"

    if os.path.exists(os.path.join(ds_dir, 'train.csv')):
        print(f"Loading preprocessed data from {ds_dir}")
        train_df = pd.read_csv(os.path.join(ds_dir, 'train.csv'))
        valid_df = pd.read_csv(os.path.join(ds_dir, 'valid.csv'))
        test_df = pd.read_csv(os.path.join(ds_dir, 'test.csv'))
    else:
        print("Preprocessed data not found, running preprocessing...")
        train_df, valid_df, test_df = prepare_dataset(config)
        save_splits(train_df, valid_df, test_df, ds_dir)

    train_loader, valid_loader, test_loader = create_dataloaders(
        config, train_df, valid_df, test_df)

    # frozen encoder (Step-1 best_model.pt for this dataset/split)
    ckpt = resolve_checkpoint(config, dataset_name, split_seed)
    encoder = load_frozen_encoder(config, ckpt, device)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(
        config['experiment']['output_dir'],
        f"step2/{dataset_name}/seed_{split_seed}/{timestamp}",
    )

    trainer = Step2Trainer(config=config, device=device, experiment_dir=exp_dir)
    results = trainer.train(train_loader, valid_loader, test_loader, encoder)
    print(f"\nResults saved to: {exp_dir}")
    return results


def _std(arr: np.ndarray) -> float:
    """Sample std; 0 when there is only one run (avoids ddof=1 NaN)."""
    return float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def _summarize_multiseed(dataset_name: str, per_seed: list, config: dict):
    if not per_seed:
        print("\nNo successful runs to summarize.")
        return

    is_cls = (config['dataset']['task_type'] == 'classification')
    seeds = [r['split_seed'] for r in per_seed]

    print("\n" + "=" * 72)
    print(f"MULTI-SEED SUMMARY  {dataset_name}  (n={len(per_seed)} splits: {seeds})")
    print("=" * 72)

    if is_cls:
        va = np.array([r['valid_metrics']['auc'] for r in per_seed])
        vacc = np.array([r['valid_metrics']['acc'] for r in per_seed])
        ta = np.array([r['test_metrics']['auc'] for r in per_seed])
        tacc = np.array([r['test_metrics']['acc'] for r in per_seed])

        print(f"  {'split':<7}{'val_auc':>10}{'val_acc':>10}{'test_auc':>11}{'test_acc':>10}")
        print("  " + "-" * 48)
        for r in per_seed:
            v, t = r['valid_metrics'], r['test_metrics']
            print(f"  {r['split_seed']:<7}{v['auc']:>10.4f}{v['acc']:>10.4f}"
                  f"{t['auc']:>11.4f}{t['acc']:>10.4f}")
        print("  " + "-" * 48)
        print(f"  {'mean':<7}{np.nanmean(va):>10.4f}{np.nanmean(vacc):>10.4f}"
              f"{np.nanmean(ta):>11.4f}{np.nanmean(tacc):>10.4f}")
        print(f"  {'std':<7}{_std(va):>10.4f}{_std(vacc):>10.4f}"
              f"{_std(ta):>11.4f}{_std(tacc):>10.4f}")
        print("=" * 72)
        print(f"\n{dataset_name}: Test AUC = {np.nanmean(ta):.4f} +/- {_std(ta):.4f}  "
              f"| Test ACC = {np.nanmean(tacc):.4f} +/- {_std(tacc):.4f}  (n={len(per_seed)})")

        summary = {
            'dataset': dataset_name, 'task': 'classification',
            'split_seeds': seeds, 'n_runs': len(per_seed),
            'per_seed': [
                {'split_seed': r['split_seed'], 'tau': r.get('tau'), 'gamma': r.get('gamma'),
                 'valid_metrics': r['valid_metrics'], 'test_metrics': r['test_metrics']}
                for r in per_seed
            ],
            'mean': {'val_auc': float(np.nanmean(va)), 'val_acc': float(np.nanmean(vacc)),
                     'test_auc': float(np.nanmean(ta)), 'test_acc': float(np.nanmean(tacc))},
            'std': {'val_auc': _std(va), 'val_acc': _std(vacc),
                    'test_auc': _std(ta), 'test_acc': _std(tacc)},
        }
    else:
        vr = np.array([r['valid_metrics']['rmse'] for r in per_seed])
        vm = np.array([r['valid_metrics']['mae'] for r in per_seed])
        tr = np.array([r['test_metrics']['rmse'] for r in per_seed])
        tm = np.array([r['test_metrics']['mae'] for r in per_seed])

        print(f"  {'split':<7}{'val_rmse':>10}{'val_mae':>10}{'test_rmse':>11}{'test_mae':>10}")
        print("  " + "-" * 48)
        for r in per_seed:
            v, t = r['valid_metrics'], r['test_metrics']
            print(f"  {r['split_seed']:<7}{v['rmse']:>10.4f}{v['mae']:>10.4f}"
                  f"{t['rmse']:>11.4f}{t['mae']:>10.4f}")
        print("  " + "-" * 48)
        print(f"  {'mean':<7}{vr.mean():>10.4f}{vm.mean():>10.4f}"
              f"{tr.mean():>11.4f}{tm.mean():>10.4f}")
        print(f"  {'std':<7}{_std(vr):>10.4f}{_std(vm):>10.4f}"
              f"{_std(tr):>11.4f}{_std(tm):>10.4f}")
        print("=" * 72)
        print(f"\n{dataset_name}: Test RMSE = {tr.mean():.4f} +/- {_std(tr):.4f}  "
              f"| Test MAE = {tm.mean():.4f} +/- {_std(tm):.4f}  (n={len(per_seed)})")

        summary = {
            'dataset': dataset_name, 'task': 'regression',
            'split_seeds': seeds, 'n_runs': len(per_seed),
            'per_seed': [
                {'split_seed': r['split_seed'], 'tau': r.get('tau'),
                 'valid_metrics': r['valid_metrics'], 'test_metrics': r['test_metrics']}
                for r in per_seed
            ],
            'mean': {'val_rmse': float(vr.mean()), 'val_mae': float(vm.mean()),
                     'test_rmse': float(tr.mean()), 'test_mae': float(tm.mean())},
            'std': {'val_rmse': _std(vr), 'val_mae': _std(vm),
                    'test_rmse': _std(tr), 'test_mae': _std(tm)},
        }

    out_dir = os.path.join(config['experiment']['output_dir'], 'step2', dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'multiseed_summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved multi-seed summary to: {out_path}")


def _print_single(dataset_name: str, split_seed: int, results: dict, is_cls: bool):
    tm = results.get('test_metrics', {})
    if is_cls:
        print(f"\n{dataset_name} (split {split_seed}): "
              f"Test AUC={tm.get('auc'):.4f}, ACC={tm.get('acc'):.4f}, "
              f"logloss={tm.get('logloss'):.4f}")
    else:
        print(f"\n{dataset_name} (split {split_seed}): "
              f"Test RMSE={tm.get('rmse'):.4f}, MAE={tm.get('mae'):.4f}")


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    set_determinism(int(cfg.get('random_seed_train', 42)))

    dataset_name = cfg.dataset_name
    assert dataset_name in cfg.datasets, (
        f"Unknown dataset: {dataset_name}. "
        f"Choose from: {list(cfg.datasets.keys())}"
    )

    config = OmegaConf.to_container(cfg, resolve=True)
    config['dataset'] = config['datasets'][dataset_name]
    # merge built-in step2 defaults with anything present in the yaml/CLI
    config['step2'] = _deep_merge(DEFAULT_STEP2, config.get('step2', {}))
    is_cls = (config['dataset']['task_type'] == 'classification')

    device = _pick_device(cfg)

    # resolve which split seeds to run (default: the single data.random_seed_split)
    ss = cfg.get('split_seeds', None)
    if ss is None:
        split_seeds = [int(config['data']['random_seed_split'])]
    elif isinstance(ss, int):
        split_seeds = [ss]
    else:
        split_seeds = [int(x) for x in ss]

    # single seed -> behaves exactly like before
    if len(split_seeds) == 1:
        cfg_i = copy.deepcopy(config)
        cfg_i['data']['random_seed_split'] = split_seeds[0]
        results = run_step2(cfg_i, device)
        _print_single(dataset_name, split_seeds[0], results, is_cls)
        return

    # multi-seed sweep in one command
    print("\n" + "#" * 70)
    print(f"# MULTI-SEED RUN  {dataset_name}  splits={split_seeds}")
    print("#  each split needs its own Step-1 checkpoint; missing ones are skipped")
    print("#" * 70)

    per_seed = []
    for sd in split_seeds:
        print("\n" + "#" * 70)
        print(f"# SPLIT SEED {sd}")
        print("#" * 70)
        cfg_i = copy.deepcopy(config)
        cfg_i['data']['random_seed_split'] = int(sd)
        try:
            res = run_step2(cfg_i, device)
        except FileNotFoundError as e:
            print(f"  !! Skipping split {sd} (no Step-1 checkpoint?): {e}")
            continue
        except Exception as e:  # noqa: BLE001 - one bad split shouldn't kill the sweep
            print(f"  !! Split {sd} failed: {e!r}")
            continue
        res['split_seed'] = int(sd)
        per_seed.append(res)

    _summarize_multiseed(dataset_name, per_seed, config)


if __name__ == "__main__":
    main()