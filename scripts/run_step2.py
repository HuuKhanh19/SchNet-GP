#!/usr/bin/env python
"""Step 2: frozen SchNet + energy-MoE of MFC (EvoGP) experts.

Usage (same data conventions as run_step1.py):
    python scripts/run_step2.py dataset_name=esol
    python scripts/run_step2.py dataset_name=lipo step2.gate.num_experts=3
    python scripts/run_step2.py dataset_name=freesolv step2.delta_learning=false

If a `step2:` block is absent from configs/base.yaml, the built-in DEFAULT_STEP2
below is used (and can be overridden from the CLI as shown).
"""

import os
import sys
import time
import copy

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.data_loader import prepare_dataset, save_splits, create_dataloaders
from src.models.embedding_extractor import load_frozen_encoder, resolve_checkpoint
from src.trainers.step2_trainer import Step2Trainer
from src.utils.utils import seed_everything


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
        'elite_rate': 0.01,                    # sr_test
        'const_prob': 0.5,
        'out_prob': 0.5,
        'layer_leaf_prob': 0.2,
        'using_funcs': ['+', '-', '*', '/'],   # symbols in FUNCS_NAMES; add 'sin','cos','exp','log' to ablate
        'const_range': [-3.0, 3.0],            # standardized embeddings ~ +/-3 sigma
        'sample_cnt': 8,                       # # of constant candidates (sr_test)
        'ridge_alphas': [0.001, 0.01, 0.1, 1.0, 10.0],
        'gp_seed': 0,
    },
    'agg': 'learned_softmax',      # mean | boltzmann | learned_softmax
    'tau_init': 0.593,             # RT (kcal/mol)
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


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


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    dataset_name = cfg.dataset_name
    assert dataset_name in cfg.datasets, (
        f"Unknown dataset: {dataset_name}. Choose from: {list(cfg.datasets.keys())}"
    )

    config = OmegaConf.to_container(cfg, resolve=True)
    config['dataset'] = config['datasets'][dataset_name]
    # merge built-in step2 defaults with anything present in the yaml/CLI
    config['step2'] = _deep_merge(DEFAULT_STEP2, config.get('step2', {}))

    gpu = cfg.get('gpu', 0)
    if torch.cuda.is_available() and gpu >= 0:
        device = torch.device(f"cuda:{gpu}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    results = run_step2(config, device)
    tm = results.get('test_metrics', {})
    print(f"\n{dataset_name}: Test RMSE={tm.get('rmse'):.4f}, MAE={tm.get('mae'):.4f}")


if __name__ == "__main__":
    main()