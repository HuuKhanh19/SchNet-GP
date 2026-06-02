"""
Embedding extraction from the frozen Step-1 SchNet encoder (CONAN-SchNet Step 2).

PHASE 0 of the pipeline: run the frozen encoder once over every conformer and
pool the per-atom 128-d hidden states into one vector per conformer.

    h = model(batch, return_atom_emb_only=True)['atom_embeddings']   # (N_atoms, 128)
    conf_emb = pool(h, _idx_atom_to_conf)                            # (num_confs, 128)

Edges are built per-conformer inside the encoder (radius_graph keyed by
_idx_atom_to_conf), so conformers never leak into each other.
"""

import glob
import os
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.models.schnet import build_schnet_model
from src.utils.scatter import scatter_add, scatter_mean


def load_frozen_encoder(config: dict, checkpoint_path: str,
                        device: torch.device) -> torch.nn.Module:
    """Build a SchNet with the Step-1 architecture and load frozen weights."""
    model = build_schnet_model(config)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  Loaded frozen encoder: {checkpoint_path}")
    return model


def resolve_checkpoint(config: dict, dataset_name: str, split_seed: int) -> str:
    """Resolve step2.encoder_checkpoint.

    'auto'  -> newest experiments/step1/{dataset}/seed_{split}/*/best_model.pt
    else    -> treated as an explicit path.
    """
    spec = config['step2'].get('encoder_checkpoint', 'auto')
    if spec and spec != 'auto':
        if not os.path.exists(spec):
            raise FileNotFoundError(f"encoder_checkpoint not found: {spec}")
        return spec

    out_dir = config['experiment']['output_dir']
    pattern = os.path.join(out_dir, 'step1', dataset_name,
                           f'seed_{split_seed}', '*', 'best_model.pt')
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError(
            f"No Step-1 checkpoint found under pattern: {pattern}\n"
            f"Train Step 1 first, or set step2.encoder_checkpoint explicitly."
        )
    return matches[-1]


@torch.no_grad()
def extract_conf_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    pool: str = 'mean',
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Forward the whole loader once and return conformer-level tensors.

    Returns:
        emb       : (total_confs, 128) float32  -- per-conformer embedding
        conf2mol  : (total_confs,)    int64     -- global molecule index per conf
        dE        : (total_confs,)    float32   -- relative MMFF energy per conf
        y         : (n_mol,)          float32   -- target per molecule
    """
    assert pool in ('mean', 'add')
    model.eval()
    emb_list, conf2mol_list, dE_list, y_list = [], [], [], []
    mol_offset = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        h = model(batch, return_atom_emb_only=True)['atom_embeddings']  # (N_atoms,128)

        atom2conf = batch['_idx_atom_to_conf']           # (N_atoms,)
        conf2mol = batch['_idx_conf_to_mol']             # (num_confs,)
        num_confs = conf2mol.shape[0]

        if pool == 'add':
            conf_emb = scatter_add(h, atom2conf, dim_size=num_confs)
        else:
            conf_emb = scatter_mean(h, atom2conf, dim_size=num_confs)

        emb_list.append(conf_emb.float().cpu())
        conf2mol_list.append((conf2mol + mol_offset).cpu())
        dE_list.append(batch['conf_energies'].float().cpu())
        y_list.append(batch['target'].float().cpu())
        mol_offset += int(batch['target'].shape[0])

    emb = torch.cat(emb_list, dim=0)
    conf2mol = torch.cat(conf2mol_list, dim=0).long()
    dE = torch.cat(dE_list, dim=0)
    y = torch.cat(y_list, dim=0)
    print(f"  Extracted: {emb.shape[0]} confs over {y.shape[0]} molecules "
          f"({pool}-pool, dim={emb.shape[1]})")
    return emb, conf2mol, dE, y