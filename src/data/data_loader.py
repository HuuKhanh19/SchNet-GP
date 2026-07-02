"""
Data Loading and Preprocessing for CONAN-SchNet.

Pipeline: CSV -> preprocess -> scaffold split -> conformer generation -> DataLoader
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Optional, Tuple, List

from .splitter import random_split, random_scaffold_split
from .conformer import inner_smi2coords


# =============================================================================
# Column detection
# =============================================================================

def detect_smiles_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect the SMILES column by common naming conventions."""
    common = ['smiles', 'smi', 'smile', 'canonical_smiles', 'mol']
    for col in df.columns:
        if col.lower() in common:
            return col
    for col in df.columns:
        for name in common:
            if name in col.lower():
                return col
    return None


def detect_target_column(
    df: pd.DataFrame, smiles_col: str, task_type: str
) -> Optional[str]:
    """Auto-detect the target column from numeric columns."""
    common = ['target', 'label', 'y', 'activity', 'value', 'measured', 'class']
    candidates = []
    for col in df.columns:
        if col == smiles_col:
            continue
        try:
            pd.to_numeric(df[col], errors='raise')
            candidates.append(col)
        except Exception:
            pass
    if not candidates:
        return None
    for col in candidates:
        if col.lower() in common:
            return col
    if task_type == 'classification':
        for col in candidates:
            if df[col].dropna().nunique() <= 10:
                return col
    return candidates[0] if candidates else None


# =============================================================================
# Preprocessing
# =============================================================================

def validate_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid using RDKit."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False
    finally:
        RDLogger.EnableLog('rdApp.*')


def target_column_names(n_tasks: int) -> List[str]:
    """Tên cột nhãn trong DataFrame đã tiền xử lý.

    - 1 task  -> ['target']            (giữ nguyên convention cũ, cache tương thích)
    - >1 task -> ['target_0', ...]     (multi-label)
    """
    if n_tasks == 1:
        return ['target']
    return [f'target_{i}' for i in range(n_tasks)]


def preprocess_dataframe(
    df: pd.DataFrame,
    smiles_column: str,
    label_columns: List[str],
    task_type: str,
) -> pd.DataFrame:
    """Clean DataFrame: validate SMILES, remove duplicates, chuẩn hoá cột nhãn.

    Nhãn được đổi tên thành 'target' (single-task) hoặc 'target_0..' (multi-label).
    - single-task (regression/classification): bỏ hàng thiếu nhãn.
    - multi-label: GIỮ ô trống (= nhãn thiếu, xử lý bằng mask ở loss); chỉ bỏ hàng
      không có nhãn nào.
    """
    df = df.copy()
    initial = len(df)
    n_tasks = len(label_columns)

    # Ép nhãn về số (ô trống -> NaN).
    for c in label_columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    if task_type == 'multilabel':
        # Bỏ hàng thiếu SMILES hoặc không có bất kỳ nhãn nào.
        df = df.dropna(subset=[smiles_column])
        df = df[df[label_columns].notna().any(axis=1)]
    else:
        df = df.dropna(subset=[smiles_column] + label_columns)
    dropped = initial - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with NaN")

    valid_mask = df[smiles_column].apply(validate_smiles)
    invalid = (~valid_mask).sum()
    if invalid:
        df = df[valid_mask]
        print(f"  Removed {invalid} invalid SMILES")

    dups = df.duplicated(subset=[smiles_column], keep='first').sum()
    if dups:
        df = df.drop_duplicates(subset=[smiles_column], keep='first')
        print(f"  Removed {dups} duplicate SMILES")

    out_cols = target_column_names(n_tasks)
    df = df[[smiles_column] + label_columns].copy()
    df.columns = ['smiles'] + out_cols

    # Nhãn classification giữ dạng float (0.0/1.0, NaN cho thiếu) để dùng
    # BCEWithLogits + mask; regression cũng float.
    for c in out_cols:
        df[c] = df[c].astype(float)

    return df.reset_index(drop=True)


# =============================================================================
# High-level pipeline
# =============================================================================

def prepare_dataset(
    config: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw CSV, preprocess, and split into train/valid/test DataFrames."""
    random_seed_split = config['data']['random_seed_split']
    split_type = config['data']['split_method']
    print(f"Preparing dataset: split={split_type}, seed={random_seed_split}")

    raw_dir = config['data']['raw_dir']
    filename = config['dataset']['file']
    filepath = os.path.join(raw_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filename}")

    ds_cfg = config['dataset']
    task_type = ds_cfg['task_type']

    # SMILES column
    smiles_col = ds_cfg.get('smiles_column')
    if smiles_col is None or smiles_col not in df.columns:
        smiles_col = detect_smiles_column(df)
    if smiles_col is None:
        raise ValueError(f"Cannot detect SMILES column from {list(df.columns)}")

    # Label columns
    if task_type == 'multilabel':
        label_cols = list(ds_cfg['label_columns'])
    else:
        target_col = ds_cfg.get('target_column')
        if target_col is None or target_col not in df.columns:
            target_col = detect_target_column(df, smiles_col, task_type)
        if target_col is None:
            raise ValueError(f"Cannot detect target column from {list(df.columns)}")
        label_cols = [target_col]

    missing = [c for c in label_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Label columns not found in CSV: {missing}")
    print(f"  SMILES col: {smiles_col}, label cols: {label_cols}")

    df = preprocess_dataframe(df, smiles_col, label_cols, task_type)
    print(f"After preprocessing: {len(df)} molecules")

    # Split
    if split_type == "random":
        train_df, valid_df, test_df = random_split(
            df, ratio_test=0.1, ration_valid=0.1, random_seed=random_seed_split
        )
    elif split_type == "random_scaffold":
        train_df, valid_df, test_df = random_scaffold_split(
            df, df['smiles'].values,
            ratio_test=0.1, ration_valid=0.1,
            random_seed=random_seed_split, dataframe=True,
        )
    else:
        raise ValueError(f"Unknown split method: {split_type}")

    print(f"Split: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")
    return train_df, valid_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Save train/valid/test DataFrames to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    valid_df.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    print(f"Saved splits to {output_dir}/")


# =============================================================================
# SchNet Dataset
# =============================================================================

class SchNetMolDataset(Dataset):
    """PyTorch Dataset that generates/caches 3D conformers for SchNet.

    Each item returns:
        atomic_numbers: (n_atoms,)    int64
        positions:      (k, n_atoms, 3)  float32  (k = num_conformers)
        target:         scalar  float32
    """

    def __init__(self, config: Dict, df: pd.DataFrame, cache_path: str = None):
        self.random_seed_gen = config['conformer']['random_seed_gen']
        self.num_conformers = config['conformer']['num_conformers']
        self.optimize_mmff = config['conformer']['optimize_mmff']
        self.prune_rms_thresh = config['conformer'].get('prune_rms_thresh', 0.0)

        self.n_tasks = config['dataset'].get('n_tasks', 1)
        target_cols = target_column_names(self.n_tasks)

        self.smiles = df['smiles'].tolist()
        # targets: (N, T) float32; NaN = nhãn thiếu (multi-label). mask suy ra
        # trong __getitem__ (1 = có nhãn, 0 = thiếu).
        self.targets = df[target_cols].values.astype(np.float32)

        self.atomic_numbers = []
        self.positions = []

        if cache_path and os.path.exists(cache_path):
            print(f"  Loading conformer cache: {cache_path}")
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            self.atomic_numbers = cache["atomic_numbers"]
            self.positions = cache["positions"]
        else:
            self._generate_conformers()
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(
                        {"atomic_numbers": self.atomic_numbers,
                         "positions": self.positions},
                        f,
                    )
                print(f"  Saved conformer cache: {cache_path}")

        # Remove failed molecules
        self._filter_failures()

    def _generate_conformers(self):
        """Generate 3D conformers for all molecules, sorted by energy."""
        from rdkit.Chem import GetPeriodicTable
        pt = GetPeriodicTable()
 
        print(f"  Generating conformers for {len(self.smiles)} molecules...")
        for i, smi in enumerate(self.smiles):
            # Call with return_energy=True to get energies for sorting
            result = inner_smi2coords(
                smi=smi,
                seed=self.random_seed_gen,
                mode='fast',
                optimize=self.optimize_mmff,
                n_confs=self.num_conformers,
                prune_rms_thresh=self.prune_rms_thresh,
                return_energy=True,
            )
 
            # return_energy=True returns 3 values: (atoms_list, coords_list, energies)
            atoms_list, coords_list, energies = result
 
            if (
                atoms_list is None or len(atoms_list) == 0
                or atoms_list[0] is None or len(atoms_list[0]) == 0
                or coords_list is None or len(coords_list) == 0
            ):
                self.atomic_numbers.append(None)
                self.positions.append(None)
            else:
                z = np.array(
                    [pt.GetAtomicNumber(s) for s in atoms_list[0]],
                    dtype=np.int64,
                )
                n_atoms = len(z)
 
                # Collect valid conformers WITH their energies
                valid_coords = []
                valid_energies = []
                for conf_idx, conf in enumerate(coords_list):
                    conf = np.asarray(conf, dtype=np.float32)
                    if conf.ndim == 2 and conf.shape == (n_atoms, 3):
                        valid_coords.append(conf)
                        # Get energy for this conformer
                        if conf_idx < len(energies):
                            valid_energies.append(float(energies[conf_idx]))
                        else:
                            valid_energies.append(float('inf'))
 
                if len(valid_coords) == 0:
                    self.atomic_numbers.append(None)
                    self.positions.append(None)
                else:
                    # Sort conformers by energy (ascending)
                    # x0 = lowest energy conformer, x_{K-1} = highest
                    energy_arr = np.array(valid_energies, dtype=np.float32)
                    sort_order = np.argsort(energy_arr)
                    sorted_coords = [valid_coords[j] for j in sort_order]
 
                    self.atomic_numbers.append(z)
                    self.positions.append(np.stack(sorted_coords, axis=0))
 
            if (i + 1) % 100 == 0:
                print(f"    Conformer generation: {i+1}/{len(self.smiles)}")

    def _filter_failures(self):
        """Remove molecules where conformer generation failed.

        Bỏ cả phân tử có toạ độ toàn số 0 (fallback khi embed thất bại hoàn toàn)
        vì đồ thị bán kính trên các điểm trùng nhau chỉ là nhiễu.
        """
        def _ok(z, p):
            if z is None or p is None:
                return False
            return bool(np.any(p))  # toàn 0 -> loại

        valid_mask = [_ok(z, p) for z, p in zip(self.atomic_numbers, self.positions)]
        if not all(valid_mask):
            n_fail = sum(1 for v in valid_mask if not v)
            print(f"  WARNING: {n_fail} molecules failed conformer generation")
            self.smiles = [s for s, v in zip(self.smiles, valid_mask) if v]
            self.targets = self.targets[valid_mask]
            self.atomic_numbers = [z for z, v in zip(self.atomic_numbers, valid_mask) if v]
            self.positions = [p for p, v in zip(self.positions, valid_mask) if v]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        z = torch.from_numpy(self.atomic_numbers[idx])
        pos = torch.from_numpy(self.positions[idx])
        y = torch.from_numpy(self.targets[idx])          # (T,)
        mask = torch.isfinite(y).float()                 # (T,) 1=có nhãn
        y = torch.nan_to_num(y, nan=0.0)                 # nhãn thiếu -> 0 (bị mask)

        return {
            "atomic_numbers": z,       # (n_atoms,)
            "positions": pos,          # (k, n_atoms, 3)
            "target": y,               # (T,)
            "target_mask": mask,       # (T,)
            "num_atoms": torch.tensor(z.shape[0], dtype=torch.long),
            "num_conformers": torch.tensor(pos.shape[0], dtype=torch.long),
        }


# =============================================================================
# Collate functions
# =============================================================================

def collate_multi_conformer(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate multi-conformer molecules into a flat batch.

    Output keys (matching SchNet.forward expected inputs):
        _atomic_numbers:   (total_atoms_all_confs,)
        _positions:        (total_atoms_all_confs, 3)
        _idx_atom_to_conf: (total_atoms_all_confs,)
        _idx_conf_to_mol:  (num_total_confs,)
        target:            (batch_size, n_tasks)
        target_mask:       (batch_size, n_tasks)
        num_atoms_per_mol: (batch_size,)
        num_confs_per_mol: (batch_size,)
    """
    atomic_numbers_all = []
    positions_all = []
    atom_to_conf_all = []
    conf_to_mol_all = []
    targets = []
    masks = []
    num_atoms_per_mol = []
    num_confs_per_mol = []

    conf_global_idx = 0

    for mol_idx, item in enumerate(batch):
        z = item["atomic_numbers"]     # (n_atoms,)
        pos = item["positions"]        # (k, n_atoms, 3)
        y = item["target"]

        n_atoms = z.shape[0]
        k = pos.shape[0]

        num_atoms_per_mol.append(n_atoms)
        num_confs_per_mol.append(k)
        targets.append(y)
        masks.append(item["target_mask"])

        for conf_idx in range(k):
            atomic_numbers_all.append(z)
            positions_all.append(pos[conf_idx])   # (n_atoms, 3)
            atom_to_conf_all.append(
                torch.full((n_atoms,), conf_global_idx, dtype=torch.long)
            )
            conf_to_mol_all.append(mol_idx)
            conf_global_idx += 1

    return {
        "_atomic_numbers": torch.cat(atomic_numbers_all, dim=0),
        "_positions": torch.cat(positions_all, dim=0),
        "_idx_atom_to_conf": torch.cat(atom_to_conf_all, dim=0),
        "_idx_conf_to_mol": torch.tensor(conf_to_mol_all, dtype=torch.long),
        "target": torch.stack(targets, dim=0),          # (B, T)
        "target_mask": torch.stack(masks, dim=0),        # (B, T)
        "num_atoms_per_mol": torch.tensor(num_atoms_per_mol, dtype=torch.long),
        "num_confs_per_mol": torch.tensor(num_confs_per_mol, dtype=torch.long),
    }


# =============================================================================
# DataLoader factory
# =============================================================================

def create_dataloaders(
    config: Dict,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/valid/test DataLoaders with conformer caching."""
    dataset_name = config['dataset']['name']
    seed = config['data']['random_seed_split']
    split_method = config['data']['split_method']
    n_conf = config['conformer']['num_conformers']

    # Cache key gồm split_method: split khác nhau -> thành phần train/valid/test
    # khác nhau, nên không được dùng chung cache (bug cũ: đổi split_method mà
    # kết quả không đổi vì load lại cache của split cũ).
    cache_dir = os.path.join(
        config['data']['processed_dir'],
        dataset_name,
        split_method,
        f"seed_{seed}",
        f"{n_conf}_conformers",
    )

    datasets = {}
    for split_name, df in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
        cache_path = os.path.join(cache_dir, f'{split_name}.pkl')
        datasets[split_name] = SchNetMolDataset(
            config=config, df=df, cache_path=cache_path,
        )

    bs = config['training']['batch_size']

    # Create a seeded generator for reproducible shuffling in train loader
    train_seed = config.get('random_seed_train', 0)
    g = torch.Generator()
    g.manual_seed(train_seed)

    train_loader = DataLoader(
        datasets['train'], batch_size=bs, shuffle=True,
        collate_fn=collate_multi_conformer, num_workers=0, pin_memory=True,
        generator=g,
    )
    valid_loader = DataLoader(
        datasets['valid'], batch_size=bs, shuffle=False,
        collate_fn=collate_multi_conformer, num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        datasets['test'], batch_size=bs, shuffle=False,
        collate_fn=collate_multi_conformer, num_workers=0, pin_memory=True,
    )

    return train_loader, valid_loader, test_loader