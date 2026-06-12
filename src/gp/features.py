"""PHASE 2 — extract + cache feature cho GP head.

Quy trình (mỗi (dataset, split_method, seed, K)):
  1. Sinh K conformer/phân tử bằng RDKit (giữ RDKit mol để tính energy + desc3d).
     Conformer sort theo energy tăng dần -> hạng i = bin i (routing PHASE 3).
  2. Encoder SchNet đã FREEZE -> embedding atom (128) -> mean-pool atom->conf -> conf_emb.
     (mean pool: chỉ dùng cho extract, KHÔNG đụng training step 1.)
  3. desc3d (mức conformer), desc2d (mức phân tử) từ RDKit.
  4. Standardize conf_emb / desc3d / desc2d / target theo TRAIN; lưu stats.
  5. Chọn num_2d descriptor theo |corr| target TRÊN TRAIN (chống leak).

Cache ra disk: dict numpy keyed theo (dataset, split, seed, K). Mảng đã pad về K
cố định (phân tử sinh < K conformer: lặp lại conformer năng-lượng-cao-nhất; conf_mask
đánh dấu conformer thật).
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torch import Tensor

from .descriptors import (
    DESC2D_NAMES,
    DESC3D_NAMES,
    compute_desc2d,
    compute_desc3d,
)

RDLogger.DisableLog("rdApp.*")

# Reuse force-field minimizer của pipeline gốc (MMFF, fallback UFF).
from src.data.conformer import _minimize_energy  # noqa: E402


# =============================================================================
# 1. Conformer generation (giữ RDKit mol -> energy + desc3d)
# =============================================================================

@dataclass
class MolConfData:
    """Conformer của 1 phân tử, đã sort theo energy tăng dần (rank 0 = thấp nhất)."""
    z: np.ndarray                 # (n_atoms,) int64, gồm H
    coords: List[np.ndarray]      # m phần tử, mỗi (n_atoms, 3) float32
    energies: np.ndarray          # (m,) float32, đã sort tăng dần (ΔE)
    desc3d: np.ndarray            # (m, n3d) float64, khớp thứ tự conformer
    desc2d: np.ndarray            # (n2d,)  float64


def _embed_confs(mol: Chem.Mol, seed: int, n_confs: int) -> List[int]:
    """ETKDGv3 + retry; fallback random coords. Trả list confId."""
    ps = AllChem.ETKDGv3()
    ps.randomSeed = int(seed)
    ps.numThreads = 0
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(n_confs), params=ps))
    if len(cids) == 0:
        ps.maxIterations = 1000
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(n_confs), params=ps))
    if len(cids) == 0:
        ps.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(n_confs), params=ps))
    return cids


def gen_mol_confs(
    smi: str, seed: int, n_confs: int, optimize: bool = True
) -> Optional[MolConfData]:
    """Sinh tối đa `n_confs` conformer cho 1 SMILES; trả MolConfData hoặc None nếu fail.

    Khớp pipeline training: giữ hydrogen, geometry tối ưu MMFF/UFF (cùng `_minimize_energy`).
    """
    mol_no_h = Chem.MolFromSmiles(smi)
    if mol_no_h is None:
        return None
    mol = AllChem.AddHs(mol_no_h)
    if mol.GetNumAtoms() > 400:
        return None  # quá lớn -> bỏ (khớp guard của inner_smi2coords)

    cids = _embed_confs(mol, seed=seed, n_confs=n_confs)
    if len(cids) == 0:
        return None

    energies = []
    for cid in cids:
        e = _minimize_energy(mol, conf_id=cid) if optimize else np.inf
        energies.append(float(e))
    energies = np.asarray(energies, dtype=np.float64)
    # ΔE theo phân tử (giống inner_smi2coords) -> ổn hơn cho standardize
    if np.isfinite(energies).any():
        energies = energies - np.nanmin(energies[np.isfinite(energies)])

    # Sort conformer theo energy tăng dần (stable; tiebreak = index gốc).
    order = np.argsort(energies, kind="stable")

    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    coords, e_sorted, d3d = [], [], []
    for rank_idx in order:
        cid = int(cids[rank_idx])
        c = mol.GetConformer(cid).GetPositions().astype(np.float32)
        coords.append(c)
        e_sorted.append(energies[rank_idx])
        d3d.append(compute_desc3d(mol, cid))

    desc2d = compute_desc2d(mol_no_h)

    return MolConfData(
        z=z,
        coords=coords,
        energies=np.asarray(e_sorted, dtype=np.float32),
        desc3d=np.asarray(d3d, dtype=np.float64),
        desc2d=desc2d,
    )


# =============================================================================
# 2. Encoder embedding: atom (128) -> mean-pool atom->conf -> conf_emb
# =============================================================================

def _build_encoder_input(items: List[MolConfData]) -> Dict[str, Tensor]:
    """Dựng batch flat (giống collate_multi_conformer) từ list MolConfData."""
    z_all, pos_all, atom_to_conf, conf_to_mol = [], [], [], []
    num_atoms_per_mol, num_confs_per_mol = [], []
    conf_g = 0
    for mol_idx, it in enumerate(items):
        z = torch.from_numpy(it.z)
        n = z.shape[0]
        m = len(it.coords)
        num_atoms_per_mol.append(n)
        num_confs_per_mol.append(m)
        for ci in range(m):
            z_all.append(z)
            pos_all.append(torch.from_numpy(it.coords[ci]))
            atom_to_conf.append(torch.full((n,), conf_g, dtype=torch.long))
            conf_to_mol.append(mol_idx)
            conf_g += 1
    return {
        "_atomic_numbers": torch.cat(z_all, dim=0),
        "_positions": torch.cat(pos_all, dim=0),
        "_idx_atom_to_conf": torch.cat(atom_to_conf, dim=0),
        "_idx_conf_to_mol": torch.tensor(conf_to_mol, dtype=torch.long),
        "num_atoms_per_mol": torch.tensor(num_atoms_per_mol, dtype=torch.long),
        "num_confs_per_mol": torch.tensor(num_confs_per_mol, dtype=torch.long),
    }


@torch.no_grad()
def extract_conf_embeddings(
    model: torch.nn.Module,
    items: List[MolConfData],
    device: torch.device,
    batch_mols: int = 64,
) -> List[np.ndarray]:
    """Trả list (m_i, 128) conf-embedding cho từng phân tử (mean-pool atom->conf)."""
    model.eval()
    out: List[np.ndarray] = []
    for start in range(0, len(items), batch_mols):
        chunk = items[start:start + batch_mols]
        batch = _build_encoder_input(chunk)
        batch = {k: v.to(device) for k, v in batch.items()}
        res = model(batch, return_atom_emb_only=True)
        atom_emb = res["atom_embeddings"]                 # (A, 128)
        idx = batch["_idx_atom_to_conf"]                  # (A,) -> [0, C)
        n_conf = int(idx.max().item()) + 1
        h = atom_emb.shape[1]
        sums = torch.zeros(n_conf, h, device=device, dtype=atom_emb.dtype)
        sums.index_add_(0, idx, atom_emb)
        counts = torch.zeros(n_conf, device=device, dtype=atom_emb.dtype)
        counts.index_add_(0, idx, torch.ones_like(idx, dtype=atom_emb.dtype))
        conf_emb = (sums / counts.clamp(min=1.0).unsqueeze(1)).cpu().numpy()
        # Split lại theo từng phân tử
        m_per_mol = batch["num_confs_per_mol"].cpu().numpy()
        pos = 0
        for m in m_per_mol:
            out.append(conf_emb[pos:pos + m])
            pos += m
    return out


# =============================================================================
# 3. Pad về K cố định + đóng gói 1 split
# =============================================================================

@dataclass
class SplitArrays:
    conf_emb: np.ndarray   # (N, K, 128) float32
    desc3d: np.ndarray     # (N, K, n3d) float32
    desc2d: np.ndarray     # (N, n2d)    float32
    energy: np.ndarray     # (N, K)      float32
    target: np.ndarray     # (N,)        float32
    conf_mask: np.ndarray  # (N, K)      bool   (True = conformer thật)
    smiles: List[str] = field(default_factory=list)


def _assemble_split(
    items: List[MolConfData],
    conf_embs: List[np.ndarray],
    targets: np.ndarray,
    smiles: List[str],
    K: int,
) -> SplitArrays:
    """Pad mỗi phân tử về đúng K conformer (lặp conformer năng-lượng-cao-nhất)."""
    N = len(items)
    n3d = items[0].desc3d.shape[1]
    h = conf_embs[0].shape[1]
    conf_emb = np.zeros((N, K, h), dtype=np.float32)
    desc3d = np.zeros((N, K, n3d), dtype=np.float32)
    energy = np.zeros((N, K), dtype=np.float32)
    desc2d = np.zeros((N, len(DESC2D_NAMES)), dtype=np.float32)
    conf_mask = np.zeros((N, K), dtype=bool)

    for i, (it, emb) in enumerate(zip(items, conf_embs)):
        m = min(len(it.coords), K)
        conf_emb[i, :m] = emb[:m]
        desc3d[i, :m] = it.desc3d[:m]
        energy[i, :m] = it.energies[:m]
        conf_mask[i, :m] = True
        # Pad: lặp conformer hạng cao nhất hiện có (rank m-1)
        if m < K:
            conf_emb[i, m:] = emb[m - 1]
            desc3d[i, m:] = it.desc3d[m - 1]
            energy[i, m:] = it.energies[m - 1]
        desc2d[i] = it.desc2d

    return SplitArrays(
        conf_emb=conf_emb, desc3d=desc3d, desc2d=desc2d,
        energy=energy, target=targets.astype(np.float32),
        conf_mask=conf_mask, smiles=smiles,
    )


# =============================================================================
# 4. Standardize (fit TRAIN) + 5. chọn desc2d theo |corr|
# =============================================================================

@dataclass
class FeatureStats:
    emb_mean: np.ndarray
    emb_std: np.ndarray
    d3d_mean: np.ndarray
    d3d_std: np.ndarray
    d2d_mean: np.ndarray
    d2d_std: np.ndarray
    target_mean: float
    target_std: float
    desc2d_idx: np.ndarray   # index 2D được chọn (theo |corr| train)
    desc2d_names: List[str]
    desc3d_names: List[str]


def _fit_stats(train: SplitArrays, num_2d: int) -> FeatureStats:
    eps = 1e-6
    # Chỉ fit trên conformer THẬT (mask) để pad không lệch thống kê.
    mask = train.conf_mask
    emb_flat = train.conf_emb[mask]            # (sum_m, 128)
    d3d_flat = train.desc3d[mask]              # (sum_m, n3d)
    emb_mean, emb_std = emb_flat.mean(0), emb_flat.std(0) + eps
    d3d_mean, d3d_std = d3d_flat.mean(0), d3d_flat.std(0) + eps
    d2d_mean, d2d_std = train.desc2d.mean(0), train.desc2d.std(0) + eps
    t_mean, t_std = float(train.target.mean()), float(train.target.std()) + eps

    # Chọn num_2d descriptor theo |Pearson corr| với target TRÊN TRAIN.
    d2d_z = (train.desc2d - d2d_mean) / d2d_std
    t_z = (train.target - t_mean) / t_std
    corr = np.array([
        abs(np.corrcoef(d2d_z[:, j], t_z)[0, 1]) if np.std(d2d_z[:, j]) > 0 else 0.0
        for j in range(d2d_z.shape[1])
    ])
    corr = np.nan_to_num(corr)
    num_2d = int(min(max(num_2d, 1), d2d_z.shape[1]))
    desc2d_idx = np.argsort(-corr)[:num_2d]
    desc2d_idx = np.sort(desc2d_idx)

    return FeatureStats(
        emb_mean=emb_mean, emb_std=emb_std,
        d3d_mean=d3d_mean, d3d_std=d3d_std,
        d2d_mean=d2d_mean, d2d_std=d2d_std,
        target_mean=t_mean, target_std=t_std,
        desc2d_idx=desc2d_idx,
        desc2d_names=[DESC2D_NAMES[j] for j in desc2d_idx],
        desc3d_names=list(DESC3D_NAMES),
    )


def _apply_stats(s: SplitArrays, st: FeatureStats) -> Dict[str, np.ndarray]:
    """Standardize + select desc2d -> dict mảng cho GP (target standardized)."""
    return {
        "conf_emb": (s.conf_emb - st.emb_mean) / st.emb_std,
        "desc3d": (s.desc3d - st.d3d_mean) / st.d3d_std,
        "desc2d": ((s.desc2d - st.d2d_mean) / st.d2d_std)[:, st.desc2d_idx],
        "energy": s.energy,
        "target": (s.target - st.target_mean) / st.target_std,
        "target_raw": s.target,
        "conf_mask": s.conf_mask,
        "smiles": np.array(s.smiles, dtype=object),
    }


# =============================================================================
# Orchestration + cache I/O
# =============================================================================

def feature_cache_dir(processed_dir: str, dataset: str, split_method: str,
                      seed: int, K: int) -> str:
    return os.path.join(processed_dir, dataset, split_method,
                        f"seed_{seed}", f"gp_K{K}")


def encoder_ckpt_dir(processed_dir: str, dataset: str, split_method: str,
                     seed: int) -> str:
    """Checkpoint encoder freeze key theo (ds, split, seed) — KHÔNG theo K vì encoder
    train 1-conf độc lập với K của GP. Train MỘT lần, các run sau dùng lại."""
    return os.path.join(processed_dir, dataset, split_method,
                        f"seed_{seed}", "encoder_1conf")


def build_feature_cache(
    model: torch.nn.Module,
    dfs: Dict[str, "pd.DataFrame"],   # noqa: F821  {'train','valid','test'}
    seed_gen: int,
    K: int,
    num_2d: int,
    device: torch.device,
    optimize_mmff: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """Sinh conformer + extract embedding + descriptor cho cả 3 split, fit stats TRAIN.

    Trả dict: {'train','valid','test': arrays}, 'stats': FeatureStats.
    """
    raw_splits: Dict[str, SplitArrays] = {}
    for name in ("train", "valid", "test"):
        df = dfs[name]
        smiles = df["smiles"].tolist()
        targets = df["target"].values.astype(np.float32)

        items: List[MolConfData] = []
        keep_targets, keep_smiles = [], []
        n_fail = 0
        for smi, y in zip(smiles, targets):
            d = gen_mol_confs(smi, seed=seed_gen, n_confs=K, optimize=optimize_mmff)
            if d is None:
                n_fail += 1
                continue
            items.append(d)
            keep_targets.append(y)
            keep_smiles.append(smi)
        if verbose:
            print(f"  [{name}] {len(items)} mols ok, {n_fail} fail conformer-gen")

        conf_embs = extract_conf_embeddings(model, items, device)
        raw_splits[name] = _assemble_split(
            items, conf_embs, np.asarray(keep_targets, dtype=np.float32),
            keep_smiles, K,
        )

    stats = _fit_stats(raw_splits["train"], num_2d=num_2d)
    if verbose:
        print(f"  desc2d chọn ({len(stats.desc2d_names)}): {stats.desc2d_names}")
    out = {name: _apply_stats(raw_splits[name], stats) for name in raw_splits}
    out["stats"] = stats
    out["K"] = K
    return out


def save_feature_cache(cache: Dict[str, object], cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "features.pkl")
    with open(path, "wb") as f:
        pickle.dump(cache, f)
    print(f"  Saved feature cache: {path}")


def load_feature_cache(cache_dir: str) -> Optional[Dict[str, object]]:
    path = os.path.join(cache_dir, "features.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
