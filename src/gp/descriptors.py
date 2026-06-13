"""RDKit descriptor 2D (mức phân tử) + 3D (mức conformer) cho GP head.

- desc2d: tính 1 lần/phân tử từ đồ thị 2D (không phụ thuộc conformer).
- desc3d: tính cho TỪNG conformer (phụ thuộc hình học 3D) qua confId.

Tên cột được giữ ổn định (list có thứ tự) để cache + chọn feature reproduce được.
"""

from __future__ import annotations

from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Descriptors3D, rdMolDescriptors

# --- 2D descriptor (mức phân tử) ---------------------------------------------
# Mỗi entry: (tên, hàm(mol_no_h) -> float). Mol đã bỏ H (canonical 2D).
DESC2D_FUNCS = [
    # ("MolWt", Descriptors.MolWt),
    # ("MolLogP", Crippen.MolLogP),
    # ("TPSA", rdMolDescriptors.CalcTPSA),
    # ("NumHDonors", rdMolDescriptors.CalcNumHBD),
    # ("NumHAcceptors", rdMolDescriptors.CalcNumHBA),
    # ("NumRotatableBonds", rdMolDescriptors.CalcNumRotatableBonds),
    # ("NumAromaticRings", rdMolDescriptors.CalcNumAromaticRings),
    # ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
    # ("NumRings", rdMolDescriptors.CalcNumRings),
    # ("NumHeteroatoms", rdMolDescriptors.CalcNumHeteroatoms),
    # ("LabuteASA", rdMolDescriptors.CalcLabuteASA),
    # ("NumHeavyAtoms", lambda m: float(m.GetNumHeavyAtoms())),
]
DESC2D_NAMES: List[str] = [name for name, _ in DESC2D_FUNCS]

# --- 3D descriptor (mức conformer) -------------------------------------------
# Mỗi entry: (tên, hàm(mol_with_Hs, confId) -> float).
DESC3D_FUNCS = [
    ("RadiusOfGyration", Descriptors3D.RadiusOfGyration),
    ("Asphericity", Descriptors3D.Asphericity),
    ("Eccentricity", Descriptors3D.Eccentricity),
    ("InertialShapeFactor", Descriptors3D.InertialShapeFactor),
    ("PMI1", Descriptors3D.PMI1),
    ("PMI2", Descriptors3D.PMI2),
    ("PMI3", Descriptors3D.PMI3),
    ("NPR1", Descriptors3D.NPR1),
    ("NPR2", Descriptors3D.NPR2),
    ("SpherocityIndex", Descriptors3D.SpherocityIndex),
]
DESC3D_NAMES: List[str] = [name for name, _ in DESC3D_FUNCS]


def compute_desc2d(mol_no_h: Chem.Mol) -> np.ndarray:
    """Vector descriptor 2D (mức phân tử), thứ tự theo DESC2D_NAMES."""
    out = np.empty(len(DESC2D_FUNCS), dtype=np.float64)
    for i, (_, fn) in enumerate(DESC2D_FUNCS):
        try:
            out[i] = float(fn(mol_no_h))
        except Exception:
            out[i] = 0.0
    return out


def compute_desc3d(mol_with_hs: Chem.Mol, conf_id: int) -> np.ndarray:
    """Vector descriptor 3D cho 1 conformer (confId), thứ tự theo DESC3D_NAMES."""
    out = np.empty(len(DESC3D_FUNCS), dtype=np.float64)
    for i, (_, fn) in enumerate(DESC3D_FUNCS):
        try:
            out[i] = float(fn(mol_with_hs, confId=int(conf_id)))
        except Exception:
            out[i] = 0.0
    return out
