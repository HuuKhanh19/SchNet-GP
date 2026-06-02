#!/usr/bin/env python
"""
Pre-flight environment check for CONAN-SchNet Step 2.
(MFC + ridge head + energy-attention aggregation + energy-routed MoE, on EvoGP)

Run on the TARGET server (the machine with the RTX 5070 Ti GPUs):

    python check_environment.py

Verifies: Python, core numerics, PyTorch + CUDA (incl. Blackwell sm_120 support),
batched ridge on GPU (the Level-1 combiner), PyTorch-Geometric (radius_graph),
the RDKit conformer pipeline, schnetpack, and EvoGP.

Every check is isolated in try/except, so one missing package will NOT abort the rest.
Output is ASCII-only (safe for Windows PowerShell).
"""

import sys
import platform
import importlib

RESULTS = []  # list of (name, status, detail)


def record(name, ok, detail=""):
    # ok: True -> OK, "warn" -> WARN, anything else (incl. False) -> FAIL
    status = "OK" if ok is True else ("WARN" if ok == "warn" else "FAIL")
    RESULTS.append((name, status, detail))
    tag = {"OK": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[status]
    print(f"{tag} {name}" + (f"  ->  {detail}" if detail else ""))


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
section("0. System / Python")
print(f"  Platform : {platform.platform()}")
print(f"  Python   : {sys.version.split()[0]}  ({sys.executable})")
record("Python >= 3.10",
       sys.version_info[:2] >= (3, 10) or "warn",
       f"found {sys.version_info.major}.{sys.version_info.minor}")

# ---------------------------------------------------------------------------
section("1. Core numerics (numpy / pandas / scipy / scikit-learn)")
for pkg in ["numpy", "pandas", "scipy", "sklearn"]:
    try:
        m = importlib.import_module(pkg)
        record(pkg, True, getattr(m, "__version__", "?"))
    except Exception as e:
        record(pkg, False, repr(e))

try:
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.metrics import roc_auc_score
    Xr = np.random.randn(50, 4)
    yr = Xr @ np.array([1.0, -2.0, 0.5, 0.0]) + 0.1
    Ridge(alpha=1.0).fit(Xr, yr)
    _ = roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    record("sklearn Ridge + roc_auc smoke", True)
except Exception as e:
    record("sklearn Ridge + roc_auc smoke", False, repr(e))

# ---------------------------------------------------------------------------
section("2. PyTorch + CUDA  (Blackwell sm_120 is the KEY risk)")
torch = None
try:
    import torch
    record("torch import", True, torch.__version__)
    print(f"  Built with CUDA : {torch.version.cuda}")
    try:
        print(f"  cuDNN           : {torch.backends.cudnn.version()}")
    except Exception:
        pass
    print(f"  Arch list       : {torch.cuda.get_arch_list()}")
    record("CUDA available", torch.cuda.is_available(),
           "" if torch.cuda.is_available() else "torch.cuda.is_available() == False")
except Exception as e:
    record("torch import", False, repr(e))

if torch is not None and torch.cuda.is_available():
    n = torch.cuda.device_count()
    record("GPU count", n >= 1, str(n))
    arch_list = torch.cuda.get_arch_list()
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)        # (12, 0) for RTX 50-series
        total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
        print(f"  cuda:{i}  {name}  cc={cap[0]}.{cap[1]}  {total:.1f} GiB")
        sm = f"sm_{cap[0]}{cap[1]}"
        if sm not in arch_list:
            record(f"cuda:{i} arch in torch build", "warn",
                   f"device is {sm} but arch_list lacks it -> kernels may fail")
    # The DECISIVE test: actually launch a CUDA kernel on each GPU.
    for i in range(n):
        try:
            a = torch.randn(512, 512, device=f"cuda:{i}")
            b = torch.randn(512, 512, device=f"cuda:{i}")
            _ = (a @ b).sum().item()
            torch.cuda.synchronize(i)
            record(f"cuda:{i} matmul smoke (REAL kernel test)", True)
        except Exception as e:
            record(f"cuda:{i} matmul smoke (REAL kernel test)", False, repr(e))

# ---------------------------------------------------------------------------
section("3. Batched ridge on GPU  (the Level-1 in-loop combiner)")
# Exactly the op the design needs: solve (Phi^T Phi + lam*I) w = Phi^T y,
# batched over the GP population P, for q constructed features.
if torch is not None and torch.cuda.is_available():
    try:
        dev = "cuda:0"
        P, Nc, q, lam = 256, 400, 8, 1e-2   # 256 individuals, 400 confs, 8 CFs
        Phi = torch.randn(P, Nc, q, device=dev)
        y = torch.randn(P, Nc, 1, device=dev)
        A = Phi.transpose(1, 2) @ Phi + lam * torch.eye(q, device=dev)   # [P,q,q]
        bvec = Phi.transpose(1, 2) @ y                                   # [P,q,1]
        w = torch.linalg.solve(A, bvec)                                  # [P,q,1]
        _ = Phi @ w
        torch.cuda.synchronize()
        record("batched torch.linalg.solve [P,q,q] on GPU", True,
               f"P={P}, q={q}, conf={Nc}")
    except Exception as e:
        record("batched torch.linalg.solve on GPU", False, repr(e))
else:
    record("batched ridge on GPU", "warn", "skipped (no CUDA)")

# ---------------------------------------------------------------------------
section("4. PyTorch-Geometric  (SchNet uses radius_graph + MessagePassing)")
try:
    import torch_geometric as pyg
    record("torch_geometric import", True, pyg.__version__)
    from torch_geometric.nn import radius_graph
    dev = "cuda:0" if (torch is not None and torch.cuda.is_available()) else "cpu"
    pos = torch.randn(40, 3, device=dev)
    batch = torch.zeros(40, dtype=torch.long, device=dev)
    ei = radius_graph(pos, r=5.0, batch=batch, max_num_neighbors=32)
    # radius_graph relies on torch_cluster CUDA kernels -> also an sm_120 check.
    record("radius_graph smoke", True, f"edges={ei.shape[1]} on {dev}")
except Exception as e:
    record("torch_geometric / radius_graph", False, repr(e))

# ---------------------------------------------------------------------------
section("5. RDKit conformer pipeline  (ETKDGv3 + MMFF energy)")
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import rdkit
    record("rdkit import", True, rdkit.__version__)
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))   # aspirin
    n_atoms = mol.GetNumAtoms()
    ps = AllChem.ETKDGv3()
    ps.randomSeed = 42
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=5, params=ps))
    energies = []
    if cids and AllChem.MMFFHasAllMoleculeParams(mol):
        mp = AllChem.MMFFGetMoleculeProperties(mol)
        for cid in cids:
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=cid)
            ff.Minimize()
            energies.append(ff.CalcEnergy())
    if cids:
        detail = f"{len(cids)} confs, {n_atoms} atoms"
        if energies:
            detail += f", E in [{min(energies):.1f}, {max(energies):.1f}] kcal/mol"
        record("ETKDGv3 embed + MMFF minimize", True, detail)
    else:
        record("ETKDGv3 embed + MMFF minimize", False, "no conformers embedded")
except Exception as e:
    record("rdkit conformer pipeline", False, repr(e))

# ---------------------------------------------------------------------------
section("6. schnetpack  (listed in requirements.txt)")
try:
    import schnetpack
    record("schnetpack import", True, getattr(schnetpack, "__version__", "?"))
except Exception as e:
    record("schnetpack import", "warn", repr(e))

# ---------------------------------------------------------------------------
section("7. EvoGP  (HIGHEST RISK: custom CUDA kernels must support sm_120)")
try:
    import evogp
    record("evogp import", True, getattr(evogp, "__version__", "?"))
    print(f"  evogp top-level names: "
          f"{[x for x in dir(evogp) if not x.startswith('_')]}")
    # Surface the tree / SR / algorithm API so you can confirm multi-output support.
    for sub in ["tree", "algorithm", "problem", "operator", "Forest", "GP"]:
        try:
            obj = getattr(evogp, sub, None)
            if obj is None:
                obj = importlib.import_module(f"evogp.{sub}")
            print(f"    found: evogp.{sub}  ->  {type(obj)}")
        except Exception:
            pass
    print("\n  NOTE: the DEFINITIVE EvoGP kernel test is to run its bundled example,")
    print("        e.g.  python example/custom_sr.py   (adapt to your install path).")
    print("        EvoGP compiles its own CUDA kernels separately from torch, so even")
    print("        if section 2's matmul passed, run the EvoGP example once on GPU.")
    print("        Then confirm: (i) multi-output trees, (ii) a custom fitness hook")
    print("        that exposes per-tree outputs as a tensor (needed for in-loop ridge).")
except Exception as e:
    record("evogp import", False, repr(e))

# ---------------------------------------------------------------------------
section("SUMMARY")
width = max(len(n) for n, _, _ in RESULTS)
n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
n_warn = sum(1 for _, s, _ in RESULTS if s == "WARN")
for name, status, detail in RESULTS:
    tag = {"OK": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[status]
    print(f"  {tag}  {name:<{width}}  {detail}")
print(f"\n  {n_fail} FAIL, {n_warn} WARN, {len(RESULTS) - n_fail - n_warn} OK")
if n_fail == 0:
    print("  -> Environment looks ready for Step 2.")
else:
    print("  -> Resolve the FAIL items before implementing Step 2.")