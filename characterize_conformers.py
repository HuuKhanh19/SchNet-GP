#!/usr/bin/env python
"""
Characterize conformer ENERGY spread for the CONAN-SchNet Step-2 gate.

The question this answers BEFORE we commit to K and the energy-routed MoE:
  Does relative MMFF energy (dE) actually VARY across conformers on these
  datasets? If most molecules are rigid (all dE ~ 0), an energy gate has
  little signal -- which is itself a finding, and would push us toward
  fewer experts or a different gate variable.

Also reports: valid-conformer counts, inf-energy (MMFF-failure) rate, and
suggested GLOBAL quantile bin edges (kcal/mol) for B in {2, 3}, compared
against RT and 2RT.

IMPORTANT: put this file in your repo's `scripts/` folder (next to
run_step1.py) so the `from src.data.conformer import ...` import resolves.
No GPU needed.

    python scripts/characterize_conformers.py            # K=20, 150 mols/dataset
    python scripts/characterize_conformers.py --K 10 --n_sample 100
    python scripts/characterize_conformers.py --datasets esol freesolv
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data.conformer import inner_smi2coords  # ETKDGv3 + MMFF, return_energy=True

DATASETS = {
    "esol":     ("refined_ESOL.csv",          "smiles"),
    "freesolv": ("refined_FreeSolv.csv",      "smiles"),
    "lipo":     ("refined_Lipophilicity.csv", "smiles"),
    "bace":     ("refined_BACE.csv",          "smiles"),
}
RT = 0.593  # kcal/mol at 298 K


def smiles_col(df, hint):
    if hint in df.columns:
        return hint
    for c in df.columns:
        if c.lower() in ("smiles", "smi", "mol", "canonical_smiles"):
            return c
    return df.columns[0]


def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def characterize(name, path, hint, K, n_sample, seed, raw_dir, out_dir):
    fp = os.path.join(raw_dir, path)
    if not os.path.exists(fp):
        print(f"\n[skip] {name}: {fp} not found")
        return
    df = pd.read_csv(fp)
    col = smiles_col(df, hint)
    smis = df[col].dropna().tolist()
    random.Random(seed).shuffle(smis)
    smis = smis[:n_sample]
    print(f"\n=== {name}  (sampling {len(smis)} / {len(df)} molecules, K={K}) ===")

    n_valid_confs, max_dE, std_dE = [], [], []
    all_dE = []
    n_total_inf = n_total_conf = 0

    for i, smi in enumerate(smis):
        try:
            atoms, coords, energies = inner_smi2coords(
                smi, seed=seed, mode="fast", optimize=True,
                n_confs=K, return_energy=True,
            )
        except Exception:
            continue
        if atoms is None or atoms[0] is None or not coords:
            continue
        e = np.asarray(energies, dtype=float)          # already dE (min-subtracted)
        n_total_conf += e.size
        finite = e[np.isfinite(e)]
        n_total_inf += (e.size - finite.size)
        n_valid_confs.append(int(finite.size))
        if finite.size:
            max_dE.append(float(finite.max()))
            std_dE.append(float(finite.std()))
            all_dE.extend(finite.tolist())
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(smis)}")

    max_dE = np.array(max_dE)
    std_dE = np.array(std_dE)
    all_dE = np.array(all_dE)

    if not n_valid_confs:
        print("  no valid molecules processed")
        return

    print(f"  valid confs/mol : mean={np.mean(n_valid_confs):.1f}  "
          f"min={np.min(n_valid_confs)}  (requested K={K})")
    print(f"  inf-energy rate : {n_total_inf}/{n_total_conf} confs "
          f"({100 * n_total_inf / max(n_total_conf, 1):.1f}%)")
    print(f"  per-mol max dE  : median={pct(max_dE, 50):.2f}  "
          f"p90={pct(max_dE, 90):.2f} kcal/mol")
    print(f"  per-mol std dE  : median={pct(std_dE, 50):.2f} kcal/mol")
    frac_rigid = float(np.mean(max_dE < RT))
    print(f"  'rigid' (max dE < RT={RT}): {100 * frac_rigid:.0f}% of molecules")
    print(f"     -> a high rigid % means weak energy-gate signal on {name}")

    for B in (2, 3):
        edges = [pct(all_dE, 100 * j / B) for j in range(1, B)]
        edges_s = ", ".join(f"{x:.2f}" for x in edges)
        print(f"  B={B} quantile edges (kcal/mol): [{edges_s}]   "
              f"(RT={RT}, 2RT={2 * RT:.2f})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(out_dir, exist_ok=True)
        cut = pct(all_dE, 99)
        plt.figure(figsize=(6, 4))
        plt.hist(all_dE[all_dE < cut], bins=50)
        plt.axvline(RT, color="r", ls="--", label=f"RT={RT}")
        plt.axvline(2 * RT, color="orange", ls="--", label="2RT")
        plt.xlabel("relative conformer energy dE (kcal/mol)")
        plt.ylabel("conformer count")
        plt.title(f"{name}: dE distribution (K={K})")
        plt.legend()
        plt.tight_layout()
        fpng = os.path.join(out_dir, f"dE_hist_{name}.png")
        plt.savefig(fpng, dpi=120)
        plt.close()
        print(f"  saved {fpng}")
    except Exception as ex:
        print(f"  (histogram skipped: {ex!r})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--n_sample", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--raw_dir", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--out_dir",
                    default=os.path.join(ROOT, "experiments", "conformer_probe"))
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    a = ap.parse_args()

    print("=" * 72)
    print("Conformer energy characterization  (Step-2 MoE gate design)")
    print("=" * 72)
    for name in a.datasets:
        if name not in DATASETS:
            print(f"[skip] unknown dataset {name}")
            continue
        path, hint = DATASETS[name]
        characterize(name, path, hint, a.K, a.n_sample, a.seed, a.raw_dir, a.out_dir)
    print("\nDONE.")


if __name__ == "__main__":
    main()