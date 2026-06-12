"""Phase 2: extract + cache embedding conformer (+ 3D descriptor) cho GP head.

Với encoder SchNet ĐÓNG BĂNG, chạy extract_conf_embeddings trên TỪNG phân tử (K
conformer energy-ranked) -> emb (k_eff, hidden). Đồng thời tính 3D descriptor mỗi
conformer. Pad về đúng K bằng cách lặp conformer hạng cao nhất (k_eff < K khi RDKit
sinh thiếu) -> bin i dùng min(i, k_eff-1).

Cache ra .npz, key theo (encoder_conformers, seed, split, k) để GP chạy lại không
phải gọi encoder.
"""

import os
from typing import List, Tuple

import numpy as np
import torch

from src.data.data_loader import SchNetMolDataset, collate_multi_conformer
from .descriptors import compute_3d_descriptors


def _cache_path(cache_dir: str, dataset: str, split_method: str,
                encoder_conformers: str, seed: int, k: int, split: str) -> str:
    return os.path.join(
        cache_dir, dataset, split_method,
        f"enc_{encoder_conformers}", f"seed_{seed}", f"k{k}", f"{split}.npz",
    )


@torch.no_grad()
def extract_split_features(
    model: torch.nn.Module,
    sch_dataset: SchNetMolDataset,
    desc3d_names: List[str],
    k: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trả (emb (N,K,H), desc3d (N,K,n3d), targets (N,)). Pad về K bằng repeat-last."""
    model.eval()
    n = len(sch_dataset)
    hidden = int(model.hidden_channels)
    n3d = len(desc3d_names)

    emb_all = np.zeros((n, k, hidden), dtype=np.float32)
    d3_all = np.zeros((n, k, n3d), dtype=np.float32)
    targets = np.asarray(sch_dataset.targets, dtype=np.float32)

    for idx in range(n):
        item = sch_dataset[idx]
        batch = collate_multi_conformer([item])
        batch = {kk: v.to(device) for kk, v in batch.items()}
        conf_emb = model.extract_conf_embeddings(batch).cpu().numpy()  # (k_eff, H)
        k_eff = conf_emb.shape[0]

        z = sch_dataset.atomic_numbers[idx]
        pos = sch_dataset.positions[idx]                              # (k_eff, n_atoms, 3)
        d3 = np.stack(
            [compute_3d_descriptors(z, pos[c], desc3d_names) for c in range(k_eff)],
            axis=0,
        )                                                            # (k_eff, n3d)

        # Pad/clamp về K: lặp hạng cao nhất (conformer cuối) nếu thiếu; cắt nếu thừa.
        for i in range(k):
            src = min(i, k_eff - 1)
            emb_all[idx, i] = conf_emb[src]
            d3_all[idx, i] = d3[src]

        if (idx + 1) % 200 == 0:
            print(f"    extract embeddings: {idx + 1}/{n}")

    return emb_all, d3_all, targets


def get_split_features(
    model: torch.nn.Module,
    sch_dataset: SchNetMolDataset,
    desc3d_names: List[str],
    k: int,
    device: torch.device,
    cache_dir: str,
    dataset: str,
    split_method: str,
    encoder_conformers: str,
    seed: int,
    split: str,
    smiles: List[str],
) -> dict:
    """Load cache nếu có, không thì extract rồi lưu. Trả dict {emb, desc3d, targets, smiles}."""
    path = _cache_path(cache_dir, dataset, split_method, encoder_conformers, seed, k, split)
    if os.path.exists(path):
        print(f"  Load feature cache: {path}")
        z = np.load(path, allow_pickle=True)
        return {"emb": z["emb"], "desc3d": z["desc3d"],
                "targets": z["targets"], "smiles": list(z["smiles"])}

    emb, d3, targets = extract_split_features(model, sch_dataset, desc3d_names, k, device)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path, emb=emb, desc3d=d3, targets=targets,
        smiles=np.array(smiles, dtype=object), desc3d_names=np.array(desc3d_names, dtype=object),
    )
    print(f"  Saved feature cache: {path}")
    return {"emb": emb, "desc3d": d3, "targets": targets, "smiles": smiles}
