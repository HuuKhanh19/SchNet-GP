"""Utility functions for CONAN-SchNet."""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True):
    """Set all random seeds for reproducibility.

    Controls:
        - Python's random module
        - PYTHONHASHSEED environment variable
        - NumPy's global random state
        - PyTorch CPU & CUDA random states
        - cuDNN deterministic mode
        - (deterministic=True) deterministic CUDA algorithms for scatter/atomic
          ops used by message passing & readout

    Why `deterministic`:
        Seeding alone does NOT make GPU runs reproducible. SchNet's message
        passing and hierarchical readout use scatter-add / atomic reductions,
        and CUDA floating-point reductions are non-associative (order-dependent).
        Without `torch.use_deterministic_algorithms(True)` the run-to-run results
        drift even with a fixed seed. Enabling it trades some speed for exact
        reproducibility (needed for the fixed-train-seed experiment design).

    Note:
        When using DataLoader with num_workers > 0, each worker needs
        its own seed via worker_init_fn. See get_worker_init_fn().
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if deterministic:
        # Required for deterministic cuBLAS (matmul) with CUDA >= 10.2.
        # Must be set before the first cuBLAS call -> set here, early.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        # warn_only=True: if an op lacks a deterministic CUDA kernel, warn
        # instead of crashing the run (it will simply stay non-deterministic).
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_worker_init_fn(base_seed: int):
    """Return a worker_init_fn for DataLoader reproducibility.
    
    Usage:
        DataLoader(..., worker_init_fn=get_worker_init_fn(42))
    """
    def worker_init_fn(worker_id: int):
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return worker_init_fn