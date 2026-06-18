"""Cấu hình trung tâm: registry dataset + lắp ráp config từ argparse.

Thay thế cho Hydra/YAML. Pipeline downstream (model / data / trainer) vẫn nhận
một dict config cùng cấu trúc như trước, nên không phải sửa.
"""

# =============================================================================
# Registry dataset — thêm dataset mới ở đây
# =============================================================================
DATASETS = {
    "esol": dict(
        name="esol", file="refined_ESOL.csv",
        smiles_column="smiles", target_column="measured",
        task_type="regression", metric="rmse",
    ),
    "freesolv": dict(
        name="freesolv", file="refined_FreeSolv.csv",
        smiles_column="smiles", target_column="measured",
        task_type="regression", metric="rmse",
    ),
    "lipo": dict(
        name="lipo", file="refined_Lipophilicity.csv",
        smiles_column="smiles", target_column="measured",
        task_type="regression", metric="rmse",
    ),
    "bace": dict(
        name="bace", file="refined_BACE.csv",
        smiles_column="smiles", target_column="class",
        task_type="classification", metric="auc",
    ),
}


def build_config(args) -> dict:
    """Lắp `argparse.Namespace` thành dict config cho toàn pipeline.

    Giữ đúng cấu trúc dict cũ (thời Hydra) để model/data/trainer dùng lại y nguyên.
    """
    if args.dataset not in DATASETS:
        raise ValueError(
            f"Dataset không hợp lệ: '{args.dataset}'. Chọn từ: {list(DATASETS)}"
        )

    # --seed-split có thể là list (quét nhiều seed) -> lấy 1 scalar; vòng lặp
    # ngoài (run_step1.main) sẽ ghi đè field này cho từng seed.
    seed_split = args.seed_split
    if isinstance(seed_split, (list, tuple)):
        seed_split = seed_split[0]

    return {
        # --- Global ---
        "dataset_name": args.dataset,
        "random_seed_train": args.seed_train,
        "gpu": args.gpu,
        "deterministic": args.deterministic,

        # --- Experiment ---
        "experiment": {
            "output_dir": args.output_dir,
            "log_level": "INFO",
            "verbose": args.verbose,
            "save": args.save,
            # Đường dẫn xuất encoder state_dict (warm-start cho eggroll). None = không
            # xuất. Có thể chứa "{seed}" -> thay bằng split seed (per-seed, tránh leak).
            "encoder_out": getattr(args, "encoder_out", None),
        },

        # --- Dataset (đã resolve) ---
        "dataset": DATASETS[args.dataset],

        # --- Data / Splitting ---
        "data": {
            "raw_dir": args.raw_dir,
            "processed_dir": args.processed_dir,
            "split_ratio": [0.81, 0.09, 0.10],  # chỉ để hiển thị (test=0.1, valid=0.1*0.9)
            "split_method": args.split_method,
            "random_seed_split": seed_split,
        },

        # --- Conformer ---
        "conformer": {
            "num_conformers": args.num_conformers,
            "max_attempts": args.max_attempts,
            "prune_rms_thresh": args.prune_rms_thresh,
            "use_random_coords": args.use_random_coords,
            "optimize_mmff": args.optimize_mmff,
            "random_seed_gen": args.seed_gen,
        },

        # --- SchNet ---
        "schnet": {
            "n_atom_basis": args.n_atom_basis,
            "n_interactions": args.n_interactions,
            "n_rbf": args.n_rbf,
            "n_filters": args.n_filters,
            "cutoff": args.cutoff,
            "atomref": None,
            "conf_readout": args.conf_readout,
        },

        # --- Training (Adam) ---
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "scheduler": "reduce_on_plateau",
            "scheduler_patience": args.scheduler_patience,
            "scheduler_factor": args.scheduler_factor,
            "early_stopping_patience": args.early_stopping_patience,
            "save_checkpoints": args.save,
            "gradient_clip": args.gradient_clip,
        },
    }


def build_eggroll_config(args) -> dict:
    """Lắp config cho pipeline eggroll (hard-count head trên SchNet warm-start).

    Dùng lại registry DATASETS + các section data/conformer/schnet giống build_config
    (để dựng đúng kiến trúc SchNet khớp encoder đã xuất ở Step 0), cộng thêm section
    `eggroll` chứa siêu tham số riêng. run_eggroll.py có argparse riêng cấp các field này.
    """
    if args.dataset not in DATASETS:
        raise ValueError(
            f"Dataset không hợp lệ: '{args.dataset}'. Chọn từ: {list(DATASETS)}"
        )

    seed_split = args.seed_split
    if isinstance(seed_split, (list, tuple)):
        seed_split = seed_split[0]

    return {
        # --- Global ---
        "dataset_name": args.dataset,
        "gpu": args.gpu,
        "deterministic": args.deterministic,

        # --- Dataset (đã resolve) ---
        "dataset": DATASETS[args.dataset],

        # --- Data / Splitting (phải khớp split của encoder per-seed) ---
        "data": {
            "raw_dir": args.raw_dir,
            "processed_dir": args.processed_dir,
            "split_ratio": [0.81, 0.09, 0.10],
            "split_method": args.split_method,
            "random_seed_split": seed_split,
        },

        # --- Conformer ---
        "conformer": {
            "num_conformers": args.num_conformers,
            "max_attempts": args.max_attempts,
            "prune_rms_thresh": args.prune_rms_thresh,
            "use_random_coords": args.use_random_coords,
            "optimize_mmff": args.optimize_mmff,
            "random_seed_gen": args.seed_gen,
        },

        # --- SchNet (phải khớp kiến trúc encoder đã train ở Step 0) ---
        "schnet": {
            "n_atom_basis": args.n_atom_basis,
            "n_interactions": args.n_interactions,
            "n_rbf": args.n_rbf,
            "n_filters": args.n_filters,
            "cutoff": args.cutoff,
            "atomref": None,
            "conf_readout": args.conf_readout,
        },

        # --- Training (chỉ batch_size dùng cho loader eval) ---
        "training": {"batch_size": args.batch_size},

        # --- Eggroll (head hard-count + LoRA + ES) ---
        "eggroll": {
            "encoder": args.encoder,          # path template, có thể chứa "{seed}"
            "seed_train": args.seed_train,
            "H": args.H,                      # số rule (hard-count head)
            "ridge_lambda": args.ridge_lambda,
            "counts_only": args.counts_only,  # bỏ tầng-1 (sàn linear trên e_pooled)
            "pop_size": args.pop_size,        # N quần thể ES
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "sigma_adapter": args.sigma_adapter,
            "sigma_head": args.sigma_head,
            "es_lr": args.es_lr,
            "p1_epochs": args.p1_epochs,      # head warm-up (encoder frozen)
            "p2_epochs": args.p2_epochs,      # gentle co-adapt (LoRA)
            "smoke": args.smoke,              # subset nhỏ + ES rút gọn để smoke CPU
        },
    }
