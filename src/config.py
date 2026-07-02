"""Cấu hình trung tâm: registry dataset + lắp ráp config từ argparse.

Thay thế cho Hydra/YAML. Pipeline downstream (model / data / trainer) vẫn nhận
một dict config cùng cấu trúc như trước, nên không phải sửa.
"""

# =============================================================================
# Registry dataset — thêm dataset mới ở đây
# =============================================================================
# Mỗi entry mô tả 1 dataset. Quy ước cột nhãn:
#   - single-task (regression / classification): dùng `target_column` (1 cột).
#   - multi-label classification: dùng `label_columns` (list các cột nhãn 0/1,
#     cho phép ô trống = nhãn thiếu -> loss có mask).
# `n_tasks` = số đầu ra của model (số cột nhãn). Suy ra tự động bên dưới.
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
    # Multi-label: 12 nhiệm vụ độc lính (Tox21). Ô trống trong CSV = nhãn thiếu.
    "tox21": dict(
        name="tox21", file="refined_Tox21.csv",
        smiles_column="SMILES",
        label_columns=[
            "NR-AhR", "NR-AR-LBD", "NR-AR", "NR-Aromatase", "NR-ER-LBD",
            "NR-ER", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE",
            "SR-MMP", "SR-p53",
        ],
        task_type="multilabel", metric="auc",
    ),
}


def dataset_label_columns(ds: dict) -> list:
    """Danh sách cột nhãn của 1 dataset (nguồn CSV).

    - multilabel: lấy `label_columns`.
    - single-task: [target_column].
    """
    if ds["task_type"] == "multilabel":
        return list(ds["label_columns"])
    return [ds["target_column"]]


def dataset_n_tasks(ds: dict) -> int:
    """Số đầu ra của model = số cột nhãn."""
    return len(dataset_label_columns(ds))


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

    # Bổ sung n_tasks cho dataset (số đầu ra của model). Copy để không đụng
    # vào registry gốc.
    ds_cfg = dict(DATASETS[args.dataset])
    ds_cfg["n_tasks"] = dataset_n_tasks(ds_cfg)

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
        },

        # --- Dataset (đã resolve) ---
        "dataset": ds_cfg,

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
