#!/usr/bin/env python
"""Preprocess Data: Load raw CSV -> split -> save to data/processed/.

Tuỳ chọn — run_step1.py tự lo bước này nếu chưa có cache. Script này chỉ để tiền
xử lý sẵn (vd. tất cả dataset) trước khi train.

Ví dụ:
    python scripts/preprocess_data.py --dataset all --seed-split 0
    python scripts/preprocess_data.py --dataset esol --split-method random --seed-split 3
"""

import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config import DATASETS
from src.data import prepare_dataset, save_splits


def preprocess_single(dataset_name: str, args) -> tuple:
    print(f"\n{'='*50}\nProcessing: {dataset_name.upper()}\n{'='*50}")

    # prepare_dataset chỉ cần 'dataset' và 'data'.
    config = {
        "dataset": DATASETS[dataset_name],
        "data": {
            "raw_dir": args.raw_dir,
            "processed_dir": args.processed_dir,
            "split_method": args.split_method,
            "random_seed_split": args.seed_split,
        },
    }

    train_df, valid_df, test_df = prepare_dataset(config)

    processed_dir = (f"{args.processed_dir}/{dataset_name}/"
                     f"{args.split_method}/seed_{args.seed_split}")
    save_splits(train_df, valid_df, test_df, processed_dir)
    return len(train_df), len(valid_df), len(test_df)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tiền xử lý + chia split cho dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="all",
                   choices=["all"] + list(DATASETS),
                   help="Dataset cần xử lý ('all' = tất cả).")
    p.add_argument("--split-method", default="random_scaffold",
                   choices=["random_scaffold", "random"], dest="split_method",
                   help="Cách chia train/valid/test (~81/9/10).")
    p.add_argument("--seed-split", type=int, default=0, dest="seed_split",
                   help="Seed chia split.")
    p.add_argument("--raw-dir", default="data/raw", dest="raw_dir",
                   help="Thư mục chứa CSV gốc.")
    p.add_argument("--processed-dir", default="data/processed", dest="processed_dir",
                   help="Thư mục đầu ra.")
    return p


def main():
    args = build_parser().parse_args()
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]

    print("=" * 60)
    print("CONAN-SchNet - Data Preprocessing")
    print("=" * 60)

    results = {}
    for ds in datasets:
        try:
            results[ds] = preprocess_single(ds, args)
        except FileNotFoundError as e:
            print(f"Warning: {e}\nSkipping {ds} - upload raw data first.")

    print("\n" + "=" * 60 + "\nSummary:")
    for ds, (tr, va, te) in results.items():
        print(f"  {ds}: Train={tr}, Valid={va}, Test={te}")


if __name__ == "__main__":
    main()
