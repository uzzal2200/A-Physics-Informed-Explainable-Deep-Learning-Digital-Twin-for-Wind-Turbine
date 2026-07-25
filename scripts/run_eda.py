"""
scripts/run_eda.py
──────────────────
Build the main dataset and generate all EDA figures.

Usage
-----
# Step 1: Build main_dataset.csv from raw CARE CSVs
python scripts/run_eda.py --step build --care_root data/CARE

# Step 2: Generate all EDA figures from existing main_dataset.csv
python scripts/run_eda.py --step eda --dataset data/main_dataset.csv

# Both steps at once
python scripts/run_eda.py --step all --care_root data/CARE
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eda.dataset_builder import CAREDatasetBuilder
from eda.eda_plots import EDAPlotter
from utils.utils import setup_logging

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Build CARE dataset and generate EDA figures."
    )
    parser.add_argument(
        "--step", choices=["build", "eda", "all"], default="all",
        help="Which step to run."
    )
    parser.add_argument(
        "--care_root", type=str, default="data/CARE",
        help="Path to CARE root directory."
    )
    parser.add_argument(
        "--dataset", type=str, default="data/main_dataset.csv",
        help="Path to main_dataset.csv (for eda step)."
    )
    parser.add_argument(
        "--save_dir", type=str, default="figures/EDA",
        help="Directory for saving EDA figures."
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display figures interactively."
    )
    args = parser.parse_args()

    setup_logging(log_file="results/eda.log")
    logger = logging.getLogger(__name__)

    # ── Step 1: Build dataset ────────────────────────────────────────────
    if args.step in ("build", "all"):
        logger.info("=" * 60)
        logger.info("  Building CARE Main Dataset")
        logger.info("=" * 60)
        builder = CAREDatasetBuilder(root=args.care_root)
        df = builder.build(save_path=args.dataset)
        logger.info(f"Dataset saved → {args.dataset}")

    # ── Step 2: EDA figures ──────────────────────────────────────────────
    if args.step in ("eda", "all"):
        logger.info("=" * 60)
        logger.info("  Generating EDA Figures")
        logger.info("=" * 60)

        if args.step == "eda":
            logger.info(f"Loading dataset from {args.dataset} …")
            df = pd.read_csv(args.dataset)
            if "time_stamp" in df.columns:
                df["time_stamp"] = pd.to_datetime(df["time_stamp"])

        plotter = EDAPlotter(df, save_dir=args.save_dir, show=args.show)
        plotter.run_all()
        logger.info(f"All figures saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
