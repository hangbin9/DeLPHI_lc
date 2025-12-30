#!/usr/bin/env python3
"""
Generate diagnostic plots for period predictions.

Usage:
    python -m scripts.plot_period_diagnostics \
        --predictions results/period_predictions.csv \
        --groundtruth data/damit_groundtruth.csv \
        --out-dir results/plots

This script generates:
1. Period parity plot (true vs predicted)
2. Uncertainty vs error plot
3. Error histogram
"""

import argparse
import sys
from pathlib import Path

# Ensure matplotlib uses non-interactive backend for saving
import matplotlib
matplotlib.use('Agg')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lc_pipeline import (
    load_groundtruth,
    load_predictions_csv,
    plot_period_parity,
    plot_uncertainty_vs_error,
    plot_error_histogram,
)
from lc_pipeline.io_utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generate diagnostic plots for period predictions."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions CSV file."
    )
    parser.add_argument(
        "--groundtruth",
        type=str,
        required=True,
        help="Path to ground truth CSV file."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/plots",
        help="Output directory for plots."
    )
    parser.add_argument(
        "--max-period",
        type=float,
        default=100.0,
        help="Maximum period for parity plot (default: 100.0 hours)."
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output file format (default: png)."
    )

    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    groundtruth_path = Path(args.groundtruth)
    out_dir = Path(args.out_dir)

    # Load files
    if not predictions_path.exists():
        print(f"Error: Predictions file not found: {predictions_path}")
        sys.exit(1)

    if not groundtruth_path.exists():
        print(f"Error: Ground truth file not found: {groundtruth_path}")
        sys.exit(1)

    print(f"Loading predictions: {predictions_path}")
    predictions_df = load_predictions_csv(predictions_path)
    print(f"  Loaded {len(predictions_df)} predictions")

    print(f"Loading ground truth: {groundtruth_path}")
    truth_df = load_groundtruth(groundtruth_path)
    print(f"  Loaded {len(truth_df)} ground truth entries")

    # Ensure output directory exists
    ensure_dir(out_dir)

    fmt = args.format

    # Generate plots
    print("\nGenerating plots...")

    # 1. Parity plot
    parity_path = out_dir / f"period_parity.{fmt}"
    print(f"  Creating parity plot: {parity_path}")
    plot_period_parity(
        truth_df=truth_df,
        preds_df=predictions_df,
        out_path=parity_path,
        max_period_hours=args.max_period
    )

    # 2. Uncertainty vs error
    uncertainty_path = out_dir / f"uncertainty_vs_error.{fmt}"
    print(f"  Creating uncertainty plot: {uncertainty_path}")
    plot_uncertainty_vs_error(
        truth_df=truth_df,
        preds_df=predictions_df,
        out_path=uncertainty_path
    )

    # 3. Error histogram
    histogram_path = out_dir / f"error_histogram.{fmt}"
    print(f"  Creating histogram: {histogram_path}")
    plot_error_histogram(
        truth_df=truth_df,
        preds_df=predictions_df,
        out_path=histogram_path
    )

    print(f"\nPlots saved to: {out_dir}")


if __name__ == "__main__":
    main()
