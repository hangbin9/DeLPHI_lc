#!/usr/bin/env python3
"""
Evaluate period predictions against ground truth.

Usage:
    python -m scripts.evaluate_period_predictions \
        --predictions results/period_predictions.csv \
        --groundtruth data/damit_groundtruth.csv

This script:
1. Loads predictions and ground truth
2. Computes comprehensive evaluation metrics
3. Prints formatted report
4. Optionally saves metrics to JSON
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lc_pipeline import (
    load_groundtruth,
    load_predictions_csv,
    evaluate_predictions,
    format_metrics_report,
)
from lc_pipeline.io_utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate period predictions against ground truth."
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
        "--out-json",
        type=str,
        default=None,
        help="Optional path to save metrics as JSON."
    )
    parser.add_argument(
        "--pred-col",
        type=str,
        default="period_hours",
        help="Column name for predicted period (default: period_hours)."
    )
    parser.add_argument(
        "--true-col",
        type=str,
        default="period_hours",
        help="Column name for true period (default: period_hours)."
    )

    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    groundtruth_path = Path(args.groundtruth)

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

    # Evaluate
    print("\nEvaluating predictions...")
    metrics = evaluate_predictions(
        predictions_df=predictions_df,
        truth_df=truth_df,
        pred_col=args.pred_col,
        true_col=args.true_col
    )

    # Print report
    report = format_metrics_report(metrics)
    print("\n" + report)

    # Save JSON if requested
    if args.out_json:
        out_path = Path(args.out_json)
        ensure_dir(out_path.parent)

        # Convert numpy types to Python types for JSON serialization
        json_metrics = {}
        for key, value in metrics.items():
            if hasattr(value, 'item'):
                json_metrics[key] = value.item()
            else:
                json_metrics[key] = value

        with open(out_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        print(f"\nMetrics saved to: {out_path}")


if __name__ == "__main__":
    main()
