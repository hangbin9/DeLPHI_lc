#!/usr/bin/env python3
"""
Extract ground truth periods from DAMIT lightcurve files.

Usage:
    python -m scripts.extract_groundtruth_from_damit \
        --manifest data/damit_manifest.csv \
        --out-groundtruth data/damit_groundtruth.csv

This script reads the 'rot_per' column from each unique object's
lightcurve CSV file to build a ground truth period table.

The rot_per column in DAMIT files is already in hours.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lc_pipeline.data import (
    load_manifest,
    extract_groundtruth_from_lightcurves
)
from lc_pipeline.config import ColumnConfig
from lc_pipeline.io_utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(
        description="Extract ground truth periods from DAMIT lightcurve files."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest CSV file."
    )
    parser.add_argument(
        "--out-groundtruth",
        type=str,
        default="data/damit_groundtruth.csv",
        help="Output path for ground truth CSV."
    )
    parser.add_argument(
        "--period-col",
        type=str,
        default="rot_per",
        help="Column name containing period values (default: rot_per)."
    )
    parser.add_argument(
        "--period-unit",
        type=str,
        choices=["hours", "days"],
        default="hours",
        help="Unit of period values in source files (default: hours)."
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_path = Path(args.out_groundtruth)

    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}")
        sys.exit(1)

    print(f"Loading manifest: {manifest_path}")
    manifest_df = load_manifest(manifest_path)
    print(f"Found {len(manifest_df)} entries for {manifest_df['object_id'].nunique()} objects")

    # Configure column mapping
    column_config = ColumnConfig(
        period_col=args.period_col,
        period_unit=args.period_unit
    )

    print(f"\nExtracting ground truth periods...")
    print(f"  Period column: {args.period_col}")
    print(f"  Period unit: {args.period_unit}")

    # Extract ground truth
    gt_df = extract_groundtruth_from_lightcurves(manifest_df, column_config)

    print(f"\nExtracted periods for {len(gt_df)} objects")

    if len(gt_df) > 0:
        print(f"\nPeriod statistics:")
        print(f"  Min:    {gt_df['period_hours'].min():.4f} hours")
        print(f"  Max:    {gt_df['period_hours'].max():.4f} hours")
        print(f"  Median: {gt_df['period_hours'].median():.4f} hours")
        print(f"  Mean:   {gt_df['period_hours'].mean():.4f} hours")

    # Save
    ensure_dir(out_path.parent)
    gt_df.to_csv(out_path, index=False)
    print(f"\nGround truth saved to: {out_path}")

    # Show sample
    print("\nSample entries:")
    print(gt_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
