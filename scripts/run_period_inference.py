#!/usr/bin/env python3
"""
Run period inference on lightcurve data.

Usage:
    python -m scripts.run_period_inference \
        --manifest data/damit_manifest.csv \
        --out-predictions results/period_predictions.csv

This script:
1. Loads the manifest and groups epochs by object
2. Runs multi-epoch consensus period estimation
3. Saves predictions with uncertainty estimates
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lc_pipeline import (
    PeriodConfig,
    ColumnConfig,
    ConsensusEngine,
    load_manifest,
    group_epochs_by_object,
    save_predictions_csv,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run period inference on lightcurve data."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest CSV file."
    )
    parser.add_argument(
        "--out-predictions",
        type=str,
        default="results/period_predictions.csv",
        help="Output path for predictions CSV."
    )
    parser.add_argument(
        "--min-period",
        type=float,
        default=2.0,
        help="Minimum search period in hours (default: 2.0)."
    )
    parser.add_argument(
        "--max-period",
        type=float,
        default=100.0,
        help="Maximum search period in hours (default: 100.0)."
    )
    parser.add_argument(
        "--n-freq",
        type=int,
        default=20000,
        help="Number of frequency grid points (default: 20000)."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=64,
        help="Number of top candidates per epoch (default: 64)."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=10.0,
        help="Softmax temperature (default: 10.0)."
    )
    parser.add_argument(
        "--no-alias-injection",
        action="store_true",
        help="Disable alias candidate injection."
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default="jd",
        help="Column name for time (default: jd)."
    )
    parser.add_argument(
        "--flux-col",
        type=str,
        default="relative_brightness",
        help="Column name for flux (default: relative_brightness)."
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging."
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    manifest_path = Path(args.manifest)
    out_path = Path(args.out_predictions)

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    # Configure
    column_config = ColumnConfig(
        time_col=args.time_col,
        flux_col=args.flux_col,
    )

    period_config = PeriodConfig(
        min_period_hours=args.min_period,
        max_period_hours=args.max_period,
        n_freq=args.n_freq,
        top_k=args.top_k,
        temperature=args.temperature,
        alias_injection=not args.no_alias_injection,
        column_config=column_config,
    )

    logger.info(f"Loading manifest: {manifest_path}")
    manifest_df = load_manifest(manifest_path)
    logger.info(f"Found {len(manifest_df)} entries")

    logger.info("Grouping epochs by object...")
    objects = group_epochs_by_object(manifest_df, column_config, validate=True)
    logger.info(f"Loaded {len(objects)} objects")

    # Period estimation
    logger.info("Running period estimation...")
    logger.info(f"  Period range: [{args.min_period:.1f}, {args.max_period:.1f}] hours")
    logger.info(f"  Frequency grid: {args.n_freq} points")
    logger.info(f"  Top-K: {args.top_k}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Alias injection: {not args.no_alias_injection}")

    start_time = time.time()

    engine = ConsensusEngine(period_config)
    predictions_df = engine.predict_many(
        objects,
        show_progress=not args.no_progress
    )

    elapsed = time.time() - start_time
    logger.info(f"Completed in {elapsed:.1f} seconds")

    # Summary statistics
    n_success = predictions_df["success"].sum()
    n_failed = len(predictions_df) - n_success
    logger.info(f"\nResults: {n_success} successful, {n_failed} failed")

    if n_success > 0:
        periods = predictions_df.loc[predictions_df["success"], "period_hours"]
        logger.info(f"Period range: [{periods.min():.2f}, {periods.max():.2f}] hours")
        logger.info(f"Median period: {periods.median():.2f} hours")

    # Save
    save_predictions_csv(predictions_df, out_path)
    logger.info(f"\nPredictions saved to: {out_path}")


if __name__ == "__main__":
    main()
