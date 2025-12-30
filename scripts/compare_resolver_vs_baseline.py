#!/usr/bin/env python3
"""
Compare period predictions with and without the physical alias resolver.

Usage:
    python -m scripts.compare_resolver_vs_baseline \
        --manifest data/damit_manifest.csv \
        --groundtruth data/damit_groundtruth.csv \
        --out-dir results/resolver_comparison

This script:
1. Runs period inference WITHOUT the resolver (baseline)
2. Runs period inference WITH the resolver enabled
3. Compares metrics side-by-side
4. Identifies cases where the resolver helped/hurt
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lc_pipeline import (
    PeriodConfig,
    PhysicalAliasConfig,
    ConsensusEngine,
    load_manifest,
    load_groundtruth,
    group_epochs_by_object,
    evaluate_predictions,
    format_metrics_report,
    save_predictions_csv,
    ensure_dir,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compare resolver vs baseline period predictions."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest CSV file."
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
        default="results/resolver_comparison",
        help="Output directory for comparison results."
    )
    parser.add_argument(
        "--min-chi2-improvement",
        type=float,
        default=0.15,
        help="Min relative chi2 improvement for resolver (default: 0.15)."
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
    gt_path = Path(args.groundtruth)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Load data
    logger.info(f"Loading manifest: {manifest_path}")
    manifest_df = load_manifest(manifest_path)
    logger.info(f"Found {len(manifest_df)} entries")

    logger.info(f"Loading ground truth: {gt_path}")
    truth_df = load_groundtruth(gt_path)
    logger.info(f"Found {len(truth_df)} ground truth entries")

    logger.info("Grouping epochs by object...")
    objects = group_epochs_by_object(manifest_df, validate=False)
    logger.info(f"Loaded {len(objects)} objects")

    # Common period config
    period_config = PeriodConfig()

    # === Run BASELINE (resolver disabled) ===
    logger.info("\n" + "="*60)
    logger.info("Running BASELINE (resolver disabled)...")
    logger.info("="*60)

    baseline_alias_config = PhysicalAliasConfig(enabled=False)
    baseline_engine = ConsensusEngine(
        config=period_config,
        alias_config=baseline_alias_config
    )
    baseline_preds = baseline_engine.predict_many(objects, show_progress=True)
    baseline_preds.to_csv(out_dir / "predictions_baseline.csv", index=False)

    baseline_metrics = evaluate_predictions(baseline_preds, truth_df)
    logger.info("\nBaseline metrics:")
    print(format_metrics_report(baseline_metrics))

    # === Run WITH RESOLVER ===
    logger.info("\n" + "="*60)
    logger.info("Running WITH RESOLVER enabled...")
    logger.info("="*60)

    resolver_alias_config = PhysicalAliasConfig(
        enabled=True,
        min_chi2_rel_improvement=args.min_chi2_improvement,
        min_epochs_for_fit=2,
        min_points_per_epoch=15,
    )
    resolver_engine = ConsensusEngine(
        config=period_config,
        alias_config=resolver_alias_config
    )
    resolver_preds = resolver_engine.predict_many(objects, show_progress=True)
    resolver_preds.to_csv(out_dir / "predictions_with_resolver.csv", index=False)

    resolver_metrics = evaluate_predictions(resolver_preds, truth_df)
    logger.info("\nResolver metrics:")
    print(format_metrics_report(resolver_metrics))

    # === Comparison ===
    logger.info("\n" + "="*60)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*60)

    comparison = {
        "baseline": baseline_metrics,
        "with_resolver": resolver_metrics,
        "delta": {}
    }

    for key in ["acc_5pct", "acc_10pct", "acc_20pct", "median_rel_err_alias", "mean_rel_err_alias"]:
        if key in baseline_metrics and key in resolver_metrics:
            delta = resolver_metrics[key] - baseline_metrics[key]
            comparison["delta"][key] = delta

    # Print comparison
    print("\nMetric               | Baseline | Resolver | Delta")
    print("-" * 55)
    for key in ["acc_5pct", "acc_10pct", "acc_20pct"]:
        b = baseline_metrics.get(key, 0) * 100
        r = resolver_metrics.get(key, 0) * 100
        d = comparison["delta"].get(key, 0) * 100
        sign = "+" if d >= 0 else ""
        print(f"{key:20} | {b:7.1f}% | {r:7.1f}% | {sign}{d:.1f}%")

    for key in ["median_rel_err_alias", "mean_rel_err_alias"]:
        b = baseline_metrics.get(key, 0) * 100
        r = resolver_metrics.get(key, 0) * 100
        d = comparison["delta"].get(key, 0) * 100
        sign = "+" if d >= 0 else ""
        print(f"{key:20} | {b:7.2f}% | {r:7.2f}% | {sign}{d:.2f}%")

    # Count resolver actions
    if "resolver_applied" in resolver_preds.columns:
        n_resolved = resolver_preds["resolver_applied"].sum()
        n_total = len(resolver_preds)
        print(f"\nResolver applied to: {n_resolved}/{n_total} objects ({100*n_resolved/n_total:.1f}%)")

    # Save comparison JSON
    comparison_json = {k: float(v) if hasattr(v, 'item') else v
                      for k, v in comparison["delta"].items()}
    comparison_json["n_resolved"] = int(resolver_preds.get("resolver_applied", []).sum())
    comparison_json["n_total"] = len(resolver_preds)

    with open(out_dir / "comparison_summary.json", "w") as f:
        json.dump(comparison_json, f, indent=2)

    logger.info(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
