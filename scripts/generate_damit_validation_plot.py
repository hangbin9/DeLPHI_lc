#!/usr/bin/env python3
"""
Generate True vs Predicted validation plot for complete DAMIT high dataset.

This script loads DAMIT ensemble predictions with ground truth and creates
a comprehensive validation visualization showing prediction accuracy with
credible interval error bars.
"""

import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.lib.honest_visualization import HonestVisualizer


def load_damit_predictions(jsonl_path):
    """
    Load DAMIT ensemble predictions from JSONL file.

    Args:
        jsonl_path: Path to predictions_damit_ensemble.jsonl

    Returns:
        List of prediction dicts with required keys:
        - period: Predicted period (hours)
        - true_period: Ground truth period (hours)
        - interval_lower: Lower bound of 68% credible interval (hours)
        - interval_upper: Upper bound of 68% credible interval (hours)
        - posterior: Model confidence (0-1) - estimated from posterior_mu
    """
    predictions = []

    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            # Extract fields
            pred_period = data.get('period_map_hr')
            true_period = data.get('P_true_hr')
            interval_lower = data.get('period_low_hr')
            interval_upper = data.get('period_high_hr')

            # Skip if any critical field is missing
            if None in [pred_period, true_period, interval_lower, interval_upper]:
                print(f"  ⚠ Skipping {data.get('object_id')} - missing fields")
                continue

            # Skip if interval is invalid (lower >= upper)
            if interval_lower >= interval_upper:
                print(f"  ⚠ Skipping {data.get('object_id')} - invalid interval")
                continue

            # Estimate posterior from posterior_mu (normalize to 0-1)
            # posterior_mu can be large, so cap at reasonable value
            posterior_mu = data.get('posterior_mu_hr', 0.5)
            posterior = min(1.0, posterior_mu / 100.0)  # Heuristic normalization
            posterior = max(0.0, posterior)

            predictions.append({
                'period': pred_period,
                'true_period': true_period,
                'interval_lower': interval_lower,
                'interval_upper': interval_upper,
                'posterior': posterior,
                'object_id': data.get('object_id'),
                'band': data.get('band'),
                'rel_err': data.get('rel_err_map'),
            })

    return predictions


def main():
    print("\n" + "="*80)
    print("DAMIT HIGH DATASET VALIDATION WITH ERROR BARS")
    print("="*80)

    # Setup paths
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    jsonl_path = artifacts_dir / "predictions_damit_ensemble.jsonl"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    if not jsonl_path.exists():
        print(f"\n✗ ERROR: Predictions file not found: {jsonl_path}")
        return

    # Load predictions
    print(f"\n[1/3] Loading DAMIT ensemble predictions...")
    predictions = load_damit_predictions(jsonl_path)
    print(f"  ✓ Loaded {len(predictions)} valid predictions")

    if not predictions:
        print("  ✗ ERROR: No valid predictions loaded")
        return

    # Basic statistics before visualization
    print(f"\n[2/3] Computing validation statistics...")
    periods = np.array([p['period'] for p in predictions])
    true_periods = np.array([p['true_period'] for p in predictions])
    rel_errors = np.abs(periods - true_periods) / true_periods

    print(f"  • N samples: {len(predictions)}")
    print(f"  • Period range: {true_periods.min():.2f} - {true_periods.max():.2f} hours")
    print(f"  • Median relative error: {np.median(rel_errors)*100:.2f}%")
    print(f"  • Mean relative error: {np.mean(rel_errors)*100:.2f}%")
    print(f"  • Accuracy <10%: {(rel_errors < 0.10).sum()}/{len(predictions)} = {(rel_errors < 0.10).mean()*100:.1f}%")

    # Check interval coverage
    in_interval = (np.array([p['interval_lower'] for p in predictions]) <= true_periods) & \
                  (true_periods <= np.array([p['interval_upper'] for p in predictions]))
    print(f"  • Interval coverage (68% expected): {in_interval.sum()}/{len(predictions)} = {in_interval.mean()*100:.1f}%")

    # Create visualization
    print(f"\n[3/3] Creating true vs predicted visualization...")
    try:
        visualizer = HonestVisualizer()
        fig = visualizer.plot_true_vs_predicted(
            predictions=predictions,
            savepath=str(results_dir / "damit_high_validation_complete.png")
        )
        print(f"  ✓ Saved to results/damit_high_validation_complete.png")
    except Exception as e:
        print(f"  ✗ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return

    # Generate detailed report
    print(f"\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)

    print(f"\nError Distribution:")
    for threshold in [0.05, 0.10, 0.20, 0.50]:
        count = (rel_errors < threshold).sum()
        pct = count / len(predictions) * 100
        print(f"  • <{threshold*100:>5.0f}% error: {count:>3d}/{len(predictions)} = {pct:>5.1f}%")

    print(f"\nInterval Coverage by Accuracy:")
    for thresh_low, thresh_high in [(0, 0.10), (0.10, 0.50), (0.50, 2.0)]:
        mask = (rel_errors >= thresh_low) & (rel_errors < thresh_high)
        if mask.sum() > 0:
            coverage = in_interval[mask].mean() * 100
            label = f"{thresh_low*100:.0f}-{thresh_high*100:.0f}%"
            print(f"  • {label:>10s} error: {coverage:>5.1f}% coverage ({mask.sum()} samples)")

    print(f"\nPeriod Band Distribution:")
    bands = {}
    for p in predictions:
        band = p.get('band', 'UNKNOWN')
        if band not in bands:
            bands[band] = []
        bands[band].append(p)

    for band in sorted(bands.keys()):
        band_preds = bands[band]
        band_errors = rel_errors[[i for i, p in enumerate(predictions) if p.get('band') == band]]
        median_err = np.median(band_errors) * 100
        accuracy = (band_errors < 0.10).mean() * 100
        print(f"  • {band:>6s}: {len(band_preds):>3d} samples, median error {median_err:>6.2f}%, accuracy {accuracy:>5.1f}%")

    print(f"\n" + "="*80)
    print("✅ VALIDATION COMPLETE!")
    print("="*80)
    print(f"\nGenerated visualization:")
    print(f"  • results/damit_high_validation_complete.png")
    print(f"\nThis plot includes:")
    print(f"  1. True vs Predicted scatter with 68% credible interval error bars")
    print(f"  2. Relative error vs true period distribution")
    print(f"  3. Credible interval coverage by error category")
    print(f"  4. Summary statistics (median error, coverage, posteriors)")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
