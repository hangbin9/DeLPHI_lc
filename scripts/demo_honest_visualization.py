#!/usr/bin/env python3
"""
Demo: Visualizing Honest Uncertainty for Period Predictions

Shows how to create publication-quality plots of:
1. Single prediction with posterior, interval, and uncertainty statement
2. Portfolio view of multiple predictions
3. Why aliases are ambiguous (illustration)
4. Flawed vs Honest approach comparison
"""

import sys
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.lib.honest_visualization import HonestVisualizer, VisualizationConfig
from tools.lib.honest_uncertainty import HonestUncertaintyComputer


def demo_single_prediction():
    """Demo: Visualize a single prediction with ground truth."""
    print("\n" + "="*80)
    print("DEMO 1: Single Prediction Visualization")
    print("="*80)

    visualizer = HonestVisualizer()

    # Example prediction (with ground truth for comparison)
    period_hours = 7.45
    posterior_prob = 0.82
    interval_lower = 7.20
    interval_upper = 7.70
    top_3_candidates = [
        (7.45, 125.3),   # Best
        (3.73, 95.2),    # Half-period alias
        (14.90, 88.1),   # Double-period alias
    ]
    true_period = 7.48  # Ground truth (0.4% error)

    fig = visualizer.plot_single_prediction(
        period_hours=period_hours,
        posterior_prob=posterior_prob,
        interval_lower=interval_lower,
        interval_upper=interval_upper,
        top_3_candidates=top_3_candidates,
        true_period=true_period,
        title="Example: Asteroid Period Prediction (Honest Approach)",
        savepath="results/honest_single_prediction.png"
    )

    print(f"✓ Single prediction plot saved to: results/honest_single_prediction.png")
    print(f"\nPrediction Details:")
    print(f"  Best Period: {period_hours:.2f} hours")
    print(f"  Posterior Confidence: {posterior_prob:.1%}")
    print(f"  68% Credible Interval: [{interval_lower:.2f}, {interval_upper:.2f}] hours")
    print(f"  Ground Truth: {true_period:.2f} hours")
    print(f"  Error: {abs(period_hours - true_period)/true_period*100:.1f}%")
    print(f"  In Interval: {'YES ✓' if interval_lower <= true_period <= interval_upper else 'NO'}")


def demo_portfolio():
    """Demo: Visualize portfolio of predictions."""
    print("\n" + "="*80)
    print("DEMO 2: Portfolio Visualization (Multiple Predictions)")
    print("="*80)

    visualizer = HonestVisualizer()

    # Generate synthetic portfolio of 50 predictions
    np.random.seed(42)
    n_predictions = 50
    predictions = []

    for i in range(n_predictions):
        # Vary confidence and uncertainty
        posterior = np.random.beta(8, 2)  # Skewed toward high confidence
        width = np.random.gamma(2, 0.5)   # Variable interval widths

        period = np.random.uniform(2, 50)
        lower = max(0.1, period - width/2)
        upper = period + width/2

        # Add some ground truth (with ~80% accuracy)
        if np.random.rand() < 0.80:
            # Good prediction
            true_period = period + np.random.normal(0, period * 0.03)
        else:
            # Bad prediction (often an alias)
            true_period = period / 2 if np.random.rand() < 0.5 else period * 2

        predictions.append({
            'period': period,
            'posterior': posterior,
            'interval_lower': lower,
            'interval_upper': upper,
            'true_period': true_period,
        })

    fig = visualizer.plot_portfolio(
        predictions=predictions,
        savepath="results/honest_portfolio.png"
    )

    print(f"✓ Portfolio plot saved to: results/honest_portfolio.png")
    print(f"\nPortfolio Statistics (n={n_predictions}):")
    posteriors = [p['posterior'] for p in predictions]
    widths = [p['interval_upper'] - p['interval_lower'] for p in predictions]
    errors = [abs(p['period'] - p['true_period'])/p['true_period'] for p in predictions]

    print(f"  Mean Posterior: {np.mean(posteriors):.2%}")
    print(f"  Mean Interval Width: {np.mean(widths):.2f} hours")
    print(f"  Accuracy <10%: {(np.array(errors) < 0.10).sum()/n_predictions*100:.1f}%")


def demo_alias_ambiguity():
    """Demo: Illustrate why aliases are ambiguous."""
    print("\n" + "="*80)
    print("DEMO 3: Why Aliases Are Ambiguous")
    print("="*80)

    visualizer = HonestVisualizer()

    period = 7.5  # hours

    fig = visualizer.plot_alias_ambiguity_illustration(
        period_hours=period,
        savepath="results/honest_alias_ambiguity.png"
    )

    print(f"✓ Alias ambiguity illustration saved to: results/honest_alias_ambiguity.png")
    print(f"\nShows why single-epoch data cannot distinguish:")
    print(f"  • True period: {period:.2f}h")
    print(f"  • Half-period alias: {period/2:.2f}h")
    print(f"  • Double-period alias: {period*2:.2f}h")
    print(f"\nAll three periods create similar folded lightcurves!")


def demo_honest_vs_flawed():
    """Demo: Compare flawed vs honest approaches."""
    print("\n" + "="*80)
    print("DEMO 4: Flawed vs Honest Approach Comparison")
    print("="*80)

    visualizer = HonestVisualizer()

    # Same prediction, shown two ways
    period_hours = 7.45
    posterior_prob = 0.82
    interval_lower = 7.20
    interval_upper = 7.70
    top_3_candidates = [
        (7.45, 125.3),
        (3.73, 95.2),
        (14.90, 88.1),
    ]

    fig = visualizer.plot_honest_vs_flawed(
        period_hours=period_hours,
        posterior_prob=posterior_prob,
        interval_lower=interval_lower,
        interval_upper=interval_upper,
        top_3_candidates=top_3_candidates,
        savepath="results/honest_vs_flawed.png"
    )

    print(f"✓ Comparison plot saved to: results/honest_vs_flawed.png")
    print(f"\nFlawed Approach (v2.0):")
    print(f"  • Uses R_alias_family (correlation: 0.0136)")
    print(f"  • Returns GREEN/GRAY flags")
    print(f"  • 19.6% of GREEN are actually wrong")
    print(f"  • 60.5% of GRAY are actually correct")
    print(f"\nHonest Approach:")
    print(f"  • Returns: Period + Posterior + Interval + Alternatives")
    print(f"  • Posterior {posterior_prob:.0%}: Model confidence among candidates")
    print(f"  • Interval: Formal ±σ uncertainty")
    print(f"  • Explicit: 'Cannot distinguish P from aliases'")


def demo_honest_uncertainty_computation():
    """Demo: Compute honest uncertainty from data."""
    print("\n" + "="*80)
    print("DEMO 5: Computing Honest Uncertainty")
    print("="*80)

    # Simulate ensemble output for 5 candidates
    cand_periods = np.array([5.0, 7.5, 10.0, 15.0, 20.0])
    total_scores = np.array([50.0, 100.0, 80.0, 60.0, 40.0])  # Best is 7.5h

    # Compute honest uncertainty
    result = HonestUncertaintyComputer.compute_single_epoch(
        best_idx=1,
        best_period_hours=7.5,
        best_log10_period=np.log10(7.5),
        cand_periods_hours=cand_periods,
        total_scores=total_scores,
    )

    print(f"\nInput:")
    print(f"  Candidates: {cand_periods}")
    print(f"  Scores: {total_scores}")
    print(f"\nOutput:")
    print(f"  Best Period: {result.period_hours:.2f} hours")
    print(f"  Posterior Prob: {result.posterior_prob:.1%}")
    print(f"  68% Interval: [{result.interval_lower:.2f}, {result.interval_upper:.2f}]")
    print(f"  Top 3 Candidates:")
    for i, (p, s) in enumerate(result.top_3_candidates):
        print(f"    #{i+1}: {p:.2f}h (score {s:.1f})")

    # Show JSON output
    result_dict = result.to_dict()
    print(f"\nJSON Output:")
    import json
    print(json.dumps(result_dict, indent=2))


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("HONEST UNCERTAINTY VISUALIZATION DEMOS")
    print("="*80)

    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)

    # Run demos
    demo_single_prediction()
    demo_portfolio()
    demo_alias_ambiguity()
    demo_honest_vs_flawed()
    demo_honest_uncertainty_computation()

    print("\n" + "="*80)
    print("ALL DEMOS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  • results/honest_single_prediction.png - Single prediction visualization")
    print("  • results/honest_portfolio.png - Portfolio of multiple predictions")
    print("  • results/honest_alias_ambiguity.png - Why aliases are ambiguous")
    print("  • results/honest_vs_flawed.png - Flawed vs Honest approach comparison")
    print("\nNext steps:")
    print("  1. View the generated PNG files")
    print("  2. Incorporate into presentations/papers")
    print("  3. Customize colors/fonts via VisualizationConfig")


if __name__ == "__main__":
    main()
