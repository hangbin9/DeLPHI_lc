#!/usr/bin/env python3
"""
Example usage of the lc_pipeline end-to-end analysis.

This demonstrates how to use the complete pipeline for asteroid lightcurve analysis.
"""
import numpy as np
from lc_pipeline import analyze, LightcurvePipeline


def load_damit_lc(filepath):
    """
    Load DAMIT lc.json format.

    Args:
        filepath: Path to lc.json file

    Returns:
        List of (N, 8) arrays [JD, brightness, sun_xyz, obs_xyz]
    """
    import json
    from pathlib import Path

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath) as f:
        data = json.load(f)

    epochs = []
    for epoch in data:
        points = np.array([
            list(map(float, line.split()))
            for line in epoch['points'].strip().split('\n')
        ])
        epochs.append(points)

    return epochs


def example_simple():
    """Simple one-shot analysis with known period."""
    print("=" * 80)
    print("EXAMPLE 1: Simple Analysis with Known Period")
    print("=" * 80)

    # Create synthetic epoch data for demonstration
    # In real usage, load from DAMIT format: epochs = load_damit_lc("asteroid_1017/lc.json")
    epochs = create_synthetic_epochs(period_hours=8.5, n_epochs=3)

    # Analyze with known period
    result = analyze(epochs, "asteroid_example", period_hours=8.5, fold=0)

    # Results
    print(f"\nPeriod: {result.period.period_hours:.2f} ± {result.period.uncertainty_hours:.2f} h")
    print(f"Best pole: λ={result.best_pole.lambda_deg:.1f}°, β={result.best_pole.beta_deg:.1f}°")
    print(f"Uncertainty: spread={result.uncertainty.spread_deg:.1f}°, confidence={result.uncertainty.confidence:.2f}")

    print(f"\nAll {len(result.poles)} candidates:")
    for i, pole in enumerate(result.poles[:5]):  # Show top 5
        print(f"  {i+1}. λ={pole.lambda_deg:6.1f}°, β={pole.beta_deg:5.1f}°, "
              f"P={pole.alias:6s} ({pole.period_hours:5.2f}h), score={pole.score:.3f}")


def example_period_estimation():
    """Full pipeline with period estimation."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Full Pipeline (Period Estimation + Pole Prediction)")
    print("=" * 80)

    # Create synthetic epoch data
    epochs = create_synthetic_epochs(period_hours=12.3, n_epochs=5)

    # Analyze without providing period (will estimate)
    result = analyze(epochs, "asteroid_example_2", fold=0)

    # Results
    print(f"\nEstimated Period: {result.period.period_hours:.2f} h")
    print(f"Period CI: [{result.period.ci_low_hours:.2f}, {result.period.ci_high_hours:.2f}] h")
    print(f"Best pole: λ={result.best_pole.lambda_deg:.1f}°, β={result.best_pole.beta_deg:.1f}°")

    # Show period alias candidates
    print(f"\nPole candidates by period alias:")
    for alias in ["base", "double", "half"]:
        alias_poles = [p for p in result.poles if p.alias == alias]
        if alias_poles:
            best = alias_poles[0]
            print(f"  {alias:6s}: P={best.period_hours:6.2f}h, "
                  f"λ={best.lambda_deg:6.1f}°, β={best.beta_deg:5.1f}°, score={best.score:.3f}")


def example_custom_config():
    """Example with custom configuration."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Custom Configuration")
    print("=" * 80)

    from lc_pipeline import PoleConfig
    from pathlib import Path

    # Custom configuration
    pole_config = PoleConfig(
        checkpoint_dir=Path("lc_pipeline/checkpoints"),
        device="cpu",  # Force CPU even if GPU available
        n_windows=8,
        tokens_per_window=256
    )

    # Create pipeline with custom config
    pipeline = LightcurvePipeline(pole_config=pole_config)

    # Analyze
    epochs = create_synthetic_epochs(period_hours=7.8, n_epochs=4)
    result = pipeline.analyze(epochs, "asteroid_example_3", period_hours=7.8, fold=1)

    print(f"\nUsing fold 1 checkpoint")
    print(f"Period: {result.period.period_hours:.2f} h")
    print(f"Best pole: λ={result.best_pole.lambda_deg:.1f}°, β={result.best_pole.beta_deg:.1f}°")


def create_synthetic_epochs(period_hours=8.5, n_epochs=3, n_points=50):
    """
    Create synthetic DAMIT-format epochs for demonstration.

    Returns:
        List of (N, 8) arrays [JD, brightness, sun_xyz, obs_xyz]
    """
    np.random.seed(42)
    epochs = []

    for epoch_idx in range(n_epochs):
        # Time points over one rotation
        jd_start = 2460000 + epoch_idx * 30  # 30-day spacing
        times = np.linspace(jd_start, jd_start + period_hours/24, n_points)

        # Synthetic brightness (double-peaked)
        phase = (times - jd_start) / (period_hours / 24)
        brightness = 1.0 + 0.3 * np.sin(2 * np.pi * phase) + 0.1 * np.sin(4 * np.pi * phase)
        brightness += np.random.normal(0, 0.02, n_points)

        # Geometry (simplified)
        sun_vec = np.array([1.0, 0.0, 0.0]) + np.random.normal(0, 0.1, (n_points, 3))
        obs_vec = np.array([0.0, 1.0, 0.0]) + np.random.normal(0, 0.1, (n_points, 3))

        # Normalize
        sun_vec = sun_vec / np.linalg.norm(sun_vec, axis=1, keepdims=True)
        obs_vec = obs_vec / np.linalg.norm(obs_vec, axis=1, keepdims=True)

        # Assemble epoch [JD, brightness, sun_xyz, obs_xyz]
        epoch = np.column_stack([times, brightness, sun_vec, obs_vec])
        epochs.append(epoch)

    return epochs


if __name__ == "__main__":
    # Run examples
    example_simple()
    example_period_estimation()
    example_custom_config()

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
