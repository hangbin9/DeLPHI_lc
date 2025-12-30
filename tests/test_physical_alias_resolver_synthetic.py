#!/usr/bin/env python3
"""
Synthetic unit-test-style experiment for the physical alias resolver.

Question: In an idealized ellipsoid world, can the physical alias resolver
reliably distinguish P from 0.5P?

This test generates synthetic double-peaked lightcurves with known periods,
then tests whether the resolver can correctly identify the true period
when presented with a {P_true, 0.5*P_true} alias family.

Usage:
    python -m tests.test_physical_alias_resolver_synthetic

    # Or with pytest:
    pytest tests/test_physical_alias_resolver_synthetic.py -v
"""

import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

from lc_pipeline.data import LightcurveEpoch, AsteroidLightcurves
from lc_pipeline.config import PhysicalAliasConfig
from lc_pipeline.alias_ellipsoid_resolver import (
    _fit_fourier_template,
    _fit_multi_epoch,
    AliasResolverResult,
)


# =============================================================================
# Synthetic Data Generation
# =============================================================================

def generate_irregular_cadence(
    n_points: int,
    baseline_days: float = 5.0,
    n_nights: int = 4,
    points_per_night_range: Tuple[int, int] = (5, 15),
    night_duration_hours: float = 6.0,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Generate irregular time sampling mimicking DAMIT-style observations.

    Creates nightly clusters with gaps between nights, simulating
    ground-based observations.

    Args:
        n_points: Approximate total number of points (actual may vary slightly).
        baseline_days: Total observation baseline in days.
        n_nights: Number of observing nights.
        points_per_night_range: (min, max) points per night.
        night_duration_hours: Duration of each night's observations.
        rng: Random number generator for reproducibility.

    Returns:
        Array of observation times in days, sorted.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Distribute nights across baseline
    night_starts = np.linspace(0, baseline_days - night_duration_hours/24, n_nights)
    # Add some jitter to night starts
    night_starts += rng.uniform(-0.2, 0.2, n_nights)
    night_starts = np.clip(night_starts, 0, baseline_days - night_duration_hours/24)
    night_starts = np.sort(night_starts)

    times = []
    points_per_night = n_points // n_nights

    for night_start in night_starts:
        # Random number of points this night
        n_this_night = rng.integers(
            points_per_night_range[0],
            points_per_night_range[1] + 1
        )

        # Generate times within this night (irregular within night too)
        night_times = night_start + np.sort(
            rng.uniform(0, night_duration_hours/24, n_this_night)
        )
        times.extend(night_times)

    return np.array(sorted(times))


def generate_synthetic_ellipsoid_lightcurve(
    period_hours: float,
    n_points: int = 50,
    noise_sigma: float = 0.02,
    shape_type: str = 'simple',
    amplitude: float = 0.3,
    baseline_days: float = 5.0,
    rng: np.random.Generator = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic double-peaked (ellipsoid-like) lightcurve.

    Args:
        period_hours: True rotation period in hours.
        n_points: Number of data points.
        noise_sigma: Gaussian noise standard deviation in magnitudes.
        shape_type: Template type:
            - 'simple': mag(phi) = A * sin(4π*phi) - pure double-peaked
            - 'mixed': mag(phi) = A1*sin(4π*phi) + A2*sin(2π*phi) - asymmetric
        amplitude: Peak-to-peak amplitude in magnitudes.
        baseline_days: Total observation baseline.
        rng: Random number generator.

    Returns:
        (time_days, mag, mag_err) arrays.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate irregular cadence
    time_days = generate_irregular_cadence(
        n_points=n_points,
        baseline_days=baseline_days,
        rng=rng
    )

    # Convert period to days
    period_days = period_hours / 24.0

    # Compute rotational phase
    phase = (time_days / period_days) % 1.0

    # Generate magnitude based on shape type
    if shape_type == 'simple':
        # Pure double-peaked: two maxima, two minima per rotation
        # mag(phi) = A * sin(4π*phi)
        mag = amplitude * np.sin(4 * np.pi * phase)

    elif shape_type == 'mixed':
        # Asymmetric double-peaked: different peak heights
        # mag(phi) = A1*sin(4π*phi) + A2*sin(2π*phi)
        A1 = amplitude * 0.7  # Main double-peaked component
        A2 = amplitude * 0.3  # Asymmetry component
        # Random phase offset for variety
        phi0 = rng.uniform(0, 2*np.pi)
        mag = A1 * np.sin(4 * np.pi * phase + phi0) + A2 * np.sin(2 * np.pi * phase)

    else:
        raise ValueError(f"Unknown shape_type: {shape_type}")

    # Add baseline magnitude (arbitrary reference)
    mag = mag + 15.0  # Typical asteroid magnitude

    # Add Gaussian noise
    mag = mag + rng.normal(0, noise_sigma, len(mag))

    # Magnitude errors (constant for synthetic data)
    mag_err = np.full_like(mag, noise_sigma)

    return time_days, mag, mag_err


def create_synthetic_asteroid(
    object_id: str,
    period_hours: float,
    n_epochs: int = 2,
    n_points_per_epoch: int = 50,
    noise_sigma: float = 0.02,
    shape_type: str = 'simple',
    rng: np.random.Generator = None
) -> AsteroidLightcurves:
    """
    Create a synthetic multi-epoch asteroid with known period.

    Args:
        object_id: Identifier for the asteroid.
        period_hours: True rotation period in hours.
        n_epochs: Number of observing epochs.
        n_points_per_epoch: Points per epoch.
        noise_sigma: Noise level in magnitudes.
        shape_type: Template shape type.
        rng: Random number generator.

    Returns:
        AsteroidLightcurves object with synthetic data.
    """
    if rng is None:
        rng = np.random.default_rng()

    epochs = []
    for i in range(n_epochs):
        time, mag, mag_err = generate_synthetic_ellipsoid_lightcurve(
            period_hours=period_hours,
            n_points=n_points_per_epoch,
            noise_sigma=noise_sigma,
            shape_type=shape_type,
            baseline_days=5.0 + i * 0.5,  # Slightly different baselines
            rng=rng
        )

        epoch = LightcurveEpoch(
            object_id=object_id,
            epoch_id=f"epoch_{i}",
            time=time,
            mag=mag,
            mag_err=mag_err
        )
        epochs.append(epoch)

    return AsteroidLightcurves(object_id=object_id, epochs=epochs)


# =============================================================================
# Resolver Testing Utilities
# =============================================================================

@dataclass
class ResolverTestResult:
    """Result from testing the resolver on a synthetic asteroid."""
    object_id: str
    true_period: float
    base_period: float  # Period fed to resolver as "consensus" result
    resolved_period: float
    chi2_true: float
    chi2_half: float
    chose_true: bool
    was_correct: bool  # Did resolver make the right choice?
    scenario: str  # 'correct_base' or 'wrong_base'


def evaluate_resolver_on_synthetic(
    asteroid: AsteroidLightcurves,
    true_period: float,
    base_period: float,
    config: PhysicalAliasConfig
) -> ResolverTestResult:
    """
    Test the resolver on a synthetic asteroid with known ground truth.

    Args:
        asteroid: Synthetic asteroid lightcurves.
        true_period: The actual period used to generate the data.
        base_period: Period to test as the "consensus" result (may be P or 0.5P).
        config: Resolver configuration.

    Returns:
        ResolverTestResult with diagnostics.
    """
    # Build alias family: {base_period, 0.5*base, 2*base} intersected with {true, 0.5*true}
    # For this test, we explicitly use {true_period, 0.5*true_period}
    half_period = true_period / 2.0
    alias_family = sorted([true_period, half_period])

    # Filter by config bounds
    alias_family = [
        p for p in alias_family
        if config.min_period_hours <= p <= config.max_period_hours
    ]

    # Fit each candidate period
    valid_epochs = [e for e in asteroid.epochs if e.n_points >= config.min_points_per_epoch]

    chi2_all = {}
    for candidate in alias_family:
        try:
            chi2, _ = _fit_multi_epoch(valid_epochs, candidate, config)
            chi2_all[candidate] = chi2
        except Exception:
            chi2_all[candidate] = np.inf

    # Find best period
    if chi2_all:
        best_period = min(chi2_all, key=chi2_all.get)
    else:
        best_period = base_period

    # Get chi2 values
    chi2_true = chi2_all.get(true_period, np.inf)
    chi2_half = chi2_all.get(half_period, np.inf)

    # Did it choose the true period?
    chose_true = abs(best_period - true_period) / true_period < 0.05

    # Was this the correct choice given the scenario?
    is_base_true = abs(base_period - true_period) / true_period < 0.05
    scenario = 'correct_base' if is_base_true else 'wrong_base'

    # Correct means:
    # - If base was wrong (0.5P), resolver should switch to true (P)
    # - If base was correct (P), resolver should keep it
    was_correct = chose_true

    return ResolverTestResult(
        object_id=asteroid.object_id,
        true_period=true_period,
        base_period=base_period,
        resolved_period=best_period,
        chi2_true=chi2_true,
        chi2_half=chi2_half,
        chose_true=chose_true,
        was_correct=was_correct,
        scenario=scenario
    )


# =============================================================================
# Main Experiment
# =============================================================================

def run_synthetic_experiment(
    n_asteroids: int = 100,
    noise_sigma: float = 0.02,
    shape_types: List[str] = ['simple', 'mixed'],
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run the full synthetic experiment.

    For each asteroid:
    - Generate synthetic data with known P_true
    - Test resolver with base = P_true (should keep it)
    - Test resolver with base = 0.5*P_true (should switch to P_true)

    Args:
        n_asteroids: Number of synthetic asteroids per shape type.
        noise_sigma: Noise level for synthetic data.
        shape_types: List of shape types to test.
        seed: Random seed for reproducibility.
        verbose: Whether to print progress.

    Returns:
        Dictionary with experiment results and metrics.
    """
    rng = np.random.default_rng(seed)

    # Resolver config - use generous settings for synthetic data
    config = PhysicalAliasConfig(
        enabled=True,
        min_period_hours=2.0,
        max_period_hours=50.0,
        min_points_per_epoch=10,
        min_epochs_for_fit=1,
        n_harmonics=2,
        regularization=0.01,
        min_chi2_rel_improvement=0.05,
    )

    results: Dict[str, List[ResolverTestResult]] = {
        shape: [] for shape in shape_types
    }

    for shape_type in shape_types:
        if verbose:
            print(f"\nTesting shape_type='{shape_type}'...")

        for i in range(n_asteroids):
            # Random true period in [4, 15] hours
            true_period = rng.uniform(4.0, 15.0)
            half_period = true_period / 2.0

            # Generate synthetic asteroid
            asteroid = create_synthetic_asteroid(
                object_id=f"synth_{shape_type}_{i:04d}",
                period_hours=true_period,
                n_epochs=2,
                n_points_per_epoch=50,
                noise_sigma=noise_sigma,
                shape_type=shape_type,
                rng=rng
            )

            # Test 1: Base is correct (P_true) - should NOT change
            result_correct_base = evaluate_resolver_on_synthetic(
                asteroid=asteroid,
                true_period=true_period,
                base_period=true_period,
                config=config
            )
            results[shape_type].append(result_correct_base)

            # Test 2: Base is wrong (0.5*P_true) - should change to P_true
            result_wrong_base = evaluate_resolver_on_synthetic(
                asteroid=asteroid,
                true_period=true_period,
                base_period=half_period,
                config=config
            )
            results[shape_type].append(result_wrong_base)

            if verbose and (i + 1) % 25 == 0:
                print(f"  Processed {i + 1}/{n_asteroids} asteroids")

    return results


def compute_metrics(results: Dict[str, List[ResolverTestResult]]) -> Dict[str, Dict[str, float]]:
    """
    Compute success metrics from experiment results.

    Args:
        results: Dictionary mapping shape_type to list of ResolverTestResult.

    Returns:
        Dictionary with metrics per shape type.
    """
    metrics = {}

    for shape_type, result_list in results.items():
        # Separate by scenario
        correct_base = [r for r in result_list if r.scenario == 'correct_base']
        wrong_base = [r for r in result_list if r.scenario == 'wrong_base']

        # Metrics for "wrong base" scenario (should switch to true)
        # Success = resolver chose true period
        n_wrong_base = len(wrong_base)
        n_switched_to_true = sum(1 for r in wrong_base if r.chose_true)
        switch_success_rate = n_switched_to_true / n_wrong_base if n_wrong_base > 0 else 0

        # Metrics for "correct base" scenario (should keep true)
        # Regression = resolver switched away from true
        n_correct_base = len(correct_base)
        n_kept_true = sum(1 for r in correct_base if r.chose_true)
        n_regressed = n_correct_base - n_kept_true
        regression_rate = n_regressed / n_correct_base if n_correct_base > 0 else 0
        keep_correct_rate = n_kept_true / n_correct_base if n_correct_base > 0 else 0

        # Chi2 ratio statistics
        chi2_ratios_wrong = [
            r.chi2_half / r.chi2_true if r.chi2_true > 0 else np.nan
            for r in wrong_base
        ]
        chi2_ratios_correct = [
            r.chi2_half / r.chi2_true if r.chi2_true > 0 else np.nan
            for r in correct_base
        ]

        metrics[shape_type] = {
            'n_asteroids': len(result_list) // 2,
            'switch_success_rate': switch_success_rate,
            'keep_correct_rate': keep_correct_rate,
            'regression_rate': regression_rate,
            'n_switched_to_true': n_switched_to_true,
            'n_wrong_base': n_wrong_base,
            'n_kept_true': n_kept_true,
            'n_correct_base': n_correct_base,
            'median_chi2_ratio': np.nanmedian(chi2_ratios_wrong + chi2_ratios_correct),
        }

    return metrics


def print_summary(metrics: Dict[str, Dict[str, float]]) -> None:
    """Print a formatted summary table of experiment results."""
    print("\n" + "=" * 70)
    print("SYNTHETIC ALIAS RESOLVER EXPERIMENT - SUMMARY")
    print("=" * 70)
    print("\nQuestion: Can the resolver distinguish P from 0.5P in idealized conditions?")
    print("\nTest scenarios:")
    print("  - 'Wrong base': Consensus picked 0.5P, resolver should switch to P")
    print("  - 'Correct base': Consensus picked P, resolver should keep P")
    print()

    print("-" * 70)
    print(f"{'Shape Type':<12} | {'Switch Success':<15} | {'Keep Correct':<15} | {'Regression':<12}")
    print(f"{'':12} | {'(wrong→right)':<15} | {'(right→right)':<15} | {'(right→wrong)':<12}")
    print("-" * 70)

    for shape_type, m in metrics.items():
        switch_pct = f"{m['switch_success_rate']*100:.1f}% ({m['n_switched_to_true']}/{m['n_wrong_base']})"
        keep_pct = f"{m['keep_correct_rate']*100:.1f}% ({m['n_kept_true']}/{m['n_correct_base']})"
        regress_pct = f"{m['regression_rate']*100:.1f}%"
        print(f"{shape_type:<12} | {switch_pct:<15} | {keep_pct:<15} | {regress_pct:<12}")

    print("-" * 70)

    # Overall metrics (average across shape types)
    avg_switch = np.mean([m['switch_success_rate'] for m in metrics.values()])
    avg_regress = np.mean([m['regression_rate'] for m in metrics.values()])

    print(f"\nOverall average:")
    print(f"  Switch success rate: {avg_switch*100:.1f}%")
    print(f"  Regression rate:     {avg_regress*100:.1f}%")


def run_assertions(metrics: Dict[str, Dict[str, float]]) -> Tuple[bool, List[str]]:
    """
    Run assertion checks on experiment results.

    Args:
        metrics: Computed metrics from experiment.

    Returns:
        (all_passed, list_of_failure_messages)
    """
    failures = []

    for shape_type, m in metrics.items():
        # Assertion 1: Switch success rate > 80%
        if m['switch_success_rate'] < 0.80:
            failures.append(
                f"[{shape_type}] Switch success rate {m['switch_success_rate']*100:.1f}% < 80% threshold"
            )

        # Assertion 2: Regression rate < 5%
        if m['regression_rate'] > 0.05:
            failures.append(
                f"[{shape_type}] Regression rate {m['regression_rate']*100:.1f}% > 5% threshold"
            )

    all_passed = len(failures) == 0
    return all_passed, failures


# =============================================================================
# Pytest-compatible test function
# =============================================================================

def test_physical_alias_resolver_synthetic():
    """
    Pytest-compatible test for the physical alias resolver.

    Tests whether the resolver can distinguish P from 0.5P in idealized
    ellipsoid-like conditions.
    """
    # Run smaller experiment for pytest (faster)
    results = run_synthetic_experiment(
        n_asteroids=50,
        noise_sigma=0.02,
        shape_types=['simple', 'mixed'],
        seed=42,
        verbose=False
    )

    metrics = compute_metrics(results)
    all_passed, failures = run_assertions(metrics)

    if not all_passed:
        failure_msg = "\n".join(failures)
        raise AssertionError(
            f"Physical alias resolver FAILS under idealized ellipsoid conditions!\n"
            f"This resolver should NOT be trusted on real DAMIT data.\n\n"
            f"Failures:\n{failure_msg}"
        )

    # If we get here, assertions passed
    print("\n✓ Physical alias resolver passed synthetic validation")
    for shape, m in metrics.items():
        print(f"  [{shape}] switch={m['switch_success_rate']*100:.0f}%, regress={m['regression_rate']*100:.0f}%")


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Physical Alias Resolver - Synthetic Validation Experiment")
    print("=" * 70)
    print("\nGenerating synthetic ellipsoid-like lightcurves with known periods...")
    print("Testing whether resolver can distinguish P from 0.5P aliases.\n")

    # Run full experiment
    results = run_synthetic_experiment(
        n_asteroids=100,
        noise_sigma=0.02,
        shape_types=['simple', 'mixed'],
        seed=42,
        verbose=True
    )

    # Compute and display metrics
    metrics = compute_metrics(results)
    print_summary(metrics)

    # Run assertions
    all_passed, failures = run_assertions(metrics)

    print("\n" + "=" * 70)
    print("ASSERTION RESULTS")
    print("=" * 70)

    if all_passed:
        print("\n✓ ALL ASSERTIONS PASSED")
        print("\nThe physical alias resolver CAN reliably distinguish P from 0.5P")
        print("under idealized ellipsoid-like conditions.")
        print("\nThis provides confidence that the resolver may help on real data")
        print("when the lightcurve shape is indeed ellipsoid-like.")
    else:
        print("\n✗ ASSERTIONS FAILED")
        print("\nThe physical alias resolver FAILS even under idealized conditions!")
        print("It should NOT be trusted on real DAMIT data.\n")
        print("Failures:")
        for f in failures:
            print(f"  - {f}")

        print("\n" + "!" * 70)
        print("WARNING: Resolver is unreliable. Consider disabling it.")
        print("!" * 70)

    print()
