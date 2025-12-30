#!/usr/bin/env python3
"""
Unit tests for the Lomb-Scargle and consensus pipeline.

Run with:
    pytest tests/test_ls_and_consensus.py -v

Or:
    python -m pytest tests/test_ls_and_consensus.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lc_pipeline import (
    PeriodConfig,
    LightcurveEpoch,
    AsteroidLightcurves,
    ConsensusEngine,
    lomb_scargle_period_search,
    scores_to_probs,
    cluster_periods,
    compute_credible_interval,
    posterior_summary,
    relative_error,
    alias_aware_relative_error,
)


class TestLombScargle:
    """Tests for Lomb-Scargle period search."""

    def test_sinusoid_recovery(self):
        """
        Test that LS correctly recovers a known sinusoidal period.

        Generate a synthetic lightcurve with P_true = 7.0 hours,
        verify that the best candidate is within 1% of true period.
        """
        # True period in hours
        P_true_hours = 7.0
        P_true_days = P_true_hours / 24.0

        # Generate synthetic lightcurve over 3 days
        np.random.seed(42)
        n_points = 100
        time = np.sort(np.random.uniform(0, 3.0, n_points))  # days

        # Sinusoidal signal with small noise
        phase = 2 * np.pi * time / P_true_days
        mag = np.sin(phase) + 0.01 * np.random.randn(n_points)
        mag_err = np.full(n_points, 0.02)

        # Configure search
        config = PeriodConfig(
            min_period_hours=2.0,
            max_period_hours=24.0,
            n_freq=10000,
            top_k=10,
        )

        # Run LS search
        periods, powers = lomb_scargle_period_search(
            time=time,
            mag=mag,
            mag_err=mag_err,
            config=config
        )

        # Best candidate should be very close to true period
        best_period = periods[0]
        rel_err = abs(best_period - P_true_hours) / P_true_hours

        assert rel_err < 0.01, (
            f"Best period {best_period:.4f}h is {rel_err*100:.2f}% off "
            f"from true {P_true_hours}h"
        )

    def test_minimum_points_error(self):
        """Test that LS raises error with too few points."""
        time = np.array([0.0, 1.0])  # Only 2 points
        mag = np.array([1.0, 1.1])
        mag_err = np.array([0.02, 0.02])

        with pytest.raises(ValueError, match="at least 3"):
            lomb_scargle_period_search(time, mag, mag_err)

    def test_returns_correct_shape(self):
        """Test that LS returns arrays of expected shape."""
        np.random.seed(123)
        time = np.sort(np.random.uniform(0, 5, 50))
        mag = np.random.randn(50)
        mag_err = np.full(50, 0.02)

        config = PeriodConfig(top_k=20)
        periods, powers = lomb_scargle_period_search(time, mag, mag_err, config)

        assert len(periods) == 20
        assert len(powers) == 20
        assert np.all(np.isfinite(periods))
        assert np.all(np.isfinite(powers))


class TestPosterior:
    """Tests for posterior computation functions."""

    def test_softmax_normalization(self):
        """Test that softmax produces normalized probabilities."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        temperature = 1.0

        probs = scores_to_probs(scores, temperature)

        assert len(probs) == len(scores)
        assert np.isclose(probs.sum(), 1.0)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_softmax_temperature_effect(self):
        """Test that higher temperature makes distribution more uniform."""
        scores = np.array([1.0, 10.0])

        probs_low_T = scores_to_probs(scores, temperature=1.0)
        probs_high_T = scores_to_probs(scores, temperature=100.0)

        # High temperature should be more uniform
        entropy_low = -np.sum(probs_low_T * np.log(probs_low_T + 1e-10))
        entropy_high = -np.sum(probs_high_T * np.log(probs_high_T + 1e-10))

        assert entropy_high > entropy_low

    def test_cluster_periods(self):
        """Test period clustering."""
        # Periods with some close together
        periods = np.array([10.0, 10.1, 10.2, 20.0, 20.05, 30.0])
        tol = 0.03  # 3% tolerance

        clusters = cluster_periods(periods, tol)

        # Should cluster into ~3 groups
        assert len(clusters) <= 4
        assert len(clusters) >= 2

    def test_credible_interval_contains_map(self):
        """Test that credible interval contains the MAP estimate."""
        periods = np.array([5.0, 7.0, 9.0, 11.0, 13.0])
        probs = np.array([0.1, 0.3, 0.4, 0.15, 0.05])  # MAP at 9.0

        ci_low, ci_high = compute_credible_interval(periods, probs, mass=0.68)

        map_idx = np.argmax(probs)
        map_period = periods[map_idx]

        assert ci_low <= map_period <= ci_high

    def test_credible_interval_mass(self):
        """Test that credible interval contains at least target mass."""
        periods = np.linspace(5, 15, 50)
        # Gaussian-like distribution
        probs = np.exp(-0.5 * ((periods - 10) / 1.5) ** 2)
        probs = probs / probs.sum()

        target_mass = 0.68
        ci_low, ci_high = compute_credible_interval(periods, probs, target_mass)

        # Compute actual mass in interval
        mask = (periods >= ci_low) & (periods <= ci_high)
        actual_mass = probs[mask].sum()

        assert actual_mass >= target_mass - 0.01  # Allow small tolerance


class TestConsensusEngine:
    """Tests for multi-epoch consensus engine."""

    def test_multi_epoch_improves_estimate(self):
        """
        Test that multi-epoch consensus produces accurate estimates.

        Create two epochs of synthetic data with the same true period,
        verify that consensus finds the correct period.
        """
        P_true_hours = 7.0
        P_true_days = P_true_hours / 24.0

        np.random.seed(42)

        # Create two epochs
        epochs = []
        for epoch_idx in range(2):
            # Each epoch covers ~3 days, offset by 30 days
            t_offset = epoch_idx * 30.0
            n_points = 80
            time = np.sort(np.random.uniform(0, 3.0, n_points)) + t_offset

            phase = 2 * np.pi * time / P_true_days
            mag = 0.3 * np.sin(phase) + 0.02 * np.random.randn(n_points)
            mag_err = np.full(n_points, 0.02)

            epochs.append(LightcurveEpoch(
                object_id="test_asteroid",
                epoch_id=f"epoch_{epoch_idx}",
                time=time,
                mag=mag,
                mag_err=mag_err
            ))

        asteroid = AsteroidLightcurves(
            object_id="test_asteroid",
            epochs=epochs
        )

        # Run consensus
        config = PeriodConfig(
            min_period_hours=2.0,
            max_period_hours=24.0,
            n_freq=10000,
            top_k=32,
            temperature=5.0,
        )
        engine = ConsensusEngine(config)
        result = engine.predict_multi_epoch(asteroid)

        assert result["success"], f"Consensus failed: {result.get('error')}"
        assert result["n_epochs_used"] == 2

        pred_period = result["period"]
        rel_err = abs(pred_period - P_true_hours) / P_true_hours

        assert rel_err < 0.01, (
            f"Predicted period {pred_period:.4f}h is {rel_err*100:.2f}% off "
            f"from true {P_true_hours}h"
        )

    def test_posterior_sums_to_one(self):
        """Test that the posterior probabilities sum to 1."""
        P_true_hours = 8.0
        P_true_days = P_true_hours / 24.0

        np.random.seed(123)

        # Single epoch
        n_points = 60
        time = np.sort(np.random.uniform(0, 2.0, n_points))
        phase = 2 * np.pi * time / P_true_days
        mag = 0.2 * np.sin(phase) + 0.03 * np.random.randn(n_points)
        mag_err = np.full(n_points, 0.02)

        epoch = LightcurveEpoch(
            object_id="test",
            epoch_id="epoch_0",
            time=time,
            mag=mag,
            mag_err=mag_err
        )

        asteroid = AsteroidLightcurves(
            object_id="test",
            epochs=[epoch]
        )

        engine = ConsensusEngine()
        result = engine.predict_multi_epoch(asteroid)

        assert result["success"]

        probs = result["probs"]
        assert np.isclose(probs.sum(), 1.0, atol=1e-6), (
            f"Posterior sums to {probs.sum()}, not 1.0"
        )

    def test_credible_interval_valid(self):
        """Test that credible interval is valid and contains MAP."""
        P_true_hours = 6.0
        P_true_days = P_true_hours / 24.0

        np.random.seed(456)

        n_points = 100
        time = np.sort(np.random.uniform(0, 4.0, n_points))
        phase = 2 * np.pi * time / P_true_days
        mag = 0.4 * np.sin(phase) + 0.02 * np.random.randn(n_points)
        mag_err = np.full(n_points, 0.02)

        epoch = LightcurveEpoch(
            object_id="test",
            epoch_id="epoch_0",
            time=time,
            mag=mag,
            mag_err=mag_err
        )

        asteroid = AsteroidLightcurves(
            object_id="test",
            epochs=[epoch]
        )

        engine = ConsensusEngine()
        result = engine.predict_multi_epoch(asteroid)

        assert result["success"]

        ci_low = result["ci_low"]
        ci_high = result["ci_high"]
        map_period = result["period"]

        assert np.isfinite(ci_low)
        assert np.isfinite(ci_high)
        assert ci_low <= map_period <= ci_high


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_relative_error(self):
        """Test relative error computation."""
        assert relative_error(10.0, 10.0) == 0.0
        assert relative_error(11.0, 10.0) == 0.1
        assert relative_error(9.0, 10.0) == 0.1
        assert np.isclose(relative_error(15.0, 10.0), 0.5)

    def test_alias_aware_error(self):
        """Test alias-aware error considers half and double periods."""
        true_period = 10.0

        # Exact match
        assert alias_aware_relative_error(10.0, true_period) == 0.0

        # Half-period alias (should be close to 0)
        assert alias_aware_relative_error(5.0, true_period) < 0.01

        # Double-period alias (should be close to 0)
        assert alias_aware_relative_error(20.0, true_period) < 0.01

        # Non-alias (should have larger error)
        err_non_alias = alias_aware_relative_error(7.0, true_period)
        assert err_non_alias > 0.2

    def test_alias_aware_picks_best(self):
        """Test that alias-aware error picks the best match."""
        true_period = 10.0

        # Slightly off from half-period
        pred = 5.1  # 2% off from 5.0 (half of 10)
        err = alias_aware_relative_error(pred, true_period)
        assert err < 0.03  # Should match the 0.5*true alias


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
