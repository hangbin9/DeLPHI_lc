#!/usr/bin/env python3
"""
Tests for phase-stratified sampling.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pole_synth.phase_stratified_sampling import (
    compute_phase_bins,
    phase_stratified_indices,
    random_indices,
    maybe_phase_stratified_indices,
    compute_bin_coverage,
)


class TestComputePhaseBins:
    """Tests for compute_phase_bins function."""

    def test_basic_binning(self):
        """Basic phase binning with known values."""
        # Period of 24 hours = 1 day
        # t=0 -> phase 0 -> bin 0
        # t=0.5 -> phase 0.5 -> bin 4 (for n_bins=8)
        t = np.array([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
        period_hours = 24.0

        bins = compute_phase_bins(t, period_hours, n_bins=8, t0=0.0)

        expected = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        np.testing.assert_array_equal(bins, expected)

    def test_wrapping(self):
        """Phase should wrap around after one period."""
        t = np.array([0.0, 1.0, 2.0, 2.5])  # days
        period_hours = 24.0  # 1 day

        bins = compute_phase_bins(t, period_hours, n_bins=8, t0=0.0)

        # t=0 -> phase 0 -> bin 0
        # t=1 -> phase 0 (wrapped) -> bin 0
        # t=2 -> phase 0 (wrapped) -> bin 0
        # t=2.5 -> phase 0.5 -> bin 4
        expected = np.array([0, 0, 0, 4])
        np.testing.assert_array_equal(bins, expected)

    def test_custom_t0(self):
        """Custom t0 should shift phase."""
        t = np.array([1.0, 1.125, 1.25])
        period_hours = 24.0

        # With t0=1.0, t=1.0 should be phase 0
        bins = compute_phase_bins(t, period_hours, n_bins=8, t0=1.0)
        expected = np.array([0, 1, 2])
        np.testing.assert_array_equal(bins, expected)

    def test_default_t0_is_deterministic(self):
        """Default t0=t.min() ensures determinism."""
        t = np.array([5.0, 5.125, 5.25])
        period_hours = 24.0

        bins1 = compute_phase_bins(t, period_hours, n_bins=8)
        bins2 = compute_phase_bins(t, period_hours, n_bins=8)

        np.testing.assert_array_equal(bins1, bins2)

    def test_invalid_period_raises(self):
        """Zero or negative period should raise."""
        t = np.array([0.0, 0.5])

        with pytest.raises(ValueError):
            compute_phase_bins(t, period_hours=0.0, n_bins=8)

        with pytest.raises(ValueError):
            compute_phase_bins(t, period_hours=-1.0, n_bins=8)

    def test_bins_in_valid_range(self):
        """All bins should be in [0, n_bins-1]."""
        rng = np.random.default_rng(42)
        t = rng.uniform(0, 100, 1000)
        period_hours = 7.3

        for n_bins in [4, 8, 16]:
            bins = compute_phase_bins(t, period_hours, n_bins=n_bins)
            assert bins.min() >= 0
            assert bins.max() < n_bins


class TestPhaseStratifiedIndices:
    """Tests for phase_stratified_indices function."""

    def test_determinism(self):
        """Same seed should produce same indices."""
        t = np.linspace(0, 10, 100)
        period_hours = 5.0

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        idx1 = phase_stratified_indices(t, period_hours, n_select=20, rng=rng1)
        idx2 = phase_stratified_indices(t, period_hours, n_select=20, rng=rng2)

        np.testing.assert_array_equal(idx1, idx2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different indices."""
        t = np.linspace(0, 10, 100)
        period_hours = 5.0

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(123)

        idx1 = phase_stratified_indices(t, period_hours, n_select=20, rng=rng1)
        idx2 = phase_stratified_indices(t, period_hours, n_select=20, rng=rng2)

        # Very unlikely to be identical
        assert not np.array_equal(idx1, idx2)

    def test_correct_count(self):
        """Should return exactly n_select indices."""
        t = np.linspace(0, 10, 100)
        period_hours = 5.0

        for n_select in [5, 10, 20, 50]:
            rng = np.random.default_rng(42)
            idx = phase_stratified_indices(t, period_hours, n_select=n_select, rng=rng)
            assert len(idx) == n_select

    def test_no_duplicates(self):
        """Selected indices should be unique."""
        t = np.linspace(0, 10, 100)
        period_hours = 5.0
        rng = np.random.default_rng(42)

        idx = phase_stratified_indices(t, period_hours, n_select=50, rng=rng)

        assert len(idx) == len(set(idx))

    def test_valid_indices(self):
        """All indices should be valid for the array."""
        t = np.linspace(0, 10, 100)
        period_hours = 5.0
        rng = np.random.default_rng(42)

        idx = phase_stratified_indices(t, period_hours, n_select=50, rng=rng)

        assert idx.min() >= 0
        assert idx.max() < len(t)

    def test_bin_coverage(self):
        """Selected indices should cover multiple bins when possible."""
        # Create data that spans multiple phases
        t = np.linspace(0, 10, 200)  # 200 points over 10 days
        period_hours = 24.0  # 1 day period
        rng = np.random.default_rng(42)

        idx = phase_stratified_indices(t, period_hours, n_select=64, n_bins=8, rng=rng)

        # Check how many bins are covered
        coverage = compute_bin_coverage(idx, t, period_hours, n_bins=8)

        # With 64 samples from 8 bins, should hit all or most bins
        assert coverage >= 6  # At least 6 of 8 bins

    def test_raises_when_n_select_too_large(self):
        """Should raise when n_select > len(t)."""
        t = np.array([0.0, 1.0, 2.0])
        period_hours = 5.0

        with pytest.raises(ValueError):
            phase_stratified_indices(t, period_hours, n_select=10, rng=np.random.default_rng(42))

    def test_empty_selection(self):
        """n_select=0 should return empty array."""
        t = np.linspace(0, 10, 100)
        period_hours = 5.0
        rng = np.random.default_rng(42)

        idx = phase_stratified_indices(t, period_hours, n_select=0, rng=rng)

        assert len(idx) == 0


class TestRandomIndices:
    """Tests for random_indices function."""

    def test_determinism(self):
        """Same seed should produce same indices."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        idx1 = random_indices(100, 20, rng=rng1)
        idx2 = random_indices(100, 20, rng=rng2)

        np.testing.assert_array_equal(idx1, idx2)

    def test_correct_count(self):
        """Should return exactly n_select indices."""
        rng = np.random.default_rng(42)
        idx = random_indices(100, 30, rng=rng)
        assert len(idx) == 30

    def test_no_duplicates(self):
        """Should sample without replacement."""
        rng = np.random.default_rng(42)
        idx = random_indices(100, 50, rng=rng)
        assert len(idx) == len(set(idx))


class TestMaybePhaseStratifiedIndices:
    """Tests for maybe_phase_stratified_indices function."""

    def test_with_period_uses_stratified(self):
        """When period is available, use phase-stratified sampling."""
        t = np.linspace(0, 10, 100)
        period_hours = 5.0
        rng = np.random.default_rng(42)

        idx = maybe_phase_stratified_indices(
            t, period_hours, n_select=20, rng=rng, allow_missing_periods=False
        )

        # Should succeed and return correct count
        assert len(idx) == 20

    def test_missing_period_strict_raises(self):
        """Missing period with strict mode should raise."""
        t = np.linspace(0, 10, 100)

        with pytest.raises(ValueError) as excinfo:
            maybe_phase_stratified_indices(
                t,
                period_hours=None,
                n_select=20,
                allow_missing_periods=False,
                object_id="damit_101_101",
            )

        assert "Period not available" in str(excinfo.value)
        assert "damit_101_101" in str(excinfo.value)

    def test_missing_period_allowed_uses_random(self):
        """Missing period with allow=True should use random sampling."""
        t = np.linspace(0, 10, 100)
        rng = np.random.default_rng(42)

        idx = maybe_phase_stratified_indices(
            t,
            period_hours=None,
            n_select=20,
            rng=rng,
            allow_missing_periods=True,
        )

        # Should succeed with random sampling
        assert len(idx) == 20
        assert len(set(idx)) == 20  # No duplicates

    def test_determinism_with_missing_period(self):
        """Random fallback should still be deterministic."""
        t = np.linspace(0, 10, 100)

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        idx1 = maybe_phase_stratified_indices(
            t, period_hours=None, n_select=20, rng=rng1, allow_missing_periods=True
        )
        idx2 = maybe_phase_stratified_indices(
            t, period_hours=None, n_select=20, rng=rng2, allow_missing_periods=True
        )

        np.testing.assert_array_equal(idx1, idx2)


class TestComputeBinCoverage:
    """Tests for compute_bin_coverage function."""

    def test_full_coverage(self):
        """All bins should be covered when indices span all phases."""
        # Create indices that hit each bin
        t = np.array([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
        period_hours = 24.0
        indices = np.arange(8)

        coverage = compute_bin_coverage(indices, t, period_hours, n_bins=8)
        assert coverage == 8

    def test_partial_coverage(self):
        """Should count unique bins correctly."""
        t = np.array([0.0, 0.1, 0.2])  # All in bin 0 or 1
        period_hours = 24.0
        indices = np.array([0, 1, 2])

        coverage = compute_bin_coverage(indices, t, period_hours, n_bins=8)
        assert coverage <= 2

    def test_empty_indices(self):
        """Empty indices should have 0 coverage."""
        t = np.linspace(0, 10, 100)
        period_hours = 5.0

        coverage = compute_bin_coverage(np.array([]), t, period_hours, n_bins=8)
        assert coverage == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
