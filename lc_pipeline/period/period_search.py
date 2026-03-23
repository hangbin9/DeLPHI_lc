"""
Lomb-Scargle period search functionality.

This module provides:
- lomb_scargle_period_search: main LS search returning top-K candidates
- inject_alias_candidates: add 0.5P and 2P alias periods
- sigma_clip: outlier removal helper
"""

from typing import Optional, Tuple

import numpy as np
from astropy.timeseries import LombScargle

from .config import PeriodConfig, DEFAULT_PERIOD_CONFIG


def sigma_clip(
    data: np.ndarray,
    sigma: float = 5.0,
    max_iter: int = 5
) -> np.ndarray:
    """
    Perform iterative sigma-clipping on data.

    Args:
        data: Input data array.
        sigma: Number of standard deviations for clipping threshold.
        max_iter: Maximum number of iterations.

    Returns:
        Boolean mask where True indicates valid (non-clipped) data.
    """
    data = np.asarray(data, dtype=float)
    mask = np.ones(len(data), dtype=bool)

    for _ in range(max_iter):
        valid_data = data[mask]
        if len(valid_data) == 0:
            break

        median = np.median(valid_data)
        std = np.std(valid_data)

        if std == 0:
            break

        new_mask = np.abs(data - median) < sigma * std
        if np.array_equal(mask, new_mask):
            break
        mask = new_mask

    return mask


def _has_close(
    periods: np.ndarray,
    p_new: float,
    tol_rel: float
) -> bool:
    """
    Check if any period in the array is within relative tolerance of p_new.

    Args:
        periods: Array of existing periods.
        p_new: New period to check.
        tol_rel: Relative tolerance (e.g., 0.02 for 2%).

    Returns:
        True if a close period exists, False otherwise.
    """
    if len(periods) == 0:
        return False
    return np.any(np.abs(periods - p_new) / p_new < tol_rel)


def lomb_scargle_period_search(
    time: np.ndarray,
    mag: np.ndarray,
    mag_err: Optional[np.ndarray],
    config: Optional[PeriodConfig] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Lomb-Scargle periodogram and return top-K candidate periods.

    Performs:
    1. Outlier removal via sigma-clipping
    2. Median subtraction for detrending
    3. LS periodogram computation
    4. Selection of top-K candidates by power

    Args:
        time: Array of observation times (days, e.g., JD).
        mag: Array of magnitudes.
        mag_err: Array of magnitude errors (optional, can be None).
        config: Period search configuration.

    Returns:
        periods_hours: Array of top-K candidate periods in hours.
        powers: Array of corresponding LS powers.

    Raises:
        ValueError: If input arrays are empty or too short.
    """
    if config is None:
        config = DEFAULT_PERIOD_CONFIG

    time = np.asarray(time, dtype=float)
    mag = np.asarray(mag, dtype=float)

    if len(time) < 3:
        raise ValueError(f"Need at least 3 data points, got {len(time)}")

    # Sigma-clip outliers
    clip_mask = sigma_clip(mag, sigma=5.0)
    time_clean = time[clip_mask]
    mag_clean = mag[clip_mask]

    if mag_err is not None:
        mag_err = np.asarray(mag_err, dtype=float)
        mag_err_clean = mag_err[clip_mask]
    else:
        mag_err_clean = None

    if len(time_clean) < 3:
        raise ValueError(f"After sigma-clipping, only {len(time_clean)} points remain")

    # Detrend by subtracting median
    mag_clean = mag_clean - np.median(mag_clean)

    # Convert period range from hours to days (time is in days)
    min_period_days = config.min_period_hours / 24.0
    max_period_days = config.max_period_hours / 24.0

    # Frequency range (cycles per day)
    f_min = 1.0 / max_period_days
    f_max = 1.0 / min_period_days

    # Create frequency grid
    freq = np.linspace(f_min, f_max, config.n_freq)

    # Compute Lomb-Scargle periodogram
    if mag_err_clean is not None and np.any(mag_err_clean > 0):
        ls = LombScargle(time_clean, mag_clean, mag_err_clean)
    else:
        ls = LombScargle(time_clean, mag_clean)

    power = ls.power(freq)

    # Handle any NaN/inf in power
    power = np.where(np.isfinite(power), power, 0.0)

    # Select top-K candidates
    n_candidates = min(config.top_k, len(freq))
    top_indices = np.argsort(power)[::-1][:n_candidates]

    # Convert frequencies to periods in hours
    periods_days = 1.0 / freq[top_indices]
    periods_hours = periods_days * 24.0
    top_powers = power[top_indices]

    return periods_hours, top_powers


def detect_harmonic_fundamental(
    periods: np.ndarray,
    powers: np.ndarray,
    config: Optional[PeriodConfig] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect if the strongest peak is a harmonic and correct to fundamental frequency.

    For asteroid lightcurves, the 2f₀ harmonic (half-period) is often strongest
    due to double-peaked brightness curves. This function checks if:
    - The strongest peak has a corresponding half-period (2×frequency) peak
    - The half-period peak is significantly weaker (indicates 2f₀ is strongest)
    - If detected, promotes the fundamental (2×period) as the true period

    Args:
        periods: Array of candidate periods (hours).
        powers: Array of corresponding powers.
        config: Configuration with harmonic detection threshold.

    Returns:
        corrected_periods: Periods with harmonic correction applied.
        corrected_powers: Powers reordered to reflect correction.
    """
    if config is None:
        config = DEFAULT_PERIOD_CONFIG

    periods = np.asarray(periods, dtype=float)
    powers = np.asarray(powers, dtype=float)

    if len(periods) == 0:
        return periods.copy(), powers.copy()

    # Find strongest peak
    strongest_idx = np.argmax(powers)
    strongest_period = periods[strongest_idx]
    strongest_power = powers[strongest_idx]

    # Check for fundamental frequency (double the detected period)
    # This checks if strongest peak might be a harmonic (2f₀)
    fundamental_period = strongest_period * 2.0
    tol = config.match_tol_rel

    # Also check if strongest peak might be the fundamental (f₀)
    # by looking for its harmonic at half-period
    harmonic_period = strongest_period / 2.0

    # Search for fundamental and harmonic in existing candidates
    fundamental_idx = None
    harmonic_idx = None

    for i, p in enumerate(periods):
        if abs(p - fundamental_period) / fundamental_period < tol:
            fundamental_idx = i
        if abs(p - harmonic_period) / harmonic_period < tol:
            harmonic_idx = i

    # Harmonic detection criterion:
    # Only promote fundamental if ALL conditions met:
    # 1. Fundamental exists (2× strongest period)
    # 2. Strongest is much stronger than fundamental (>1.5×)
    # 3. Strongest does NOT have a strong harmonic at half-period
    #
    # Rationale: If strongest peak has a significant harmonic below it,
    # then strongest is already f₀ (fundamental), not 2f₀
    if fundamental_idx is not None:
        fundamental_power = powers[fundamental_idx]
        harmonic_ratio = strongest_power / (fundamental_power + 1e-10)

        # Check if strongest peak has a significant harmonic at half-period
        # Threshold: harmonic power > 50% of strongest
        # If yes, strongest is already the fundamental
        if harmonic_idx is not None:
            harmonic_power = powers[harmonic_idx]
            if harmonic_power > strongest_power * 0.5:
                # Strongest has significant harmonic → strongest is f₀, not 2f₀
                return periods.copy(), powers.copy()

        # Strongest peak is >1.5× stronger than fundamental
        # AND doesn't have a strong harmonic below it
        # → Strongest is likely 2f₀ (harmonic), promote fundamental (f₀)
        if harmonic_ratio > 1.5:
            corrected_powers = powers.copy()
            corrected_powers[fundamental_idx] = strongest_power * 1.1

            return periods, corrected_powers

    return periods.copy(), powers.copy()


def inject_alias_candidates(
    periods: np.ndarray,
    powers: np.ndarray,
    config: Optional[PeriodConfig] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject alias candidates (0.5P, 2P) into the candidate list.

    For each period P, adds P/2 and 2P as candidates if:
    - They fall within the search range [min_period, max_period]
    - They are not already present within match_tol_rel

    Injected aliases receive a small power: min(powers) * alias_base_power_scale

    Args:
        periods: Array of candidate periods (hours).
        powers: Array of corresponding powers.
        config: Configuration with alias settings.

    Returns:
        augmented_periods: Combined array with original and alias candidates.
        augmented_powers: Combined array with original and alias powers.
    """
    if config is None:
        config = DEFAULT_PERIOD_CONFIG

    if not config.alias_injection:
        return periods.copy(), powers.copy()

    periods = np.asarray(periods, dtype=float)
    powers = np.asarray(powers, dtype=float)

    if len(periods) == 0:
        return periods.copy(), powers.copy()

    min_p = config.min_period_hours
    max_p = config.max_period_hours
    tol = config.match_tol_rel

    # Base power for injected aliases
    alias_power = powers.min() * config.alias_base_power_scale

    # Collect new alias candidates
    new_periods = []
    new_powers = []

    # Collect all current periods (will grow as we add aliases)
    all_periods = list(periods)

    for p in periods:
        # Half-period alias
        p_half = p / 2.0
        if min_p <= p_half <= max_p:
            if not _has_close(np.array(all_periods), p_half, tol):
                new_periods.append(p_half)
                new_powers.append(alias_power)
                all_periods.append(p_half)

        # Double-period alias
        p_double = p * 2.0
        if min_p <= p_double <= max_p:
            if not _has_close(np.array(all_periods), p_double, tol):
                new_periods.append(p_double)
                new_powers.append(alias_power)
                all_periods.append(p_double)

    # Combine original and new candidates
    if new_periods:
        augmented_periods = np.concatenate([periods, np.array(new_periods)])
        augmented_powers = np.concatenate([powers, np.array(new_powers)])
    else:
        augmented_periods = periods.copy()
        augmented_powers = powers.copy()

    return augmented_periods, augmented_powers
