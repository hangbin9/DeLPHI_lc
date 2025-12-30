"""
Posterior computation and aggregation for period estimation.

This module provides:
- scores_to_probs: softmax conversion of LS powers to probabilities
- cluster_periods: merge similar periods across epochs
- aggregate_multi_epoch_posterior: product-of-experts combination
- compute_credible_interval: smallest interval containing target mass
- posterior_summary: convenience function for MAP, CI, sigma_eff

All functions return purely descriptive statistics without any
classification or quality flags.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import PeriodConfig, DEFAULT_PERIOD_CONFIG
from .period_search import inject_alias_candidates


def scores_to_probs(
    scores: np.ndarray,
    temperature: float
) -> np.ndarray:
    """
    Convert raw LS powers to probabilities via temperature-scaled softmax.

    Args:
        scores: Array of LS power values.
        temperature: Softmax temperature (higher = more uniform).

    Returns:
        Normalized probability distribution over candidates.
    """
    s = np.asarray(scores, dtype=float)

    if len(s) == 0:
        return np.array([])

    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    # Numerical stability: subtract max
    s = s - np.max(s)

    # Apply softmax with temperature
    exp_s = np.exp(s / temperature)
    total = exp_s.sum()

    if total == 0 or not np.isfinite(total):
        # Fallback to uniform if numerical issues
        return np.ones(len(s)) / len(s)

    probs = exp_s / total
    return probs


def cluster_periods(
    all_periods: np.ndarray,
    match_tol_rel: float
) -> np.ndarray:
    """
    Cluster periods within relative tolerance and return representatives.

    Groups periods that are within match_tol_rel of each other,
    using the median of each cluster as the representative.

    Args:
        all_periods: Array of candidate periods (any unit).
        match_tol_rel: Relative tolerance for grouping (e.g., 0.02 for 2%).

    Returns:
        Array of unique representative periods (cluster medians).
    """
    if len(all_periods) == 0:
        return np.array([])

    periods = np.sort(np.asarray(all_periods, dtype=float))

    clusters: List[List[float]] = []
    current_cluster: List[float] = [periods[0]]

    for p in periods[1:]:
        # Compare to cluster center (using first element or median)
        cluster_center = np.median(current_cluster)
        rel_diff = abs(p - cluster_center) / cluster_center

        if rel_diff < match_tol_rel:
            # Same cluster
            current_cluster.append(p)
        else:
            # New cluster
            clusters.append(current_cluster)
            current_cluster = [p]

    # Don't forget the last cluster
    clusters.append(current_cluster)

    # Return median of each cluster
    representatives = np.array([np.median(c) for c in clusters])
    return representatives


def aggregate_multi_epoch_posterior(
    epoch_periods: List[np.ndarray],
    epoch_scores: List[np.ndarray],
    config: Optional[PeriodConfig] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate per-epoch LS results into a single joint posterior.

    Implements product-of-experts aggregation:
    1. Optionally inject alias candidates per epoch
    2. Convert scores to per-epoch probabilities with softmax
    3. Cluster all periods to unique centers
    4. For each unique period, multiply probabilities from all epochs
    5. Renormalize to obtain final posterior

    Args:
        epoch_periods: List of period arrays, one per epoch.
        epoch_scores: List of power arrays, one per epoch.
        config: Configuration for aggregation parameters.

    Returns:
        periods: Array of unique period candidates.
        probs: Normalized posterior probabilities.
    """
    if config is None:
        config = DEFAULT_PERIOD_CONFIG

    if len(epoch_periods) == 0:
        return np.array([]), np.array([])

    # Process each epoch: inject aliases, convert to probabilities
    epoch_data: List[Tuple[np.ndarray, np.ndarray]] = []

    for periods, scores in zip(epoch_periods, epoch_scores):
        if len(periods) == 0:
            continue

        # Optionally inject alias candidates
        if config.alias_injection:
            aug_periods, aug_scores = inject_alias_candidates(
                periods, scores, config
            )
        else:
            aug_periods, aug_scores = periods.copy(), scores.copy()

        # Convert to probabilities
        probs = scores_to_probs(aug_scores, config.temperature)
        epoch_data.append((aug_periods, probs))

    if len(epoch_data) == 0:
        return np.array([]), np.array([])

    # Collect all periods
    all_periods = np.concatenate([p for p, _ in epoch_data])

    # Cluster to unique representatives
    unique_periods = cluster_periods(all_periods, config.match_tol_rel)

    if len(unique_periods) == 0:
        return np.array([]), np.array([])

    # Product of experts: multiply probabilities across epochs
    # Small epsilon for periods not found in an epoch
    epsilon = 1e-12

    joint_probs = np.ones(len(unique_periods))

    for p_epoch, prob_epoch in epoch_data:
        for i, P_star in enumerate(unique_periods):
            # Find matching periods within tolerance
            rel_diffs = np.abs(p_epoch - P_star) / P_star
            mask = rel_diffs < config.match_tol_rel

            if np.any(mask):
                # Use maximum probability among matches
                joint_probs[i] *= prob_epoch[mask].max()
            else:
                # No match: multiply by small epsilon
                joint_probs[i] *= epsilon

    # Renormalize
    total = joint_probs.sum()
    if total > 0 and np.isfinite(total):
        joint_probs = joint_probs / total
    else:
        # Fallback to uniform
        joint_probs = np.ones(len(unique_periods)) / len(unique_periods)

    return unique_periods, joint_probs


def compute_credible_interval(
    periods: np.ndarray,
    probs: np.ndarray,
    mass: float
) -> Tuple[float, float]:
    """
    Compute the smallest period interval containing at least `mass` probability.

    Uses a sliding window approach to find the tightest credible interval.

    Args:
        periods: Array of candidate periods (sorted or unsorted).
        probs: Corresponding normalized probabilities.
        mass: Target credible mass (e.g., 0.68 for 68%).

    Returns:
        (P_low, P_high): Bounds of the credible interval.
        Returns (NaN, NaN) if computation fails.
    """
    if len(periods) == 0 or len(probs) == 0:
        return (np.nan, np.nan)

    if mass <= 0 or mass > 1:
        raise ValueError(f"mass must be in (0, 1], got {mass}")

    periods = np.asarray(periods, dtype=float)
    probs = np.asarray(probs, dtype=float)

    # Sort by period
    sort_idx = np.argsort(periods)
    sorted_periods = periods[sort_idx]
    sorted_probs = probs[sort_idx]

    # Cumulative probability
    cumsum = np.cumsum(sorted_probs)

    n = len(sorted_periods)
    best_width = np.inf
    best_interval = (sorted_periods[0], sorted_periods[-1])

    # Sliding window to find smallest interval with >= mass
    for i in range(n):
        # Find smallest j such that cumsum[j] - cumsum[i-1] >= mass
        if i == 0:
            cum_before_i = 0.0
        else:
            cum_before_i = cumsum[i - 1]

        target = cum_before_i + mass

        # Binary search for j
        j = np.searchsorted(cumsum, target, side='left')
        if j >= n:
            j = n - 1

        # Verify we have enough mass
        interval_mass = cumsum[j] - cum_before_i
        if interval_mass >= mass - 1e-10:  # Small tolerance for floating point
            width = sorted_periods[j] - sorted_periods[i]
            if width < best_width:
                best_width = width
                best_interval = (sorted_periods[i], sorted_periods[j])

    return best_interval


def posterior_summary(
    periods: np.ndarray,
    probs: np.ndarray,
    config: Optional[PeriodConfig] = None
) -> Dict[str, Any]:
    """
    Compute summary statistics of the discrete posterior.

    Returns purely descriptive statistics without any classification.

    Args:
        periods: Array of candidate periods.
        probs: Normalized posterior probabilities.
        config: Configuration for credible mass.

    Returns:
        Dictionary with:
            - map_period: Maximum a posteriori period
            - map_prob: Probability of MAP period
            - ci_low: Lower bound of credible interval
            - ci_high: Upper bound of credible interval
            - sigma_eff: Effective uncertainty (half CI width)
            - entropy: Shannon entropy of posterior (bits)
    """
    if config is None:
        config = DEFAULT_PERIOD_CONFIG

    periods = np.asarray(periods, dtype=float)
    probs = np.asarray(probs, dtype=float)

    result: Dict[str, Any] = {
        "map_period": np.nan,
        "map_prob": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "sigma_eff": np.nan,
        "entropy": np.nan,
    }

    if len(periods) == 0 or len(probs) == 0:
        return result

    # MAP estimate
    map_idx = np.argmax(probs)
    result["map_period"] = periods[map_idx]
    result["map_prob"] = probs[map_idx]

    # Credible interval
    ci_low, ci_high = compute_credible_interval(
        periods, probs, config.credible_mass
    )
    result["ci_low"] = ci_low
    result["ci_high"] = ci_high

    # Effective sigma (half CI width)
    if np.isfinite(ci_low) and np.isfinite(ci_high):
        result["sigma_eff"] = 0.5 * (ci_high - ci_low)

    # Shannon entropy (in bits)
    valid_probs = probs[probs > 0]
    if len(valid_probs) > 0:
        entropy = -np.sum(valid_probs * np.log2(valid_probs))
        result["entropy"] = entropy

    return result
