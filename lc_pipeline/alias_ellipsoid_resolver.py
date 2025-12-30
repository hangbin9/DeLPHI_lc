"""
EXPERIMENTAL: Physical alias resolver using ellipsoid-like templates.

This module provides an OPTIONAL post-processing step that attempts to
resolve factor-of-2 period aliases by fitting simple physically-constrained
templates (double-peaked, ellipsoid-like lightcurves) to multi-epoch data.

DISABLED BY DEFAULT. Only activated when PhysicalAliasConfig.enabled=True.

The approach:
1. Given a consensus period P, consider the alias family {P, 0.5P, 2P}
2. For each candidate, fit a low-order Fourier template across all epochs
3. Compare chi-squared fits and select the best if improvement is significant

This is a "last-mile" disambiguation step, not a replacement for LS+consensus.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from .config import PhysicalAliasConfig, DEFAULT_PHYSICAL_ALIAS_CONFIG
from .data import LightcurveEpoch, AsteroidLightcurves

logger = logging.getLogger(__name__)


@dataclass
class AliasResolverResult:
    """
    Result from the physical alias resolver.

    Attributes:
        original_period: The consensus period before resolution (hours).
        resolved_period: The period after resolution (hours). Same as original if no change.
        was_resolved: Whether the resolver changed the period.
        was_ambiguous: Whether the case was considered ambiguous (triggered fitting).
        reason: Human-readable reason for the decision.
        chi2_original: Chi-squared for original period fit.
        chi2_resolved: Chi-squared for resolved period fit (if changed).
        alias_family: The {P, 0.5P, 2P} candidates considered.
        chi2_all: Chi-squared values for all candidates in alias_family.
    """
    original_period: float
    resolved_period: float
    was_resolved: bool
    was_ambiguous: bool
    reason: str
    chi2_original: Optional[float] = None
    chi2_resolved: Optional[float] = None
    alias_family: Optional[List[float]] = None
    chi2_all: Optional[Dict[float, float]] = None


def _build_fourier_design_matrix(
    phases: np.ndarray,
    n_harmonics: int
) -> np.ndarray:
    """
    Build design matrix for Fourier template fitting.

    For n_harmonics=2:
        columns = [1, cos(2π*phase), sin(2π*phase), cos(4π*phase), sin(4π*phase)]

    This represents a double-peaked lightcurve (ellipsoid-like), which is
    physically expected for triaxial asteroid rotation.

    Args:
        phases: Array of rotational phases in [0, 1).
        n_harmonics: Number of Fourier harmonics.

    Returns:
        Design matrix of shape (n_points, 1 + 2*n_harmonics).
    """
    n = len(phases)
    n_cols = 1 + 2 * n_harmonics
    X = np.zeros((n, n_cols))

    # DC offset (mean magnitude)
    X[:, 0] = 1.0

    for k in range(1, n_harmonics + 1):
        # cos(2π * k * phase) and sin(2π * k * phase)
        angle = 2 * np.pi * k * phases
        X[:, 2 * k - 1] = np.cos(angle)
        X[:, 2 * k] = np.sin(angle)

    return X


def _fit_fourier_template(
    time: np.ndarray,
    mag: np.ndarray,
    mag_err: np.ndarray,
    period_hours: float,
    n_harmonics: int,
    regularization: float
) -> Tuple[float, np.ndarray]:
    """
    Fit a Fourier template to lightcurve data at a given period.

    Uses weighted least squares with optional L2 regularization.

    Args:
        time: Observation times (days).
        mag: Magnitudes.
        mag_err: Magnitude errors.
        period_hours: Trial period in hours.
        n_harmonics: Number of Fourier harmonics.
        regularization: L2 regularization strength.

    Returns:
        chi2: Reduced chi-squared of the fit.
        coeffs: Fitted Fourier coefficients.
    """
    period_days = period_hours / 24.0

    # Compute phases
    phases = (time / period_days) % 1.0

    # Build design matrix
    X = _build_fourier_design_matrix(phases, n_harmonics)

    # Weights from errors
    weights = 1.0 / (mag_err ** 2 + 1e-10)
    W = np.diag(weights)

    # Weighted least squares with regularization: (X'WX + λI)β = X'Wy
    XtW = X.T @ W
    XtWX = XtW @ X
    XtWy = XtW @ mag

    # Add regularization (skip DC term)
    reg_matrix = regularization * np.eye(X.shape[1])
    reg_matrix[0, 0] = 0  # Don't regularize DC offset

    try:
        coeffs = np.linalg.solve(XtWX + reg_matrix, XtWy)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse
        coeffs = np.linalg.lstsq(XtWX + reg_matrix, XtWy, rcond=None)[0]

    # Compute residuals and chi-squared
    residuals = mag - X @ coeffs
    chi2 = np.sum(weights * residuals ** 2)

    # Degrees of freedom
    dof = max(1, len(mag) - len(coeffs))
    chi2_reduced = chi2 / dof

    return chi2_reduced, coeffs


def _fit_multi_epoch(
    epochs: List[LightcurveEpoch],
    period_hours: float,
    config: PhysicalAliasConfig
) -> Tuple[float, List[np.ndarray]]:
    """
    Fit a single period across multiple epochs.

    Each epoch gets its own DC offset (mean magnitude can vary),
    but shares the same periodic template shape.

    Args:
        epochs: List of lightcurve epochs.
        period_hours: Trial period in hours.
        config: Physical alias configuration.

    Returns:
        total_chi2: Sum of chi-squared across all epochs.
        all_coeffs: List of coefficient arrays per epoch.
    """
    total_chi2 = 0.0
    all_coeffs = []

    for epoch in epochs:
        if epoch.n_points < config.min_points_per_epoch:
            continue

        chi2, coeffs = _fit_fourier_template(
            time=epoch.time,
            mag=epoch.mag,
            mag_err=epoch.mag_err,
            period_hours=period_hours,
            n_harmonics=config.n_harmonics,
            regularization=config.regularization
        )

        total_chi2 += chi2
        all_coeffs.append(coeffs)

    return total_chi2, all_coeffs


def _build_alias_family(
    period_hours: float,
    config: PhysicalAliasConfig
) -> List[float]:
    """
    Build the alias family {P, 0.5P, 2P} within allowed range.

    Args:
        period_hours: Central period.
        config: Configuration with period bounds.

    Returns:
        List of candidate periods within [min_period, max_period].
    """
    candidates = [period_hours]

    half_period = period_hours / 2.0
    if config.min_period_hours <= half_period <= config.max_period_hours:
        candidates.append(half_period)

    double_period = period_hours * 2.0
    if config.min_period_hours <= double_period <= config.max_period_hours:
        candidates.append(double_period)

    return sorted(set(candidates))


def check_ambiguity(
    posterior_result: Dict[str, Any],
    config: PhysicalAliasConfig
) -> bool:
    """
    Check if the consensus result is ambiguous enough to warrant resolution.

    Args:
        posterior_result: Result dict from ConsensusEngine.predict_multi_epoch.
        config: Physical alias configuration.

    Returns:
        True if the case is considered ambiguous.
    """
    # Get posterior probabilities if available
    probs = posterior_result.get("probs", np.array([]))
    periods = posterior_result.get("periods", np.array([]))

    if len(probs) < 2 or len(periods) < 2:
        return False

    # Sort by probability
    sorted_idx = np.argsort(probs)[::-1]
    top_prob = probs[sorted_idx[0]]
    second_prob = probs[sorted_idx[1]]

    # Check score margin
    if top_prob > 0:
        margin = (top_prob - second_prob) / top_prob
        if margin > config.max_score_margin_for_ambiguity:
            return False  # Clear winner, not ambiguous

    # Check if second-best is an alias of the best
    top_period = periods[sorted_idx[0]]
    second_period = periods[sorted_idx[1]]

    ratio = max(top_period, second_period) / min(top_period, second_period)
    is_alias = abs(ratio - 2.0) < 0.1 or abs(ratio - 1.0) < 0.1

    if not is_alias:
        return False  # Second-best isn't an alias, not our problem

    # Power ratio check (if available)
    # For now, if we got here, consider it ambiguous
    return True


def resolve_alias(
    asteroid: AsteroidLightcurves,
    consensus_result: Dict[str, Any],
    config: Optional[PhysicalAliasConfig] = None
) -> AliasResolverResult:
    """
    Attempt to resolve period aliases using physical template fitting.

    This is the main entry point for the alias resolver.

    Args:
        asteroid: Multi-epoch lightcurve data.
        consensus_result: Result dict from ConsensusEngine.predict_multi_epoch.
        config: Physical alias configuration. Uses default if None.

    Returns:
        AliasResolverResult with resolution decision and diagnostics.
    """
    if config is None:
        config = DEFAULT_PHYSICAL_ALIAS_CONFIG

    original_period = consensus_result.get("period", np.nan)

    # Default result: no change
    result = AliasResolverResult(
        original_period=original_period,
        resolved_period=original_period,
        was_resolved=False,
        was_ambiguous=False,
        reason="Resolver disabled or not triggered"
    )

    # Master switch check
    if not config.enabled:
        result.reason = "Resolver disabled (config.enabled=False)"
        return result

    # Validate input
    if not np.isfinite(original_period):
        result.reason = "Invalid original period (NaN)"
        return result

    # Period range check
    if not (config.min_period_hours <= original_period <= config.max_period_hours):
        result.reason = f"Period {original_period:.2f}h outside fitting range"
        return result

    # Data quality gate
    valid_epochs = [e for e in asteroid.epochs if e.n_points >= config.min_points_per_epoch]
    if len(valid_epochs) < config.min_epochs_for_fit:
        result.reason = f"Insufficient epochs ({len(valid_epochs)} < {config.min_epochs_for_fit})"
        return result

    # Ambiguity gate
    is_ambiguous = check_ambiguity(consensus_result, config)
    result.was_ambiguous = is_ambiguous

    if not is_ambiguous:
        result.reason = "Not ambiguous (clear consensus winner)"
        return result

    # Build alias family
    alias_family = _build_alias_family(original_period, config)
    result.alias_family = alias_family

    if len(alias_family) < 2:
        result.reason = "No valid alias candidates in range"
        return result

    # Fit each candidate
    chi2_all: Dict[float, float] = {}
    for candidate_period in alias_family:
        try:
            chi2, _ = _fit_multi_epoch(valid_epochs, candidate_period, config)
            chi2_all[candidate_period] = chi2
        except Exception as e:
            logger.warning(f"Fitting failed for period {candidate_period:.2f}h: {e}")
            chi2_all[candidate_period] = np.inf

    result.chi2_all = chi2_all

    if not chi2_all or all(np.isinf(v) for v in chi2_all.values()):
        result.reason = "All fits failed"
        return result

    # Find best candidate
    best_period = min(chi2_all, key=chi2_all.get)
    best_chi2 = chi2_all[best_period]
    original_chi2 = chi2_all.get(original_period, np.inf)

    result.chi2_original = original_chi2
    result.chi2_resolved = best_chi2

    # Decision gate: is the improvement significant?
    if np.isfinite(original_chi2) and original_chi2 > 0:
        rel_improvement = (original_chi2 - best_chi2) / original_chi2
        abs_improvement = original_chi2 - best_chi2
    else:
        rel_improvement = 0.0
        abs_improvement = 0.0

    # Check if best is different from original
    period_ratio = max(best_period, original_period) / min(best_period, original_period)
    is_different = period_ratio > 1.05  # More than 5% different

    if not is_different:
        result.reason = "Best fit matches original period"
        return result

    # Check improvement thresholds
    if rel_improvement < config.min_chi2_rel_improvement:
        result.reason = (
            f"Insufficient relative improvement "
            f"({rel_improvement*100:.1f}% < {config.min_chi2_rel_improvement*100:.1f}%)"
        )
        return result

    if abs_improvement < config.min_chi2_abs_improvement:
        result.reason = (
            f"Insufficient absolute improvement "
            f"({abs_improvement:.2f} < {config.min_chi2_abs_improvement:.2f})"
        )
        return result

    # Check that the change is within allowed factor
    if period_ratio > config.max_allowed_period_factor:
        result.reason = f"Period change too large (factor {period_ratio:.2f})"
        return result

    # Accept the resolution
    result.resolved_period = best_period
    result.was_resolved = True
    result.reason = (
        f"Resolved: {original_period:.3f}h -> {best_period:.3f}h "
        f"(chi2 {original_chi2:.2f} -> {best_chi2:.2f}, "
        f"{rel_improvement*100:.1f}% improvement)"
    )

    logger.info(f"[{asteroid.object_id}] {result.reason}")

    return result
