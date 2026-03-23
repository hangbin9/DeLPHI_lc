"""
Configuration dataclasses for the period prediction pipeline.

This module defines all tunable parameters for:
- Period search (frequency grid, candidate selection)
- Posterior aggregation (temperature, clustering tolerance)
- Data column mapping (DAMIT-specific defaults, but configurable)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ColumnConfig:
    """
    Configuration for CSV column names and data interpretation.

    Designed for DAMIT-style data by default, but fully configurable
    for other data formats.

    Attributes:
        time_col: Column name for time (JD or MJD). Default: "jd"
        flux_col: Column name for flux/brightness. Used if mag_col is None.
        mag_col: Column name for magnitude. If provided, used directly instead of flux.
        mag_err_col: Column name for magnitude error. If None, uses default_mag_err.
        period_col: Column name for ground truth period. Default: "rot_per"
        period_unit: Unit of period_col values: "hours" or "days". Default: "hours"
        default_mag_err: Default magnitude error when mag_err_col is None.
    """
    time_col: str = "jd"
    flux_col: str = "relative_brightness"
    mag_col: Optional[str] = None
    mag_err_col: Optional[str] = None
    period_col: str = "rot_per"
    period_unit: str = "hours"  # "hours" or "days"
    default_mag_err: float = 0.02


@dataclass
class PeriodConfig:
    """
    Configuration for period search and posterior aggregation.

    Attributes:
        min_period_hours: Minimum period to search (hours).
        max_period_hours: Maximum period to search (hours).
        n_freq: Number of frequency grid points for Lomb-Scargle.
        top_k: Number of top candidates to keep per epoch.
        min_points_per_epoch: Minimum data points required for an epoch.
        temperature: Softmax temperature for converting LS powers to probabilities.
        match_tol_rel: Relative tolerance for clustering periods (fraction).
        alias_injection: Whether to inject 0.5P and 2P alias candidates.
        alias_base_power_scale: Scale factor for injected alias powers (relative to min power).
        credible_mass: Target mass for credible interval (e.g., 0.68 for 68%).
        max_period_ratio_for_plot: Max ratio for plotting (used in diagnostics).
        column_config: Column mapping configuration.
    """
    # Search range in hours
    min_period_hours: float = 2.0
    max_period_hours: float = 200.0  # Covers 99.9% of known asteroids (median ~10h)

    # Frequency grid
    # For max_period=200h, n_freq=20000 gives:
    #   ~0.02% resolution at 10h, ~0.25% at 100h, ~0.5% at 200h
    # For extended range (max_period=2000h for slow rotators), increase n_freq
    # proportionally (e.g., n_freq=200000) to maintain resolution.
    n_freq: int = 20000

    # Candidate selection
    top_k: int = 64
    min_points_per_epoch: int = 10

    # Posterior / aggregation
    temperature: float = 10.0
    match_tol_rel: float = 0.02  # 2% relative tolerance
    alias_injection: bool = True
    alias_base_power_scale: float = 0.1

    # Uncertainty
    credible_mass: float = 0.68

    # Plotting
    max_period_ratio_for_plot: float = 4.0

    # Column configuration
    column_config: ColumnConfig = field(default_factory=ColumnConfig)


@dataclass
class PhysicalAliasConfig:
    """
    Configuration for the OPTIONAL physical alias resolver.

    This is an EXPERIMENTAL post-processing step that attempts to resolve
    factor-of-2 aliases using simple ellipsoid-like template fits.

    DISABLED BY DEFAULT. Set enabled=True to activate.

    Attributes:
        enabled: Master switch. Default False (resolver completely skipped).
        min_period_hours: Minimum period for template fitting.
        max_period_hours: Maximum period for template fitting.

        # Ambiguity gate: when to consider calling the resolver
        max_power_ratio_for_ambiguity: If top LS power / alias power < this, consider ambiguous.
        max_score_margin_for_ambiguity: Normalized posterior score margin threshold.

        # Data quality gate
        min_points_per_epoch: Minimum points per epoch for template fitting.
        min_epochs_for_fit: Minimum epochs required to attempt fitting.

        # Decision gate: when to accept resolver's answer
        min_chi2_rel_improvement: Minimum relative chi2 improvement to switch (e.g., 0.15 = 15%).
        min_chi2_abs_improvement: Minimum absolute chi2 improvement.
        max_allowed_period_factor: Only consider periods within this factor (2.1 for {P, 0.5P, 2P}).

        # Template parameters
        n_harmonics: Number of Fourier harmonics (2 = ellipsoid-like double-peaked).
        regularization: L2 regularization for template coefficients.
    """
    enabled: bool = False  # MASTER SWITCH - disabled by default

    # Period range for fitting
    min_period_hours: float = 2.0
    max_period_hours: float = 50.0

    # Ambiguity gate
    max_power_ratio_for_ambiguity: float = 1.5
    max_score_margin_for_ambiguity: float = 0.20

    # Data quality gate
    min_points_per_epoch: int = 15
    min_epochs_for_fit: int = 2

    # Decision gate
    min_chi2_rel_improvement: float = 0.15
    min_chi2_abs_improvement: float = 1.0
    max_allowed_period_factor: float = 2.1

    # Template parameters
    n_harmonics: int = 2  # 2 harmonics = double-peaked ellipsoid-like
    regularization: float = 0.01


# Default configuration singletons
DEFAULT_PERIOD_CONFIG = PeriodConfig()
DEFAULT_COLUMN_CONFIG = ColumnConfig()
DEFAULT_PHYSICAL_ALIAS_CONFIG = PhysicalAliasConfig()
