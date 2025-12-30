"""
Multi-epoch consensus period estimation.

This module provides the ConsensusEngine class that orchestrates
the full period estimation pipeline:
1. Per-epoch Lomb-Scargle search
2. Multi-epoch posterior aggregation
3. Batch prediction for multiple asteroids
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import (
    PeriodConfig,
    PhysicalAliasConfig,
    DEFAULT_PERIOD_CONFIG,
    DEFAULT_PHYSICAL_ALIAS_CONFIG,
)
from .data import LightcurveEpoch, AsteroidLightcurves
from .period_search import lomb_scargle_period_search
from .posterior import aggregate_multi_epoch_posterior, posterior_summary

logger = logging.getLogger(__name__)


class ConsensusEngine:
    """
    Multi-epoch consensus period estimation engine.

    Combines Lomb-Scargle period search across multiple epochs
    using product-of-experts aggregation.

    Attributes:
        config: PeriodConfig with search and aggregation parameters.
    """

    def __init__(
        self,
        config: Optional[PeriodConfig] = None,
        alias_config: Optional[PhysicalAliasConfig] = None
    ):
        """
        Initialize the consensus engine.

        Args:
            config: Configuration for period search and aggregation.
                   Uses DEFAULT_PERIOD_CONFIG if not provided.
            alias_config: OPTIONAL configuration for physical alias resolver.
                         Disabled by default (alias_config.enabled=False).
        """
        self.config = config or DEFAULT_PERIOD_CONFIG
        self.alias_config = alias_config or DEFAULT_PHYSICAL_ALIAS_CONFIG

    def predict_single_epoch(
        self,
        lc: LightcurveEpoch
    ) -> Dict[str, Any]:
        """
        Run Lomb-Scargle search on a single epoch.

        Args:
            lc: LightcurveEpoch with time, mag, mag_err arrays.

        Returns:
            Dictionary with:
                - object_id: str
                - epoch_id: str
                - periods: np.ndarray of candidate periods (hours)
                - powers: np.ndarray of LS powers
                - n_points: int number of data points
                - success: bool whether search succeeded
                - error: str error message if failed
        """
        result: Dict[str, Any] = {
            "object_id": lc.object_id,
            "epoch_id": lc.epoch_id,
            "periods": np.array([]),
            "powers": np.array([]),
            "n_points": lc.n_points,
            "success": False,
            "error": None,
        }

        # Check minimum points
        if lc.n_points < self.config.min_points_per_epoch:
            result["error"] = (
                f"Insufficient points: {lc.n_points} < "
                f"{self.config.min_points_per_epoch}"
            )
            return result

        try:
            periods, powers = lomb_scargle_period_search(
                time=lc.time,
                mag=lc.mag,
                mag_err=lc.mag_err,
                config=self.config
            )
            result["periods"] = periods
            result["powers"] = powers
            result["success"] = True
        except Exception as e:
            result["error"] = str(e)
            logger.warning(
                f"LS search failed for {lc.object_id}/{lc.epoch_id}: {e}"
            )

        return result

    def predict_multi_epoch(
        self,
        asteroid: AsteroidLightcurves
    ) -> Dict[str, Any]:
        """
        Perform multi-epoch consensus period estimation for one asteroid.

        Combines LS results from all valid epochs using product-of-experts
        aggregation to produce a single posterior distribution.

        Args:
            asteroid: AsteroidLightcurves with multiple epochs.

        Returns:
            Dictionary with:
                - object_id: str
                - period: float MAP period (hours)
                - periods: np.ndarray posterior support
                - probs: np.ndarray posterior probabilities
                - ci_low: float lower CI bound (hours)
                - ci_high: float upper CI bound (hours)
                - sigma_eff: float effective uncertainty (hours)
                - n_epochs_used: int number of epochs used
                - n_epochs_total: int total epochs available
                - success: bool whether estimation succeeded
                - error: str error message if failed
        """
        result: Dict[str, Any] = {
            "object_id": asteroid.object_id,
            "period": np.nan,
            "periods": np.array([]),
            "probs": np.array([]),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "sigma_eff": np.nan,
            "n_epochs_used": 0,
            "n_epochs_total": asteroid.n_epochs,
            "success": False,
            "error": None,
        }

        # Run LS on each epoch
        epoch_periods: List[np.ndarray] = []
        epoch_powers: List[np.ndarray] = []

        for epoch in asteroid.epochs:
            epoch_result = self.predict_single_epoch(epoch)
            if epoch_result["success"] and len(epoch_result["periods"]) > 0:
                epoch_periods.append(epoch_result["periods"])
                epoch_powers.append(epoch_result["powers"])

        n_epochs_used = len(epoch_periods)
        result["n_epochs_used"] = n_epochs_used

        if n_epochs_used == 0:
            result["error"] = "No valid epochs for period estimation"
            return result

        # Aggregate across epochs
        try:
            periods, probs = aggregate_multi_epoch_posterior(
                epoch_periods=epoch_periods,
                epoch_scores=epoch_powers,
                config=self.config
            )

            if len(periods) == 0:
                result["error"] = "Aggregation produced empty posterior"
                return result

            result["periods"] = periods
            result["probs"] = probs

            # Get summary statistics
            summary = posterior_summary(periods, probs, self.config)
            result["period"] = summary["map_period"]
            result["ci_low"] = summary["ci_low"]
            result["ci_high"] = summary["ci_high"]
            result["sigma_eff"] = summary["sigma_eff"]
            result["success"] = True

        except Exception as e:
            result["error"] = str(e)
            logger.warning(
                f"Posterior aggregation failed for {asteroid.object_id}: {e}"
            )

        # === OPTIONAL: Physical alias resolver hook ===
        # Only runs if alias_config.enabled=True (disabled by default)
        result = self._maybe_resolve_alias(asteroid, result)

        return result

    def _maybe_resolve_alias(
        self,
        asteroid: AsteroidLightcurves,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        OPTIONAL hook for physical alias resolution.

        Only called if alias_config.enabled=True. Otherwise returns
        result unchanged.

        Args:
            asteroid: Multi-epoch lightcurve data.
            result: Current result dict from predict_multi_epoch.

        Returns:
            Updated result dict with optional alias resolution fields.
        """
        # Initialize resolver fields (always present for consistent schema)
        result["period_pre_resolver"] = result.get("period", np.nan)
        result["resolver_applied"] = False
        result["resolver_reason"] = "Resolver disabled"

        # Early exit if resolver disabled
        if not self.alias_config.enabled:
            return result

        # Early exit if consensus failed
        if not result.get("success", False):
            result["resolver_reason"] = "Consensus failed, resolver skipped"
            return result

        # Import resolver lazily to avoid circular imports and keep it optional
        try:
            from .alias_ellipsoid_resolver import resolve_alias
        except ImportError as e:
            logger.warning(f"Could not import alias resolver: {e}")
            result["resolver_reason"] = f"Import error: {e}"
            return result

        # Run resolver
        try:
            resolver_result = resolve_alias(
                asteroid=asteroid,
                consensus_result=result,
                config=self.alias_config
            )

            # Update result with resolver output
            result["resolver_applied"] = resolver_result.was_resolved
            result["resolver_reason"] = resolver_result.reason
            result["resolver_was_ambiguous"] = resolver_result.was_ambiguous

            if resolver_result.was_resolved:
                result["period"] = resolver_result.resolved_period
                result["period_resolved"] = resolver_result.resolved_period
                result["resolver_chi2_original"] = resolver_result.chi2_original
                result["resolver_chi2_resolved"] = resolver_result.chi2_resolved

        except Exception as e:
            logger.warning(f"Alias resolver failed for {asteroid.object_id}: {e}")
            result["resolver_reason"] = f"Resolver error: {e}"

        return result

    def predict_many(
        self,
        objects: Dict[str, AsteroidLightcurves],
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Run predictions for multiple asteroids.

        Args:
            objects: Dictionary mapping object_id to AsteroidLightcurves.
            show_progress: Whether to display a progress bar.

        Returns:
            DataFrame with columns:
                - object_id
                - period_hours
                - ci_low_hours
                - ci_high_hours
                - sigma_eff_hours
                - n_epochs_used
                - n_epochs_total
                - success
        """
        records: List[Dict[str, Any]] = []

        iterator = objects.items()
        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(objects),
                desc="Period estimation"
            )

        for object_id, asteroid in iterator:
            result = self.predict_multi_epoch(asteroid)

            record = {
                "object_id": result["object_id"],
                "period_hours": result["period"],
                "ci_low_hours": result["ci_low"],
                "ci_high_hours": result["ci_high"],
                "sigma_eff_hours": result["sigma_eff"],
                "n_epochs_used": result["n_epochs_used"],
                "n_epochs_total": result["n_epochs_total"],
                "success": result["success"],
            }

            # Add resolver columns if present (always present when hook runs)
            if "period_pre_resolver" in result:
                record["period_pre_resolver_hours"] = result["period_pre_resolver"]
                record["resolver_applied"] = result.get("resolver_applied", False)

            records.append(record)

        df = pd.DataFrame(records)
        return df


def run_consensus_pipeline(
    objects: Dict[str, AsteroidLightcurves],
    config: Optional[PeriodConfig] = None,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Convenience function to run the full consensus pipeline.

    Args:
        objects: Dictionary mapping object_id to AsteroidLightcurves.
        config: Period search configuration.
        show_progress: Whether to display progress bar.

    Returns:
        DataFrame with prediction results.
    """
    engine = ConsensusEngine(config)
    return engine.predict_many(objects, show_progress)
