"""End-to-end asteroid lightcurve analysis pipeline."""
import logging
from typing import List, Optional, Union
import numpy as np

from ..period.consensus import ConsensusEngine
from ..data.loaders import LightcurveEpoch, AsteroidLightcurves
from ..schema import LightcurveData
from .schema import AnalysisResult, PeriodResult
from .forking import PeriodForker
from .pole import PoleConfig
from .uncertainty import compute_uncertainty

logger = logging.getLogger(__name__)


class LightcurvePipeline:
    """
    End-to-end asteroid lightcurve analysis.

    Example:
        ```python
        pipeline = LightcurvePipeline()
        result = pipeline.analyze(epochs, "asteroid_1017")
        print(f"Period: {result.period.period_hours:.2f} h")
        print(f"Best pole: λ={result.best_pole.lambda_deg:.1f}°, β={result.best_pole.beta_deg:.1f}°")
        ```
    """

    def __init__(
        self,
        period_config=None,
        pole_config: Optional[PoleConfig] = None
    ):
        """
        Initialize pipeline.

        Args:
            period_config: Configuration for period prediction (optional)
            pole_config: Configuration for pole prediction (optional)
        """
        self._period_engine = None
        self._forker = None
        self._period_config = period_config
        self._pole_config = pole_config

    @property
    def period_engine(self) -> ConsensusEngine:
        """Lazy-load period engine."""
        if self._period_engine is None:
            self._period_engine = ConsensusEngine(self._period_config)
        return self._period_engine

    @property
    def forker(self) -> PeriodForker:
        """Lazy-load period forker."""
        if self._forker is None:
            self._forker = PeriodForker(self._pole_config)
        return self._forker

    def analyze(
        self,
        epochs: List[np.ndarray],
        object_id: str,
        period_hours: Optional[float] = None,
        fold: int = 0,
        ensemble: bool = False
    ) -> AnalysisResult:
        """
        Analyze asteroid lightcurve data.

        Args:
            epochs: List of (N, 8) DAMIT-format arrays
            object_id: Asteroid identifier
            period_hours: Known period (skips estimation if provided)
            fold: Model fold to use (0-4 for CV177 5-fold CV). Ignored if ensemble=True.
            ensemble: If True, average predictions across all 5 folds (better but 5x slower)

        Returns:
            Complete analysis result with period, poles, and uncertainty
        """
        # Part 1: Period estimation (or use provided period)
        if period_hours is not None:
            period = PeriodResult(
                period_hours=period_hours,
                uncertainty_hours=period_hours * 0.02,
                ci_low_hours=period_hours * 0.95,
                ci_high_hours=period_hours * 1.05,
                n_epochs=len(epochs),
                success=True
            )
        else:
            period = self._estimate_period(epochs, object_id)

        # Part 2: Pole prediction with alias forking
        poles = self.forker.predict_with_aliases(epochs, period.period_hours, fold, ensemble=ensemble)

        # Compute uncertainty
        uncertainty = compute_uncertainty(poles)

        return AnalysisResult(
            object_id=object_id,
            period=period,
            poles=poles,
            best_pole=poles[0],
            uncertainty=uncertainty
        )

    def _estimate_period(self, epochs: List[np.ndarray], object_id: str) -> PeriodResult:
        """
        Convert epochs and run period estimation.

        Args:
            epochs: List of (N, 8) DAMIT-format arrays
            object_id: Asteroid identifier

        Returns:
            PeriodResult with estimated period and uncertainty
        """
        # CRITICAL: Sort all data by time first, then merge and re-split
        # The dataset does NOT preserve the epoch boundaries from user input!
        all_times = []
        all_brightness = []

        for data in epochs:
            all_times.extend(data[:, 0])
            all_brightness.extend(data[:, 1])

        all_times = np.array(all_times)
        all_brightness = np.array(all_brightness)

        # Sort by time
        sort_idx = np.argsort(all_times)
        all_times = all_times[sort_idx]
        all_brightness = all_brightness[sort_idx]

        # Re-split by 10-day gaps for period estimation (longer epochs = better)
        time_gaps = np.diff(all_times)
        gap_indices = np.where(time_gaps > 10.0)[0] + 1
        split_indices = [0] + list(gap_indices) + [len(all_times)]

        lc_epochs = []
        for i in range(len(split_indices) - 1):
            start, end = split_indices[i], split_indices[i+1]
            if end - start >= 10:  # Minimum 10 points
                time = all_times[start:end]
                mag = all_brightness[start:end]  # Use brightness directly (not log10)

                lc_epochs.append(LightcurveEpoch(
                    object_id=object_id,
                    epoch_id=f"epoch_{i}",
                    time=time,
                    mag=mag,
                    mag_err=np.full_like(mag, 0.02)
                ))

        asteroid = AsteroidLightcurves(object_id=object_id, epochs=lc_epochs)
        result = self.period_engine.predict_multi_epoch(asteroid)

        return PeriodResult(
            period_hours=result["period"],
            uncertainty_hours=result["sigma_eff"],
            ci_low_hours=result["ci_low"],
            ci_high_hours=result["ci_high"],
            n_epochs=result["n_epochs_used"],
            success=result["success"]
        )


def analyze(
    epochs: Union[List[np.ndarray], LightcurveData],
    object_id: Optional[str] = None,
    period_hours: Optional[float] = None,
    fold: int = 0,
    ensemble: bool = False
) -> AnalysisResult:
    """
    One-shot analysis function.

    Args:
        epochs: Either:
            - List of (N, 8) DAMIT-format arrays (legacy format)
            - LightcurveData object from unified schema
        object_id: Asteroid identifier (required if epochs is list, optional if LightcurveData)
        period_hours: Known period (skips estimation if provided).
            If None and epochs is LightcurveData with period_hours field, uses that.
        fold: Model fold to use (0-4 for CV177 5-fold CV). Ignored if ensemble=True.
        ensemble: If True, average predictions across all 5 folds (better but 5x slower)

    Returns:
        Complete analysis result
    """
    # Handle LightcurveData input
    if isinstance(epochs, LightcurveData):
        lc_data = epochs
        object_id = object_id or lc_data.object_id

        # Use period from LightcurveData if not explicitly provided
        if period_hours is None and lc_data.period_hours is not None:
            period_hours = lc_data.period_hours
            logger.info(f"Using period from LightcurveData: {period_hours:.3f} h")

        # Convert LightcurveData to epoch arrays
        epochs_arrays = _convert_lightcurve_data_to_epochs(lc_data)
        return LightcurvePipeline().analyze(epochs_arrays, object_id, period_hours, fold)

    # Legacy format: List of numpy arrays
    if object_id is None:
        raise ValueError("object_id is required when using legacy epoch array format")

    return LightcurvePipeline().analyze(epochs, object_id, period_hours, fold, ensemble)


def _convert_lightcurve_data_to_epochs(lc_data: LightcurveData) -> List[np.ndarray]:
    """Convert LightcurveData to legacy DAMIT-format epoch arrays.

    Args:
        lc_data: LightcurveData object

    Returns:
        List of (N, 8) arrays in DAMIT format:
        [time_jd, brightness, sun_x, sun_y, sun_z, earth_x, earth_y, earth_z]
    """
    epochs = []
    for epoch in lc_data.epochs:
        rows = []
        for obs in epoch.observations:
            row = [
                obs.time_jd,
                obs.relative_brightness,
                obs.sun_asteroid_vector[0],
                obs.sun_asteroid_vector[1],
                obs.sun_asteroid_vector[2],
                obs.earth_asteroid_vector[0],
                obs.earth_asteroid_vector[1],
                obs.earth_asteroid_vector[2],
            ]
            rows.append(row)

        epochs.append(np.array(rows))

    return epochs
