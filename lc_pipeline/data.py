"""
Data structures and I/O for lightcurve data.

This module provides:
- LightcurveEpoch: single-epoch lightcurve data
- AsteroidLightcurves: multi-epoch container for one asteroid
- Loading functions for manifests, ground truth, and epoch data
- Flux-to-magnitude conversion
"""

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .config import ColumnConfig, DEFAULT_COLUMN_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class LightcurveEpoch:
    """
    A single epoch of lightcurve observations.

    Attributes:
        object_id: Unique identifier for the asteroid.
        epoch_id: Identifier for this epoch (within the object).
        time: Array of observation times (JD or MJD, in days).
        mag: Array of magnitudes.
        mag_err: Array of magnitude errors.
    """
    object_id: str
    epoch_id: str
    time: np.ndarray
    mag: np.ndarray
    mag_err: np.ndarray

    def __post_init__(self):
        """Validate arrays have consistent lengths."""
        n = len(self.time)
        if len(self.mag) != n or len(self.mag_err) != n:
            raise ValueError(
                f"Inconsistent array lengths: time={len(self.time)}, "
                f"mag={len(self.mag)}, mag_err={len(self.mag_err)}"
            )

    @property
    def n_points(self) -> int:
        """Number of observations in this epoch."""
        return len(self.time)

    @property
    def time_span_days(self) -> float:
        """Total time span of observations in days."""
        if len(self.time) == 0:
            return 0.0
        return float(self.time.max() - self.time.min())


@dataclass
class AsteroidLightcurves:
    """
    Multi-epoch lightcurve data for a single asteroid.

    Attributes:
        object_id: Unique identifier for the asteroid.
        epochs: List of LightcurveEpoch objects.
    """
    object_id: str
    epochs: List[LightcurveEpoch] = field(default_factory=list)

    @property
    def n_epochs(self) -> int:
        """Number of epochs."""
        return len(self.epochs)

    @property
    def total_points(self) -> int:
        """Total number of observations across all epochs."""
        return sum(e.n_points for e in self.epochs)


def flux_to_mag(flux: np.ndarray) -> np.ndarray:
    """
    Convert flux (relative brightness) to magnitude.

    Uses the standard astronomical formula: mag = -2.5 * log10(flux)

    Args:
        flux: Array of flux values (must be positive).

    Returns:
        Array of magnitudes. Invalid values (flux <= 0) become NaN.
    """
    flux = np.asarray(flux, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # Handle zero/negative flux gracefully
        mag = np.where(flux > 0, -2.5 * np.log10(flux), np.nan)
    return mag


def load_manifest(manifest_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a manifest CSV file.

    Expected columns:
        - object_id: string identifier
        - epoch_id: string or int identifier
        - file_path: path to lightcurve CSV

    Optional columns:
        - survey: survey name
        - filter: photometric filter

    Args:
        manifest_path: Path to the manifest CSV.

    Returns:
        DataFrame with manifest entries.

    Raises:
        FileNotFoundError: If manifest file doesn't exist.
        ValueError: If required columns are missing.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    df = pd.read_csv(path)

    required_cols = {"object_id", "file_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    # Ensure object_id and epoch_id are strings
    df["object_id"] = df["object_id"].astype(str)
    if "epoch_id" in df.columns:
        df["epoch_id"] = df["epoch_id"].astype(str)
    else:
        # Generate epoch_id from row index if not present
        df["epoch_id"] = [f"epoch_{i}" for i in range(len(df))]

    return df


def load_groundtruth(gt_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load ground truth period CSV.

    Expected columns:
        - object_id: string identifier
        - period_hours: true rotation period in hours

    Args:
        gt_path: Path to ground truth CSV.

    Returns:
        DataFrame with ground truth periods.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If required columns are missing.
    """
    path = Path(gt_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    df = pd.read_csv(path)

    required_cols = {"object_id", "period_hours"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Ground truth missing required columns: {missing}")

    df["object_id"] = df["object_id"].astype(str)
    return df


def load_epoch_from_file(
    file_path: Union[str, Path],
    object_id: str,
    epoch_id: str,
    column_config: Optional[ColumnConfig] = None
) -> Optional[LightcurveEpoch]:
    """
    Load a single epoch from a lightcurve CSV file.

    Handles flux-to-magnitude conversion and NaN filtering.

    Args:
        file_path: Path to the lightcurve CSV.
        object_id: Object identifier.
        epoch_id: Epoch identifier.
        column_config: Column mapping configuration.

    Returns:
        LightcurveEpoch object, or None if loading fails or no valid data.
    """
    if column_config is None:
        column_config = DEFAULT_COLUMN_CONFIG

    path = Path(file_path)
    if not path.exists():
        logger.warning(f"Lightcurve file not found: {path}")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return None

    # Get time column
    if column_config.time_col not in df.columns:
        logger.warning(f"Time column '{column_config.time_col}' not found in {path}")
        return None
    time = df[column_config.time_col].values

    # Get magnitude - either directly or convert from flux
    if column_config.mag_col and column_config.mag_col in df.columns:
        mag = df[column_config.mag_col].values
    elif column_config.flux_col in df.columns:
        flux = df[column_config.flux_col].values
        mag = flux_to_mag(flux)
    else:
        logger.warning(
            f"Neither mag_col '{column_config.mag_col}' nor "
            f"flux_col '{column_config.flux_col}' found in {path}"
        )
        return None

    # Get magnitude error
    if column_config.mag_err_col and column_config.mag_err_col in df.columns:
        mag_err = df[column_config.mag_err_col].values
    else:
        mag_err = np.full_like(mag, column_config.default_mag_err)

    # Convert to float arrays
    time = np.asarray(time, dtype=float)
    mag = np.asarray(mag, dtype=float)
    mag_err = np.asarray(mag_err, dtype=float)

    # Filter out invalid values (NaN in time or mag)
    valid_mask = np.isfinite(time) & np.isfinite(mag)
    if not np.any(valid_mask):
        logger.warning(f"No valid data points in {path}")
        return None

    time = time[valid_mask]
    mag = mag[valid_mask]
    mag_err = mag_err[valid_mask]

    # Replace any remaining NaN in mag_err with default
    mag_err = np.where(np.isfinite(mag_err), mag_err, column_config.default_mag_err)

    # Sort by time
    sort_idx = np.argsort(time)
    time = time[sort_idx]
    mag = mag[sort_idx]
    mag_err = mag_err[sort_idx]

    return LightcurveEpoch(
        object_id=object_id,
        epoch_id=epoch_id,
        time=time,
        mag=mag,
        mag_err=mag_err
    )


def load_epoch_from_row(
    row: pd.Series,
    column_config: Optional[ColumnConfig] = None
) -> Optional[LightcurveEpoch]:
    """
    Load a lightcurve epoch from a manifest row.

    Args:
        row: A row from the manifest DataFrame.
        column_config: Column mapping configuration.

    Returns:
        LightcurveEpoch object, or None if loading fails.
    """
    return load_epoch_from_file(
        file_path=row["file_path"],
        object_id=str(row["object_id"]),
        epoch_id=str(row.get("epoch_id", "epoch_0")),
        column_config=column_config
    )


def group_epochs_by_object(
    manifest_df: pd.DataFrame,
    column_config: Optional[ColumnConfig] = None,
    validate: bool = True
) -> Dict[str, AsteroidLightcurves]:
    """
    Group manifest rows by object_id and load all epochs.

    Performs strict grouping: all epochs with the same object_id are
    grouped together. Validates for duplicate (object_id, epoch_id) pairs.

    Args:
        manifest_df: DataFrame with manifest entries.
        column_config: Column mapping configuration.
        validate: Whether to validate for duplicates and log warnings.

    Returns:
        Dictionary mapping object_id to AsteroidLightcurves.

    Raises:
        ValueError: If duplicate (object_id, epoch_id) pairs are found.
    """
    if column_config is None:
        column_config = DEFAULT_COLUMN_CONFIG

    # Validate for duplicates
    if validate and "epoch_id" in manifest_df.columns:
        duplicates = manifest_df.groupby(["object_id", "epoch_id"]).size()
        duplicates = duplicates[duplicates > 1]
        if len(duplicates) > 0:
            raise ValueError(
                f"Duplicate (object_id, epoch_id) pairs found: "
                f"{duplicates.index.tolist()[:5]}..."
            )

    objects: Dict[str, AsteroidLightcurves] = {}

    for object_id, group_df in manifest_df.groupby("object_id"):
        object_id = str(object_id)
        epochs = []

        for _, row in group_df.iterrows():
            epoch = load_epoch_from_row(row, column_config)
            if epoch is not None:
                epochs.append(epoch)

        if epochs:
            objects[object_id] = AsteroidLightcurves(
                object_id=object_id,
                epochs=epochs
            )

            # Warn if only single epoch (consensus degrades)
            if validate and len(epochs) == 1:
                logger.warning(
                    f"Object '{object_id}' has only 1 epoch - "
                    "consensus will degrade to single-epoch mode"
                )
        else:
            logger.warning(f"No valid epochs loaded for object '{object_id}'")

    return objects


def extract_groundtruth_from_lightcurves(
    manifest_df: pd.DataFrame,
    column_config: Optional[ColumnConfig] = None
) -> pd.DataFrame:
    """
    Extract ground truth periods from lightcurve CSV files.

    Reads the period column from each unique object's first CSV file.

    Args:
        manifest_df: DataFrame with manifest entries.
        column_config: Column mapping configuration.

    Returns:
        DataFrame with columns: object_id, period_hours
    """
    if column_config is None:
        column_config = DEFAULT_COLUMN_CONFIG

    records = []

    for object_id in manifest_df["object_id"].unique():
        # Get first file for this object
        object_rows = manifest_df[manifest_df["object_id"] == object_id]
        first_file = object_rows.iloc[0]["file_path"]

        try:
            df = pd.read_csv(first_file, nrows=1)
            if column_config.period_col in df.columns:
                period_value = df[column_config.period_col].iloc[0]

                # Convert to hours if needed
                if column_config.period_unit == "days":
                    period_hours = float(period_value) * 24.0
                else:
                    period_hours = float(period_value)

                records.append({
                    "object_id": str(object_id),
                    "period_hours": period_hours
                })
            else:
                logger.warning(
                    f"Period column '{column_config.period_col}' not found "
                    f"in {first_file}"
                )
        except Exception as e:
            logger.warning(f"Failed to extract period from {first_file}: {e}")

    return pd.DataFrame(records)
