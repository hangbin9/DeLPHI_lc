"""Convert DAMIT CSV format to unified schema.

DAMIT CSV format:
    - Columns: time, mag, x, y, z, dx, dy, dz
    - 'mag' is actually relative_brightness (centered ~1.0)
    - (x,y,z) = sun-asteroid unit vector
    - (dx,dy,dz) = earth-asteroid unit vector

Unified schema requires:
    - time_jd, relative_brightness
    - sun_asteroid_vector, earth_asteroid_vector
    - Optional: ground truth from separate JSON
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from lc_pipeline.schema import (
    LightcurveData,
    Epoch,
    Observation,
    GroundTruth,
    PoleSolution,
)

logger = logging.getLogger(__name__)


def load_damit_csv(csv_path: Path) -> pd.DataFrame:
    """Load DAMIT CSV file and validate columns."""
    df = pd.read_csv(csv_path)

    # DAMIT format columns (case-insensitive)
    required_cols = ['time', 'mag', 'x', 'y', 'z', 'dx', 'dy', 'dz']

    # Normalize column names (lowercase)
    df.columns = df.columns.str.lower()

    # Check for required columns
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing DAMIT columns in {csv_path}: {missing}")

    return df


def assign_epoch_ids(times: np.ndarray, gap_threshold_hours: float = 12.0) -> np.ndarray:
    """Assign epoch IDs based on time gaps.

    Args:
        times: Array of Julian dates
        gap_threshold_hours: Minimum gap (hours) to start new epoch

    Returns:
        Array of epoch IDs (0-indexed)
    """
    if len(times) == 0:
        return np.array([], dtype=int)

    # Sort times
    sorted_idx = np.argsort(times)
    sorted_times = times[sorted_idx]

    # Compute time differences (in days)
    diffs = np.diff(sorted_times)
    gap_threshold_days = gap_threshold_hours / 24.0

    # Find gaps that exceed threshold
    new_epoch = np.concatenate([[False], diffs > gap_threshold_days])
    epoch_ids = np.cumsum(new_epoch)

    # Restore original order
    original_order_epochs = np.empty_like(epoch_ids)
    original_order_epochs[sorted_idx] = epoch_ids

    return original_order_epochs


def load_damit_ground_truth(
    object_id: str,
    gt_json_path: Path = Path("data/pole_gt_metadata_qf_ge_3_complete.json")
) -> Optional[GroundTruth]:
    """Load ground truth pole from DAMIT metadata JSON.

    Args:
        object_id: Asteroid identifier
        gt_json_path: Path to ground truth JSON file

    Returns:
        GroundTruth object or None if not available
    """
    if not gt_json_path.exists():
        logger.warning(f"Ground truth file not found: {gt_json_path}")
        return None

    with open(gt_json_path) as f:
        gt_data = json.load(f)

    if object_id not in gt_data:
        logger.debug(f"No ground truth for {object_id}")
        return None

    obj_gt = gt_data[object_id]

    # Extract poles (stored as Cartesian unit vectors)
    poles = obj_gt.get('poles', [])
    if not poles:
        logger.warning(f"No pole solutions for {object_id}")
        return None

    pole_solutions = [
        PoleSolution(cartesian=pole)
        for pole in poles
    ]

    # Extract rotation period (if available)
    period = obj_gt.get('period_hours', obj_gt.get('rotation_period_hours'))
    if period is None:
        logger.warning(f"No rotation period for {object_id}, using default 8.0h")
        period = 8.0  # Default fallback

    # Extract quality flag
    quality = obj_gt.get('quality_flag', obj_gt.get('qf'))

    return GroundTruth(
        rotation_period_hours=float(period),
        pole_solutions=pole_solutions,
        quality_flag=quality,
        source="DAMIT"
    )


def convert_damit_to_unified(
    csv_path: Path,
    object_id: Optional[str] = None,
    gt_json_path: Optional[Path] = None,
    gap_threshold_hours: float = 12.0,
) -> LightcurveData:
    """Convert DAMIT CSV to unified LightcurveData format.

    Args:
        csv_path: Path to DAMIT CSV file
        object_id: Asteroid identifier (defaults to filename stem)
        gt_json_path: Path to ground truth JSON (optional)
        gap_threshold_hours: Time gap to split epochs

    Returns:
        LightcurveData object conforming to unified schema
    """
    # Infer object_id from filename if not provided
    if object_id is None:
        object_id = csv_path.stem

    # Load DAMIT CSV
    df = load_damit_csv(csv_path)

    # Extract data
    times = df['time'].values.astype(np.float32)
    brightness = df['mag'].values.astype(np.float32)

    # Geometry vectors
    sun_vecs = df[['x', 'y', 'z']].values.astype(np.float32)
    earth_vecs = df[['dx', 'dy', 'dz']].values.astype(np.float32)

    # Assign epoch IDs
    epoch_ids = assign_epoch_ids(times, gap_threshold_hours)

    # Group observations by epoch
    epochs = []
    for epoch_id in np.unique(epoch_ids):
        mask = epoch_ids == epoch_id

        observations = []
        for i in np.where(mask)[0]:
            obs = Observation(
                time_jd=float(times[i]),
                relative_brightness=float(brightness[i]),
                sun_asteroid_vector=sun_vecs[i].tolist(),
                earth_asteroid_vector=earth_vecs[i].tolist(),
                epoch_id=int(epoch_id),
            )
            observations.append(obs)

        epoch = Epoch(
            epoch_id=int(epoch_id),
            observations=observations
        )
        epochs.append(epoch)

    # Load ground truth (if available)
    ground_truth = None
    if gt_json_path is not None:
        ground_truth = load_damit_ground_truth(object_id, gt_json_path)

    # Create unified LightcurveData
    lc_data = LightcurveData(
        format_version="1.0",
        object_id=object_id,
        coordinate_frame="EQUATORIAL_J2000",
        epochs=epochs,
        ground_truth=ground_truth,
        metadata={
            "source": "DAMIT",
            "original_file": str(csv_path),
            "num_observations": len(times),
            "num_epochs": len(epochs),
        }
    )

    return lc_data


def load_damit_object(
    object_id: str,
    csv_dir: Path = Path("data/damit_csv_qf_ge_3"),
    gt_json_path: Path = Path("data/pole_gt_metadata_qf_ge_3_complete.json"),
    gap_threshold_hours: float = 12.0,
) -> LightcurveData:
    """Convenience function to load DAMIT object by ID.

    Args:
        object_id: Asteroid identifier
        csv_dir: Directory containing DAMIT CSV files
        gt_json_path: Path to ground truth JSON
        gap_threshold_hours: Time gap to split epochs

    Returns:
        LightcurveData object
    """
    csv_path = csv_dir / f"{object_id}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"DAMIT CSV not found: {csv_path}")

    return convert_damit_to_unified(
        csv_path=csv_path,
        object_id=object_id,
        gt_json_path=gt_json_path,
        gap_threshold_hours=gap_threshold_hours,
    )


def batch_convert_damit(
    csv_dir: Path = Path("data/damit_csv_qf_ge_3"),
    output_dir: Path = Path("data/unified_format"),
    gt_json_path: Path = Path("data/pole_gt_metadata_qf_ge_3_complete.json"),
    gap_threshold_hours: float = 12.0,
) -> Dict[str, Path]:
    """Convert all DAMIT CSVs in directory to unified format.

    Args:
        csv_dir: Directory containing DAMIT CSV files
        output_dir: Output directory for unified JSON files
        gt_json_path: Path to ground truth JSON
        gap_threshold_hours: Time gap to split epochs

    Returns:
        Dictionary mapping object_id -> output_path
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(csv_dir.glob("*.csv"))
    converted = {}

    for csv_path in csv_files:
        object_id = csv_path.stem

        try:
            lc_data = convert_damit_to_unified(
                csv_path=csv_path,
                object_id=object_id,
                gt_json_path=gt_json_path,
                gap_threshold_hours=gap_threshold_hours,
            )

            # Save as JSON
            output_path = output_dir / f"{object_id}.json"
            with open(output_path, 'w') as f:
                json.dump(lc_data.to_dict(), f, indent=2)

            converted[object_id] = output_path
            logger.info(f"Converted {object_id}: {len(lc_data.epochs)} epochs, "
                       f"{len(lc_data.get_all_observations())} observations")

        except Exception as e:
            logger.error(f"Failed to convert {object_id}: {e}")

    logger.info(f"Converted {len(converted)}/{len(csv_files)} objects")
    return converted


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Convert single object
    lc_data = load_damit_object("asteroid_101")
    print(f"Loaded {lc_data.object_id}:")
    print(f"  Epochs: {len(lc_data.epochs)}")
    print(f"  Observations: {len(lc_data.get_all_observations())}")
    print(f"  Time range: {lc_data.get_time_range()}")

    # Batch convert (commented out to avoid running by default)
    # converted = batch_convert_damit()
    # print(f"Converted {len(converted)} objects to unified format")
