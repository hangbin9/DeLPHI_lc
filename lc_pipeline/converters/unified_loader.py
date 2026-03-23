"""Load data from unified schema formats (JSON or CSV).

Supports loading LightcurveData from:
  - JSON files conforming to lc_pipeline.schema
  - CSV files conforming to SimplifiedCSVSchema
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
    SimplifiedCSVSchema,
)

logger = logging.getLogger(__name__)


def load_unified_json(json_path: Path) -> LightcurveData:
    """Load LightcurveData from unified JSON format.

    Args:
        json_path: Path to JSON file conforming to schema

    Returns:
        LightcurveData object with validation
    """
    with open(json_path) as f:
        data_dict = json.load(f)

    # Pydantic will validate schema
    lc_data = LightcurveData(**data_dict)

    logger.info(f"Loaded {lc_data.object_id}: {len(lc_data.epochs)} epochs, "
               f"{len(lc_data.get_all_observations())} observations")

    return lc_data


def load_unified_csv(
    csv_path: Path,
    object_id: Optional[str] = None,
    ground_truth_path: Optional[Path] = None,
) -> LightcurveData:
    """Load LightcurveData from simplified CSV format.

    Args:
        csv_path: Path to CSV file with required columns
        object_id: Asteroid identifier (defaults to filename stem)
        ground_truth_path: Optional path to ground truth JSON

    Returns:
        LightcurveData object
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    # Validate required columns
    SimplifiedCSVSchema.validate_csv(df)

    # Infer object_id if not provided
    if object_id is None:
        object_id = csv_path.stem

    # Extract observations
    observations = []
    for _, row in df.iterrows():
        obs = Observation(
            time_jd=float(row['time_jd']),
            relative_brightness=float(row['relative_brightness']),
            sun_asteroid_vector=[
                float(row['sun_x']),
                float(row['sun_y']),
                float(row['sun_z'])
            ],
            earth_asteroid_vector=[
                float(row['earth_x']),
                float(row['earth_y']),
                float(row['earth_z'])
            ],
            brightness_error=float(row['brightness_error']) if 'brightness_error' in row else None,
            epoch_id=int(row['epoch_id']) if 'epoch_id' in row else None,
        )
        observations.append(obs)

    # Group by epoch_id (if provided)
    if 'epoch_id' in df.columns:
        epochs = []
        for epoch_id in sorted(df['epoch_id'].unique()):
            epoch_obs = [obs for obs in observations if obs.epoch_id == epoch_id]
            epoch = Epoch(
                epoch_id=int(epoch_id),
                observations=epoch_obs
            )
            epochs.append(epoch)
    else:
        # Single epoch
        epoch = Epoch(epoch_id=0, observations=observations)
        epochs = [epoch]

    # Extract period_hours if present in CSV
    period_hours = None
    if 'period_hours' in df.columns:
        # Use first non-null value (should be same for all rows)
        period_values = df['period_hours'].dropna().unique()
        if len(period_values) > 0:
            period_hours = float(period_values[0])

    # Load ground truth (if provided)
    ground_truth = None
    if ground_truth_path is not None:
        with open(ground_truth_path) as f:
            gt_dict = json.load(f)
        ground_truth = GroundTruth(**gt_dict)

    # Create LightcurveData
    lc_data = LightcurveData(
        format_version="1.0",
        object_id=object_id,
        coordinate_frame="EQUATORIAL_J2000",
        epochs=epochs,
        period_hours=period_hours,
        ground_truth=ground_truth,
        metadata={
            "source": "CSV",
            "original_file": str(csv_path),
            "num_observations": len(observations),
            "num_epochs": len(epochs),
        }
    )

    logger.info(f"Loaded CSV {lc_data.object_id}: {len(lc_data.epochs)} epochs, "
               f"{len(observations)} observations")

    return lc_data


def save_unified_json(lc_data: LightcurveData, output_path: Path) -> None:
    """Save LightcurveData to JSON file.

    Args:
        lc_data: LightcurveData object to save
        output_path: Output JSON file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(lc_data.to_dict(), f, indent=2)

    logger.info(f"Saved {lc_data.object_id} to {output_path}")


def save_unified_csv(lc_data: LightcurveData, output_path: Path) -> None:
    """Save LightcurveData to simplified CSV format.

    Args:
        lc_data: LightcurveData object to save
        output_path: Output CSV file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten observations
    rows = []
    for obs in lc_data.get_all_observations():
        row = {
            'time_jd': obs.time_jd,
            'relative_brightness': obs.relative_brightness,
            'sun_x': obs.sun_asteroid_vector[0],
            'sun_y': obs.sun_asteroid_vector[1],
            'sun_z': obs.sun_asteroid_vector[2],
            'earth_x': obs.earth_asteroid_vector[0],
            'earth_y': obs.earth_asteroid_vector[1],
            'earth_z': obs.earth_asteroid_vector[2],
        }

        if obs.brightness_error is not None:
            row['brightness_error'] = obs.brightness_error

        if obs.epoch_id is not None:
            row['epoch_id'] = obs.epoch_id

        # Add period_hours if present (same for all rows)
        if lc_data.period_hours is not None:
            row['period_hours'] = lc_data.period_hours

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    logger.info(f"Saved {lc_data.object_id} to {output_path} ({len(rows)} rows)")


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Load from JSON
    # lc_data = load_unified_json(Path("data/unified_format/asteroid_101.json"))
    # print(f"Loaded {lc_data.object_id} from JSON")

    # Load from CSV
    # lc_data = load_unified_csv(Path("data/simplified_csv/asteroid_101.csv"))
    # print(f"Loaded {lc_data.object_id} from CSV")

    print("Unified loader module ready")
