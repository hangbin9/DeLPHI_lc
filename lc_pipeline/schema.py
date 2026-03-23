"""Unified data schema for asteroid lightcurve pole estimation.

This module defines the standard format for lightcurve data, independent of source.
All converters should output data conforming to this schema.

Format Version: 1.0
Coordinate Frame: EQUATORIAL_J2000 (unless specified otherwise)
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import numpy as np


class Observation(BaseModel):
    """Single photometric observation with geometry."""

    # REQUIRED FIELDS
    time_jd: float = Field(..., description="Julian Date of observation")
    relative_brightness: float = Field(..., description="Normalized brightness (centered ~1.0)")
    sun_asteroid_vector: List[float] = Field(..., description="Unit vector from asteroid to Sun [x,y,z]")
    earth_asteroid_vector: List[float] = Field(..., description="Unit vector from asteroid to Earth [x,y,z]")

    # OPTIONAL FIELDS
    brightness_error: Optional[float] = Field(None, description="Measurement uncertainty (1-sigma)")
    epoch_id: Optional[int] = Field(None, description="Observation epoch grouping")

    @validator('sun_asteroid_vector', 'earth_asteroid_vector')
    def validate_unit_vector(cls, v):
        """Ensure vectors are 3D and approximately unit norm."""
        if len(v) != 3:
            raise ValueError(f"Vector must have 3 components, got {len(v)}")
        norm = np.linalg.norm(v)
        if not (0.9 < norm < 1.1):
            raise ValueError(f"Vector should be unit norm, got norm={norm:.3f}")
        return v


class Epoch(BaseModel):
    """Collection of observations from a single observing epoch."""

    epoch_id: int = Field(..., description="Unique epoch identifier")
    observations: List[Observation] = Field(..., description="List of observations in this epoch")

    # Optional metadata
    observer_code: Optional[str] = Field(None, description="MPC observatory code")
    filter_band: Optional[str] = Field(None, description="Photometric filter (V, R, etc.)")

    @validator('observations')
    def validate_nonempty(cls, v):
        """Ensure at least one observation per epoch."""
        if len(v) == 0:
            raise ValueError("Epoch must contain at least one observation")
        return v


class PoleSolution(BaseModel):
    """Pole orientation solution (ground truth or prediction)."""

    # Can provide EITHER Cartesian OR spherical coordinates
    cartesian: Optional[List[float]] = Field(None, description="3D unit vector [x,y,z] in J2000 frame")
    lambda_deg: Optional[float] = Field(None, description="Ecliptic longitude (degrees)")
    beta_deg: Optional[float] = Field(None, description="Ecliptic latitude (degrees)")

    # Optional quality metrics
    uncertainty_deg: Optional[float] = Field(None, description="Angular uncertainty (degrees)")
    confidence: Optional[float] = Field(None, description="Confidence score [0,1]")

    @validator('cartesian')
    def validate_cartesian(cls, v):
        """Ensure Cartesian pole is 3D unit vector."""
        if v is not None:
            if len(v) != 3:
                raise ValueError(f"Cartesian pole must have 3 components, got {len(v)}")
            norm = np.linalg.norm(v)
            if not (0.9 < norm < 1.1):
                raise ValueError(f"Pole should be unit norm, got norm={norm:.3f}")
        return v

    @validator('lambda_deg', 'beta_deg')
    def validate_angles(cls, v):
        """Ensure angles are in valid range."""
        if v is not None:
            if not (-360 <= v <= 360):
                raise ValueError(f"Angle out of range: {v}")
        return v


class GroundTruth(BaseModel):
    """Ground truth pole and rotation period (for training/evaluation only)."""

    # BOTH are OPTIONAL - can provide period only, poles only, or both
    rotation_period_hours: Optional[float] = Field(
        None,
        description="Ground truth sidereal rotation period (hours)"
    )

    pole_solutions: Optional[List[PoleSolution]] = Field(
        None,
        description="List of pole solutions (handles ambiguity). Provide for pole training/evaluation."
    )

    # OPTIONAL metadata
    quality_flag: Optional[int] = Field(None, description="DAMIT quality factor (1-3)")
    source: Optional[str] = Field(None, description="Data source (DAMIT, ATLAS, etc.)")
    reference: Optional[str] = Field(None, description="Citation/DOI")

    @validator('rotation_period_hours')
    def validate_period(cls, v):
        """Ensure period is physically reasonable."""
        if v is not None and not (0.1 < v < 1000):
            raise ValueError(f"Unrealistic rotation period: {v} hours")
        return v

    @validator('pole_solutions')
    def validate_poles_nonempty(cls, v):
        """Ensure at least one pole solution if provided."""
        if v is not None and len(v) == 0:
            raise ValueError("If providing pole_solutions, must have at least one pole")
        return v

    def __init__(self, **data):
        """Validate that at least ONE of period or poles is provided."""
        super().__init__(**data)
        if self.rotation_period_hours is None and self.pole_solutions is None:
            raise ValueError("GroundTruth must provide at least one of: rotation_period_hours, pole_solutions")


class LightcurveData(BaseModel):
    """Complete lightcurve dataset for a single asteroid."""

    format_version: str = Field("1.0", description="Schema version")
    object_id: str = Field(..., description="Unique asteroid identifier")
    coordinate_frame: str = Field("EQUATORIAL_J2000", description="Reference frame for vectors")

    # REQUIRED: observation data
    epochs: List[Epoch] = Field(..., description="List of observation epochs")

    # OPTIONAL: user-provided period (for inference with known period)
    period_hours: Optional[float] = Field(
        None,
        description="Known rotation period (hours). If provided, period estimation is skipped."
    )

    # OPTIONAL: ground truth (for training/evaluation only)
    ground_truth: Optional[GroundTruth] = Field(
        None,
        description="Ground truth for training/evaluation. NOT needed for inference."
    )

    # OPTIONAL: asteroid metadata
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @validator('period_hours')
    def validate_period_hours(cls, v):
        """Ensure period is physically reasonable."""
        if v is not None and not (0.1 < v < 1000):
            raise ValueError(f"Unrealistic rotation period: {v} hours")
        return v

    @validator('epochs')
    def validate_epochs_nonempty(cls, v):
        """Ensure at least one epoch."""
        if len(v) == 0:
            raise ValueError("Must provide at least one epoch")
        return v

    def get_all_observations(self) -> List[Observation]:
        """Flatten all observations across epochs."""
        obs = []
        for epoch in self.epochs:
            obs.extend(epoch.observations)
        return obs

    def get_time_range(self) -> tuple[float, float]:
        """Get min and max Julian dates."""
        all_times = [obs.time_jd for epoch in self.epochs for obs in epoch.observations]
        return min(all_times), max(all_times)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)."""
        return self.dict(exclude_none=True)


class SimplifiedCSVSchema:
    """Simplified CSV format for users who prefer flat files.

    CSV columns (REQUIRED):
        - time_jd: Julian Date
        - relative_brightness: Normalized brightness
        - sun_x, sun_y, sun_z: Unit vector from asteroid to Sun
        - earth_x, earth_y, earth_z: Unit vector from asteroid to Earth

    CSV columns (OPTIONAL):
        - brightness_error: Measurement uncertainty
        - epoch_id: Observation epoch grouping
        - period_hours: Known rotation period (for inference)

    Ground truth (separate file or header comment):
        - rotation_period_hours: Sidereal period (for training/evaluation)
        - pole_lambda_deg, pole_beta_deg: Ecliptic pole coordinates
        - pole_x, pole_y, pole_z: Cartesian pole (alternative)
    """

    REQUIRED_COLUMNS = [
        'time_jd',
        'relative_brightness',
        'sun_x', 'sun_y', 'sun_z',
        'earth_x', 'earth_y', 'earth_z'
    ]

    OPTIONAL_COLUMNS = [
        'brightness_error',
        'epoch_id',
        'period_hours'
    ]

    @staticmethod
    def validate_csv(df) -> bool:
        """Check if DataFrame has required columns."""
        import pandas as pd
        missing = set(SimplifiedCSVSchema.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True


# Example usage documentation
EXAMPLE_JSON_INFERENCE = """
Example 1: Inference with known period (no ground truth)
{
  "format_version": "1.0",
  "object_id": "asteroid_101",
  "coordinate_frame": "EQUATORIAL_J2000",

  "period_hours": 8.34,

  "epochs": [
    {
      "epoch_id": 0,
      "observations": [
        {
          "time_jd": 2433827.771536,
          "relative_brightness": 0.9882,
          "brightness_error": 0.02,
          "sun_asteroid_vector": [0.554, -0.742, 0.479],
          "earth_asteroid_vector": [0.512, -0.578, 0.614],
          "epoch_id": 0
        }
      ]
    }
  ]
}
"""

EXAMPLE_JSON_TRAINING_POLES = """
Example 2: Training/evaluation with POLE ground truth only (period will be estimated)
{
  "format_version": "1.0",
  "object_id": "asteroid_101",
  "coordinate_frame": "EQUATORIAL_J2000",

  "epochs": [
    {
      "epoch_id": 0,
      "observations": [
        {
          "time_jd": 2433827.771536,
          "relative_brightness": 0.9882,
          "brightness_error": 0.02,
          "sun_asteroid_vector": [0.554, -0.742, 0.479],
          "earth_asteroid_vector": [0.512, -0.578, 0.614],
          "epoch_id": 0
        }
      ]
    }
  ],

  "ground_truth": {
    "pole_solutions": [
      {
        "cartesian": [0.35, -0.19, 0.92],
        "lambda_deg": 210.5,
        "beta_deg": 62.3
      }
    ],
    "quality_flag": 3,
    "source": "DAMIT"
  }
}
"""

EXAMPLE_JSON_TRAINING_PERIOD = """
Example 3: Evaluation with PERIOD ground truth only (poles will be predicted)
{
  "format_version": "1.0",
  "object_id": "asteroid_101",
  "coordinate_frame": "EQUATORIAL_J2000",

  "epochs": [
    {
      "epoch_id": 0,
      "observations": [
        {
          "time_jd": 2433827.771536,
          "relative_brightness": 0.9882,
          "brightness_error": 0.02,
          "sun_asteroid_vector": [0.554, -0.742, 0.479],
          "earth_asteroid_vector": [0.512, -0.578, 0.614],
          "epoch_id": 0
        }
      ]
    }
  ],

  "ground_truth": {
    "rotation_period_hours": 8.34,
    "quality_flag": 3,
    "source": "DAMIT"
  }
}
"""

EXAMPLE_JSON_FULL = """
Example 4: Training with FULL ground truth (period + poles)
{
  "format_version": "1.0",
  "object_id": "asteroid_101",
  "coordinate_frame": "EQUATORIAL_J2000",

  "epochs": [
    {
      "epoch_id": 0,
      "observations": [
        {
          "time_jd": 2433827.771536,
          "relative_brightness": 0.9882,
          "brightness_error": 0.02,
          "sun_asteroid_vector": [0.554, -0.742, 0.479],
          "earth_asteroid_vector": [0.512, -0.578, 0.614],
          "epoch_id": 0
        }
      ]
    }
  ],

  "ground_truth": {
    "rotation_period_hours": 8.34,
    "pole_solutions": [
      {
        "cartesian": [0.35, -0.19, 0.92],
        "lambda_deg": 210.5,
        "beta_deg": 62.3
      }
    ],
    "quality_flag": 3,
    "source": "DAMIT"
  }
}
"""

EXAMPLE_CSV = """
time_jd,relative_brightness,brightness_error,sun_x,sun_y,sun_z,earth_x,earth_y,earth_z,epoch_id
2433827.771536,0.9882,0.02,0.554,-0.742,0.479,0.512,-0.578,0.614,0
2433827.781536,1.0124,0.02,0.554,-0.742,0.479,0.512,-0.578,0.614,0
2433827.791536,0.9956,0.02,0.554,-0.742,0.479,0.512,-0.578,0.614,0
"""
