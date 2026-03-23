"""Ecliptic coordinate conversions."""
import numpy as np
from typing import Tuple

def xyz_to_ecliptic(x: float, y: float, z: float) -> Tuple[float, float]:
    """Convert unit vector to ecliptic (lambda, beta) in degrees."""
    beta = np.degrees(np.arcsin(np.clip(z, -1, 1)))
    lam = np.degrees(np.arctan2(y, x)) % 360
    return lam, beta

def ecliptic_to_xyz(lam_deg: float, beta_deg: float) -> Tuple[float, float, float]:
    """Convert ecliptic coordinates to unit vector."""
    lam, beta = np.radians(lam_deg), np.radians(beta_deg)
    x = np.cos(beta) * np.cos(lam)
    y = np.cos(beta) * np.sin(lam)
    z = np.sin(beta)
    return x, y, z
