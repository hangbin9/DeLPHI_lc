"""
Coordinate frame transformations for asteroid rotation pole estimation.

Supports conversions between:
- Equatorial (EQ): Standard J2000 celestial equatorial frame
- Ecliptic (ECL): Ecliptic plane frame (Earth's orbital plane)

Key constant: J2000 mean obliquity = 23.439281°
"""

import numpy as np
from typing import Union


# J2000 mean obliquity (angle between equatorial and ecliptic planes)
OBLIQUITY_DEG = 23.439281
OBLIQUITY_RAD = np.radians(OBLIQUITY_DEG)


def rot_x(angle_rad: float) -> np.ndarray:
    """
    Rotation matrix around X-axis.

    Args:
        angle_rad: rotation angle in radians

    Returns:
        (3, 3) rotation matrix
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rot_y(angle_rad: float) -> np.ndarray:
    """
    Rotation matrix around Y-axis.

    Args:
        angle_rad: rotation angle in radians

    Returns:
        (3, 3) rotation matrix
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rot_z(angle_rad: float) -> np.ndarray:
    """
    Rotation matrix around Z-axis.

    Args:
        angle_rad: rotation angle in radians

    Returns:
        (3, 3) rotation matrix
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def equatorial_to_ecliptic(v: np.ndarray) -> np.ndarray:
    """
    Convert vector(s) from equatorial (J2000) to ecliptic frame.

    Rotates by -obliquity around X-axis (makes Y-Z plane match ecliptic).

    Args:
        v: (..., 3) vector(s) in equatorial frame

    Returns:
        (..., 3) vector(s) in ecliptic frame
    """
    v = np.atleast_1d(v)
    original_shape = v.shape

    # Flatten to (N, 3)
    if v.ndim == 1:
        v = v.reshape(1, 3)
    else:
        v = v.reshape(-1, 3)

    # Rotation: -obliquity around X
    R = rot_x(-OBLIQUITY_RAD)

    # Apply: v_ecl = R @ v_eq
    v_ecl = (R @ v.T).T  # (N, 3)

    # Restore shape
    v_ecl = v_ecl.reshape(original_shape)

    return v_ecl


def ecliptic_to_equatorial(v: np.ndarray) -> np.ndarray:
    """
    Convert vector(s) from ecliptic to equatorial (J2000) frame.

    Rotates by +obliquity around X-axis.

    Args:
        v: (..., 3) vector(s) in ecliptic frame

    Returns:
        (..., 3) vector(s) in equatorial frame
    """
    v = np.atleast_1d(v)
    original_shape = v.shape

    # Flatten to (N, 3)
    if v.ndim == 1:
        v = v.reshape(1, 3)
    else:
        v = v.reshape(-1, 3)

    # Rotation: +obliquity around X
    R = rot_x(OBLIQUITY_RAD)

    # Apply: v_eq = R @ v_ecl
    v_eq = (R @ v.T).T  # (N, 3)

    # Restore shape
    v_eq = v_eq.reshape(original_shape)

    return v_eq


def canonicalize_hemisphere(v: np.ndarray, method: str = "z") -> np.ndarray:
    """
    Map vectors to a canonical hemisphere.

    Args:
        v: (..., 3) vector(s)
        method: "z" maps to z >= 0, "dot_z" maps so dot(v, [0,0,1]) >= 0 (equivalent to "z")

    Returns:
        (..., 3) canonicalized vector(s)
    """
    v = np.atleast_1d(v)
    original_shape = v.shape

    if v.ndim == 1:
        v = v.reshape(1, 3)
    else:
        v = v.reshape(-1, 3)

    if method == "z":
        # Map to z >= 0
        flip_mask = v[:, 2] < 0
        v[flip_mask] *= -1
    elif method == "dot_z":
        # Map so dot(v, +Z) >= 0 (same as z >= 0)
        flip_mask = v[:, 2] < 0
        v[flip_mask] *= -1
    else:
        raise ValueError(f"Unknown canonicalization method: {method}")

    v = v.reshape(original_shape)
    return v
