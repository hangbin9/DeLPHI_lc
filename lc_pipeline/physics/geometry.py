"""Geometry utilities including antipode-aware angle computations."""

import numpy as np


def antipode_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute antipode-aware angle between two unit vectors.

    The angle is the minimum of the angle between v1 and v2,
    and the angle between v1 and -v2 (antipode).

    Args:
        v1: (3,) unit vector
        v2: (3,) unit vector

    Returns:
        Angle in degrees (0-90, accounting for antipode)
    """
    # Normalize to unit vectors
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)

    # Angle between v1 and v2
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))

    # Angle with antipode (180 - angle)
    antipode_angle = 180.0 - angle

    # Return minimum (accounting for ambiguity)
    return float(min(angle, antipode_angle))


def best_solution_error(prediction: np.ndarray, solutions) -> float:
    """
    Find best angular error between prediction and any solution (antipode-aware).

    Args:
        prediction: (3,) predicted pole unit vector
        solutions: List or array of (3,) solution unit vectors

    Returns:
        Minimum antipode-aware angle to any solution
    """
    solutions = np.atleast_2d(solutions)
    if len(solutions) == 0:
        return float('inf')
    errors = [antipode_angle(prediction, sol) for sol in solutions]
    return float(min(errors))


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (v / norm).astype(np.float32)
