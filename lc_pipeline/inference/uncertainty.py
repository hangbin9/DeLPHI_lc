"""Uncertainty quantification for pole predictions."""
import numpy as np
from typing import List

from .schema import PoleCandidate, PoleUncertainty
from ..physics.geometry import antipode_angle


def angular_distance(p1: PoleCandidate, p2: PoleCandidate) -> float:
    """
    Angular distance in degrees (antipode-aware).

    Args:
        p1: First pole candidate
        p2: Second pole candidate

    Returns:
        Angular distance in degrees [0, 90]
    """
    return antipode_angle(np.array(p1.xyz), np.array(p2.xyz))


def compute_spread(candidates: List[PoleCandidate], top_n: int = 3) -> float:
    """
    Average pairwise angular distance of top candidates.

    Args:
        candidates: List of pole candidates (should be sorted by score)
        top_n: Number of top candidates to consider

    Returns:
        Average pairwise angular distance in degrees
    """
    top = candidates[:top_n]
    if len(top) < 2:
        return 0.0

    distances = [
        angular_distance(top[i], top[j])
        for i in range(len(top))
        for j in range(i + 1, len(top))
    ]

    return float(np.mean(distances))


def compute_confidence(candidates: List[PoleCandidate]) -> float:
    """
    Confidence based on score gap between #1 and #2.

    Args:
        candidates: List of pole candidates (should be sorted by score)

    Returns:
        Confidence score [0, 1], where:
        - 1.0 = very confident (large gap)
        - 0.0 = not confident (small gap)
    """
    if len(candidates) < 2:
        return 1.0

    gap = candidates[0].score - candidates[1].score

    # Map gap to confidence: 0.3 gap = full confidence
    confidence = min(1.0, gap / 0.3)

    return float(confidence)


def compute_uncertainty(candidates: List[PoleCandidate]) -> PoleUncertainty:
    """
    Compute comprehensive uncertainty metrics.

    Args:
        candidates: List of pole candidates (should be sorted by score)

    Returns:
        PoleUncertainty object with spread and confidence metrics
    """
    return PoleUncertainty(
        spread_deg=compute_spread(candidates),
        confidence=compute_confidence(candidates)
    )
