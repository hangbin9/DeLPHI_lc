"""Output data structures for the lightcurve analysis pipeline."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class PoleCandidate:
    """A single pole candidate with antipode (poles are ambiguous up to 180°)."""
    lambda_deg: float      # Ecliptic longitude [0, 360)
    beta_deg: float        # Ecliptic latitude [-90, 90]
    xyz: Tuple[float, float, float]  # Unit vector
    period_hours: float    # Period used
    alias: str             # "base", "double", "half"
    score: float           # Quality score [0, 1]
    slot: int              # K=3 slot index

    @property
    def antipode_lambda_deg(self) -> float:
        """Antipode ecliptic longitude [0, 360)."""
        return (self.lambda_deg + 180.0) % 360.0

    @property
    def antipode_beta_deg(self) -> float:
        """Antipode ecliptic latitude [-90, 90]."""
        return -self.beta_deg

    @property
    def antipode_xyz(self) -> Tuple[float, float, float]:
        """Antipode unit vector."""
        return (-self.xyz[0], -self.xyz[1], -self.xyz[2])

@dataclass
class PeriodResult:
    """Period estimation result."""
    period_hours: float
    uncertainty_hours: float  # sigma_eff
    ci_low_hours: float
    ci_high_hours: float
    n_epochs: int
    success: bool

@dataclass
class PoleUncertainty:
    """Uncertainty metrics for pole predictions."""
    spread_deg: float      # Angular spread of top candidates
    confidence: float      # Quality-based confidence [0, 1]

@dataclass
class AnalysisResult:
    """Complete pipeline result."""
    object_id: str
    period: PeriodResult
    poles: List[PoleCandidate]  # All 9, sorted by score
    best_pole: PoleCandidate
    uncertainty: PoleUncertainty
