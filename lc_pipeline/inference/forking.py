"""Period alias forking for robust pole estimation."""
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .pole import PoleInference, PoleConfig
from .schema import PoleCandidate
from .coordinates import xyz_to_ecliptic


def generate_aliases(period_base: float, min_period: float = 4.0) -> List[Tuple[float, str]]:
    """
    Generate period alias set: [P, 2P, 0.5P].

    Args:
        period_base: Base period in hours
        min_period: Minimum period threshold (don't generate half-period below this)

    Returns:
        List of (period_hours, alias_name) tuples
    """
    aliases = [
        (period_base, "base"),
        (2 * period_base, "double"),
    ]

    # Only add half-period if it's above minimum threshold
    if period_base >= 2 * min_period:
        aliases.append((0.5 * period_base, "half"))

    return aliases


class PeriodForker:
    """Run pole inference across period aliases."""

    def __init__(self, config: Optional[PoleConfig] = None):
        self.inference = PoleInference(config)

    def predict_with_aliases(
        self,
        epochs: List[np.ndarray],
        period_base: float,
        fold: int = 0,
        ensemble: bool = False
    ) -> List[PoleCandidate]:
        """
        Run forked inference for all period aliases.

        Returns 9 candidates (3 periods x 3 slots), sorted by quality.

        Args:
            epochs: List of (N, 8) DAMIT-format arrays
            period_base: Base period in hours
            fold: Model fold to use (0-4 for CV177 5-fold CV). Ignored if ensemble=True.
            ensemble: If True, average predictions across all 5 folds (better but slower)

        Returns:
            List of 9 PoleCandidate objects, sorted by score descending
        """
        aliases = generate_aliases(period_base)
        candidates = []
        all_logits = []
        has_quality = True

        # Run inference for each period alias
        for period, alias in aliases:
            poles, logits = self.inference.predict(epochs, period, fold, ensemble=ensemble)

            if logits is None:
                has_quality = False
            else:
                all_logits.append(logits)

            # Create candidates for each of the 3 slots
            for slot in range(3):
                lam, beta = xyz_to_ecliptic(*poles[slot])
                candidates.append(PoleCandidate(
                    lambda_deg=lam,
                    beta_deg=beta,
                    xyz=tuple(poles[slot]),
                    period_hours=period,
                    alias=alias,
                    score=0.0,  # Filled below
                    slot=slot
                ))

        if has_quality and all_logits:
            # Global softmax normalization across all candidates
            all_logits_concat = np.concatenate(all_logits)
            scores = F.softmax(torch.tensor(all_logits_concat), dim=0).numpy()
            for i, c in enumerate(candidates):
                c.score = float(scores[i])
        else:
            # No quality head: assign equal scores
            n = len(candidates)
            for c in candidates:
                c.score = 1.0 / n if n > 0 else 0.0

        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)

        return candidates
