"""Pole prediction inference engine."""
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .model import PolePredictor
from .tokenizer import tokenize_lightcurve

# Default checkpoint location
DEFAULT_CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"


@dataclass
class PoleConfig:
    """Configuration for pole prediction."""
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    device: str = "auto"  # "cuda", "cpu", or "auto"
    n_windows: int = 8
    tokens_per_window: int = 256


class PoleInference:
    """Pole prediction inference engine."""

    def __init__(self, config: Optional[PoleConfig] = None):
        self.config = config or PoleConfig()
        self._models = {}  # Lazy-loaded per fold
        self._device = self._resolve_device()

    def _resolve_device(self) -> torch.device:
        """Resolve device (cuda/cpu)."""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)

    def _load_fold(self, fold: int) -> PolePredictor:
        """Load model for a specific fold (lazy loading)."""
        if fold not in self._models:
            checkpoint_path = self.config.checkpoint_dir / f"fold_{fold}.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}\n"
                    f"Please ensure model checkpoints are in {self.config.checkpoint_dir}"
                )
            model = PolePredictor.load(str(checkpoint_path), str(self._device))
            self._models[fold] = model
        return self._models[fold]

    def predict(
        self,
        epochs: List[np.ndarray],
        period_hours: float,
        fold: int = 0,
        ensemble: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict poles for given lightcurve data.

        Args:
            epochs: List of (N, 8) DAMIT-format arrays
            period_hours: Rotation period
            fold: Model fold to use (0-4 for CV177 5-fold CV). Ignored if ensemble=True.
            ensemble: If True, average predictions across all 5 folds (better but slower)

        Returns:
            poles: (3, 3) unit vectors [slot, xyz]
            quality: (3,) quality logits [slot] or None if model has no quality head
        """
        if ensemble:
            all_poles = []
            all_quality = []

            for fold_idx in range(5):
                poles, quality = self._predict_single_fold(epochs, period_hours, fold_idx)
                all_poles.append(poles)
                if quality is not None:
                    all_quality.append(quality)

            # Average pole vectors and renormalize
            poles_avg = np.mean(all_poles, axis=0)
            poles_avg = poles_avg / np.linalg.norm(poles_avg, axis=1, keepdims=True)

            # Average quality logits if available
            quality_avg = np.mean(all_quality, axis=0) if all_quality else None

            return poles_avg, quality_avg
        else:
            return self._predict_single_fold(epochs, period_hours, fold)

    def _predict_single_fold(
        self,
        epochs: List[np.ndarray],
        period_hours: float,
        fold: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict using a single fold model."""
        model = self._load_fold(fold)

        # Tokenize
        tokens, mask = tokenize_lightcurve(
            epochs,
            period_hours=period_hours,
            n_windows=self.config.n_windows,
            tokens_per_window=self.config.tokens_per_window
        )

        # Run inference
        with torch.no_grad():
            tokens_t = torch.from_numpy(tokens).unsqueeze(0).to(self._device)
            mask_t = torch.from_numpy(mask).unsqueeze(0).float().to(self._device)
            poles, quality = model(tokens_t, mask_t)

        poles_np = poles[0].cpu().numpy()
        quality_np = quality[0].cpu().numpy() if quality is not None else None
        return poles_np, quality_np
