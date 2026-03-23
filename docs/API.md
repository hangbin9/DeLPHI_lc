# API Reference

Complete API documentation for the lc_pipeline package.

---

## Main API

### `analyze()`

Main entry point for lightcurve analysis.

```python
def analyze(
    epochs: List[np.ndarray],
    object_id: str,
    period_hours: Optional[float] = None,
    fold: int = 0,
    pole_config: Optional[PoleConfig] = None,
    period_config: Optional[PeriodConfig] = None
) -> AnalysisResult
```

**Parameters**:
- `epochs`: List of (N, 8) arrays [JD, brightness, sun_xyz, obs_xyz]
- `object_id`: Asteroid identifier (for logging/caching)
- `period_hours`: Known period (optional, estimated if not provided)
- `fold`: Cross-validation fold (0, 1, or 2)
- `pole_config`: Pole inference configuration (optional)
- `period_config`: Period estimation configuration (optional)

**Returns**: `AnalysisResult` with period and pole predictions

**Example**:
```python
from lc_pipeline import analyze

result = analyze(epochs, 'asteroid_1017', fold=0)
print(result.period.period_hours)
print(result.best_pole.lambda_deg, result.best_pole.beta_deg)
```

---

## Classes

### `LightcurvePipeline`

Main pipeline class for batch processing.

```python
class LightcurvePipeline:
    def __init__(
        self,
        pole_config: Optional[PoleConfig] = None,
        period_config: Optional[PeriodConfig] = None
    )

    def analyze(
        self,
        epochs: List[np.ndarray],
        object_id: str,
        period_hours: Optional[float] = None,
        fold: int = 0
    ) -> AnalysisResult
```

**Example**:
```python
from lc_pipeline import LightcurvePipeline

pipeline = LightcurvePipeline()
result = pipeline.analyze(epochs, 'asteroid_1017', fold=0)
```

---

### `PoleConfig`

Configuration for pole inference.

```python
@dataclass
class PoleConfig:
    device: str = 'auto'                    # 'auto', 'cpu', 'cuda'
    checkpoint_dir: Path = Path('lc_pipeline/checkpoints')
    n_windows: int = 8                      # Number of time windows
    tokens_per_window: int = 256            # Tokens per window
    max_seq_length: int = 2048             # Maximum sequence length
```

**Example**:
```python
from lc_pipeline import PoleConfig

config = PoleConfig(
    device='cpu',
    n_windows=8,
    tokens_per_window=256
)
```

---

### `PeriodConfig`

Configuration for period estimation.

```python
@dataclass
class PeriodConfig:
    min_period_hours: float = 2.0          # Minimum period to search
    max_period_hours: float = 50.0         # Maximum period to search
    n_periods: int = 10000                  # Number of period trials
    min_observations: int = 20              # Minimum observations per epoch
    false_alarm_probability: float = 0.01   # Lomb-Scargle FAP threshold
```

**Example**:
```python
from lc_pipeline import PeriodConfig

config = PeriodConfig(
    min_period_hours=5.0,
    max_period_hours=30.0,
    n_periods=20000
)
```

---

## Result Classes

### `AnalysisResult`

Complete analysis result.

```python
@dataclass
class AnalysisResult:
    period: PeriodResult                   # Period estimation result
    best_pole: PoleCandidate               # Best pole prediction
    poles: List[PoleCandidate]             # All pole candidates (sorted by score)
    uncertainty: PoleUncertainty           # Uncertainty estimates
    object_id: str                         # Asteroid identifier
```

**Attributes**:
- `period`: Period estimation (see `PeriodResult`)
- `best_pole`: Top-ranked pole candidate (see `PoleCandidate`)
- `poles`: All pole candidates sorted by confidence score
- `uncertainty`: Uncertainty quantification (see `PoleUncertainty`)
- `object_id`: Input object identifier

---

### `PeriodResult`

Period estimation result.

```python
@dataclass
class PeriodResult:
    period_hours: float                    # Best period estimate (hours)
    uncertainty_hours: float               # 1-sigma uncertainty
    ci_low_hours: float                    # 95% confidence interval (low)
    ci_high_hours: float                   # 95% confidence interval (high)
    power: float                           # Lomb-Scargle power
    false_alarm_prob: float                # False alarm probability
    n_epochs_used: int                     # Number of epochs used
```

**Example**:
```python
result = analyze(epochs, 'asteroid', fold=0)

print(f"Period: {result.period.period_hours:.2f} ± {result.period.uncertainty_hours:.2f} h")
print(f"95% CI: [{result.period.ci_low_hours:.2f}, {result.period.ci_high_hours:.2f}] h")
print(f"Power: {result.period.power:.3f}, FAP: {result.period.false_alarm_prob:.1e}")
```

---

### `PoleCandidate`

Single pole prediction.

```python
@dataclass
class PoleCandidate:
    lambda_deg: float                      # Ecliptic longitude (degrees)
    beta_deg: float                        # Ecliptic latitude (degrees)
    period_hours: float                    # Period for this pole
    alias: str                             # Period alias ('base', 'double', 'half')
    score: float                           # Confidence score (0-1)
    rank: int                              # Rank among all candidates
```

**Example**:
```python
result = analyze(epochs, 'asteroid', fold=0)

# Best pole
pole = result.best_pole
print(f"Pole: (λ={pole.lambda_deg:.1f}°, β={pole.beta_deg:.1f}°)")
print(f"Period: {pole.period_hours:.2f}h ({pole.alias} alias)")
print(f"Score: {pole.score:.3f}, Rank: {pole.rank}")

# All candidates
for pole in result.poles[:5]:  # Top 5
    print(f"{pole.rank}. λ={pole.lambda_deg:6.1f}°, β={pole.beta_deg:5.1f}°, "
          f"score={pole.score:.3f}")
```

---

### `PoleUncertainty`

Uncertainty quantification.

```python
@dataclass
class PoleUncertainty:
    spread_deg: float                      # Angular spread of top candidates (degrees)
    confidence: float                      # Overall confidence (0-1)
    top_k_separation_deg: float            # Separation to k-th candidate
    entropy: float                         # Score distribution entropy
```

**Example**:
```python
result = analyze(epochs, 'asteroid', fold=0)

print(f"Uncertainty: spread={result.uncertainty.spread_deg:.1f}°, "
      f"confidence={result.uncertainty.confidence:.2f}")
```

---

## Period Estimation

### `ConsensusEngine`

Multi-epoch period estimation.

```python
class ConsensusEngine:
    def __init__(self, config: Optional[PeriodConfig] = None)

    def estimate_period(
        self,
        epochs: List[np.ndarray],
        object_id: str
    ) -> PeriodResult
```

**Example**:
```python
from lc_pipeline import ConsensusEngine

engine = ConsensusEngine()
period_result = engine.estimate_period(epochs, 'asteroid_1017')

print(f"Period: {period_result.period_hours:.2f}h")
```

---

## Pole Inference

### `PoleInference`

Neural network pole prediction.

```python
class PoleInference:
    def __init__(self, config: Optional[PoleConfig] = None)

    def predict_poles(
        self,
        epochs: List[np.ndarray],
        period_hours: float,
        fold: int = 0,
        n_candidates: int = 100
    ) -> List[PoleCandidate]
```

**Example**:
```python
from lc_pipeline import PoleInference

engine = PoleInference()
poles = engine.predict_poles(epochs, period_hours=8.5, fold=0)

for pole in poles[:10]:
    print(f"λ={pole.lambda_deg:.1f}°, β={pole.beta_deg:.1f}°, score={pole.score:.3f}")
```

---

## Data Loading

### `load_manifest()`

Load DAMIT lightcurve manifest.

```python
def load_manifest(damit_dir: str = 'data/damit') -> Dict[str, Path]
```

**Returns**: Dictionary mapping asteroid IDs to lc.json file paths

**Example**:
```python
from lc_pipeline.data import load_manifest

manifest = load_manifest('data/damit')
print(f"Found {len(manifest)} asteroids")

for asteroid_id, lc_path in list(manifest.items())[:5]:
    print(f"{asteroid_id}: {lc_path}")
```

---

### `group_epochs_by_object()`

Group DAMIT epochs by asteroid.

```python
def group_epochs_by_object(
    damit_dir: str = 'data/damit'
) -> Dict[str, List[np.ndarray]]
```

**Returns**: Dictionary mapping asteroid IDs to epoch lists

**Example**:
```python
from lc_pipeline.data import group_epochs_by_object

asteroids = group_epochs_by_object('data/damit')

for asteroid_id, epochs in list(asteroids.items())[:3]:
    print(f"{asteroid_id}: {len(epochs)} epochs")
```

---

## Evaluation Metrics

### `pole_error()`

Calculate angular error between predicted and ground truth poles.

```python
def pole_error(
    pred_lambda: float,
    pred_beta: float,
    gt_lambda: float,
    gt_beta: float
) -> float
```

**Parameters**:
- `pred_lambda`: Predicted ecliptic longitude (degrees)
- `pred_beta`: Predicted ecliptic latitude (degrees)
- `gt_lambda`: Ground truth longitude (degrees)
- `gt_beta`: Ground truth latitude (degrees)

**Returns**: Angular error in degrees

**Example**:
```python
from lc_pipeline.evaluation.metrics import pole_error

error = pole_error(
    pred_lambda=162.5, pred_beta=-43.2,
    gt_lambda=162.0, gt_beta=-43.0
)
print(f"Pole error: {error:.2f}°")
```

---

## Utilities

### Coordinate Conversions

```python
from lc_pipeline.utils.coordinates import (
    ecliptic_to_equatorial,
    equatorial_to_ecliptic,
    pole_to_xyz,
    xyz_to_pole
)

# Convert ecliptic to equatorial
ra, dec = ecliptic_to_equatorial(lambda_deg=162.0, beta_deg=-43.0)

# Convert to unit vector
x, y, z = pole_to_xyz(lambda_deg=162.0, beta_deg=-43.0)
```

---

## Constants

```python
from lc_pipeline import __version__

print(f"lc_pipeline version: {__version__}")
```

---

## Type Hints

For type checking:

```python
from typing import List, Optional
import numpy as np
from lc_pipeline import (
    AnalysisResult,
    PeriodResult,
    PoleCandidate,
    PoleUncertainty,
    PoleConfig,
    PeriodConfig
)

def process_asteroid(
    epochs: List[np.ndarray],
    object_id: str,
    config: Optional[PoleConfig] = None
) -> AnalysisResult:
    from lc_pipeline import analyze
    return analyze(epochs, object_id, pole_config=config, fold=0)
```

---

## Next Steps

- See [USER_GUIDE.md](USER_GUIDE.md) for practical examples and troubleshooting
- See [examples/](../examples/) for sample input files
- See [DATA_FORMAT.md](DATA_FORMAT.md) for input data specification
