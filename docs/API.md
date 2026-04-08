# API Reference

Reference for the public API exposed by `lc_pipeline`.

## Main Entry Point

### `analyze()`

```python
def analyze(
    epochs,
    object_id: Optional[str] = None,
    period_hours: Optional[float] = None,
    fold: int = 0,
    ensemble: bool = False,
) -> AnalysisResult
```

Accepted input forms for `epochs`:

- `List[np.ndarray]` where each array is `(N, 8)` in DAMIT-style column order
- `LightcurveData` from the unified schema

Parameters:

- `epochs`: lightcurve data
- `object_id`: required for the legacy list-of-arrays input; optional for `LightcurveData`
- `period_hours`: known period in hours; if omitted, the package estimates it
- `fold`: checkpoint index, `0` through `4`
- `ensemble`: if `True`, average predictions across all 5 fold checkpoints

Example:

```python
import pandas as pd
from lc_pipeline import analyze

df = pd.read_csv("examples/asteroid_101.csv", comment="#")
result = analyze([df.values], "asteroid_101", fold=0)
```

## Pipeline Class

### `LightcurvePipeline`

```python
class LightcurvePipeline:
    def __init__(self, period_config=None, pole_config: Optional[PoleConfig] = None)

    def analyze(
        self,
        epochs,
        object_id: str,
        period_hours: Optional[float] = None,
        fold: int = 0,
        ensemble: bool = False,
    ) -> AnalysisResult
```

Use this class if you want to reuse one pipeline instance across many calls.

## Configuration

### `PoleConfig`

```python
@dataclass
class PoleConfig:
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    device: str = "auto"
    n_windows: int = 8
    tokens_per_window: int = 256
```

Notes:

- `device` can be `"auto"`, `"cpu"`, or `"cuda"`
- the inference path uses up to **8** windows and up to **256** tokens per window

## Result Types

### `AnalysisResult`

```python
@dataclass
class AnalysisResult:
    object_id: str
    period: PeriodResult
    poles: List[PoleCandidate]
    best_pole: PoleCandidate
    uncertainty: PoleUncertainty
```

### `PeriodResult`

```python
@dataclass
class PeriodResult:
    period_hours: float
    uncertainty_hours: float
    ci_low_hours: float
    ci_high_hours: float
    n_epochs: int
    success: bool
```

### `PoleCandidate`

```python
@dataclass
class PoleCandidate:
    lambda_deg: float
    beta_deg: float
    xyz: Tuple[float, float, float]
    period_hours: float
    alias: str
    score: float
    slot: int
```

Additional computed properties:

- `antipode_lambda_deg`
- `antipode_beta_deg`
- `antipode_xyz`

Interpretation notes:

- `alias` is one of `base`, `double`, or `half`
- `slot` is the model output head index
- candidates are returned sorted by `score`
- the `score` is **not** a guarantee that the first candidate is physically correct

### `PoleUncertainty`

```python
@dataclass
class PoleUncertainty:
    spread_deg: float
    confidence: float
```

Interpretation notes:

- `spread_deg` summarizes how far apart the top candidates are
- `confidence` is derived from score separation and should be treated as a convenience heuristic

## Unified Data Input

The package also accepts the unified schema defined in `lc_pipeline.schema`.

Common entry points:

- `LightcurveData`
- `Epoch`
- `Observation`
- `GroundTruth`
- `PoleSolution`

If you pass a `LightcurveData` object to `analyze()`:

- `object_id` is taken from the object if not provided explicitly
- `period_hours` is taken from the object if present and not overridden

## Period-Only Usage

```python
from lc_pipeline import ConsensusEngine

engine = ConsensusEngine()
# Single epoch:
result = engine.predict_single_epoch(lc_epoch)
# Multiple epochs:
result = engine.predict_multi_epoch(asteroid_lightcurves)
```

Use this when you only want the classical period-estimation stage.

## Related CLI

```bash
python run_pole_prediction.py --input examples/asteroid_101.csv
```

This calls the same inference pipeline documented here.
