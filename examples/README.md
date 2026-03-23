# Example Files

Sample input files for lc_pipeline.

## Files

| File | Description | Ground Truth |
|------|-------------|--------------|
| `asteroid_101.json` | Asteroid 101 Helena (DAMIT) | ✅ Period + Pole |
| `asteroid_101.csv` | Same data in CSV format | ✅ In header comments |
| `inference_only.json` | Minimal inference example | ❌ None |

---

## Usage Examples

### 1. Inference Only (No Ground Truth)

```python
from lc_pipeline import analyze
from lc_pipeline.converters import load_unified_json

# Load - no ground truth needed
lc_data = load_unified_json("examples/inference_only.json")

# Analyze - period estimated automatically
result = analyze(lc_data.to_epochs(), lc_data.object_id, fold=0)

print(f"Estimated period: {result.period.period_hours:.2f} h")
print(f"Best pole: λ={result.best_pole.lambda_deg:.1f}°, β={result.best_pole.beta_deg:.1f}°")
```

### 2. With Ground Truth (Training/Evaluation)

```python
from lc_pipeline import analyze
from lc_pipeline.converters import load_unified_json

# Load with ground truth
lc_data = load_unified_json("examples/asteroid_101.json")

# Analyze
result = analyze(lc_data.to_epochs(), lc_data.object_id, fold=0)

# Compare to ground truth
gt = lc_data.ground_truth
print(f"Predicted period: {result.period.period_hours:.2f} h")
print(f"True period: {gt.rotation_period_hours:.2f} h")
print(f"True pole: λ={gt.pole_solutions[0].lambda_deg}°, β={gt.pole_solutions[0].beta_deg}°")
```

### 3. CSV Format

```python
import pandas as pd
from lc_pipeline import analyze

# Load CSV (comment lines contain ground truth metadata)
df = pd.read_csv("examples/asteroid_101.csv", comment='#')
epochs = [df.values]

# Analyze
result = analyze(epochs, "asteroid_101", fold=0)
print(f"Period: {result.period.period_hours:.2f} h")
```

---

## Ground Truth (Asteroid 101 Helena)

From DAMIT database:

| Property | Value |
|----------|-------|
| **Rotation period** | 5.223 hours |
| **Pole (ecliptic)** | λ = 354.0°, β = 82.0° |
| **Quality flag** | 3 |
| **Source** | DAMIT |

---

## Quick Installation Test

```bash
python -c "
from lc_pipeline import analyze
import pandas as pd

df = pd.read_csv('examples/asteroid_101.csv', comment='#')
result = analyze([df.values], 'asteroid_101', fold=0)

print('✅ Installation verified!')
print(f'Period: {result.period.period_hours:.2f} h')
print(f'Pole: λ={result.best_pole.lambda_deg:.1f}°, β={result.best_pole.beta_deg:.1f}°')
"
```

---

## Data Format Reference

See [docs/DATA_FORMAT.md](../docs/DATA_FORMAT.md) for complete schema documentation.

### Required Fields

| Field | Description |
|-------|-------------|
| `time_jd` | Julian Date |
| `relative_brightness` | Normalized brightness (~1.0) |
| `sun_asteroid_vector` | [x, y, z] unit vector to Sun |
| `earth_asteroid_vector` | [x, y, z] unit vector to Earth |

### Optional Ground Truth

| Field | Description |
|-------|-------------|
| `ground_truth.rotation_period_hours` | True period (for evaluation) |
| `ground_truth.pole_solutions` | True poles (for training) |
