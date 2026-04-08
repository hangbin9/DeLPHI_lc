# Quick Start

DeLPHI predicts asteroid rotation poles from lightcurve observations. You give it photometric time-series data, and it returns pole orientation candidates.

## Option 1: Google Colab (no installation)

1. Open the inference notebook in Colab:
   [run_pole_prediction.ipynb](../run_pole_prediction.ipynb)

2. The notebook includes a synthetic data demo that runs immediately. To use your own data, replace the synthetic epoch array with your CSV:

```python
import pandas as pd
data = pd.read_csv("your_asteroid.csv").dropna().values
epochs = [data]
result = analyze(epochs, "your_asteroid", fold=0)
```

## Option 2: Local Installation

```bash
git clone https://github.com/hangbin9/DeLPHI_lc.git
cd DeLPHI_lc
pip install -e .
```

Verify:
```bash
python -c "from lc_pipeline import analyze; print('OK')"
```

### Run inference from the command line

```bash
# With automatic period estimation
python run_pole_prediction.py --input examples/asteroid_101.csv

# With a known period (faster, skips estimation)
python run_pole_prediction.py --input examples/asteroid_101.csv --period 8.5

# Save results as JSON
python run_pole_prediction.py --input examples/asteroid_101.csv --output results.json --format json

# Process a directory of CSV files
python run_pole_prediction.py --input-dir my_asteroids/ --output results.json --format json
```

### Run inference from Python

```python
from lc_pipeline import analyze
import numpy as np

data = np.loadtxt("your_asteroid.csv", delimiter=',', skiprows=1)
result = analyze([data], "your_asteroid", fold=0)

print(f"Period: {result.period.period_hours:.2f} h")
for pole in result.poles:
    print(f"  lambda={pole.lambda_deg:.1f}, beta={pole.beta_deg:.1f} ({pole.alias})")
```

## Input Format

An 8-column CSV with a header row:

```
time_jd,relative_brightness,sun_x,sun_y,sun_z,earth_x,earth_y,earth_z
2433827.77,0.988,-1.524,2.562,-1.655,-1.376,1.556,-1.655
```

- Column 0: Julian Date
- Column 1: Relative brightness (not magnitudes, centered around 1.0)
- Columns 2-4: Sun-asteroid vector (x, y, z)
- Columns 5-7: Observer-asteroid vector (x, y, z)

See `examples/asteroid_101.csv` for a working example.

## Output

The pipeline returns 6-9 pole candidates:
- 2-3 period aliases (P, 2P, and P/2 if P >= 8h)
- 3 poles per alias (K=3 multiple hypothesis output)

Each candidate has ecliptic longitude (lambda) and latitude (beta). The model outputs unranked hypotheses. The package displays them sorted by a heuristic score, but the top candidate is not guaranteed to be the correct pole. Use external constraints (radar, occultations, other inversions) to select among candidates.

## What Next

- [USER_GUIDE.md](USER_GUIDE.md) for training, evaluation, and advanced usage
- [DATA_FORMAT.md](DATA_FORMAT.md) for detailed input format reference
- [ARCHITECTURE.md](ARCHITECTURE.md) for how the model works
