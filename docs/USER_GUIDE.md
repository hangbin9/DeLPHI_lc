# DeLPHI User Guide

Beginner-focused guide for running the `lc_pipeline` package on asteroid lightcurve data.

## What This Package Does

`lc_pipeline` takes asteroid lightcurve observations and returns:

- an estimated rotation period
- several candidate pole directions
- simple uncertainty summaries

It does **not** produce a final physical shape model by itself. The intended use is to narrow the search space before a more detailed inversion workflow.

## Before You Start

### What data you need

For each observation, you need:

- observation time in Julian Date
- relative brightness
- Sun-to-asteroid unit vector `(x, y, z)`
- observer-to-asteroid unit vector `(x, y, z)`

The simplest supported input is a CSV with 8 columns:

```text
time_jd,brightness,sun_x,sun_y,sun_z,obs_x,obs_y,obs_z
```

### Data requirements in practice

The pipeline and manuscript support the following guidance:

- minimum for pole inference: **3 observations in an observing epoch**
- minimum for period estimation: **10 observations in an observing epoch**
- validated rotation-period range: **2.4 to 67.5 h**
- search range used by the period estimator: **2 to 200 h**

Best results come from **multiple observing epochs / apparitions**, not from a single dense night.

## Installation

```bash
git clone https://github.com/hangbin9/DeLPHI_lc.git
cd DeLPHI_lc
pip install -e .
```

### Verify the install

```bash
python -c "from lc_pipeline import analyze, __version__; print(__version__)"
```

## Step-By-Step: Run The Example

### 1. Run the example file from the command line

```bash
python run_pole_prediction.py --input examples/asteroid_101.csv
```

You should see:

- the estimated period
- the top candidate pole
- all returned candidates
- uncertainty summaries

### 2. Run the example from Python

```python
import pandas as pd
from lc_pipeline import analyze

df = pd.read_csv("examples/asteroid_101.csv", comment="#")
epochs = [df.values]

result = analyze(epochs, "asteroid_101", fold=0)

print(f"Period: {result.period.period_hours:.2f} h")
print(f"Top candidate: λ={result.best_pole.lambda_deg:.1f}°, β={result.best_pole.beta_deg:.1f}°")
print(f"Candidates returned: {len(result.poles)}")
```

## Run Your Own Data

### CSV input

```bash
python run_pole_prediction.py --input my_asteroid.csv
```

### JSON input

The script accepts:

- the legacy DAMIT-style JSON format
- the unified JSON format documented in [docs/DATA_FORMAT.md](DATA_FORMAT.md)

```bash
python run_pole_prediction.py --input my_asteroid.json
```

### If you already know the period

```bash
python run_pole_prediction.py --input my_asteroid.csv --period 8.5
```

This skips period estimation and uses your supplied period for pole prediction.

## Understanding The Output

### Period

The package reports:

- `period_hours`
- `uncertainty_hours`
- `ci_low_hours`
- `ci_high_hours`

### Pole candidates

The package returns **6 to 9** candidates because it combines:

- **3 pole hypotheses**
- the base period `P`
- the doubled period `2P`
- the half-period `P/2` when `P >= 8 h`

Each candidate includes:

- ecliptic longitude `lambda_deg`
- ecliptic latitude `beta_deg`
- the tested period alias
- a `score`
- the output `slot`

### Important interpretation rules

- The model outputs unranked hypotheses. The package displays them sorted by a heuristic score, but the top candidate is not guaranteed to be the correct pole.
- The main manuscript metric is an **oracle** metric, which assumes the best candidate is selected in hindsight.
- The antipode is physically equivalent for many use cases and should also be checked.
- Inspect several candidates, not just the first one.

## Recommended Beginner Workflow

1. Start with `run_pole_prediction.py`.
2. Confirm that your file loads and the period estimate is reasonable.
3. Save the JSON output if you want to inspect all candidates later.
4. Check multiple pole candidates and their antipodes.
5. Use external constraints or classical inversion to choose among them.

## Python API: Minimal Reference

### Main entry point

```python
result = analyze(
    epochs,
    object_id="asteroid_name",
    period_hours=None,
    fold=0,
    ensemble=False,
)
```

Notes:

- `fold` can be `0` through `4`
- `ensemble=True` averages predictions across all 5 fold checkpoints
- if `period_hours` is omitted, the package estimates the period first

For the full API, see [docs/API.md](API.md).

## Troubleshooting

### Import error

```bash
python -c "from lc_pipeline import analyze"
```

If this fails, reinstall with:

```bash
pip install -e .
```

### Checkpoint not found

The pre-trained checkpoints live in `lc_pipeline/checkpoints/`. If they were removed, the inference API cannot run.

### Results look unstable

Check the basics first:

- make sure the file columns are in the documented order
- make sure the geometry vectors are present
- make sure there are enough observations for period estimation
- make sure you inspect multiple returned candidates

### One dense lightcurve gives poor results

This is expected. The manuscript shows that multi-apparition coverage matters more than very dense sampling in one observing window.

## Advanced / Developer Topics

- [docs/ARCHITECTURE.md](ARCHITECTURE.md): model architecture overview
- [docs/MODULE_REFERENCE.md](MODULE_REFERENCE.md): module map
- [docs/DATA_FORMAT.md](DATA_FORMAT.md): schema details

Training:

- `train_pole_model.py` trains the GeoHierK3Transformer model using 5-fold cross-validation.
- See [docs/TRAINING_MULTIEPOCH.md](TRAINING_MULTIEPOCH.md) for multi-epoch training details.
