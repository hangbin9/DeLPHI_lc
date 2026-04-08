# DeLPHI

**Deep Learning Photometry-based Hypothesis Inference**

`lc_pipeline` is the Python package for DeLPHI, a two-stage pipeline for asteroid lightcurve analysis:

1. estimate the rotation period from photometry with a classical Lomb-Scargle based method
2. predict multiple candidate spin-axis directions with a Transformer model

The package is designed to help a user narrow the search space before running a full classical inversion workflow.

## Status

> **Note**: This repository is under active development. A companion paper describing the method is in preparation.

## Published Results

These are the results reported in [paper/manuscript.tex](paper/manuscript.tex) for the published configuration:

- **5-fold cross-validation on 174 DAMIT asteroids**:
  - mean oracle error: **19.02° ± 2.68°**
  - pooled median oracle error: **16.61°**
- **End-to-end pipeline with estimated periods**:
  - mean oracle error: **18.90°**
  - median oracle error: **17.25°**
- **Period estimation on the same 174 asteroids**:
  - median alias-aware relative error: **5.3%**
- **External validation on 163 ZTF asteroids**:
  - mean oracle error: **18.82° ± 1.02°**
  - median oracle error: **16.31°**
- **Runtime**:
  - approximately **0.3 s per asteroid** on a single RTX 4070 GPU

Important context:

- The model outputs **unranked hypotheses**. The package displays them sorted by a heuristic score, but the top candidate is not guaranteed to be the correct pole.
- The main paper metric is an **oracle** metric: it measures how good the best candidate is in hindsight.

## Quick Start

### 1. Install

```bash
git clone https://github.com/hangbin9/DeLPHI_lc.git
cd DeLPHI_lc
pip install -e .
```

### 2. Verify the install

```bash
python -c "from lc_pipeline import analyze, __version__; print(__version__)"
```

### 3. Run on the example file

```bash
python run_pole_prediction.py --input examples/asteroid_101.csv
```

This uses the pre-trained checkpoints in `lc_pipeline/checkpoints/`.

### 4. Run on your own CSV file

```bash
python run_pole_prediction.py --input my_asteroid.csv
```

Expected CSV columns:

```text
time_jd,brightness,sun_x,sun_y,sun_z,obs_x,obs_y,obs_z
```

If you already know the rotation period, you can skip period estimation:

```bash
python run_pole_prediction.py --input my_asteroid.csv --period 8.5
```

## Python Example

```python
import pandas as pd
from lc_pipeline import analyze

df = pd.read_csv("examples/asteroid_101.csv", comment="#")
epochs = [df.values]

result = analyze(epochs, "asteroid_101", fold=0)

print(f"Period: {result.period.period_hours:.2f} ± {result.period.uncertainty_hours:.2f} h")
print(f"Top candidate: λ={result.best_pole.lambda_deg:.1f}°, β={result.best_pole.beta_deg:.1f}°")
print(f"Returned candidates: {len(result.poles)}")
```

## What The Pipeline Returns

The package returns:

- a period estimate with uncertainty
- **6 to 9** pole candidates
- antipodal equivalents for each pole direction
- simple uncertainty summaries (`spread_deg`, `confidence`)

Why 6 to 9 candidates:

- the model predicts **3 pole hypotheses**
- inference tests the base period `P` and `2P`
- it also tests `P/2` when the base period is at least `8 h`

Practical guidance:

- inspect more than the top-ranked candidate
- treat the antipode as physically equivalent unless you have external constraints
- use external information, classical inversion, radar, or occultations to choose among candidates

## Minimum Data Guidance

From the manuscript and the production pipeline behavior:

- hard minimum per observing epoch for pole inference: **3 observations**
- hard minimum per observing epoch for period estimation: **10 observations**
- validated period range: **2.4 to 67.5 h**
- search range used by the pre-trained period estimator: **2 to 200 h**
- best results come from **multiple apparitions / observing epochs**

The manuscript’s degradation study shows that the pipeline works best when data cover multiple apparitions. A single dense night is usually less informative than sparser coverage across different viewing geometries.

## Documentation

- [docs/QUICKSTART.md](docs/QUICKSTART.md): fastest path to running inference
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md): step-by-step beginner guide
- [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md): input schema and examples
- [docs/API.md](docs/API.md): public Python API
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): technical overview of the production model
- [docs/MODULE_REFERENCE.md](docs/MODULE_REFERENCE.md): package/module map
- [examples/README.md](examples/README.md): example files and quick commands

## Training

- `run_pole_prediction.py` is the recommended entry point for inference.
- `train_pole_model.py` trains the GeoHierK3Transformer model using 5-fold cross-validation.
- See [docs/TRAINING_MULTIEPOCH.md](docs/TRAINING_MULTIEPOCH.md) for multi-epoch training details.

## Project Layout

```text
.
├── README.md
├── run_pole_prediction.py
├── train_pole_model.py
├── docs/
├── examples/
└── lc_pipeline/
```

## Citation

If you cite the package, cite the manuscript and describe the software as the `lc_pipeline` implementation of DeLPHI.
