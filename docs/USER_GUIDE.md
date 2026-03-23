# DeLPHI User Guide

Complete guide for using DeLPHI (`lc_pipeline`) to predict asteroid rotation poles from lightcurve data.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Usage - Inference](#usage---inference)
4. [Usage - Training](#usage---training)
5. [Understanding Results](#understanding-results)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 30-Second Demo

```python
from lc_pipeline import analyze
import numpy as np

# Load your lightcurve data (N observations × 8 columns)
# Format: [time_jd, brightness, sun_x, sun_y, sun_z, obs_x, obs_y, obs_z]
epochs = [np.loadtxt("asteroid_data.csv", delimiter=',', skiprows=1)]

# Analyze
result = analyze(epochs, "my_asteroid", fold=0)

# Results
print(f"Period: {result.period.period_hours:.2f} h")
print(f"Pole: λ={result.best_pole.lambda_deg:.1f}°, β={result.best_pole.beta_deg:.1f}°")
```

That's it.

---

## Installation

### From Source

```bash
git clone https://github.com/hangbin9/DeLPHI_lc.git
cd DeLPHI_lc
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy, SciPy, pandas
- See `requirements.txt` for full list

### Verify Installation

```python
from lc_pipeline import analyze, __version__
print(f"lc_pipeline version: {__version__}")
```

---

## Usage - Inference

### Command-Line (Easiest)

Use the provided scripts for quick analysis:

```bash
# Single asteroid
python run_pole_prediction.py --input asteroid_101.csv

# Batch processing
python run_pole_prediction.py --input-dir asteroids/ --output results.json

# With known period
python run_pole_prediction.py --input asteroid_101.csv --period 8.5
```

Run `python run_pole_prediction.py --help` for all options.

### Python API

```python
from lc_pipeline import analyze
import pandas as pd

# Load DAMIT-style CSV
df = pd.read_csv("asteroid_101.csv", comment='#').dropna()
epoch = df.values  # (N, 8) array

# Analyze
result = analyze(
    epochs=[epoch],
    object_id="asteroid_101",
    period_hours=None,  # Auto-estimate, or provide known period
    fold=0  # Which checkpoint to use (0-4)
)

# Access results
print(f"Period: {result.period.period_hours:.2f} ± {result.period.uncertainty_hours:.2f} h")
print(f"Best pole: λ={result.best_pole.lambda_deg:.1f}°, β={result.best_pole.beta_deg:.1f}°")
print(f"Quality score: {result.best_pole.score:.3f}")
print(f"Confidence: {result.uncertainty.confidence:.2f}")

# All candidates (6-9 total: 2-3 period aliases × 3 poles)
for i, pole in enumerate(result.poles, 1):
    print(f"{i}. λ={pole.lambda_deg:.1f}°, β={pole.beta_deg:.1f}°, "
          f"alias={pole.alias}, score={pole.score:.3f}")
```

### Jupyter Notebook (Google Colab)

Open `run_pole_prediction.ipynb` for a complete interactive tutorial:
- Works with synthetic data (no downloads)
- Visualization of results
- Batch processing examples
- Export to JSON

---

## Usage - Training

### Command-Line Training

```bash
python train_pole_model.py \
    --csv-dir data/damit_csv_qf_ge_3 \
    --spin-root data/damit_spins_complete \
    --period-json data/periods.json \
    --outdir checkpoints_new \
    --folds 0,1,2,3,4
```

### Hyperparameters

- `--epochs`: Maximum training epochs (default: 200)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 3e-4)
- `--patience`: Early stopping patience (default: 50)
- `--seed`: Random seed (default: 777)
- `--device`: `cuda` or `cpu` (default: cuda)

Model architecture (d_model=128, 4 heads, 4 layers) and loss weights are fixed to production values. See `lc_pipeline/scripts/train_k3.py` for the full set of tunable parameters.

### Jupyter Notebook Training

Open `train_pole_model.ipynb` for interactive training in Google Colab:
- Google Drive integration
- Real-time progress bars
- Training curve visualization
- Checkpoint management

### Data Requirements

**For Training**, you need:

1. **Lightcurve CSV files** (one per asteroid):
   ```csv
   time,mag,x,y,z,dx,dy,dz
   2433827.771536,0.9882,-1.524,2.561,-1.654,-1.376,1.555,-1.655
   ```

2. **Ground truth JSON** (all asteroids):
   ```json
   {
     "asteroid_101": {
       "poles": [[0.35, -0.19, 0.92]],
       "period_hours": 8.34
     }
   }
   ```

See [DATA_FORMAT.md](DATA_FORMAT.md) for complete format specification.

---

## Understanding Results

### What You Get

The `analyze()` function returns:

```python
result = AnalysisResult(
    object_id="asteroid_101",
    period=PeriodResult(...),      # Period estimation
    poles=[...],                    # 6-9 candidates (sorted by score)
    best_pole=PoleCandidate(...),  # Top-scoring pole
    uncertainty=PoleUncertainty(...) # Confidence metrics
)
```

### Period Estimation

```python
result.period.period_hours        # Estimated period (hours)
result.period.uncertainty_hours   # 1-sigma uncertainty
result.period.ci_low_hours        # 95% CI lower bound
result.period.ci_high_hours       # 95% CI upper bound
```

**Method**: Multi-epoch Lomb-Scargle + Bayesian consensus

### Pole Predictions

The model outputs **6-9 candidates** (depending on period):
- 3 period aliases (base P, double 2P, half 0.5P)
- 3 poles per alias (K=3)

```python
for pole in result.poles:
    pole.lambda_deg     # Ecliptic longitude [0, 360)
    pole.beta_deg       # Ecliptic latitude [-90, 90]
    pole.xyz            # Cartesian unit vector
    pole.period_hours   # Period used for this pole
    pole.alias          # "base", "double", "half"
    pole.score          # Quality score [0, 1]
    pole.slot           # K=3 slot index
```

### Candidate Selection

The production model does **not** include a quality head. All K=3 candidates are unranked by the model. Selection among candidates requires external evaluation (oracle selection against ground truth, or domain-specific criteria).

### Uncertainty Metrics

```python
result.uncertainty.spread_deg    # Angular spread of top candidates
result.uncertainty.confidence    # Quality-based confidence [0, 1]
```

- **Low spread (<20°)**: Poles are clustered → likely convergence
- **High confidence (>0.7)**: Quality head is confident (but can be wrong!)

### Performance Expectations

**From 5-fold cross-validation on 174 DAMIT asteroids:**

| Metric | Value | What It Means |
|--------|-------|---------------|
| **Oracle@K=3 Mean** | 19.02° ± 2.68° | Best of 3 poles (if you knew which) |
| **Oracle@K=3 Median** | 16.61° | Robust central estimate |

**Key Insight**: Candidates are unranked. Oracle error represents the best achievable if the correct pole is selected externally.

---

## Examples

### Example 1: Simple Analysis

```python
from lc_pipeline import analyze
import pandas as pd

# Load data
df = pd.read_csv("examples/asteroid_101.csv", comment='#').dropna()
epochs = [df.values]

# Analyze
result = analyze(epochs, "asteroid_101", fold=0)

# Print results
print(f"Period: {result.period.period_hours:.2f} h")
print(f"Pole: λ={result.best_pole.lambda_deg:.1f}°, β={result.best_pole.beta_deg:.1f}°")
```

### Example 2: With Known Period

```python
# Skip period estimation if you already know it
result = analyze(
    epochs=epochs,
    object_id="asteroid_101",
    period_hours=8.34,  # Known from external source
    fold=0
)
```

### Example 3: Batch Processing

```python
from pathlib import Path
import json

results = {}

for csv_file in Path("asteroids/").glob("*.csv"):
    object_id = csv_file.stem
    df = pd.read_csv(csv_file, comment='#').dropna()

    result = analyze([df.values], object_id, fold=0)

    results[object_id] = {
        "period": result.period.period_hours,
        "pole_lambda": result.best_pole.lambda_deg,
        "pole_beta": result.best_pole.beta_deg,
        "score": result.best_pole.score,
    }

# Save
with open("batch_results.json", 'w') as f:
    json.dump(results, f, indent=2)
```

### Example 4: Using Different Folds

```python
# Try multiple folds for ensemble
folds_results = []

for fold in range(5):
    result = analyze(epochs, "asteroid_101", fold=fold)
    folds_results.append(result.best_pole)

# Ensemble: average predictions
import numpy as np
avg_lambda = np.mean([p.lambda_deg for p in folds_results])
avg_beta = np.mean([p.beta_deg for p in folds_results])

print(f"Ensemble pole: λ={avg_lambda:.1f}°, β={avg_beta:.1f}°")
```

### Example 5: Visualization

```python
import matplotlib.pyplot as plt

# Plot all candidates
fig, ax = plt.subplots(figsize=(10, 6))

for i, pole in enumerate(result.poles, 1):
    color = 'red' if i == 1 else 'blue'
    size = 100 if i == 1 else 30
    ax.scatter(pole.lambda_deg, pole.beta_deg,
               s=size, c=color, alpha=0.7, edgecolors='black')
    ax.text(pole.lambda_deg + 5, pole.beta_deg + 5, str(i), fontsize=8)

ax.set_xlabel('Ecliptic Longitude λ (°)')
ax.set_ylabel('Ecliptic Latitude β (°)')
ax.set_xlim(0, 360)
ax.set_ylim(-90, 90)
ax.grid(True, alpha=0.3)
ax.set_title(f'Pole Candidates: {result.object_id}')
plt.show()
```

---

## Troubleshooting

### Import Error: "No module named 'lc_pipeline'"

**Solution**: Install the package first:
```bash
pip install -e /path/to/lc_pipeline
```

### Period Estimation Fails

**Symptoms**: `result.period.success = False`

**Causes**:
- Too few observations (<50)
- Very noisy data
- Irregular sampling

**Solution**:
- Provide known period with `period_hours=X.X`
- Use more epochs
- Check data quality

### All Poles Have Same Score

**Symptoms**: `result.best_pole.score ≈ result.poles[1].score`

**Meaning**: The production model does not include a quality head, so scores may not be meaningful for ranking. Candidates are unranked by design.

**Solution**: This is expected. Use oracle selection or domain-specific criteria to choose among candidates.

### CUDA Out of Memory

**Solution**: The model will automatically fall back to CPU. If you want to force CPU:
```python
# Set device before importing
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from lc_pipeline import analyze
```

### NaN Results

**Cause**: Invalid input data (negative brightness, non-finite values)

**Solution**: Clean your data:
```python
df = df.dropna()  # Remove missing values
df = df[df.iloc[:, 1] > 0]  # Remove negative brightness
```

### Results Don't Match Ground Truth

**This is expected!** Remember:
- Oracle@K=3 error is 19.02° mean, 16.61° median (best case)
- Candidates are unranked (no quality head)
- Some asteroids are inherently difficult (especially equator-on geometries)
- Model is not perfect

**What to do**:
1. Check if ground truth is in the candidates (oracle evaluation)
2. Try different folds (ensemble)
3. Validate with external data if critical

---

## Best Practices

### Data Preparation

1. **Quality**: Use QF≥3 data for training
2. **Coverage**: More epochs = better results
3. **Sampling**: Uniform time sampling helps
4. **Units**: Ensure brightness is normalized (not magnitude!)

### Inference

1. **Checkpoints**: Try multiple folds, use ensemble
2. **Validation**: Always check uncertainty metrics
3. **Critical use**: Validate with external data
4. **Interpretation**: Remember oracle vs selector gap

### Training

1. **Start simple**: Use default hyperparameters first
2. **Monitor**: Watch for NaN, exploding gradients
3. **Patience**: Training takes time (~50-100 epochs to converge)
4. **Validation**: Check oracle error on validation set

---

## Getting Help

- **Documentation**: Check [API.md](API.md), [DATA_FORMAT.md](DATA_FORMAT.md), [ARCHITECTURE.md](ARCHITECTURE.md)
- **Examples**: See `examples/` directory and notebooks
- **Issues**: Report bugs on GitHub
- **Questions**: Open a discussion on GitHub

---

## What's Next?

1. Try the [inference examples](#examples)
2. Run the [Jupyter notebooks](../run_pole_prediction.ipynb)
3. Train your own model with custom data
4. Check [ARCHITECTURE.md](ARCHITECTURE.md) for technical details

For technical details, see [ARCHITECTURE.md](ARCHITECTURE.md).
