# Asteroid Lightcurve Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

End-to-end pipeline for asteroid pole determination from multi-epoch photometric lightcurve observations using physics-informed deep learning.

**Performance**: **19.02 +/- 2.68 deg** mean oracle error, **16.61 deg** pooled median (5-fold CV, 174 DAMIT asteroids, 2,987 training epochs)

**ZTF External Validation**: **18.82 +/- 1.06 deg** on 163 asteroids

---

## Quick Start

### Installation

```bash
git clone https://github.com/hangbin9/lc_dl.git
cd lc_dl
pip install -e .
```

Verify the installation:

```bash
python -c "from lc_pipeline import analyze; print('Ready')"
```

**Requirements**: Python 3.8+, PyTorch 1.12+, ~70 MB disk space (includes 5 pre-trained checkpoints)

### Basic Usage

```python
from lc_pipeline import analyze
import numpy as np

# Load lightcurve data (DAMIT 8-column CSV format)
data = np.loadtxt("asteroid_101.csv", delimiter=',', skiprows=1)
epochs = [data]

# Run analysis (period estimation + pole prediction)
result = analyze(epochs, "asteroid_101", fold=0)

# Results
print(f"Period: {result.period.period_hours:.2f} +/- {result.period.uncertainty_hours:.2f} h")
print(f"Best pole: lambda={result.best_pole.lambda_deg:.1f} deg, beta={result.best_pole.beta_deg:.1f} deg")

# All 9 candidates (3 period aliases x 3 poles)
for i, pole in enumerate(result.poles, 1):
    print(f"  {i}. lambda={pole.lambda_deg:.1f}, beta={pole.beta_deg:.1f}, "
          f"P={pole.period_hours:.2f}h ({pole.alias})")
```

If you already know the rotation period, pass it directly to skip period estimation:

```python
result = analyze(epochs, "asteroid_101", period_hours=8.5, fold=0)
```

See [examples/](examples/) for sample input files.

---

## How It Works

This pipeline combines classical physics with deep learning in two stages:

**Stage 1: Period Estimation** (Classical Bayesian)
- Multi-epoch Lomb-Scargle periodogram analysis
- Product-of-experts posterior fusion across epochs
- Output: period with 95% credible interval
- Median error: 5.3% (alias-aware)

**Stage 2: Pole Prediction** (Deep Learning)
- Transformer neural network (d_model=128, 4 layers, 4 heads, ~994K parameters)
- 13-dimensional feature space per observation:
  - 3 temporal features (normalized time, time delta, normalized brightness)
  - 6 geometry slots (set to zeros by design)
  - 4 period features (rotation phase sin/cos, log period, cumulative rotations)
- K=3 pole hypotheses (unranked, evaluated by oracle selection)
- Period alias expansion: P, 2P, and P/2 (if P >= 8h) give 6-9 total candidates

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for technical details.

---

## Performance

### DAMIT Cross-Validation (174 QF>=3 Asteroids, 5-Fold)

| Metric | Value |
|--------|-------|
| **Oracle@K=3 Mean** | **19.02 deg** |
| **Oracle@K=3 Pooled Median** | **16.61 deg** |
| **Across-Fold Std** | **+/- 2.68 deg** |
| **ZTF External (163 asteroids)** | **18.82 +/- 1.06 deg** |

**Per-Fold Results**:

| Fold | Val Asteroids | Mean Oracle | Median Oracle |
|------|---------------|-------------|---------------|
| 0 | 35 | 19.51 deg | - |
| 1 | 35 | 14.88 deg | - |
| 2 | 35 | 18.32 deg | - |
| 3 | 35 | 22.05 deg | - |
| 4 | 34 | 20.34 deg | - |

### Period Estimation

| Metric | Value |
|--------|-------|
| Median alias-aware error | 5.3% |
| Success rate | 100% (174/174) |

### Computational

| Task | Time |
|------|------|
| Inference (single asteroid) | ~0.3s |
| Training (per fold, GPU) | ~5 min |
| Training epochs to convergence | median 6 (range 1-41) |

---

## Data Requirements

### Input Format

The pipeline accepts DAMIT-style lightcurve data as numpy arrays with 8 columns:

| Column | Description |
|--------|-------------|
| 0 | Julian Date (time) |
| 1 | Relative brightness |
| 2-4 | Sun position vector (x, y, z) |
| 5-7 | Observer position vector (x, y, z) |

Each observing epoch is a separate numpy array. Pass a list of epoch arrays to `analyze()`.

### Downloading DAMIT Data

Training and replication require data from the [DAMIT database](https://astro.troja.mff.cuni.cz/projects/damit/).

1. Visit https://astro.troja.mff.cuni.cz/projects/damit/
2. Download lightcurve data for asteroids with Quality Flag >= 3
3. Download spin solutions (pole orientations) for ground truth

Place the data in a `data/` directory (excluded from git by default):

```
data/
├── damit_csv_qf_ge_3/              # Lightcurve CSV files (one per asteroid)
│   ├── asteroid_101.csv
│   ├── asteroid_102.csv
│   └── ...
├── damit_spins_complete/            # Ground truth pole solutions (JSON)
│   ├── asteroid_101.json
│   ├── asteroid_102.json
│   └── ...
└── periods.json                     # Rotation periods: {"asteroid_101": 8.34, ...}
```

See [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for detailed data preparation instructions.

---

## Command-Line Scripts

### Inference

```bash
# Single asteroid
python run_pole_prediction.py --input examples/asteroid_101.csv

# With known period
python run_pole_prediction.py --input examples/asteroid_101.csv --period 8.5

# Batch processing
python run_pole_prediction.py --input-dir my_asteroids/ --output results.json --format json
```

### Training

```bash
# Train all 5 folds (production recipe, ~25 min on GPU)
python train_pole_model.py \
    --csv-dir data/damit_csv_qf_ge_3 \
    --spin-root data/damit_spins_complete \
    --period-json data/periods.json \
    --outdir checkpoints_new \
    --folds 0,1,2,3,4

# Single fold for testing
python train_pole_model.py \
    --csv-dir data/damit_csv_qf_ge_3 \
    --spin-root data/damit_spins_complete \
    --period-json data/periods.json \
    --outdir test_output \
    --folds 0 --epochs 5
```

### Tests

```bash
pytest tests/
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | Installation, usage, data preparation, troubleshooting |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Model architecture and design decisions |
| [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) | Input data schema and format reference |
| [docs/MODULE_REFERENCE.md](docs/MODULE_REFERENCE.md) | File-by-file module reference |
| [docs/API.md](docs/API.md) | Python API reference |
| [docs/TRAINING_MULTIEPOCH.md](docs/TRAINING_MULTIEPOCH.md) | Multi-epoch training details |

---

## Project Structure

```
lc_dl/
├── README.md                    # This file
├── setup.py                     # Package installation
├── requirements.txt             # Python dependencies
├── run_pole_prediction.py       # Command-line inference script
├── train_pole_model.py          # Command-line training script
├── *.ipynb                      # Colab notebooks
│
├── docs/                        # Documentation
├── examples/                    # Sample input files
├── tests/                       # Tests with bundled fixtures
│
└── lc_pipeline/                 # Main package
    ├── __init__.py              # Public API
    ├── schema.py                # Data schema (Pydantic)
    ├── inference/               # End-to-end inference pipeline
    ├── period/                  # Period estimation (Lomb-Scargle + Bayesian)
    ├── models/                  # Neural network (GeoHierK3Transformer)
    ├── training/                # Loss functions
    ├── evaluation/              # Metrics and evaluation
    ├── data/                    # Dataset classes
    ├── physics/                 # Geometry and coordinate transforms
    ├── converters/              # Data format conversion
    ├── utils/                   # Utilities
    ├── scripts/                 # Internal training/eval scripts
    └── checkpoints/             # Pre-trained models (5 folds, ~12 MB each)
```

See [docs/MODULE_REFERENCE.md](docs/MODULE_REFERENCE.md) for file-by-file documentation.

---

## Advanced Usage

### Period Estimation Only

```python
from lc_pipeline import ConsensusEngine

engine = ConsensusEngine()
# epochs: list of (N, 8) numpy arrays
period_result = engine.predict_single(epochs, "asteroid_101")
print(f"Period: {period_result['period']:.2f} h")
```

### Pole Prediction with Known Period

```python
from lc_pipeline import PoleInference

inf = PoleInference()
poles, quality = inf.predict(epochs, period_hours=8.5, fold=0)
# poles: list of 3 unit vectors [(x, y, z), ...]
```

### Using the Pipeline Class

```python
from lc_pipeline import LightcurvePipeline

pipeline = LightcurvePipeline()
result = pipeline.analyze(epochs, "asteroid_101", period_hours=8.5, fold=0)
```

---

## Known Limitations

1. **Geometry features disabled by design**: The 6 geometry slots in the 13-feature input are set to zeros. Experiments showed this performs better than active geometry on DAMIT data.

2. **Candidates are unranked**: The model outputs K=3 pole candidates without quality scores. Users should evaluate all candidates using external constraints (radar, occultations, etc.).

3. **Period alias ambiguity**: Symmetric lightcurves make P vs 2P indistinguishable. The pipeline returns candidates for multiple period aliases (6-9 total).

4. **Training data scope**: Validated on 174 DAMIT asteroids (QF>=3). Performance on other populations may vary.

5. **Pole latitude effect**: Equator-on asteroids (low ecliptic latitude |beta|) have inherently higher errors (~45 deg) due to geometric ambiguity. This is a physical limitation, not a model deficiency.

---

## Citation

```bibtex
@software{asteroid_lc_pipeline,
  title = {Asteroid Lightcurve Pipeline: Deep Learning for Pole Determination},
  author = {[Author Names]},
  year = {2026},
  url = {https://github.com/hangbin9/lc_dl},
  note = {19.02 +/- 2.68 deg mean oracle error on 174 DAMIT asteroids (5-fold CV)}
}
```

**DAMIT Database**: https://astro.troja.mff.cuni.cz/projects/damit/

---

## License

MIT License - see [LICENSE](LICENSE) file
