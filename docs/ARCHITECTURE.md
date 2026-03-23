# DeLPHI Architecture

Technical documentation of the DeLPHI system architecture, model design, and implementation. The Python package is imported as `lc_pipeline`.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Model Architecture](#model-architecture)
4. [Training Strategy](#training-strategy)
5. [Implementation Details](#implementation-details)
6. [Performance Characteristics](#performance-characteristics)

---

## System Overview

### Design Philosophy

lc_pipeline is a **hybrid classical-ML system** that combines:
- **Classical physics** for period estimation (Lomb-Scargle + Bayesian consensus)
- **Deep learning** for pole prediction (Hierarchical Transformer with K=3 output)

### Why Hybrid?

**Period Estimation**: Classical methods are robust, interpretable, and well-validated. ML adds no value here.

**Pole Prediction**: High-dimensional problem requiring pattern recognition across lightcurve shape, geometry, and physics. ML excels here.

### Two-Stage Pipeline

```
┌─────────────────────────┐
│  Stage 1: Period        │
│  (Classical Physics)    │
│                         │
│  • Lomb-Scargle        │
│  • Bayesian Posterior  │
│  • Multi-epoch Fusion  │
└───────────┬─────────────┘
            │
            ▼ Period P
┌─────────────────────────┐
│  Stage 2: Pole          │
│  (Deep Learning)        │
│                         │
│  • Tokenization        │
│  • Transformer Encoder │
│  • K=3 Pole Heads      │
│  • Unranked Output     │
└─────────────────────────┘
            │
            ▼ 9 Pole Candidates
```

**Advantages**:
- Independent optimization
- Use period from external sources
- Clear error attribution
- Interpretable period estimates

---

## Pipeline Architecture

### Stage 1: Period Estimation

**Module**: `lc_pipeline.period`

#### Components

```
LightcurveEpoch(s) → PeriodSearch → Posterior → ConsensusEngine → PeriodResult
```

**1. PeriodSearch** (`period_search.py`)
- Lomb-Scargle periodogram per epoch
- Frequency grid: [0.1, 100] hours
- Returns: posterior distribution P(period | data)

**2. BayesianPosterior** (`posterior.py`)
- Aggregates multi-epoch posteriors
- Product-of-experts fusion
- Credible intervals (95% CI)

**3. ConsensusEngine** (`consensus.py`)
- Final period selection
- Alias-aware merging
- Uncertainty quantification

#### Algorithm

```python
# For each epoch
posterior_i = lomb_scargle_posterior(epoch_i)

# Combine epochs (Bayesian product)
joint_posterior = product(posterior_1, posterior_2, ..., posterior_N)

# Extract period
period = argmax(joint_posterior)
uncertainty = effective_sigma(joint_posterior)
ci_low, ci_high = credible_interval(joint_posterior, level=0.95)
```

#### Output

```python
PeriodResult(
    period_hours=8.34,
    uncertainty_hours=0.12,
    ci_low_hours=8.10,
    ci_high_hours=8.58,
    n_epochs=3,
    success=True
)
```

---

### Stage 2: Pole Prediction

**Module**: `lc_pipeline.inference`

#### Components

```
Epochs + Period → Tokenizer → PolePredictor → PeriodForker → PoleInference
```

**1. Tokenizer** (`tokenizer.py`)

Converts raw lightcurve to structured input:

```python
# Input: List of epoch arrays (N_obs, 8)
# Output: Token tensor (n_windows, tokens_per_window, 13)

tokens = tokenize_lightcurve(
    epochs,
    period_hours=8.34,
    max_gap_days=1.0,      # CRITICAL: must be 1.0
    n_windows=8,           # CRITICAL: must be 8
    tokens_per_window=256  # CRITICAL: must be 256
)
```

**Tokenization Process**:
1. Phase-fold observations by period P
2. Split into observation windows (max_gap_days threshold)
3. Select up to n_windows windows
4. Sample tokens_per_window points per window
5. Compute 13 features per token:
   - Temporal (3): normalized time, delta time, MAD-normalized brightness
   - Geometry (6): reserved placeholders (always zero)
   - Period (4): rotations elapsed, log period, sin/cos rotation phase

**2. PolePredictor** (`model.py`)

Hierarchical Transformer model:

```
Input: (n_windows, tokens_per_window, 13)
    ↓
Token Projection (13 → d_model)
    ↓
Transformer Encoder (4 layers, 4 heads)
    ↓
Attention Pooling + Mean Pooling across windows
    ↓
K=3 Pole Heads (3 separate heads)
    ↓
Output: poles (3, 3)
```

**Architecture Details**:
- `d_model`: 128 (model dimension)
- `n_heads`: 4 (attention heads)
- `n_layers`: 4 (encoder layers)
- Activation: GELU
- No cross-window encoder (removed, it caused collapse)
- No quality head (candidates are unranked)

**3. PeriodForker** (`forking.py`)

Handles factor-of-2 photometric-rotation alias:

```python
# Run model 3 times with different periods
poles_base = model(tokens_P)     # Period P
poles_double = model(tokens_2P)  # Period 2P
poles_half = model(tokens_P/2)   # Period 0.5P

# Combine: 2-3 periods × 3 poles = 6-9 candidates
all_poles = [
    poles_base[0], poles_base[1], poles_base[2],
    poles_double[0], poles_double[1], poles_double[2],
    poles_half[0], poles_half[1], poles_half[2]
]
```

**Why?**: Asteroids with symmetric lightcurves have identical appearance at P and 2P. Model cannot distinguish without external constraints.

**4. PoleInference** (`pole.py`)

Orchestrates the full inference pipeline:

```python
def predict(tokens):
    # Forward pass
    poles = model(tokens)  # (3, 3) unranked pole candidates

    # Apply period forking
    all_poles = period_forker(poles, period)

    # Compute uncertainty
    uncertainty = compute_uncertainty(all_poles)

    return all_poles, uncertainty  # Candidates are unranked
```

---

## Model Architecture

### GeoHierK3Transformer

**File**: `lc_pipeline/models/geo_hier_k3_transformer.py`

```
┌─────────────────────────────────────────┐
│  Input: (B, W, T, F)                    │
│    B = batch size                       │
│    W = n_windows (8)                    │
│    T = tokens_per_window (256)          │
│    F = features (13)                    │
└────────────────┬────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│  Token Projection                       │
│  Linear(13 → 128) + LayerNorm          │
└────────────────┬────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│  Transformer Encoder                    │
│  TransformerEncoder(4 layers, 4 heads)  │
│  Activation: GELU                       │
│  Output: (B, W, T, 128)                 │
└────────────────┬────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│  Attention Pooling (per window)         │
│  + Mean Pooling across windows          │
│  Output: (B, 128)                       │
└────────────────┬────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│  K=3 Pole Heads                         │
│  3 × MLP (128 → 3)                     │
└────────────────┬────────────────────────┘
                 ▼
           poles (B, 3, 3)
```

### K=3 Pole Heads

**Why K=3?**
- Single pole: Often far from ground truth
- K=1: Oracle error ~40°
- K=3: Oracle error ~18° (mean), ~13° (median)
- K=5: Marginal improvement, 5× candidates

**Implementation**:
```python
class PoleHead(nn.Module):
    def __init__(self, d_model):
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # Output: (x, y, z)
        )

    def forward(self, x):
        pole = self.fc(x)
        # Normalize to unit vector
        pole = pole / torch.norm(pole, dim=-1, keepdim=True)
        return pole

# K=3 separate heads
self.pole_head_0 = PoleHead(d_model)
self.pole_head_1 = PoleHead(d_model)
self.pole_head_2 = PoleHead(d_model)
```

### No Quality Head (Production)

The production model does **not** include a quality head. All K=3 pole candidates are unranked. Selection is performed externally via oracle evaluation (minimum angular distance to ground truth) or domain-specific criteria.

Earlier versions included a quality head, but it was removed because oracle selection is external and the quality head added unnecessary complexity.

---

## Training Strategy

### Direct Training (Production v1.0)

**Single-stage training**:

```
All Epochs:   All heads trained simultaneously
              Direct optimization of pole prediction + quality scoring
```

**Why?**
- Simpler training pipeline
- Better end-to-end optimization
- Quality head learns naturally from oracle signal
- No curriculum complexity needed

### Loss Function

```python
loss = oracle_softmin(poles, gt_poles)
     + λ_div × diversity_loss(poles)
     + λ_var × batch_variance_loss(poles)
     + λ_con × contrastive_loss(poles)
```

**Components**:

1. **Oracle Softmin Loss**: Minimize distance to nearest GT pole
   ```python
   # For each sample, find closest GT pole to any predicted pole
   min_error = min over k in K, min over g in GT:
                   angular_distance(pole_k, gt_g)
   oracle_loss = mean(min_error)
   ```

2. **Diversity Loss**: Continuous exponential encouraging angular separation (sigma=15 deg)

3. **Batch Variance Loss** (lambda=5.0): Prevents collapse to constant predictions

4. **Contrastive Loss** (lambda=2.0): Encourages different inputs to produce different outputs

### Hyperparameters

**Standard Configuration**:
```python
epochs = 50
batch_size = 32
learning_rate = 3e-4
weight_decay = 1e-4
patience = 50  # Early stopping

# Loss weights
lambda_var = 5.0       # Batch variance
lambda_contrastive = 2.0  # Contrastive
div_sigma_deg = 15.0   # Continuous diversity
```

---

## Implementation Details

### Coordinate Systems

**Ecliptic Coordinates** (output):
- λ (lambda): Ecliptic longitude [0°, 360°)
- β (beta): Ecliptic latitude [-90°, +90°]

**Cartesian Coordinates** (internal):
- Unit vectors (x, y, z)
- Conversion:
  ```python
  x = cos(β) * cos(λ)
  y = cos(β) * sin(λ)
  z = sin(β)
  ```

**Equatorial J2000** (input geometry):
- Sun/Earth positions in J2000 frame
- Standard astronomical reference frame

### Checkpoints

**Location**: `lc_pipeline/checkpoints/`

**Format**:
```python
checkpoint = {
    "model_state_dict": {...},
    "config": {
        "d_model": 128,
        "n_heads": 4,
        ...
    },
    "metrics": {
        "oracle_error": 19.02,
        "epoch": 6,
    }
}
```

**Loading**:
```python
from lc_pipeline.inference.pole import PoleInference, PoleConfig

config = PoleConfig(fold=0)
pole_engine = PoleInference(config)
# Automatically loads lc_pipeline/checkpoints/fold_0.pt
```

### Critical Parameters

**These MUST match training**:

```python
# Tokenization
max_gap_days = 1.0       # NOT 1.5!
n_windows = 8            # NOT 6 or 10
tokens_per_window = 256  # NOT 128 or 512
features = 13            # 3 temporal + 6 geometry (zeros) + 4 period

# Model architecture
d_model = 128
n_heads = 4
n_layers = 4
```

Changing these breaks checkpoint compatibility.

---

## Performance Characteristics

### Computational Requirements

**Inference** (single asteroid):
- Time: ~0.5-2 seconds (GPU), ~5-10 seconds (CPU)
- Memory: ~500 MB (model + data)
- GPU: Not required, but speeds up by 5-10×

**Training** (174 asteroids, 2,987 epochs, multi-epoch):
- Time: ~5 min/fold (GPU), ~25 min total (5-fold CV)
- Memory: ~2 GB GPU VRAM
- GPU: Recommended (NVIDIA with CUDA)

### Validation Performance

**5-fold Cross-Validation** (174 QF>=3 DAMIT asteroids, 2,987 training epochs):

| Metric | Value | Notes |
|--------|-------|-------|
| Mean Oracle@K=3 | 19.02° ± 2.68° | Best of 3 poles (asteroid-level) |
| Pooled Median Oracle@K=3 | 16.61° | Robust estimate |
| ZTF External | 18.82° ± 1.02° | 163 asteroids |

Candidates are unranked (no quality head). Oracle error represents the best achievable with external selection.

### Scalability

**Batch Inference**:
```python
# Process 100 asteroids
# Sequential: ~100 seconds (GPU), ~1000 seconds (CPU)
# Batch (B=10): ~20 seconds (GPU), ~200 seconds (CPU)
```

**Training Data Scaling**:
- 100 asteroids: Oracle ~30°
- 174 asteroids (2,987 epochs): Oracle ~19° (mean), ~17° (median)
- 500+ asteroids: Oracle ~12-15° (estimated)

---

## Design Decisions

### Why Not End-to-End?

**Alternative**: Single model predicting period + pole

**Why we don't**:
1. Period estimation is well-solved classically
2. ML period estimates are less interpretable
3. Independent optimization is easier
4. Can use period from other sources

### Why K=3 and Not K=1?

**K=1 Oracle Error**: ~40°
**K=3 Oracle Error**: ~18° (mean), ~13° (median)
**K=5 Oracle Error**: ~16° (estimated)

**Trade-off**: K=3 gives good oracle improvement with manageable candidates.

### Why No Quality Head?

Earlier versions included a quality head for ranking candidates. It was removed because:
1. Oracle selection is performed externally (not by the model)
2. The quality head added complexity without improving oracle error
3. Anti-collapse losses (batch variance, contrastive) replaced the quality head's regularization role

---

## Future Improvements

### Potential Enhancements

1. **Larger Training Set**: 500+ DAMIT asteroids → ~12-15° oracle
2. **Physics Priors**: Incorporate rotation dynamics
3. **Multi-modal**: Combine with spectroscopy, radar
4. **Ensemble**: Average predictions across folds
5. **Uncertainty Calibration**: Better confidence estimates
6. **Geometry Integration**: Enable geometry features after retraining

### Known Limitations

1. **Period Alias**: Cannot resolve P vs 2P from lightcurve alone
2. **Data Dependency**: Trained on DAMIT only, generalization to other datasets varies
3. **No Candidate Ranking**: Candidates are unranked (no quality head)
4. **Geometry Features Disabled**: Geometry slots set to zero (active geometry tested but degrades performance)

---

## References

### Key Components

- **Lomb-Scargle**: Lomb (1976), Scargle (1982)
- **Transformers**: Vaswani et al. (2017)
- **Curriculum Learning**: Bengio et al. (2009)

### Related Work

- DAMIT Database: https://astro.troja.mff.cuni.cz/projects/damit/
- Asteroid pole inversion: Kaasalainen & Torppa (2001)

---

**For questions or contributions, see GitHub repository.**
