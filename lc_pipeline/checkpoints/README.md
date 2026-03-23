# Model Checkpoints

This directory contains the production model checkpoints for asteroid pole prediction.

## Model Information

- **Model**: GeoHierK3Transformer (Single Transformer encoder + attention pooling + mean-pooling across windows)
- **Training**: 174 DAMIT asteroids (QF>=3), 5-fold cross-validation, multi-epoch training (2,987 epochs)
- **Performance**: 19.02° +/- 2.68° mean oracle error, 16.61° pooled median (asteroid-level)
- **ZTF External**: 18.82° +/- 1.06° on 163 asteroids
- **Parameters**: ~994K (d_model=128, n_heads=4, 4 layers, GELU)

## Files

- `fold_0.pt` - Validates on fold 0 (35 asteroids), oracle=19.51° mean, seed=777
- `fold_1.pt` - Validates on fold 1 (35 asteroids), oracle=14.88° mean, seed=777
- `fold_2.pt` - Validates on fold 2 (35 asteroids), oracle=18.32° mean, seed=777
- `fold_3.pt` - Validates on fold 3 (35 asteroids), oracle=22.05° mean, seed=42
- `fold_4.pt` - Validates on fold 4 (34 asteroids), oracle=20.34° mean, seed=777

## Usage

For end-to-end analysis, use the high-level pipeline API which handles checkpoint loading automatically:

```python
from lc_pipeline import analyze

result = analyze(epochs, "asteroid_1017", period_hours=8.5, fold=0)
```

To load a checkpoint directly:

```python
from lc_pipeline.inference.model import PolePredictor

model = PolePredictor.load('lc_pipeline/checkpoints/fold_0.pt')
```

## Model Architecture

- **Input**: Multi-window lightcurve tokens [W, T, F]
  - W = up to 8 windows
  - T = up to 256 tokens per window
  - F = 13 features (3 temporal + 6 geometry zeros + 4 period features)

- **Architecture**:
  - Token projection (F=13 -> d_model=128)
  - LayerNorm
  - Transformer encoder (4 layers, 4 heads, GELU)
  - Attention pooling per window
  - Mean pooling across windows
  - K=3 independent MLP slot heads -> 3 pole unit vectors

- **Output**: 3 unranked pole candidates (unit vectors on the sphere)
  - No quality head (candidates are not ranked by the model)
  - Evaluation uses oracle selection (best of K=3)

## Training

Loss: oracle_softmin(tau=5) + continuous_diversity(sigma=15) + batch_variance(lambda=5) + contrastive(lambda=2)

Hyperparameters: lr=3e-4, weight_decay=1e-4, batch_size=32, patience=50, CosineAnnealingLR
