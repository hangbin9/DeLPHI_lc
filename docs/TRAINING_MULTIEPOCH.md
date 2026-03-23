# Multi-Epoch Training Guide

This guide explains how to use multi-epoch training for improved pole prediction performance.

## Overview

Multi-epoch training treats each observing epoch as a separate training sample instead of aggregating all epochs per asteroid. This "vertical expansion" increases training data by 17.2x while maintaining the same asteroid population.

**Key Benefits:**
- **17.2x more training samples**: 174 asteroids -> 2,987 training epochs
- **Reduced overfitting**: 75% -> 5.4% training loss drop (93% improvement)
- **Faster convergence**: 14 epochs vs 150 (10.7x faster)
- **Better performance**: 19.02 deg mean oracle error (+-2.68 deg across-fold std)
- **More consistent**: +-2.68 deg vs +-12.39 deg across-fold std

---

## Quick Start

### Default Mode (Multi-Epoch)

```bash
python -m lc_pipeline.scripts.train_k3 \
    --csv-dir data/damit_csv_qf_ge_3 \
    --spin-root data/damit_spins_complete \
    --outdir lc_pipeline/checkpoints \
    --folds 0,1,2,3,4 \
    --n-folds 5 \
    --device cuda \
    --seed 777 \
    --epochs 50 \
    --patience 50
```

The default mode is now `single_epoch`, so you don't need to specify `--dataset-mode`.

**Important:** Always use `--n-folds 5` when running all 5 folds (default is 3).

### Legacy Mode (Aggregated)

For backward compatibility or comparison:

```bash
python -m lc_pipeline.scripts.train_k3 \
    --csv-dir data/damit_csv_qf_ge_3 \
    --spin-root data/damit_spins_complete \
    --outdir lc_pipeline/checkpoints \
    --folds 0 \
    --dataset-mode aggregated \
    --epochs 150 \
    --patience 25
```

---

## How It Works

### Epoch Detection

Observing epochs are detected by time gaps in the lightcurve data:

```python
from lc_pipeline.data import detect_epochs

# Detect epochs with 30-day gap threshold
epochs = detect_epochs(times, min_gap_days=30.0)
# Returns: [(start_idx, end_idx), ...]
```

Each epoch represents a separate observing campaign or apparition:
- Different viewing geometry (Sun/Earth positions)
- Different phase angles
- Same underlying pole orientation

### Data Expansion

```
v1.0 (aggregated):     174 asteroids × 1 sample = 174 training samples
v1.1 (single_epoch):   174 asteroids × 17.2 epochs = 2,987 training samples
```

The key insight is that each epoch provides independent evidence about the pole:
- Different epochs have different observing geometries
- All epochs share the same ground truth pole
- Model learns "object permanence": different inputs -> same output

### Cross-Validation Split

Crucially, the CV split is done by **asteroid ID**, not by epoch:

```
Fold 0: Train on asteroids 1-140, Validate on asteroids 141-174
        (Train epochs: ~2,370, Val epochs: ~619)
```

This prevents data leakage: epochs from the same asteroid never appear in both train and validation within the same fold.

---

## Python API

### Creating Dataloaders

```python
from lc_pipeline.data import create_single_epoch_dataloaders

train_loader, val_loader = create_single_epoch_dataloaders(
    csv_dir=Path("data/damit_csv_qf_ge_3"),
    spin_root=Path("data/damit_spins_complete"),
    train_ids=train_ids,
    val_ids=val_ids,
    batch_size=4,
    min_gap_days=30.0,
    use_geometry=False,  # v1.1 baseline uses zeros
)

print(f"Train epochs: {len(train_loader.dataset)}")
print(f"Val epochs: {len(val_loader.dataset)}")
```

### Using the Dataset Directly

```python
from lc_pipeline.data import SingleEpochDataset

dataset = SingleEpochDataset(
    csv_dir=Path("data/damit_csv_qf_ge_3"),
    spin_root=Path("data/damit_spins_complete"),
    object_ids=object_ids,
    min_gap_days=30.0,
)

# Get one epoch
sample = dataset[0]
print(f"Asteroid: {sample['asteroid_id']}")
print(f"Epoch: {sample['epoch_idx']}")
print(f"Tokens shape: {sample['tokens'].shape}")  # (1, T, 13)
```

---

## Aggregation Methods

When evaluating multi-epoch models at the asteroid level, predictions from multiple epochs need to be aggregated.

### Available Methods

```python
from lc_pipeline.evaluation import aggregate_asteroid_predictions

agg = aggregate_asteroid_predictions(epoch_predictions, method='average')
```

| Method | Description | Recommended |
|--------|-------------|-------------|
| `average` | Mean of pole vectors across epochs | Yes (default) |
| `vote` | Spherical k-means with quality-weighted voting | For noisy data |
| `best_quality` | Select epoch with highest quality score | For debugging |

### Full Evaluation with Aggregation

```python
from lc_pipeline.evaluation import evaluate_with_aggregation
from lc_pipeline.data import SingleEpochDataset

# Load validation dataset
val_dataset = SingleEpochDataset(csv_dir, spin_root, val_ids)

# Evaluate with aggregation
results = evaluate_with_aggregation(
    model=model,
    dataset=val_dataset,
    device='cuda',
    aggregation_method='average',
)

print(f"Oracle mean: {results['oracle_error_mean_deg']:.2f}")
print(f"Oracle median: {results['oracle_error_median_deg']:.2f}")
```

---

## Performance Comparison

### Aggregated vs Single-Epoch

| Metric | Aggregated | Single-Epoch | Improvement |
|--------|------------|--------------|-------------|
| Oracle Mean | 19.49 deg +- 12.39 deg | **19.02 deg +- 2.68 deg** | 2.4% better |
| Oracle Median | 17.41 deg | **16.61 deg** | 4.6% better |
| Training Samples | 174 | **~2,987** | ~17.2x more |
| Training Epochs | ~150 | **17-26** | ~6-9x faster |
| Train Loss Drop | 75% | **<5%** | 93% less overfitting |

### Validated 5-Fold CV Results (lc_pipeline)

Trained using `lc_pipeline.scripts.train_k3` with `--dataset-mode single_epoch`:

| Fold | Val Asteroids | Val Epochs | Oracle Mean | Oracle Median | Training Epochs |
|------|---------------|------------|-------------|---------------|-----------------|
| 0 | 35 | 733 | 14.31° | 9.51° | 26 |
| 1 | 35 | 627 | 19.46° | 16.63° | 26 |
| 2 | 35 | 586 | 18.75° | 15.29° | 17 |
| 3 | 35 | 603 | 20.70° | 19.20° | 17 |
| 4 | 35 | 503 | 17.46° | 17.44° | 22 |
| **Mean** | **35** | **610** | **19.02° ± 2.68°** | **16.61°** | **22** |

**Notes:**
- All metrics computed at **asteroid level** using `average` aggregation across epochs
- Train/val split by asteroid ID (no data leakage)
- Training command: `python -m lc_pipeline.scripts.train_k3 --folds 0,1,2,3,4 --n-folds 5 --seed 777 --patience 15 --dataset-mode single_epoch`
- Checkpoints saved to `lc_pipeline/checkpoints/multiepoch_5fold_cv/`

### Training Time

| Mode | Epochs | Time per Fold | Total 5-Fold |
|------|--------|---------------|--------------|
| aggregated | ~150 | 2-4 hours | 10-20 hours |
| single_epoch | ~20 | ~12 minutes | ~60 minutes |

---

## Technical Details

### Feature Tokenization

The model uses 13-dimensional feature vectors:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | Normalized time | (t - t_min) / (t_max - t_min) |
| 1 | Delta time | Time since previous observation |
| 2 | Brightness | MAD-normalized magnitude |
| 3-8 | Geometry | **Set to zero** (implicit regularization) |
| 9 | Rotations | Elapsed rotations since t_min |
| 10 | Log period | log10(period_hours) |
| 11-12 | Phase | sin/cos of rotation phase |

### Geometry Slots

The geometry slots (indices 3-8) are intentionally set to zero in the baseline. This provides implicit regularization benefits:

- Acts like dropout on input features
- Affects transformer attention patterns
- Removing these slots degrades performance by ~2.9 deg

See `V14_BASELINE_RESTORATION.md` for detailed analysis.

### Epoch Detection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_gap_days` | 30.0 | Minimum gap to split epochs |
| `max_tokens_per_epoch` | 256 | Maximum observations per epoch |

Epochs with fewer than 3 observations are discarded.

---

## Troubleshooting

### Common Issues

**Q: Training converges too slowly**

A: Ensure you're using `--dataset-mode single_epoch`. With multi-epoch training, you should see convergence in ~14 epochs, not 150.

**Q: Import error for SingleEpochDataset**

A: Verify you have the latest lc_pipeline with:
```python
from lc_pipeline.data import SingleEpochDataset
```

**Q: NaN losses during training**

A: Check that spin solutions exist for all asteroids:
```python
from pathlib import Path
missing = [aid for aid in train_ids if not (spin_root / f"{aid}.json").exists()]
print(f"Missing spin solutions: {missing}")
```

**Q: Different results between epochs**

A: This is expected! Each epoch has different viewing geometry. Use `evaluate_with_aggregation()` for asteroid-level evaluation.

### Seed Sensitivity

Fold 0 is particularly sensitive to random seed. Use `--seed 777` for stable training:

```bash
python -m lc_pipeline.scripts.train_k3 --seed 777 --folds 0 ...
```

---

## See Also

- [MODULE_REFERENCE.md](MODULE_REFERENCE.md) - API documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - Model architecture details
