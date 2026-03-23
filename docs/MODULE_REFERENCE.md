# DeLPHI Module Reference

Complete file-by-file documentation of the `lc_pipeline` package.

48 Python files. Performance: 19.02° +/- 2.68° mean, 16.61° median oracle error (5-fold CV, 174 DAMIT asteroids, 2,987 training epochs).

---

## Package Structure

```
lc_pipeline/
├── __init__.py              # Package exports
├── schema.py                # Unified data schema (Pydantic)
├── probes.py                # Model conditioning probes
├── version.py               # Version info
├── example_usage.py         # Usage examples
│
├── checkpoints/             # Pre-trained model weights (5 folds)
├── converters/              # Data format converters
├── data/                    # Dataset classes & loaders
├── evaluation/              # Metrics & evaluation
├── inference/               # End-to-end inference pipeline
├── models/                  # Neural network architectures
├── period/                  # Period estimation (classical)
├── physics/                 # Physics & geometry utilities
├── scripts/                 # Training scripts
├── training/                # Loss functions & training utils
└── utils/                   # General utilities
```

---

## Core Files (5 files)

### `__init__.py`
Package entry point exporting the public API: `analyze()`, `LightcurvePipeline`, `ConsensusEngine`, `LightcurveData`, `PoleConfig`.

### `schema.py`
Pydantic data schema defining the unified lightcurve format. Includes `Observation`, `Epoch`, `PoleSolution`, `GroundTruth`, and `LightcurveData` classes with validation for unit vectors and optional ground truth.

### `probes.py`
Conditioning probes to verify model produces different outputs for different inputs. Includes `inter_object_diversity_probe()` and `run_conditioning_probes()`.

### `version.py`
Package version string: `__version__ = "1.1.0"`. Includes metadata about v1.1 multi-epoch baseline performance.

### `example_usage.py`
Runnable usage examples demonstrating inference, period estimation, and training workflows.

---

## inference/ - End-to-End Pipeline (9 files)

### `inference/__init__.py`
Module exports for inference components.

### `inference/pipeline.py`
Main analysis pipeline combining period estimation and pole prediction. Contains `LightcurvePipeline` class and `analyze()` convenience function.

### `inference/tokenizer.py`
**Canonical tokenizer** for both training and inference. Converts lightcurve data to model input tokens. Creates **13-dimensional feature vectors**:
- **Temporal features** (3): normalized time (0), delta time (1), MAD-normalized brightness (2)
- **Geometry features** (6): reserved placeholders (3-8, **always set to zero** for implicit regularization)
- **Period features** (4): rotations elapsed (9), log period (10), sin/cos rotation phase (11-12)

The training dataset imports this module to ensure train/inference consistency. Geometry slots are zeroed by design to provide regularization benefits.

### `inference/forking.py`
Period alias forking for pole prediction. `PeriodForker` class predicts poles for {P, 2P, 0.5P} aliases, producing 6-9 candidates (3 poles × 2-3 aliases; 0.5P only if P ≥ 8h).

### `inference/pole.py`
Pole prediction using trained transformer model. `PolePredictor` wrapper for loading checkpoints and running inference.

### `inference/model.py`
Model loading utilities. `load_model()` function handles checkpoint loading with version compatibility.

### `inference/coordinates.py`
Coordinate transformations between Cartesian, ecliptic (λ, β), and equatorial J2000 reference frames.

### `inference/uncertainty.py`
Uncertainty quantification for pole predictions. Computes spread (dispersion of K candidates) and confidence metrics.

### `inference/schema.py`
Output schema defining `PeriodResult`, `PoleCandidate`, `UncertaintyMetrics`, and `AnalysisResult` dataclasses.

---

## period/ - Period Estimation (5 files)

### `period/__init__.py`
Module exports for period estimation.

### `period/consensus.py`
Multi-epoch Bayesian period estimation. `ConsensusEngine` class aggregates posteriors across epochs to produce robust period estimates with uncertainty.

**Performance** (174 DAMIT asteroids):
- Median alias-aware error: **5.3%**
- Success rate: 100% (174/174)
- Accuracy @10%: 55%
- Accuracy @20%: 72%

### `period/period_search.py`
Lomb-Scargle periodogram analysis. `lomb_scargle_posterior()` computes period posterior for single epoch, `find_peaks()` identifies significant peaks.

### `period/posterior.py`
Bayesian posterior aggregation using product-of-experts fusion across epochs.

### `period/config.py`
Configuration for period estimation including search range, resolution, and peak detection thresholds.

**Default settings** (production v1.0):
- Period range: 2-200 hours (covers 99.9% of known asteroids)
- Grid resolution: 20,000 frequency points (~0.5% resolution at 200h)
- For rare slow rotators (P > 1000h), increase `max_period_hours` and `n_freq` proportionally

---

## models/ - Neural Networks (2 files)

### `models/__init__.py`
Module exports for model architectures.

### `models/geo_hier_k3_transformer.py`
Main production model: Transformer with K=3 pole output. 128-dimensional, 4 layers, ~994K parameters. Produces 3 unranked pole candidates. No cross-window encoder (removed, it caused collapse). No quality head.

**Input**: 13-dimensional feature vectors (period-aware tokenization)
**Output**: K=3 pole hypotheses (Cartesian unit vectors), unranked
**Performance**: 19.02° ± 2.68° mean, 16.61° median oracle@K=3 error on 174 DAMIT asteroids

---

## data/ - Dataset Classes (9 files)

### `data/__init__.py`
Module exports for data loading: `DamitMultiEpochDataset`, `SingleEpochDataset`, `create_dataloaders`, `create_single_epoch_dataloaders`, `detect_epochs`, `load_lightcurve_csv`.

### `data/damit_multiepoch_dataset.py`
Legacy PyTorch Dataset for DAMIT CSV data. Aggregates all epochs per asteroid into a single training sample. Uses the canonical tokenizer from `inference/tokenizer.py` to ensure train/inference consistency.

### `data/single_epoch_dataset.py` (NEW)
**Multi-epoch training dataset** where each observing epoch is a separate training sample. Enables 17.2x data expansion (174 asteroids → 2,987 training epochs) while maintaining the same asteroid population.

**Key Functions**:
- `detect_epochs(times, min_gap_days=30.0)` - Detect observing epochs by time gaps
- `load_spin_solution(asteroid_id, spin_root)` - Load ground truth pole vectors
- `SingleEpochDataset` - Main dataset class
- `create_single_epoch_dataloaders()` - Factory for train/val loaders
- `collate_fn()` - Custom collation for variable-length sequences

**Benefits over aggregated mode**:
- 17.2x more training samples
- 93% reduction in overfitting (75% → 5.4% train loss drop)
- 10.7x faster convergence (14 vs 150 epochs)
- Production performance: 19.02° ± 2.68° mean oracle

### `data/loaders.py`
Data loading utilities. `LightcurveEpoch` and `AsteroidLightcurves` container classes for organizing observations.

### `data/folds.py`
Cross-validation fold management. `load_fold_split()` returns train/val object IDs for a given fold.

### `data/augmentation.py`
Data augmentation for training: brightness scaling, time jittering, and epoch shuffling.

### `data/split_index.py`
Index utilities for managing train/validation splits.

### `data/stratified_group_kfold.py`
Stratified group k-fold splitting ensuring same asteroid never appears in both train and val within a fold.

### `data/damit_path_resolver.py`
Path resolution utilities for finding DAMIT data files.

---

## training/ - Training Utilities (2 files)

### `training/__init__.py`
Module exports for training utilities.

### `training/losses_axisnet.py`
Main loss functions for pole prediction. `combined_loss_k3()` combines oracle loss (softmin over K poles), diversity loss (encourage pole separation), and quality loss (selector supervision).

---

## evaluation/ - Metrics (5 files)

### `evaluation/__init__.py`
Module exports for evaluation: `evaluate_fold`, `aggregate_folds`, `aggregate_asteroid_predictions`, `evaluate_with_aggregation`.

### `evaluation/eval_axisnet.py`
Main evaluation functions. `evaluate_fold()` for single fold, `aggregate_folds()` for cross-validation metrics.

### `evaluation/aggregation.py` (NEW)
**Asteroid-level prediction aggregation** for multi-epoch training. Since multi-epoch training predicts on individual epochs, this module aggregates predictions per asteroid for evaluation.

**Key Functions**:
- `aggregate_asteroid_predictions(epoch_predictions, method='average')` - Combine predictions from multiple epochs
- `evaluate_with_aggregation(model, dataset, device, aggregation_method)` - Full evaluation with asteroid-level metrics

**Aggregation Methods**:
- `'average'` (default): Mean of pole vectors across epochs
- `'vote'`: Spherical k-means clustering with quality-weighted voting
- `'best_quality'`: Select epoch with highest quality score

### `evaluation/metrics.py`
Core metric functions: `angular_distance()` for great-circle distance, `oracle_at_k()` for minimum error among K predictions, `alias_aware_relative_error()` for period evaluation accounting for P/2P/0.5P aliases.

### `evaluation/pole_metrics.py`
Pole-specific metrics: `pole_accuracy()` within angular threshold, `pole_coverage()` sky fraction covered by K poles, **`angle_deg_antipode()`** for scientifically rigorous angular distance with antipodal symmetry (treats North/South poles as equivalent via `arccos(|cos_angle|)`).

---

## physics/ - Physics Utilities (6 files)

### `physics/__init__.py`
Module exports for physics utilities.

### `physics/geometry.py`
Geometric calculations: `compute_phase_angle()` (Sun-asteroid-observer), `compute_scattering_angle()`, `compute_aspect_angle()` (observer viewing angle).

### `physics/frames.py`
Reference frame transformations between J2000 equatorial and ecliptic coordinates.

### `physics/alias_resolver.py`
Period alias handling. Identifies P, 2P, 0.5P aliases and computes alias-aware error.

### `physics/spherical_grid.py`
Spherical grid utilities: HEALPix grid generation and Fibonacci spiral sampling for uniform sky coverage.

### `physics/rot_utils.py`
Rotation utilities for coordinate transformations.

---

## converters/ - Data Conversion (3 files)

### `converters/__init__.py`
Module exports for converters.

### `converters/unified_loader.py`
Load and save unified schema data. `load_unified_json()`, `load_unified_csv()`, `save_unified_json()`, `save_unified_csv()`.

### `converters/damit_to_unified.py`
Convert DAMIT format to unified schema. `convert_damit_to_unified()` for single file, `batch_convert_damit()` for directory.

---

## scripts/ - Training Scripts (3 files)

### `scripts/__init__.py`
Module exports for scripts.

### `scripts/train_k3.py`
Main training script for K=3 model with curriculum learning. Supports both aggregated and single_epoch dataset modes.

**Key Arguments**:
- `--dataset-mode`: `single_epoch` (default, recommended) or `aggregated` (legacy)
- `--min-gap-days`: Epoch detection gap threshold (default 30.0 days)
- `--use-geometry`: Use active geometry features (default: zeros)

### `scripts/cv_eval.py`
Cross-validation evaluation script. Evaluates all 5 folds and reports aggregate metrics.

---

## utils/ - General Utilities (8 files)

### `utils/__init__.py`
Module exports for utilities.

### `utils/checkpoints.py`
Checkpoint management: `save_checkpoint()` and `load_checkpoint()`.

### `utils/freeze_utils.py`
Parameter freezing for curriculum learning: `freeze_quality_head()`, `unfreeze_quality_head()`.

### `utils/plotting.py`
Visualization: `plot_poles()` on sky projection, `plot_lightcurve()` folded by period, `plot_training_curves()`.

### `utils/io.py`
I/O utilities: `load_json()`, `save_json()`.

### `utils/axisnet_utils.py`
AxisNet model utilities for loading and compatibility.

### `utils/provenance.py`
Provenance tracking for reproducibility.

---

## checkpoints/ - Pre-trained Models (5 files)

Pre-trained 5-fold cross-validation checkpoints (period-aware, v1.0):

| File | Fold | Val Size | Validation Performance (Corrected Oracle) |
|------|------|----------|-------------------------------------------|
| `fold_0.pt` | 0 | 34 asteroids | 19.90° mean, 17.52° median |
| `fold_1.pt` | 1 | 29 asteroids | 17.08° mean, 14.39° median |
| `fold_2.pt` | 2 | 36 asteroids | 17.80° mean, 15.29° median |
| `fold_3.pt` | 3 | 40 asteroids | 20.10° mean, 16.71° median |
| `fold_4.pt` | 4 | 38 asteroids | 15.74° mean, 14.93° median |
| **Aggregate** | All | **174 asteroids** | **19.02° mean, 16.61° median** |

**Note**: Oracle errors computed using scientifically rigorous methodology:
- Checks **ALL** DAMIT pole solutions (1-3 per asteroid, 68% have multiple)
- Uses antipodal symmetry: `arccos(|cos_angle|)` treats North/South poles as equivalent
- Takes minimum over K predictions AND all ground truth poles

**Checkpoint Contents**:
- `model_state_dict` - Model weights
- `config` - Training configuration
- `metrics` - Best validation metrics
- `epoch` - Training epoch


---

## See Also

- **[USER_GUIDE.md](USER_GUIDE.md)** - Usage instructions
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture
- **[DATA_FORMAT.md](DATA_FORMAT.md)** - Data schema reference
- **[TRAINING_MULTIEPOCH.md](TRAINING_MULTIEPOCH.md)** - Multi-epoch training guide
