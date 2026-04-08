# lc_pipeline Module Reference

Module map for the `lc_pipeline` package.

## Public Entry Points

### Package root

`lc_pipeline/__init__.py`

Exports the main user-facing API, including:

- `analyze`
- `LightcurvePipeline`
- `ConsensusEngine`
- `PoleConfig`
- unified schema types

### Version metadata

`lc_pipeline/version.py`

Defines `__version__` and lightweight package metadata.

### Example usage

`lc_pipeline/example_usage.py`

Small runnable examples for inference-oriented workflows.

## Inference Stack

### `lc_pipeline/inference/pipeline.py`

End-to-end orchestration:

- optional period estimation
- alias expansion
- pole inference
- uncertainty summary

### `lc_pipeline/inference/pole.py`

Loads pre-trained checkpoints and runs one model fold or an ensemble across folds.

### `lc_pipeline/inference/forking.py`

Expands inference across period aliases and returns 6 to 9 `PoleCandidate` objects.

### `lc_pipeline/inference/tokenizer.py`

Canonical tokenization used by the inference pipeline:

- 13 features per token
- up to 8 windows
- up to 256 tokens per window
- geometry feature slots zeroed in the current production path

### `lc_pipeline/inference/schema.py`

Output dataclasses:

- `PeriodResult`
- `PoleCandidate`
- `PoleUncertainty`
- `AnalysisResult`

### `lc_pipeline/inference/uncertainty.py`

Computes simple summary values such as candidate spread and score-gap confidence.

### `lc_pipeline/inference/coordinates.py`

Coordinate conversions for ecliptic/cartesian representations.

## Model Definition

### `lc_pipeline/models/geo_hier_k3_transformer.py`

Shipping Transformer model for pole prediction:

- single encoder path
- attention pooling within windows
- mean pooling across windows
- `K=3` output heads

## Period Estimation

### `lc_pipeline/period/consensus.py`

Multi-epoch period estimation and posterior fusion.

### `lc_pipeline/period/period_search.py`

Per-epoch Lomb-Scargle period search.

### `lc_pipeline/period/posterior.py`

Posterior aggregation and credible-interval helpers.

### `lc_pipeline/period/config.py`

Configuration objects for the period-estimation stage.

## Unified Schema And Converters

### `lc_pipeline/schema.py`

Unified data schema for JSON/structured workflows.

### `lc_pipeline/converters/`

Loaders and converters for:

- DAMIT-style data
- unified JSON
- unified CSV

## Data / Training / Evaluation Support

These modules are primarily for research, training, and evaluation workflows rather than beginner inference use.

### `lc_pipeline/data/`

Dataset classes and loading helpers, including the single-epoch training dataset used by the production training code.

### `lc_pipeline/training/`

Loss functions and training utilities.

### `lc_pipeline/evaluation/`

Evaluation metrics and fold-aggregation helpers.

### `lc_pipeline/scripts/train_k3.py`

Research / production training entry point used for the main model family.

### `lc_pipeline/scripts/cv_eval.py`

Cross-validation evaluation for fold checkpoints.

## Physics And Utilities

### `lc_pipeline/physics/`

Geometry helpers, frame handling, and alias-resolution utilities.

### `lc_pipeline/utils/`

General utilities for I/O, plotting, checkpoints, and related support code.

## Checkpoints

### `lc_pipeline/checkpoints/`

Contains the pre-trained fold checkpoints used by the inference API.

Notes:

- the public beginner workflow expects `fold_0.pt` through `fold_4.pt`
- the directory also contains internal evaluation artifacts and historical checkpoints
- users should treat the plain `fold_*.pt` files as the supported pre-trained checkpoints

## Command-Line Scripts At Repo Root

- `run_pole_prediction.py`: recommended beginner entry point for inference
- `train_pole_model.py`: training script for GeoHierK3Transformer using 5-fold cross-validation
