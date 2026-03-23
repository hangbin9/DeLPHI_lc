"""
Evaluation metrics and utilities for lc_pipeline.

This module provides:
- evaluate_fold: Evaluate model on single fold
- aggregate_folds: Combine metrics across folds
- aggregate_asteroid_predictions: Combine epoch predictions for multi-epoch training
- evaluate_with_aggregation: Full evaluation with asteroid-level aggregation
"""

from .eval_axisnet import evaluate_fold, aggregate_folds
from .aggregation import (
    aggregate_asteroid_predictions,
    evaluate_with_aggregation,
)

__all__ = [
    # Core evaluation
    "evaluate_fold",
    "aggregate_folds",
    # Multi-epoch aggregation
    "aggregate_asteroid_predictions",
    "evaluate_with_aggregation",
]
