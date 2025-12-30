"""
Evaluation metrics for period predictions.

This module provides:
- relative_error: absolute relative error
- alias_aware_relative_error: min error over {P, 0.5P, 2P}
- accuracy_at_tol: fraction of predictions within tolerance
- evaluate_predictions: comprehensive evaluation suite

No train/test leakage: evaluation only compares pre-computed predictions
to ground truth without any fitting or tuning.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def relative_error(pred: float, true: float) -> float:
    """
    Compute absolute relative error.

    Args:
        pred: Predicted period.
        true: True period.

    Returns:
        |pred - true| / true

    Raises:
        ValueError: If true is zero.
    """
    if true == 0:
        raise ValueError("True period cannot be zero")
    return abs(pred - true) / abs(true)


def alias_aware_relative_error(pred: float, true: float) -> float:
    """
    Compute alias-aware relative error.

    Returns the minimum relative error considering period aliases:
    - true
    - 0.5 * true (half-period alias)
    - 2 * true (double-period alias)

    This accounts for common LS period aliasing where predicting
    the half or double period is physically equivalent.

    Args:
        pred: Predicted period.
        true: True period.

    Returns:
        Minimum relative error over {true, 0.5*true, 2*true}
    """
    if true == 0:
        raise ValueError("True period cannot be zero")

    candidates = [true, 0.5 * true, 2.0 * true]
    errors = [abs(pred - c) / abs(c) for c in candidates]
    return min(errors)


def accuracy_at_tol(errors: np.ndarray, tol: float) -> float:
    """
    Compute accuracy at a given tolerance threshold.

    Args:
        errors: Array of error values.
        tol: Tolerance threshold.

    Returns:
        Fraction of errors <= tol (between 0 and 1).
    """
    errors = np.asarray(errors, dtype=float)
    valid_errors = errors[np.isfinite(errors)]

    if len(valid_errors) == 0:
        return 0.0

    return float(np.mean(valid_errors <= tol))


def evaluate_predictions(
    predictions_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    pred_col: str = "period_hours",
    true_col: str = "period_hours"
) -> Dict[str, Any]:
    """
    Evaluate predictions against ground truth.

    Joins predictions with truth by object_id and computes comprehensive
    metrics. No fitting or tuning is performed.

    Args:
        predictions_df: DataFrame with columns: object_id, period_hours
        truth_df: DataFrame with columns: object_id, period_hours
        pred_col: Column name for predicted period in predictions_df
        true_col: Column name for true period in truth_df

    Returns:
        Dictionary with metrics:
            - n_objects: int, number of evaluated objects
            - n_matched: int, objects with both prediction and truth
            - n_valid: int, objects with finite error
            - median_rel_err: float, median relative error
            - mean_rel_err: float, mean relative error
            - std_rel_err: float, std of relative error
            - median_rel_err_alias: float, alias-aware median rel error
            - mean_rel_err_alias: float, alias-aware mean rel error
            - acc_5pct: float, alias-aware accuracy at 5%
            - acc_10pct: float, alias-aware accuracy at 10%
            - acc_20pct: float, alias-aware accuracy at 20%
    """
    # Ensure object_id is string in both
    preds = predictions_df.copy()
    truth = truth_df.copy()
    preds["object_id"] = preds["object_id"].astype(str)
    truth["object_id"] = truth["object_id"].astype(str)

    # Merge on object_id
    merged = preds.merge(
        truth[["object_id", true_col]],
        on="object_id",
        how="inner",
        suffixes=("_pred", "_true")
    )

    result: Dict[str, Any] = {
        "n_objects": len(truth),
        "n_matched": len(merged),
        "n_valid": 0,
        "median_rel_err": np.nan,
        "mean_rel_err": np.nan,
        "std_rel_err": np.nan,
        "median_rel_err_alias": np.nan,
        "mean_rel_err_alias": np.nan,
        "acc_5pct": 0.0,
        "acc_10pct": 0.0,
        "acc_20pct": 0.0,
    }

    if len(merged) == 0:
        return result

    # Handle column naming from merge
    if f"{true_col}_true" in merged.columns:
        true_vals = merged[f"{true_col}_true"].values
    else:
        true_vals = merged[true_col].values

    if pred_col in merged.columns:
        pred_vals = merged[pred_col].values
    elif f"{pred_col}_pred" in merged.columns:
        pred_vals = merged[f"{pred_col}_pred"].values
    else:
        # Try without suffix
        pred_vals = merged[pred_col].values

    # Compute errors
    rel_errors = []
    alias_errors = []

    for pred, true in zip(pred_vals, true_vals):
        if np.isfinite(pred) and np.isfinite(true) and true > 0:
            rel_errors.append(relative_error(pred, true))
            alias_errors.append(alias_aware_relative_error(pred, true))
        else:
            rel_errors.append(np.nan)
            alias_errors.append(np.nan)

    rel_errors = np.array(rel_errors)
    alias_errors = np.array(alias_errors)

    valid_mask = np.isfinite(rel_errors)
    n_valid = int(valid_mask.sum())
    result["n_valid"] = n_valid

    if n_valid == 0:
        return result

    valid_rel = rel_errors[valid_mask]
    valid_alias = alias_errors[valid_mask]

    # Standard metrics
    result["median_rel_err"] = float(np.median(valid_rel))
    result["mean_rel_err"] = float(np.mean(valid_rel))
    result["std_rel_err"] = float(np.std(valid_rel))

    # Alias-aware metrics
    result["median_rel_err_alias"] = float(np.median(valid_alias))
    result["mean_rel_err_alias"] = float(np.mean(valid_alias))

    # Accuracy at various tolerances (alias-aware)
    result["acc_5pct"] = accuracy_at_tol(valid_alias, 0.05)
    result["acc_10pct"] = accuracy_at_tol(valid_alias, 0.10)
    result["acc_20pct"] = accuracy_at_tol(valid_alias, 0.20)

    return result


def format_metrics_report(metrics: Dict[str, Any]) -> str:
    """
    Format metrics dictionary as a human-readable report.

    Args:
        metrics: Dictionary from evaluate_predictions.

    Returns:
        Formatted string report.
    """
    lines = [
        "=" * 50,
        "Period Prediction Evaluation Report",
        "=" * 50,
        "",
        f"Objects in ground truth:  {metrics['n_objects']}",
        f"Objects with predictions: {metrics['n_matched']}",
        f"Objects with valid error: {metrics['n_valid']}",
        "",
        "--- Relative Error (standard) ---",
        f"  Median: {metrics['median_rel_err']:.4f} ({metrics['median_rel_err']*100:.2f}%)",
        f"  Mean:   {metrics['mean_rel_err']:.4f} ({metrics['mean_rel_err']*100:.2f}%)",
        f"  Std:    {metrics['std_rel_err']:.4f}",
        "",
        "--- Relative Error (alias-aware: min over P, 0.5P, 2P) ---",
        f"  Median: {metrics['median_rel_err_alias']:.4f} ({metrics['median_rel_err_alias']*100:.2f}%)",
        f"  Mean:   {metrics['mean_rel_err_alias']:.4f} ({metrics['mean_rel_err_alias']*100:.2f}%)",
        "",
        "--- Accuracy @ Tolerance (alias-aware) ---",
        f"  Acc@5%:  {metrics['acc_5pct']:.4f} ({metrics['acc_5pct']*100:.1f}%)",
        f"  Acc@10%: {metrics['acc_10pct']:.4f} ({metrics['acc_10pct']*100:.1f}%)",
        f"  Acc@20%: {metrics['acc_20pct']:.4f} ({metrics['acc_20pct']*100:.1f}%)",
        "",
        "=" * 50,
    ]
    return "\n".join(lines)
