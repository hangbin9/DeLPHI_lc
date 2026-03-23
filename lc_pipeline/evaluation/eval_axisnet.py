"""Evaluation metrics for 29O-R: oracle@2, selector accuracy, gap analysis."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from lc_pipeline.training.losses_axisnet import eval_antipode_angle

logger = logging.getLogger(__name__)


def compute_metrics_per_object(
    poles: torch.Tensor,  # [K, 3] - K-agnostic (K=2, K=3, K≥4)
    quality_logits: Optional[torch.Tensor],  # [K] or None
    solutions: torch.Tensor,  # [S, 3]
) -> Dict:
    """
    Compute per-object metrics for K-agnostic pole selection (K=2, K=3, K≥4).

    This function supports arbitrary K (number of pole hypotheses).

    Returns:
        {
            'oracle_error_deg': float (min over K poles),
            'oracle_idx': int (0 to K-1, which pole achieved oracle error),
            'quality_error_deg': float (selected by quality head),
            'quality_idx': int (0 to K-1, which pole was selected),
            'naive0_error_deg': float (always pole 0),
            'gap_deg': float (quality_error - oracle_error),
            'selector_accuracy': float (1.0 if quality_idx == oracle_idx else 0.0),
            'k': int (number of poles used),
        }
    """
    # Detect K from input
    K = poles.shape[0]

    # Compute error for each hypothesis
    angles = []
    for k in range(K):  # K-agnostic loop
        pole = poles[k:k+1]  # [1, 3]
        angle = eval_antipode_angle(pole, solutions).min()  # scalar
        angles.append(angle.item())

    angles = np.array(angles)

    # Oracle: min error across K hypotheses
    oracle_idx = int(np.argmin(angles))
    oracle_error = float(angles[oracle_idx])

    # Naive0: always select pole 0
    naive0_error = float(angles[0])

    # Quality head: if available
    quality_idx = 0
    quality_error = float(angles[0])
    if quality_logits is not None:
        quality_idx = int(torch.argmax(quality_logits).item())
        # CRITICAL: Bounds check for K≥3 (quality_idx must be < K)
        if quality_idx >= K:
            logger.warning(
                f"Quality index {quality_idx} out of bounds for K={K}. "
                f"Defaulting to oracle selection (idx={oracle_idx})."
            )
            quality_idx = oracle_idx
        quality_error = float(angles[quality_idx])

    # Gap: difference between quality selection and oracle
    gap = quality_error - oracle_error

    # Selector accuracy: did quality head select oracle?
    selector_acc = 1.0 if quality_idx == oracle_idx else 0.0

    return {
        'oracle_error_deg': oracle_error,
        'oracle_idx': oracle_idx,
        'quality_error_deg': quality_error,
        'quality_idx': quality_idx,
        'naive0_error_deg': naive0_error,
        'gap_deg': gap,
        'selector_accuracy': selector_acc,
        'k': K,  # Track which K value produced this result
    }


def evaluate_fold(
    model: torch.nn.Module,
    val_dataloader: object,
    device: str = 'cuda',
) -> Dict:
    """
    Evaluate one fold.

    Returns:
        {
            'oracle_errors': list of floats,
            'quality_errors': list of floats,
            'naive0_errors': list of floats,
            'gaps': list of floats,
            'selector_accuracies': list of floats,
            'object_ids': list of strings,
            'metrics': {
                'oracle_median_deg': float,
                'oracle_mean_deg': float,
                'quality_median_deg': float,
                'quality_mean_deg': float,
                'naive0_median_deg': float,
                'gap_median_deg': float,
                'gap_mean_deg': float,
                'selector_accuracy': float,
            }
        }
    """
    model.eval()
    device_obj = torch.device(device)

    oracle_errors = []
    quality_errors = []
    naive0_errors = []
    gaps = []
    selector_accs = []
    object_ids = []

    with torch.no_grad():
        for batch in val_dataloader:
            tokens = batch['tokens'].to(device_obj)
            mask = batch['mask'].to(device_obj)
            batch_obj_ids = batch['object_ids']
            solutions_list = batch['solutions']

            poles, quality_logits = model(tokens, mask)  # [B, K, 3], [B, K] or None (K-agnostic)

            B = poles.shape[0]
            for b in range(B):
                if solutions_list[b] is None or len(solutions_list[b]) == 0:
                    continue

                poles_b = poles[b]  # [K, 3] - K-agnostic
                sols = solutions_list[b].to(device_obj)
                metrics = compute_metrics_per_object(
                    poles_b,
                    quality_logits[b:b+1].squeeze() if quality_logits is not None else None,
                    sols
                )

                oracle_errors.append(metrics['oracle_error_deg'])
                quality_errors.append(metrics['quality_error_deg'])
                naive0_errors.append(metrics['naive0_error_deg'])
                gaps.append(metrics['gap_deg'])
                selector_accs.append(metrics['selector_accuracy'])
                object_ids.append(batch_obj_ids[b])

    # Aggregate metrics
    if oracle_errors:
        metrics = {
            'oracle_median_deg': float(np.median(oracle_errors)),
            'oracle_mean_deg': float(np.mean(oracle_errors)),
            'quality_median_deg': float(np.median(quality_errors)),
            'quality_mean_deg': float(np.mean(quality_errors)),
            'naive0_median_deg': float(np.median(naive0_errors)),
            'gap_median_deg': float(np.median(gaps)),
            'gap_mean_deg': float(np.mean(gaps)),
            'selector_accuracy': float(np.mean(selector_accs)),
            'n_objects': len(oracle_errors),
        }
    else:
        metrics = {
            'oracle_median_deg': None,
            'oracle_mean_deg': None,
            'quality_median_deg': None,
            'quality_mean_deg': None,
            'naive0_median_deg': None,
            'gap_median_deg': None,
            'gap_mean_deg': None,
            'selector_accuracy': None,
            'n_objects': 0,
        }

    return {
        'oracle_errors': oracle_errors,
        'quality_errors': quality_errors,
        'naive0_errors': naive0_errors,
        'gaps': gaps,
        'selector_accuracies': selector_accs,
        'object_ids': object_ids,
        'metrics': metrics,
    }


def aggregate_folds(fold_results: List[Dict]) -> Dict:
    """
    Aggregate results from multiple folds.

    Returns:
        {
            'oracle_median_deg': float,
            'oracle_mean_deg': float,
            'quality_median_deg': float,
            'quality_mean_deg': float,
            'naive0_median_deg': float,
            'gap_median_deg': float (CRITICAL: median of per-object gaps, not difference of medians),
            'gap_mean_deg': float,
            'selector_accuracy': float,
            'per_fold': {...},
        }
    """
    all_oracle = []
    all_quality = []
    all_naive0 = []
    all_gaps = []
    all_selector_accs = []
    per_fold = {}

    for f_idx, fold_result in enumerate(fold_results):
        all_oracle.extend(fold_result['oracle_errors'])
        all_quality.extend(fold_result['quality_errors'])
        all_naive0.extend(fold_result['naive0_errors'])
        all_gaps.extend(fold_result['gaps'])
        all_selector_accs.extend(fold_result['selector_accuracies'])

        per_fold[f_idx] = fold_result['metrics']

    if all_oracle:
        return {
            'oracle_median_deg': float(np.median(all_oracle)),
            'oracle_mean_deg': float(np.mean(all_oracle)),
            'quality_median_deg': float(np.median(all_quality)),
            'quality_mean_deg': float(np.mean(all_quality)),
            'naive0_median_deg': float(np.median(all_naive0)),
            'gap_median_deg': float(np.median(all_gaps)),
            'gap_mean_deg': float(np.mean(all_gaps)),
            'selector_accuracy': float(np.mean(all_selector_accs)),
            'total_objects': len(all_oracle),
            'per_fold': per_fold,
        }
    else:
        return {
            'oracle_median_deg': None,
            'quality_median_deg': None,
            'gap_median_deg': None,
            'selector_accuracy': None,
            'per_fold': per_fold,
        }
