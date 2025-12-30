#!/usr/bin/env python3
"""
Evaluate observability-based gates for pole prediction.

Correct implementation:
- One row per object (not per apparition/prediction)
- Metrics computed per object consistently
- Thresholds chosen from training folds only
- No threshold leakage to test fold
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple, List
import math

# ============================================================================
# UTILITIES
# ============================================================================

def compute_angular_error_deg(p_pred: np.ndarray, p_true: np.ndarray) -> float:
    """Antipode-aware angular error in degrees."""
    p_pred = p_pred / np.linalg.norm(p_pred)
    p_true = p_true / np.linalg.norm(p_true)
    dot_prod = np.clip(np.dot(p_pred, p_true), -1.0, 1.0)
    error_rad = np.arccos(dot_prod)
    error_rad = min(error_rad, np.pi - error_rad)  # Antipode-aware
    return float(error_rad * 180.0 / math.pi)


def compute_observability_features(asteroids: List[Dict]) -> np.ndarray:
    """
    Compute observability features per asteroid.

    Returns:
        (n_asteroids, 4) array with columns:
        [time_span_days, brightness_amplitude_mag, n_apparitions, total_n_obs]
    """
    features = []

    for ast in asteroids:
        # Collect all times and mags across apparitions
        times_all = []
        mags_all = []
        n_apparitions = len(ast['apparitions'])
        total_n_obs = 0

        for app in ast['apparitions']:
            app_features = app['features']  # (n_obs, 12): [time, mag, ...]
            total_n_obs += app_features.shape[0]
            times_all.extend(app_features[:, 0])
            mags_all.extend(app_features[:, 1])

        # Compute span and amplitude
        if times_all:
            time_span = max(times_all) - min(times_all)
        else:
            time_span = 0

        if mags_all:
            # Use percentile-based amplitude (more robust to outliers)
            amp = np.percentile(mags_all, 95) - np.percentile(mags_all, 5)
        else:
            amp = 0

        features.append([time_span, amp, n_apparitions, total_n_obs])

    return np.array(features)


def evaluate_gate(
    errors: np.ndarray,
    labels: np.ndarray,
    observability: np.ndarray,
    gate_mask: np.ndarray,
    fold_idx: int
) -> Dict:
    """
    Evaluate a single gate on the test fold.

    Args:
        errors: (n_test,) angular errors in degrees
        labels: (n_test,) binary labels (1 if error ≤ 25°)
        observability: (n_test, 4) observability features
        gate_mask: (n_test,) boolean mask (True = accept)
        fold_idx: fold number for reporting

    Returns:
        Dict with coverage, median error, p90, accuracy, etc.
    """

    n_total = len(errors)
    n_accepted = gate_mask.sum()
    coverage = n_accepted / n_total if n_total > 0 else 0

    if n_accepted == 0:
        return {
            'fold': fold_idx,
            'coverage': 0.0,
            'n_accepted': 0,
            'n_test': n_total,
            'median_error': np.nan,
            'p90_error': np.nan,
            'accuracy_le25': np.nan,
            'mean_error': np.nan
        }

    errors_accepted = errors[gate_mask]
    labels_accepted = labels[gate_mask]

    median_error = float(np.median(errors_accepted))
    p90_error = float(np.percentile(errors_accepted, 90))
    mean_error = float(np.mean(errors_accepted))
    accuracy_le25 = float((labels_accepted == 1).sum() / len(labels_accepted))

    return {
        'fold': fold_idx,
        'coverage': float(coverage),
        'n_accepted': int(n_accepted),
        'n_test': int(n_total),
        'median_error': median_error,
        'p90_error': p90_error,
        'accuracy_le25': accuracy_le25,
        'mean_error': mean_error
    }


def grid_search_gates(
    train_observability: np.ndarray,
    test_errors: np.ndarray,
    test_labels: np.ndarray,
    test_observability: np.ndarray,
    fold_idx: int
) -> Dict:
    """
    Grid search over observability thresholds.

    Thresholds chosen from training fold only.
    Applied to test fold.
    """

    span_train = train_observability[:, 0]
    amp_train = train_observability[:, 1]
    n_app_train = train_observability[:, 2]

    span_test = test_observability[:, 0]
    amp_test = test_observability[:, 1]
    n_app_test = test_observability[:, 2]

    # Quantile thresholds (chosen from training)
    quantiles = [0.3, 0.5, 0.7]
    span_thresholds = [np.percentile(span_train, q * 100) for q in quantiles]
    amp_thresholds = [np.percentile(amp_train, q * 100) for q in quantiles]

    results = {
        'fold': fold_idx,
        'span_thresholds': {q: t for q, t in zip(quantiles, span_thresholds)},
        'amp_thresholds': {q: t for q, t in zip(quantiles, amp_thresholds)},
        'gates': {}
    }

    # Single-feature gates
    for q, tau_span in zip(quantiles, span_thresholds):
        mask = span_test <= tau_span
        eval_result = evaluate_gate(test_errors, test_labels, test_observability, mask, fold_idx)
        results['gates'][f'span_le_q{int(q*100)}'] = eval_result

    for q, tau_amp in zip(quantiles, amp_thresholds):
        mask = amp_test <= tau_amp
        eval_result = evaluate_gate(test_errors, test_labels, test_observability, mask, fold_idx)
        results['gates'][f'amp_le_q{int(q*100)}'] = eval_result

    # Multi-apparition gate
    mask_multiapp = n_app_test >= 2
    eval_result = evaluate_gate(test_errors, test_labels, test_observability, mask_multiapp, fold_idx)
    results['gates']['n_app_ge_2'] = eval_result

    # Conjunction gates
    for q_span, tau_span in zip(quantiles, span_thresholds):
        for q_amp, tau_amp in zip(quantiles, amp_thresholds):
            mask = (span_test <= tau_span) & (amp_test <= tau_amp)
            eval_result = evaluate_gate(test_errors, test_labels, test_observability, mask, fold_idx)
            gate_name = f'span_le_q{int(q_span*100)}_AND_amp_le_q{int(q_amp*100)}'
            results['gates'][gate_name] = eval_result

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Evaluate observability gates for all folds."""

    print("="*80)
    print("OBSERVABILITY-BASED GATE EVALUATION")
    print("="*80)
    print()

    # Load calibrator features
    features_file = Path('/tmp/phase1_calibrator_features_5fold.npz')
    if not features_file.exists():
        print(f"ERROR: {features_file} not found")
        return

    data = np.load(features_file)

    all_results = {}
    all_fold_results = []

    for fold_idx in range(5):
        print(f"{'='*80}")
        print(f"FOLD {fold_idx}")
        print(f"{'='*80}")
        print()

        # Load test set
        dataset_path = Path(f'/tmp/phase1_pole_dataset/fold_{fold_idx}/test_asteroids.pkl')
        if not dataset_path.exists():
            print(f"ERROR: {dataset_path} not found")
            continue

        with open(dataset_path, 'rb') as f:
            test_asteroids = pickle.load(f)

        # Load training set (for threshold selection)
        train_asteroids = []
        for other_fold in range(5):
            if other_fold != fold_idx:
                train_path = Path(f'/tmp/phase1_pole_dataset/fold_{other_fold}/test_asteroids.pkl')
                if train_path.exists():
                    with open(train_path, 'rb') as f:
                        train_asteroids.extend(pickle.load(f))

        print(f"Training set: {len(train_asteroids)} asteroids (from other folds)")
        print(f"Test set: {len(test_asteroids)} asteroids")
        print()

        # Compute observability features
        test_observability = compute_observability_features(test_asteroids)
        train_observability = compute_observability_features(train_asteroids)

        # Load errors and labels
        errors = data[f'errors_fold{fold_idx}']
        labels = data[f'y_fold{fold_idx}']

        print(f"Observability statistics (test fold):")
        print(f"  Time span: min={test_observability[:, 0].min():.2f}, "
              f"median={np.median(test_observability[:, 0]):.2f}, "
              f"max={test_observability[:, 0].max():.2f} days")
        print(f"  Amplitude: min={test_observability[:, 1].min():.3f}, "
              f"median={np.median(test_observability[:, 1]):.3f}, "
              f"max={test_observability[:, 1].max():.3f} mags")
        print(f"  N apparitions: {test_observability[:, 2].min():.0f} - "
              f"{test_observability[:, 2].max():.0f} "
              f"(median {np.median(test_observability[:, 2]):.0f})")
        print()

        # Grid search over gates
        fold_results = grid_search_gates(
            train_observability,
            errors,
            labels,
            test_observability,
            fold_idx
        )

        all_results[f'fold_{fold_idx}'] = fold_results

        # Print results
        print("Gate Performance:")
        print()
        print(f"{'Gate':<50} {'Cov':>6} {'Med Error':>10} {'P90':>7} {'Acc≤25°':>9}")
        print("-" * 85)

        for gate_name, gate_result in sorted(fold_results['gates'].items()):
            cov = gate_result['coverage']
            med_err = gate_result['median_error']
            p90_err = gate_result['p90_error']
            acc = gate_result['accuracy_le25']

            med_str = f"{med_err:.1f}°" if not np.isnan(med_err) else "  nan"
            p90_str = f"{p90_err:.1f}°" if not np.isnan(p90_err) else "  nan"
            acc_str = f"{acc*100:.1f}%" if not np.isnan(acc) else "  nan"

            print(f"{gate_name:<50} {cov:6.1%} {med_str:>10} {p90_str:>7} {acc_str:>9}")

        all_fold_results.append(fold_results)
        print()

    # ---- Aggregate across folds ----
    print("="*80)
    print("AGGREGATE RESULTS ACROSS 5 FOLDS")
    print("="*80)
    print()

    # Collect all gate names
    all_gate_names = set()
    for fold_result in all_fold_results:
        all_gate_names.update(fold_result['gates'].keys())

    all_gate_names = sorted(all_gate_names)

    print(f"{'Gate':<50} {'Cov':>6} {'Med Error':>10} {'P90':>7} {'Acc≤25°':>9}")
    print("-" * 85)

    for gate_name in all_gate_names:
        # Aggregate across folds
        coverages = []
        med_errors = []
        p90_errors = []
        accuracies = []

        for fold_result in all_fold_results:
            if gate_name in fold_result['gates']:
                g = fold_result['gates'][gate_name]
                coverages.append(g['coverage'])
                if not np.isnan(g['median_error']):
                    med_errors.append(g['median_error'])
                if not np.isnan(g['p90_error']):
                    p90_errors.append(g['p90_error'])
                if not np.isnan(g['accuracy_le25']):
                    accuracies.append(g['accuracy_le25'])

        if len(med_errors) > 0:
            cov_mean = np.mean(coverages)
            med_mean = np.mean(med_errors)
            p90_mean = np.mean(p90_errors)
            acc_mean = np.mean(accuracies)

            med_str = f"{med_mean:.1f}°"
            p90_str = f"{p90_mean:.1f}°"
            acc_str = f"{acc_mean*100:.1f}%"

            print(f"{gate_name:<50} {cov_mean:6.1%} {med_str:>10} {p90_str:>7} {acc_str:>9}")

    print()

    # Save results
    output_file = Path('/tmp/observability_gates_results.json')
    with open(output_file, 'w') as f:
        # Convert numpy types for JSON serialization
        json_results = json.loads(json.dumps(all_results, default=str))
        json.dump(json_results, f, indent=2)

    print(f"Detailed results saved to: {output_file}")
    print()

    # Summary
    print("="*80)
    print("INTERPRETATION")
    print("="*80)
    print()
    print("""
If best gate achieves median ≤25° at coverage ≥25%:
  → Observability gating is deployable
  → Frame as: "median X° at Y% coverage on constrainable objects"
  → Gate decision: pre-prediction, based on lightcurve quality only

If best gate achieves median ~24-25° at coverage 25-50%:
  → Useful improvement over ungated 44-55°
  → Coverage trade-off is acceptable for production

If gating shows no improvement:
  → Observability is not the dominant factor
  → Alternative: long-span failure mode analysis (phase drift, etc.)
  → Consider per-apparition encoding improvements
""")


if __name__ == '__main__':
    main()
