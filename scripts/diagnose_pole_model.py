#!/usr/bin/env python3
"""
Diagnostic script: Validate pole model quality.

Three checks to determine if weak calibration is due to:
1. Sub-optimal point estimate (MAP extraction issue)
2. Model not learning sharp distributions (distribution quality)
3. Observable stratification (some subsets learnable, others not)
"""

import numpy as np
import pickle
import torch
from pathlib import Path
from typing import Dict, Tuple
import json
import math

# ============================================================================
# 1. ANGULAR ERROR COMPUTATION (UTILITY)
# ============================================================================

def compute_angular_error_deg(p_pred: np.ndarray, p_true: np.ndarray) -> float:
    """Antipode-aware angular error in degrees."""
    p_pred = p_pred / np.linalg.norm(p_pred)
    p_true = p_true / np.linalg.norm(p_true)
    dot_prod = np.clip(np.dot(p_pred, p_true), -1.0, 1.0)
    error_rad = np.arccos(dot_prod)
    error_rad = min(error_rad, np.pi - error_rad)  # Antipode-aware
    return float(error_rad * 180.0 / math.pi)


# ============================================================================
# 2. CHECK 1: DIFFERENT POINT ESTIMATORS
# ============================================================================

def check_point_estimators(mixture_params: Dict, p_true: np.ndarray) -> Dict:
    """
    Compare different point estimators:
    - MAP: p_hat = p that maximizes p_vMF_mixture
    - Best component mean: μ of highest-posterior component
    - Spherical mean: normalize(Σ π_i μ_i)
    """

    mu = mixture_params['mu']  # (K, 3)
    kappa = mixture_params['kappa']  # (K,)
    weight = mixture_params['weight']  # (K,)

    K = mu.shape[0]

    # ---- Estimator 1: Best component mean (current method) ----
    best_idx = np.argmax(weight)
    p_best_comp = mu[best_idx]
    p_best_comp = p_best_comp / np.linalg.norm(p_best_comp)
    error_best_comp = compute_angular_error_deg(p_best_comp, p_true)

    # ---- Estimator 2: Spherical mean of mixture ----
    p_spherical = np.sum(weight[:, None] * mu, axis=0)
    p_spherical = p_spherical / np.linalg.norm(p_spherical)
    error_spherical = compute_angular_error_deg(p_spherical, p_true)

    # ---- Estimator 3: MAP via grid search ----
    # Sample from mixture and find sample with highest density
    n_samples = 10000
    samples = []
    densities = []

    for _ in range(n_samples):
        # Sample component
        k = np.random.choice(K, p=weight)

        # Sample from vMF(mu_k, kappa_k)
        # Use Von Mises-Fisher sampler
        mu_k = mu[k]
        kappa_k = kappa[k]

        # Simple VonMisesFisher sample (for unit sphere)
        # Using exponential map from Euclidean
        v = np.random.randn(3)
        v = v / np.linalg.norm(v)
        phi = np.random.uniform(0, 2 * np.pi)

        # Approx VonMisesFisher sample (crude)
        theta = 2 * np.arctan2(np.sqrt(1 - np.exp(-2 * kappa_k)), np.sqrt(np.exp(-2 * kappa_k)))
        u = np.random.randn(2)
        u = u / np.linalg.norm(u)

        # Rotate to align with mu_k
        sample = np.cos(theta) * mu_k + np.sin(theta) * np.cross(mu_k, u)
        sample = sample / np.linalg.norm(sample)

        samples.append(sample)

    samples = np.array(samples)

    # Find sample with highest posterior (mode of mixture)
    # For simplicity, use weighted likelihood
    densities = np.zeros(len(samples))
    for i, sample in enumerate(samples):
        for k in range(K):
            # VonMisesFisher density (approx)
            cos_angle = np.clip(np.dot(sample, mu[k]), -1, 1)
            log_dens_k = kappa[k] * cos_angle - np.log(4 * np.pi * np.sinh(kappa[k]))
            densities[i] += weight[k] * np.exp(log_dens_k)

    p_map_idx = np.argmax(densities)
    p_map = samples[p_map_idx]
    error_map = compute_angular_error_deg(p_map, p_true)

    return {
        'p_best_comp_error': error_best_comp,
        'p_spherical_error': error_spherical,
        'p_map_error': error_map,
        'best_method': min(
            [('best_comp', error_best_comp),
             ('spherical', error_spherical),
             ('map', error_map)],
            key=lambda x: x[1]
        )[0]
    }


# ============================================================================
# 3. CHECK 2: MASS WITHIN 25° OF TRUTH (DISTRIBUTION QUALITY)
# ============================================================================

def compute_mass_within_cap(mixture_params: Dict, p_true: np.ndarray, cap_angle_deg: float = 25.0) -> float:
    """
    Compute probability mass within a spherical cap of given angle around p_true.
    Also accounts for antipode.
    """

    mu = mixture_params['mu']  # (K, 3)
    kappa = mixture_params['kappa']  # (K,)
    weight = mixture_params['weight']  # (K,)

    cap_angle_rad = cap_angle_deg * np.pi / 180.0
    cap_angle_antipode = np.pi - cap_angle_rad

    # Sample from mixture and compute fraction in cap
    n_samples = 10000
    count_in_cap = 0

    for _ in range(n_samples):
        # Sample component
        k = np.random.choice(len(weight), p=weight)

        # Sample from vMF (crude approximation)
        # Use rejection sampling or direct method
        mu_k = mu[k]
        kappa_k = min(kappa[k], 100)  # Clamp to avoid numerical issues

        # Generate random point on sphere
        v = np.random.randn(3)
        v = v / np.linalg.norm(v)

        # Accept with probability proportional to vMF density
        cos_angle_k = np.dot(v, mu_k)

        # vMF density: exp(kappa * cos(angle))
        log_dens = kappa_k * cos_angle_k

        # Rejection (simplified)
        if np.log(np.random.rand()) < log_dens:
            # Check if in cap around truth
            angle_to_truth = np.arccos(np.clip(np.dot(v, p_true), -1, 1))
            angle_to_antipode = np.pi - angle_to_truth

            if angle_to_truth <= cap_angle_rad or angle_to_antipode <= cap_angle_rad:
                count_in_cap += 1

    return count_in_cap / n_samples


# ============================================================================
# 4. MAIN DIAGNOSTIC
# ============================================================================

def main():
    """Run diagnostic checks on Phase 1 calibrator data."""

    results_file = Path('/tmp/phase1_calibrator_results_5fold.json')
    features_file = Path('/tmp/phase1_calibrator_features_5fold.npz')

    if not results_file.exists() or not features_file.exists():
        print("ERROR: Required files not found")
        return

    # Load features and errors
    data = np.load(features_file)

    print("="*80)
    print("POLE MODEL DIAGNOSTIC CHECKS")
    print("="*80)
    print()

    # ---- Load model to extract mixture params ----
    model_path = Path('/tmp/phase1_pole_models_fold_0.pt')
    dataset_path = Path('/tmp/phase1_pole_dataset/fold_0/test_asteroids.pkl')

    if not model_path.exists() or not dataset_path.exists():
        print(f"ERROR: Model or dataset not found")
        print(f"  Model: {model_path.exists()}")
        print(f"  Dataset: {dataset_path.exists()}")
        return

    # Load test set
    with open(dataset_path, 'rb') as f:
        test_asteroids = pickle.load(f)

    print(f"Loaded {len(test_asteroids)} test asteroids")
    print()

    # ---- CHECK 1: Point estimators ----
    print("="*80)
    print("CHECK 1: DIFFERENT POINT ESTIMATORS")
    print("="*80)
    print()
    print("Testing if current extraction (best component) is optimal...")
    print()

    estimator_results = []
    for idx, asteroid in enumerate(test_asteroids[:10]):  # Sample first 10
        p_true = asteroid['metadata']['pole_true']

        # For now, extract features and simulate
        # (In practice, would re-run model inference)
        results = {
            'asteroid_id': asteroid['id'],
            'note': 'Would require re-running model inference'
        }
        estimator_results.append(results)

    print("NOTE: Full CHECK 1 requires re-running model inference on test set")
    print("      (This diagnostic needs access to mixture_params for each test object)")
    print()

    # ---- CHECK 2: Mass within 25° ----
    print("="*80)
    print("CHECK 2: DISTRIBUTION QUALITY (Mass within 25° of truth)")
    print("="*80)
    print()
    print("For correct predictions (error ≤ 25°):")
    print("  Expected: Most should have mass_true25 > 0.5 (mixture confident about truth)")
    print()
    print("For incorrect predictions (error > 25°):")
    print("  Expected: Most should have mass_true25 < 0.3 (mixture unsure)")
    print()
    print("NOTE: Full CHECK 2 also requires mixture_params re-computation")
    print()

    # ---- CHECK 3: Observable stratification ----
    print("="*80)
    print("CHECK 3: STRATIFICATION BY OBSERVABILITY")
    print("="*80)
    print()

    # Load error data
    errors_fold0 = data['errors_fold0']
    y_fold0 = data['y_fold0']

    print("Fold 0 Analysis:")
    print(f"  Total samples: {len(errors_fold0)}")
    print(f"  Correct (≤25°): {(y_fold0 == 1).sum()}")
    print(f"  Incorrect (>25°): {(y_fold0 == 0).sum()}")
    print()

    print("Ungated error distribution:")
    print(f"  Median: {np.median(errors_fold0):.1f}°")
    print(f"  Mean: {np.mean(errors_fold0):.1f}°")
    print(f"  Std: {np.std(errors_fold0):.1f}°")
    print(f"  Min: {np.min(errors_fold0):.1f}°")
    print(f"  Max: {np.max(errors_fold0):.1f}°")
    print(f"  P10: {np.percentile(errors_fold0, 10):.1f}°")
    print(f"  P25: {np.percentile(errors_fold0, 25):.1f}°")
    print(f"  P75: {np.percentile(errors_fold0, 75):.1f}°")
    print(f"  P90: {np.percentile(errors_fold0, 90):.1f}°")
    print()

    # Stratify by observability proxy: error size
    # (In practice, would use actual observability features)
    low_error = errors_fold0 <= np.percentile(errors_fold0, 33)
    mid_error = (errors_fold0 > np.percentile(errors_fold0, 33)) & (errors_fold0 <= np.percentile(errors_fold0, 67))
    high_error = errors_fold0 > np.percentile(errors_fold0, 67)

    print("Error distribution quartiles:")
    print(f"  Low (≤P33):  median={np.median(errors_fold0[low_error]):.1f}°, "
          f"correct={(y_fold0[low_error] == 1).sum()}/{len(low_error)}")
    print(f"  Mid (P33-67): median={np.median(errors_fold0[mid_error]):.1f}°, "
          f"correct={(y_fold0[mid_error] == 1).sum()}/{len(mid_error)}")
    print(f"  High (>P67): median={np.median(errors_fold0[high_error]):.1f}°, "
          f"correct={(y_fold0[high_error] == 1).sum()}/{len(high_error)}")
    print()

    # ---- Summary ----
    print("="*80)
    print("INTERPRETATION")
    print("="*80)
    print()
    print("""
If CHECK 1 shows significant improvement with MAP or spherical mean:
  → Problem is point estimator extraction
  → Fix: Use better p_hat computation method
  → Expected impact: +2-5° median improvement

If CHECK 2 shows low mass_true25 even for correct predictions:
  → Problem is distribution sharpness
  → Model doesn't learn to concentrate around truth
  → Expected: Hard to fix without stronger teacher signal

If CHECK 3 shows strong stratification (low-error subset much better):
  → Some objects are learnable, others are ambiguous
  → Solution: Use observability gating (# apparitions, span, amplitude)
  → Expected: Can deploy with 30-40% coverage at ≤25° median

Current status:
  - Calibrator AUC: 0.574 (features weak)
  - Ungated median: 44-55° (barely better than random 60°)
  - Features don't capture confidence → κ is uninformative
""")

    print()
    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print()
    print("1. Implement full CHECK 1 by re-running model on test set")
    print("   - Compare best_comp vs MAP vs spherical_mean errors")
    print()
    print("2. Implement full CHECK 2 by computing mass_true25 from mixture_params")
    print("   - Stratify by correct/incorrect predictions")
    print()
    print("3. Use test set asteroids' metadata for true observability features:")
    print("   - n_apparitions, time_span, brightness_amplitude, geometry_spread")
    print("   - Stratify error by these features")
    print()


if __name__ == '__main__':
    main()
