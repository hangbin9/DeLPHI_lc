"""
Asteroid-level prediction aggregation for multi-epoch training.

Since the model is trained on individual epochs, inference predictions
need to be aggregated per asteroid when evaluating at the asteroid level.
This module provides aggregation methods and evaluation utilities.

Key Functions:
- aggregate_asteroid_predictions(): Combine predictions from multiple epochs
- evaluate_with_aggregation(): Full evaluation with asteroid-level metrics

Aggregation Methods:
- 'average': Mean of pole vectors across epochs (default, recommended)
- 'vote': Spherical k-means clustering with quality-weighted voting
- 'best_quality': Select epoch with highest quality score

Example:
    >>> from lc_pipeline.evaluation import aggregate_asteroid_predictions
    >>> epoch_predictions = [
    ...     {'poles': np.array([[0.1, 0.2, 0.97]]*3), 'scores': np.array([0.8, 0.1, 0.1])},
    ...     {'poles': np.array([[0.15, 0.18, 0.97]]*3), 'scores': np.array([0.7, 0.2, 0.1])},
    ... ]
    >>> agg = aggregate_asteroid_predictions(epoch_predictions, method='average')
    >>> print(agg['poles'].shape)  # (3, 3)
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch

from ..physics.geometry import best_solution_error

logger = logging.getLogger(__name__)


def _spherical_kmeans(points: np.ndarray, k: int, max_iter: int = 20, seed: int = 42) -> np.ndarray:
    """
    Spherical k-means clustering for unit vectors.

    Standard k-means adapted for spherical data where distance is measured
    by angular distance (equivalently, 1 - cosine similarity). Cluster
    centers are always normalized to unit length.

    Args:
        points: (N, 3) unit vectors on the sphere
        k: Number of clusters
        max_iter: Maximum iterations (default 20, usually converges faster)
        seed: Random seed for initialization

    Returns:
        (k, 3) cluster centers (unit vectors)

    Notes:
        - Uses cosine similarity as the distance metric
        - Centers are recomputed as mean direction, then normalized
        - Empty clusters retain their previous center
    """
    N = len(points)
    if N < k:
        # Not enough points for k clusters
        return points[:k] if N > 0 else np.zeros((k, 3))

    # Initialize centers by selecting k random points
    rng = np.random.RandomState(seed)
    init_indices = rng.choice(N, size=k, replace=False)
    centers = points[init_indices].copy()

    for iteration in range(max_iter):
        # Assign each point to nearest center (cosine similarity)
        similarities = np.dot(points, centers.T)  # (N, k)
        assignments = np.argmax(similarities, axis=1)  # (N,)

        # Update centers
        new_centers = np.zeros((k, 3))
        for cluster_idx in range(k):
            cluster_points = points[assignments == cluster_idx]
            if len(cluster_points) > 0:
                # Mean direction (then normalize)
                mean_vec = cluster_points.mean(axis=0)
                norm = np.linalg.norm(mean_vec)
                if norm > 1e-8:
                    new_centers[cluster_idx] = mean_vec / norm
                else:
                    # Degenerate case: keep old center
                    new_centers[cluster_idx] = centers[cluster_idx]
            else:
                # Empty cluster: keep old center
                new_centers[cluster_idx] = centers[cluster_idx]

        # Check convergence
        center_change = np.linalg.norm(new_centers - centers)
        centers = new_centers

        if center_change < 1e-6:
            break

    return centers


def aggregate_asteroid_predictions(
    epoch_predictions: List[Dict],
    method: str = 'average'
) -> Dict:
    """
    Aggregate predictions across epochs of the same asteroid.

    Since multi-epoch training uses individual epochs as samples, we get K pole
    predictions per epoch. This function combines them into K final predictions
    for the asteroid.

    Args:
        epoch_predictions: List of dicts with 'poles' and 'scores' keys.
            Each dict contains:
                - poles: (K, 3) array of pole unit vectors
                - scores: (K,) array of quality scores
        method: Aggregation method:
            - 'average': Mean of pole vectors across epochs (default)
            - 'vote': Spherical k-means with quality-weighted voting
            - 'best_quality': Select epoch with highest quality score

    Returns:
        Dict with aggregated 'poles' (K, 3) and 'scores' (K,)

    Example:
        >>> epoch_preds = [
        ...     {'poles': poles1, 'scores': scores1},
        ...     {'poles': poles2, 'scores': scores2},
        ... ]
        >>> agg = aggregate_asteroid_predictions(epoch_preds, method='average')
        >>> print(agg['poles'].shape)  # (K, 3)
    """
    if len(epoch_predictions) == 0:
        raise ValueError("No epoch predictions to aggregate")

    if len(epoch_predictions) == 1:
        return epoch_predictions[0]

    K = len(epoch_predictions[0]['poles'])

    if method == 'average':
        # Average pole vectors across epochs
        all_poles = np.array([pred['poles'] for pred in epoch_predictions])  # (N_epochs, K, 3)
        avg_poles = all_poles.mean(axis=0)  # (K, 3)

        # Normalize to unit vectors
        norms = np.linalg.norm(avg_poles, axis=1, keepdims=True)
        avg_poles = avg_poles / (norms + 1e-8)

        # Average scores
        all_scores = np.array([pred['scores'] for pred in epoch_predictions])
        avg_scores = all_scores.mean(axis=0)

        return {
            'poles': avg_poles,
            'scores': avg_scores,
        }

    elif method == 'vote':
        # Voting aggregation via spherical k-means clustering
        # 1. Collect all K*N pole predictions
        all_poles = []
        all_scores = []
        for pred in epoch_predictions:
            all_poles.extend(pred['poles'])
            all_scores.extend(pred['scores'])

        all_poles = np.array(all_poles)  # (K*N, 3)
        all_scores = np.array(all_scores)  # (K*N,)

        # 2. Cluster into K groups using spherical k-means
        cluster_centers = _spherical_kmeans(all_poles, K, max_iter=20)

        # 3. Each epoch votes for the best cluster (highest quality score)
        cluster_votes = np.zeros(K)

        for pred in epoch_predictions:
            epoch_poles = pred['poles']  # (K, 3)
            epoch_scores = pred['scores']  # (K,)

            # For each pole in this epoch, find its nearest cluster
            for pole_idx, (pole, score) in enumerate(zip(epoch_poles, epoch_scores)):
                # Find nearest cluster
                similarities = np.dot(cluster_centers, pole)  # Cosine similarity
                nearest_cluster = np.argmax(similarities)

                # This epoch votes for this cluster with weight = quality score
                cluster_votes[nearest_cluster] += score

        # 4. Return cluster centers ranked by vote count
        ranked_indices = np.argsort(cluster_votes)[::-1]  # Descending order

        return {
            'poles': cluster_centers[ranked_indices],
            'scores': cluster_votes[ranked_indices] / (cluster_votes.sum() + 1e-8),  # Normalize votes
        }

    elif method == 'best_quality':
        # Select epoch with highest quality score
        best_idx = np.argmax([pred['scores'].max() for pred in epoch_predictions])
        return epoch_predictions[best_idx]

    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def predict_epoch(
    model: torch.nn.Module,
    epoch_sample: Dict,
    device: str = 'cuda'
) -> Dict:
    """
    Predict poles for a single epoch.

    Args:
        model: Trained GeoHierK3Transformer model
        epoch_sample: Dict from SingleEpochDataset.__getitem__
        device: CUDA device

    Returns:
        Dict with:
            - poles: (K, 3) cartesian unit vectors
            - scores: (K,) quality scores (softmax normalized)
    """
    with torch.no_grad():
        tokens = epoch_sample['tokens'].unsqueeze(0).to(device)  # (1, W, T, F)
        mask = epoch_sample['mask'].unsqueeze(0).to(device)      # (1, W, T)

        poles, quality_logits = model(tokens, mask)  # (1, K, 3), (1, K)

        poles = poles[0].cpu().numpy()  # (K, 3)
        if quality_logits is not None:
            scores = torch.softmax(quality_logits[0], dim=0).cpu().numpy()  # (K,)
        else:
            scores = np.ones(poles.shape[0]) / poles.shape[0]  # Uniform

    return {
        'poles': poles,
        'scores': scores,
    }


def evaluate_with_aggregation(
    model: torch.nn.Module,
    dataset,  # SingleEpochDataset
    device: str = 'cuda',
    aggregation_method: str = 'average',
) -> Dict:
    """
    Evaluate model on validation dataset with asteroid-level aggregation.

    This function groups epochs by asteroid, predicts on each epoch,
    aggregates predictions, and computes metrics at the asteroid level.
    This ensures fair comparison with the baseline which trains on
    aggregated asteroids.

    Args:
        model: Trained GeoHierK3Transformer model
        dataset: SingleEpochDataset (validation split)
        device: CUDA device
        aggregation_method: How to aggregate epoch predictions ('average', 'vote', 'best_quality')

    Returns:
        Dict with:
            - n_asteroids: Number of unique asteroids
            - n_epochs_total: Total number of epochs
            - aggregation_method: Method used
            - oracle_error_mean_deg: Mean oracle error across asteroids
            - oracle_error_median_deg: Median oracle error
            - oracle_error_std_deg: Standard deviation of oracle error
            - selected_error_mean_deg: Mean error of quality-selected pole
            - selected_error_median_deg: Median selected error
            - naive0_error_mean_deg: Mean error of always selecting pole 0
            - naive0_error_median_deg: Median naive0 error
            - asteroid_results: List of per-asteroid result dicts

    Example:
        >>> from lc_pipeline.data import SingleEpochDataset
        >>> from lc_pipeline.models import GeoHierK3Transformer
        >>> dataset = SingleEpochDataset(csv_dir, spin_root, val_ids)
        >>> results = evaluate_with_aggregation(model, dataset, device='cuda')
        >>> print(f"Oracle mean: {results['oracle_error_mean_deg']:.2f}")
    """
    model.eval()
    logger.info(f"Evaluating {len(dataset)} epochs...")

    # Group epochs by asteroid
    asteroid_epochs = defaultdict(list)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        asteroid_id = sample['asteroid_id']
        asteroid_epochs[asteroid_id].append((idx, sample))

    logger.info(f"Found {len(asteroid_epochs)} unique asteroids")

    # Predict and aggregate per asteroid
    asteroid_results = []

    for asteroid_id, epochs in asteroid_epochs.items():
        # Predict on each epoch
        epoch_predictions = []
        ground_truth = None

        for idx, sample in epochs:
            pred = predict_epoch(model, sample, device)
            epoch_predictions.append(pred)

            # Ground truth is same for all epochs
            if ground_truth is None:
                ground_truth = sample['solutions'][0].numpy()  # (S, 3)

        # Aggregate predictions
        agg_pred = aggregate_asteroid_predictions(epoch_predictions, method=aggregation_method)

        # Compute oracle error (best match across K poles)
        oracle_errors = []
        for pole_vec in agg_pred['poles']:
            error = best_solution_error(pole_vec, ground_truth)
            oracle_errors.append(error)

        oracle_error = min(oracle_errors)
        best_k = int(np.argmin(oracle_errors))

        # Compute selected error (quality head choice)
        selected_k = int(np.argmax(agg_pred['scores']))
        selected_error = oracle_errors[selected_k]

        # Naive baseline (always pole 0)
        naive0_error = oracle_errors[0]

        asteroid_results.append({
            'asteroid_id': asteroid_id,
            'n_epochs': len(epochs),
            'oracle_error_deg': float(oracle_error),
            'selected_error_deg': float(selected_error),
            'naive0_error_deg': float(naive0_error),
            'best_k': best_k,
            'selected_k': selected_k,
            'all_errors': [float(e) for e in oracle_errors],
        })

    # Compute summary statistics
    oracle_errors = [r['oracle_error_deg'] for r in asteroid_results]
    selected_errors = [r['selected_error_deg'] for r in asteroid_results]
    naive0_errors = [r['naive0_error_deg'] for r in asteroid_results]

    summary = {
        'n_asteroids': len(asteroid_results),
        'n_epochs_total': len(dataset),
        'aggregation_method': aggregation_method,
        'oracle_error_mean_deg': float(np.mean(oracle_errors)),
        'oracle_error_median_deg': float(np.median(oracle_errors)),
        'oracle_error_std_deg': float(np.std(oracle_errors)),
        'selected_error_mean_deg': float(np.mean(selected_errors)),
        'selected_error_median_deg': float(np.median(selected_errors)),
        'naive0_error_mean_deg': float(np.mean(naive0_errors)),
        'naive0_error_median_deg': float(np.median(naive0_errors)),
        'asteroid_results': asteroid_results,
    }

    return summary
