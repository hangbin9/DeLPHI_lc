#!/usr/bin/env python3
"""
Build pole_synth_multiepoch_v1 dataset with high geometry diversity.

This dataset explicitly optimizes for pole identifiability through:
- Fibonacci lattice for uniform sphere sampling
- Greedy sun direction selection with min separation enforcement
- Phase-controlled observer directions via Rodrigues rotation
- Per-object rejection sampling with informativeness gates

Token format (28-dimensional):
    [0:3]   sun_u (sun direction, unit vector)
    [3:6]   obs_u (observer direction, unit vector)
    [6]     brightness (pole-dependent photometry)
    [7]     log_magerr (log magnitude error)
    [8:28]  padding (zeros)

Usage:
    python scripts/build_pole_synth_multiepoch_v1_highgeo.py \\
        --out datasets/pole_synth_multiepoch_v1 \\
        --n-objects 500 \\
        --epochs 5 \\
        --seed 123
"""

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np


def generate_fibonacci_points(n: int) -> np.ndarray:
    """
    Generate n approximately uniformly distributed points on the unit sphere
    using the Fibonacci lattice method.

    Args:
        n: Number of points

    Returns:
        Array of shape (n, 3) with unit vectors
    """
    indices = np.arange(n, dtype=np.float64)
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    theta = 2 * np.pi * indices / phi
    z = 1 - (2 * indices + 1) / n

    r = np.sqrt(1 - z ** 2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return np.stack([x, y, z], axis=-1)


def rodrigues_rotation(v: np.ndarray, k: np.ndarray, theta: float) -> np.ndarray:
    """
    Rotate vector v around axis k by angle theta using Rodrigues' rotation formula.

    Args:
        v: Vector to rotate, shape (3,)
        k: Rotation axis, unit vector, shape (3,)
        theta: Rotation angle in radians

    Returns:
        Rotated vector, shape (3,)
    """
    k = k / np.linalg.norm(k)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    v_rot = v * cos_t + np.cross(k, v) * sin_t + k * np.dot(k, v) * (1 - cos_t)
    return v_rot


@dataclass
class V1GenerationConfig:
    """Configuration for V1 dataset generation."""

    n_objects: int = 500
    epochs_per_object: int = 5
    tokens_per_epoch: int = 64
    token_dim: int = 28

    # Informativeness gates
    gate_geometry_diversity: float = 0.60
    gate_phase_coverage: float = 0.70
    gate_snr: float = 0.70

    # Geometry constraints
    min_sun_separation_deg: float = 60.0
    target_phase_angles_deg: List[float] = field(
        default_factory=lambda: [10.0, 40.0, 80.0, 120.0, 160.0]
    )
    phase_jitter_deg: float = 10.0

    # Photometry parameters
    noise_std: float = 0.05
    rotation_amplitude: float = 0.3

    # Sampling
    max_rejection_tries: int = 200
    seed: int = 42

    def __post_init__(self):
        if isinstance(self.target_phase_angles_deg, tuple):
            self.target_phase_angles_deg = list(self.target_phase_angles_deg)


class GeometrySampler:
    """Sample sun and observer directions with geometry constraints."""

    def __init__(self, config: V1GenerationConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.fib_points = generate_fibonacci_points(4096)

    def sample_sun_directions(self, n_epochs: int) -> np.ndarray:
        """
        Sample n_epochs sun directions with minimum angular separation.

        Returns:
            Array of shape (n_epochs, 3) with unit vectors
        """
        min_sep_rad = np.deg2rad(self.config.min_sun_separation_deg)

        # Start with random first direction
        first_idx = self.rng.integers(0, len(self.fib_points))
        sun_dirs = [self.fib_points[first_idx]]

        # Greedy selection for remaining epochs
        for _ in range(n_epochs - 1):
            # Compute distances to all existing sun directions
            candidates = self.fib_points.copy()
            valid_mask = np.ones(len(candidates), dtype=bool)

            for existing in sun_dirs:
                cos_sim = np.dot(candidates, existing)
                angles = np.arccos(np.clip(cos_sim, -1, 1))
                valid_mask &= (angles >= min_sep_rad)

            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                # Fallback: use most distant point
                min_dists = np.inf * np.ones(len(candidates))
                for existing in sun_dirs:
                    cos_sim = np.dot(candidates, existing)
                    angles = np.arccos(np.clip(cos_sim, -1, 1))
                    min_dists = np.minimum(min_dists, angles)
                best_idx = np.argmax(min_dists)
            else:
                best_idx = self.rng.choice(valid_indices)

            sun_dirs.append(self.fib_points[best_idx])

        return np.array(sun_dirs)

    def sample_observer_for_phase(
        self, sun_dir: np.ndarray, target_phase_deg: float
    ) -> np.ndarray:
        """
        Sample observer direction to achieve target phase angle with sun.

        Phase angle = angle between sun direction and observer direction.

        Args:
            sun_dir: Sun direction, unit vector, shape (3,)
            target_phase_deg: Target phase angle in degrees

        Returns:
            Observer direction, unit vector, shape (3,)
        """
        # Add jitter to target phase
        jitter = self.rng.uniform(
            -self.config.phase_jitter_deg, self.config.phase_jitter_deg
        )
        phase_rad = np.deg2rad(target_phase_deg + jitter)

        # Find a perpendicular axis for rotation
        if abs(sun_dir[2]) < 0.9:
            perp = np.cross(sun_dir, np.array([0, 0, 1]))
        else:
            perp = np.cross(sun_dir, np.array([1, 0, 0]))
        perp = perp / np.linalg.norm(perp)

        # Random rotation around sun direction
        random_angle = self.rng.uniform(0, 2 * np.pi)
        perp = rodrigues_rotation(perp, sun_dir, random_angle)

        # Rotate sun direction by phase angle around perpendicular axis
        obs_dir = rodrigues_rotation(sun_dir, perp, phase_rad)
        obs_dir = obs_dir / np.linalg.norm(obs_dir)

        return obs_dir

    def sample_epoch_geometry(self, epoch_idx: int, sun_dir: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample geometry for one epoch.

        Returns:
            Tuple of (sun_directions, observer_directions) each shape (T, 3)
        """
        T = self.config.tokens_per_epoch

        # Get target phase for this epoch
        target_phase = self.config.target_phase_angles_deg[
            epoch_idx % len(self.config.target_phase_angles_deg)
        ]

        sun_dirs = np.tile(sun_dir, (T, 1))  # Same sun for all tokens in epoch
        obs_dirs = np.zeros((T, 3))

        for t in range(T):
            obs_dirs[t] = self.sample_observer_for_phase(sun_dir, target_phase)

        return sun_dirs, obs_dirs


class PhotometrySimulator:
    """Simulate pole-dependent photometry."""

    def __init__(self, config: V1GenerationConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

    def simulate_brightness(
        self,
        pole: np.ndarray,
        sun_dirs: np.ndarray,
        obs_dirs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate brightness observations for given geometry.

        Args:
            pole: Pole direction, unit vector, shape (3,)
            sun_dirs: Sun directions, shape (T, 3)
            obs_dirs: Observer directions, shape (T, 3)

        Returns:
            Tuple of (brightness, log_magerr), each shape (T,)
        """
        T = len(sun_dirs)

        # Base brightness: Lambert illumination from sun onto pole-facing surface
        cos_sun_pole = np.dot(sun_dirs, pole)  # (T,)
        base_brightness = np.maximum(0, cos_sun_pole)

        # Add rotation-induced variation
        rotation_phase = np.linspace(0, 2 * np.pi, T)
        rotation_term = self.config.rotation_amplitude * np.sin(rotation_phase)

        brightness = base_brightness + rotation_term

        # Add noise
        noise = self.rng.normal(0, self.config.noise_std, T)
        brightness = brightness + noise

        # Magnitude error (log scale)
        magerr = np.abs(self.rng.normal(0.02, 0.01, T))
        log_magerr = np.log(magerr + 1e-6)

        return brightness, log_magerr


class InformativenessEvaluator:
    """Evaluate informativeness of generated samples."""

    def __init__(self, config: V1GenerationConfig):
        self.config = config

    def compute_metrics(
        self,
        sun_dirs_all: np.ndarray,
        obs_dirs_all: np.ndarray,
        brightness_all: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute informativeness metrics for a sample.

        Args:
            sun_dirs_all: Sun directions, shape (E, T, 3)
            obs_dirs_all: Observer directions, shape (E, T, 3)
            brightness_all: Brightness values, shape (E, T)

        Returns:
            Dict with metric scores in [0, 1]
        """
        E, T, _ = sun_dirs_all.shape

        # 1. Geometry diversity: spread of sun directions across epochs
        epoch_sun_means = sun_dirs_all.mean(axis=1)  # (E, 3)
        pairwise_angles = []
        for i in range(E):
            for j in range(i + 1, E):
                cos_sim = np.dot(epoch_sun_means[i], epoch_sun_means[j])
                angle = np.arccos(np.clip(cos_sim, -1, 1))
                pairwise_angles.append(angle)

        mean_sep = np.mean(pairwise_angles) if pairwise_angles else 0
        geometry_diversity = min(1.0, mean_sep / np.deg2rad(90))  # Normalize to [0, 1]

        # 2. Phase coverage: spread of phase angles
        phase_angles = []
        for e in range(E):
            sun_mean = sun_dirs_all[e].mean(axis=0)
            obs_mean = obs_dirs_all[e].mean(axis=0)
            cos_phase = np.dot(sun_mean, obs_mean) / (
                np.linalg.norm(sun_mean) * np.linalg.norm(obs_mean) + 1e-8
            )
            phase = np.arccos(np.clip(cos_phase, -1, 1))
            phase_angles.append(np.rad2deg(phase))

        phase_range = max(phase_angles) - min(phase_angles) if phase_angles else 0
        phase_coverage = min(1.0, phase_range / 150.0)  # Max expected range ~150°

        # 3. SNR: signal-to-noise ratio of brightness
        brightness_flat = brightness_all.flatten()
        signal = np.std(brightness_flat)
        noise = self.config.noise_std
        snr = signal / (noise + 1e-8)
        snr_score = min(1.0, snr / 5.0)  # Normalize

        # 4. Observation quality (variance in directions)
        obs_variance = np.var(obs_dirs_all)
        obs_quality = min(1.0, obs_variance * 10)

        # 5. Matrix rank proxy (using PCA on geometry)
        geometry_flat = np.concatenate([sun_dirs_all.reshape(-1, 3), obs_dirs_all.reshape(-1, 3)], axis=1)
        try:
            _, s, _ = np.linalg.svd(geometry_flat, full_matrices=False)
            rank_ratio = np.sum(s > 0.01 * s[0]) / len(s)
        except:
            rank_ratio = 0.5

        return {
            'geometry_diversity': float(geometry_diversity),
            'phase_coverage': float(phase_coverage),
            'snr': float(snr_score),
            'observation_quality': float(obs_quality),
            'matrix_rank': float(rank_ratio),
        }

    def passes_gates(self, metrics: Dict[str, float]) -> bool:
        """Check if metrics pass all informativeness gates."""
        return (
            metrics['geometry_diversity'] >= self.config.gate_geometry_diversity
            and metrics['phase_coverage'] >= self.config.gate_phase_coverage
            and metrics['snr'] >= self.config.gate_snr
        )


class V1DatasetBuilder:
    """Build V1 dataset with rejection sampling."""

    def __init__(self, config: V1GenerationConfig, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.rng = np.random.default_rng(config.seed)

        self.geometry_sampler = GeometrySampler(config, self.rng)
        self.photometry_sim = PhotometrySimulator(config, self.rng)
        self.evaluator = InformativenessEvaluator(config)

    def sample_random_pole(self) -> np.ndarray:
        """Sample a random pole direction uniformly on the sphere."""
        # Use Fibonacci lattice point
        idx = self.rng.integers(0, len(self.geometry_sampler.fib_points))
        return self.geometry_sampler.fib_points[idx].copy()

    def generate_single_object(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Generate data for a single object with rejection sampling.

        Returns:
            Tuple of (tokens, mask, pole, metrics)
            - tokens: shape (E, T, D)
            - mask: shape (E, T)
            - pole: shape (3,)
            - metrics: informativeness metrics dict
        """
        E = self.config.epochs_per_object
        T = self.config.tokens_per_epoch
        D = self.config.token_dim

        for attempt in range(self.config.max_rejection_tries):
            # Sample pole
            pole = self.sample_random_pole()

            # Sample sun directions for all epochs
            sun_dirs_epochs = self.geometry_sampler.sample_sun_directions(E)

            # Generate each epoch
            tokens = np.zeros((E, T, D), dtype=np.float32)
            sun_dirs_all = np.zeros((E, T, 3))
            obs_dirs_all = np.zeros((E, T, 3))
            brightness_all = np.zeros((E, T))

            for e in range(E):
                sun_dirs, obs_dirs = self.geometry_sampler.sample_epoch_geometry(
                    e, sun_dirs_epochs[e]
                )
                brightness, log_magerr = self.photometry_sim.simulate_brightness(
                    pole, sun_dirs, obs_dirs
                )

                # Pack into tokens
                tokens[e, :, 0:3] = sun_dirs
                tokens[e, :, 3:6] = obs_dirs
                tokens[e, :, 6] = brightness
                tokens[e, :, 7] = log_magerr
                # [8:28] remains zeros (padding)

                sun_dirs_all[e] = sun_dirs
                obs_dirs_all[e] = obs_dirs
                brightness_all[e] = brightness

            # Evaluate informativeness
            metrics = self.evaluator.compute_metrics(
                sun_dirs_all, obs_dirs_all, brightness_all
            )

            if self.evaluator.passes_gates(metrics):
                mask = np.ones((E, T), dtype=bool)
                metrics['attempts'] = attempt + 1
                return tokens, mask, pole, metrics

        # Fallback: return last attempt even if it doesn't pass
        mask = np.ones((E, T), dtype=bool)
        metrics['attempts'] = self.config.max_rejection_tries
        metrics['passed_gates'] = False
        return tokens, mask, pole, metrics

    def build_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the complete dataset.

        Returns:
            Tuple of (all_tokens, all_masks)
            - all_tokens: shape (N, E, T, D)
            - all_masks: shape (N, E, T)
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        N = self.config.n_objects
        E = self.config.epochs_per_object
        T = self.config.tokens_per_epoch
        D = self.config.token_dim

        all_tokens = np.zeros((N, E, T, D), dtype=np.float32)
        all_masks = np.zeros((N, E, T), dtype=bool)
        all_poles = np.zeros((N, 3), dtype=np.float32)
        all_metrics = []

        for i in range(N):
            tokens, mask, pole, metrics = self.generate_single_object()
            all_tokens[i] = tokens
            all_masks[i] = mask
            all_poles[i] = pole
            all_metrics.append(metrics)

            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{N} objects")

        # Save dataset
        np.savez_compressed(
            self.output_dir / "shard_000.npz",
            tokens=all_tokens,
            masks=all_masks,
            poles=all_poles,
        )

        # Build feature names
        feature_names = (
            ["sun_u_x", "sun_u_y", "sun_u_z"]
            + ["obs_u_x", "obs_u_y", "obs_u_z"]
            + ["brightness", "log_magerr"]
            + [f"pad_{i}" for i in range(D - 8)]
        )

        # Save manifest
        manifest = {
            "version": "pole_synth_multiepoch_v1_highgeo",
            "feature_names": feature_names,
            "token_dim": D,
            "n_models": N,
            "n_epochs": E,
            "n_obs_per_epoch": T,
            "generation_config": asdict(self.config),
            "statistics": {
                "mean_attempts": float(np.mean([m.get('attempts', 1) for m in all_metrics])),
                "pass_rate": float(np.mean([m.get('passed_gates', True) for m in all_metrics])),
            },
        }

        with open(self.output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Saved dataset to {self.output_dir}")
        print(f"  - {N} objects, {E} epochs, {T} tokens/epoch")
        print(f"  - Mean attempts: {manifest['statistics']['mean_attempts']:.1f}")

        return all_tokens, all_masks


def main():
    parser = argparse.ArgumentParser(description="Build V1 dataset with high geometry diversity")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--n-objects", type=int, default=500, help="Number of objects")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per object")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = V1GenerationConfig(
        n_objects=args.n_objects,
        epochs_per_object=args.epochs,
        seed=args.seed,
    )

    print("Building V1 dataset with config:")
    print(f"  n_objects: {config.n_objects}")
    print(f"  epochs_per_object: {config.epochs_per_object}")
    print(f"  tokens_per_epoch: {config.tokens_per_epoch}")
    print(f"  seed: {config.seed}")

    builder = V1DatasetBuilder(config, args.out)
    builder.build_dataset()


if __name__ == "__main__":
    main()
